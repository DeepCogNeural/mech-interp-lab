"""H10b: reuse the frozen H10 run, changing only full-answer scoring."""
import argparse
import datetime
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
import subprocess
import torch
import h10_transfer as h10
from scoring import parse_integer

PREREG = "369373d9c64f99f93b594480fc8c624fc7cdb78d"
SOURCE = Path('workspace') / "./exp07/stage3/h10"
MANIFEST_HASH = "4b49b2aaaa4716d203880ef48a5fa8a03c9a26685d3263c1635d1bd0d3019eb3"
# A malformed comma sequence is not silently reduced to a candidate prefix.
NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)*")
VALID_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def parse_number(text):
    match = NUMBER.search(text)
    if match is None or VALID_NUMBER.fullmatch(match[0]) is None:
        return None
    return str(Decimal(match[0].replace(",", "")))


def equal(value, target):
    return value is not None and Decimal(str(value)) == Decimal(str(target).replace(",", ""))


def parsed(v, parser):
    return v["parsed_number"] if parser == "h10b" else v["frozen_unique_integer"]


class NumberHarness(h10.Harness):
    def greedy(self, *args, **kwargs):
        outputs = super().greedy(*args, **kwargs)
        for out in outputs:
            out["frozen_unique_integer"] = out["answer"]
            out["parsed_number"] = parse_number(out["text"])
        return outputs


def one_metric(base, values, target, donor_rows=None, order=None,
               parser="h10b", denominator_parser=None):
    denominator_parser = denominator_parser or parser
    gold = base["rows"][0]["query_type"][-1]
    eligible = [i for i, v in enumerate(base["outputs"])
                if equal(parsed(v, denominator_parser), v[gold])]
    n = len(eligible)
    flip = sum(equal(parsed(values[i], parser), values[i][target]) for i in eligible)
    valid = sum(any(equal(parsed(values[i], parser), values[i][k]) for k in ["N", "T"])
                for i in eligible)
    result = dict(denominator=n, flip_count=flip, flip_rate=flip/n if n else None,
                  valid_count=valid, validity=valid/n if n else None,
                  all_item_validity=sum(any(equal(parsed(v, parser), v[k]) for k in ["N", "T"])
                                        for v in values)/len(values),
                  mean_delta_d=sum(v["d"]-b["d"] for v,b in zip(values,base["outputs"]))/len(values))
    if donor_rows is not None:
        injected = sum(any(equal(parsed(values[i],parser), donor_rows[order[i]][k])
                           for k in ["N","T"]) for i in eligible)
        result.update(donor_number_injection_count=injected,
                      donor_number_injection_rate=injected/n if n else None)
    return result


def metric(base, values, target, donor_rows=None, order=None):
    result = one_metric(base,values,target,donor_rows,order)
    result["frozen_unique_integer"] = one_metric(base,values,target,donor_rows,order,parser="frozen")
    result["frozen_on_h10b_denominator"] = one_metric(base,values,target,donor_rows,order,
                                                     parser="frozen",denominator_parser="h10b")
    result["outputs"] = values
    if order is not None:
        result["donor_order"] = order
    return result


def gate(args):
    root = Path(args.output)
    runs = [h10.read(root/f"clean_{g}.json") for g in range(4)]
    assert all(r.get("complete") and r["manifest_sha256"]==MANIFEST_HASH for r in runs)
    by_parser = {}
    for parser in ["h10b","frozen"]:
        entries = []
        for query in ["Q_N","Q_T"]:
            values = [v for run in runs for v in run["outputs"][query]]
            correct = sum(equal(parsed(v,parser),v[query[-1]]) for v in values)
            valid = sum(any(equal(parsed(v,parser),v[k]) for k in ["N","T"]) for v in values)
            entries.append(dict(query=query,correct=correct,total=len(values),accuracy=correct/len(values),
                                valid_count=valid,validity=valid/len(values),
                                first_generated_is_digit=sum(v["first_is_digit"] for v in values)))
        by_parser[parser] = entries
    passed = all(v["accuracy"] >= .8 for v in by_parser["h10b"])
    result = dict(manifest_sha256=MANIFEST_HASH,query_results=by_parser["h10b"],
                  frozen_unique_integer=by_parser["frozen"],passed=passed,
                  pooled_accuracy=sum(v["correct"] for v in by_parser["h10b"])/512,
                  verdict="eligible for H10b patch test" if passed else "not testable under H10b parser",
                  clean_file_sha256={str(g):h10.sha(root/f"clean_{g}.json") for g in range(4)},
                  computed_utc=h10.now())
    h10.dump(root/"behaviour_gate.json",result)
    print(json.dumps(result,indent=2))


def verify(args):
    frozen, freeze_hash, h7_hash = h10.verify(args)
    path = h10.HERE/"ADDENDUM_H10b.md"
    assert path.read_bytes() == subprocess.check_output(
        ["git","show",f"{PREREG}:{path.relative_to(h10.REPO)}"],cwd=h10.REPO)
    subprocess.run(["git","merge-base","--is-ancestor",PREREG,args.expected_commit],cwd=h10.REPO,check=True)
    assert h10.sha(SOURCE/"manifest.json")==MANIFEST_HASH
    for g in h10.read(SOURCE/"manifest.json")["groups"]:
        assert h10.sha(SOURCE/g["filename"])==g["sha256"]
    assert Path(args.output).resolve() != SOURCE.resolve()
    return frozen,freeze_hash,h7_hash


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase",choices=["prepare","clean","gate","run","aggregate"])
    p.add_argument("--group",type=int,choices=range(4))
    for name in ["freeze","expected-commit","stimuli","h7-output","output"]:
        p.add_argument("--"+name,required=True)
    args = p.parse_args()
    frozen,freeze_hash,h7_hash = verify(args)
    root = Path(args.output); root.mkdir(parents=True,exist_ok=True)
    h10.Harness = NumberHarness
    h10.metric = metric
    if args.phase == "prepare":
        assert not (root/"manifest.json").exists()
        for name in ["manifest.json","manifest.sha256"]+[f"group_{g}.json" for g in range(4)]:
            shutil.copyfile(SOURCE/name,root/name)
        h10.dump(root/"protocol.json",dict(prereg_commit=PREREG,source_commit=args.expected_commit,
            manifest_sha256=MANIFEST_HASH,freeze_sha256=freeze_hash,prepared_utc=h10.now(),
            parser="first numeric sequence with decimal arithmetic; frozen unique integer retained"))
    elif args.phase == "clean": h10.clean(args,frozen,freeze_hash,h7_hash)
    elif args.phase == "gate": gate(args)
    elif args.phase == "run": h10.run(args,frozen,freeze_hash,h7_hash)
    else:
        h10.aggregate(args)
        (root/"h10_summary.json").rename(root/"h10b_summary.json")

if __name__ == "__main__": main()
