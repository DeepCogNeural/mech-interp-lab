"""Preregistered H9: frozen Q_N head outputs transferred into Q_F.

Run only after the complete H7 aggregate. No discovery or head reselection.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from stage2 import (REPO, effects, extra_controls, gate, head_patches,
                    intervention, read, rows_for, sha)
from patch import Harness
from stimuli import HELDOUT_SEEDS, dump


def verify(args):
    gate(args.expected_commit)
    freeze_path = Path(args.freeze).resolve()
    raw = freeze_path.read_bytes()
    committed = subprocess.check_output(
        ["git", "show", f"{args.expected_commit}:{freeze_path.relative_to(REPO)}"],
        cwd=REPO,
    )
    assert raw == committed, "FREEZE must be byte-identical to the committed file"
    frozen = json.loads(raw.decode().split("```json\n", 1)[1].split("\n```", 1)[0])
    h7_dir = Path(args.h7_output)
    sidecar = read(h7_dir / "freeze.json")
    assert sha(freeze_path) == sidecar["freeze_sha256"]
    assert all(sidecar[key] == value for key, value in frozen.items())
    assert frozen["task"] == "numeric"
    assert frozen["heldout_seeds"] == list(HELDOUT_SEEDS)
    assert len(frozen["random_sets"]) == 100
    assert all(sha(HERE / name) == digest
               for name, digest in frozen["source_sha256"].items())
    assert sha(HERE / "DESIGN.md") == frozen["design_sha256"]
    assert sha(Path(args.stimuli) / "manifest.json") == frozen["stimuli_manifest_sha256"]
    assert sha(frozen["mean_path"]) == frozen["mean_sha256"]
    summary_path = h7_dir / "heldout_summary.json"
    assert "H7_sufficiency_passes" in read(summary_path)
    for seed in HELDOUT_SEEDS:
        run = read(h7_dir / f"heldout_{seed}.json")
        assert run.get("complete") and len(run["sets"]) == 101
        assert run["freeze_sha256"] == sidecar["freeze_sha256"]
    return frozen, sidecar["freeze_sha256"], sha(summary_path)


def run_seed(args, frozen, freeze_hash, h7_hash):
    assert args.seed in HELDOUT_SEEDS
    started = time.perf_counter()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    outdir = Path(args.output)
    path = outdir / f"qf_{args.seed}.json"
    assert not path.exists(), "Never overwrite a partial or completed seed"
    with (outdir / f"qf_{args.seed}.started").open("x") as marker:
        marker.write(f"{started_utc}\nOne attempt; no silent retry.\n")
    h = Harness("Qwen/Qwen3-4B")
    rows = rows_for(args, args.seed)
    layers = sorted({layer for layer, _ in frozen["heads"]})
    captures = [(layer, "head", "prefill") for layer in layers]
    data = {}
    for query in ["Q_N", "Q_F"]:
        selected_rows = [row for row in rows if row["query_type"] == query]
        assert len(selected_rows) == 64
        outputs, cache = h.measure(selected_rows, captures=captures)
        batch, pos = h.batch(selected_rows)
        data[query] = dict(rows=selected_rows, outputs=outputs, cache=cache,
                           batch=batch, pos=pos)
    base, donor = data["Q_F"], data["Q_N"]
    assert all((b["item"], b["N"], b["T"]) == (d["item"], d["N"], d["T"])
               for b, d in zip(base["rows"], donor["rows"]))
    out = dict(seed=args.seed, freeze_sha256=freeze_hash,
               h7_summary_sha256=h7_hash, source_commit=args.expected_commit,
               started_utc=started_utc, clean_QN=donor["outputs"],
               clean_QF=base["outputs"], sets=[], extra_controls=[])
    dump(path, out)
    for index, heads in enumerate([frozen["heads"]] + frozen["random_sets"]):
        values = intervention(h, base, head_patches(donor["cache"], heads))
        out["sets"].append(dict(index=index,
            label="selected" if index == 0 else "layer-matched random",
            direction="Q_F->N", outputs=values, **effects(base, values, "N")))
        dump(path, out)
        print("H9", args.seed, "set", index, flush=True)
    out["extra_controls"] = extra_controls(
        h, base, donor, frozen["heads"], args.seed, "N")
    out["complete"] = True
    out["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out["wall_seconds"] = time.perf_counter() - started
    dump(path, out)


def aggregate(args, frozen, freeze_hash, h7_hash):
    records = []
    for seed in HELDOUT_SEEDS:
        run = read(Path(args.output) / f"qf_{seed}.json")
        assert run.get("complete") and len(run["sets"]) == 101
        assert run["freeze_sha256"] == freeze_hash
        assert run["h7_summary_sha256"] == h7_hash
        assert [entry["index"] for entry in run["sets"]] == list(range(101))
        for entry in run["sets"] + run["extra_controls"]:
            values = entry["outputs"]
            assert len(values) == 64
            assert entry["flip_rate"] == sum(v["answer"] == v["N"] for v in values) / 64
            assert entry["validity"] == sum(v["answer"] in [v["N"], v["T"]] for v in values) / 64
        selected = run["sets"][0]
        randoms = run["sets"][1:]
        maximum = max(entry["flip_rate"] for entry in randoms)
        records.append(dict(seed=seed,
            **{key: value for key, value in selected.items() if key != "outputs"},
            random_max=maximum, above_all_random=selected["flip_rate"] > maximum,
            random_validity_min=min(entry["validity"] for entry in randoms),
            random_validity_max=max(entry["validity"] for entry in randoms),
            clean_N=sum(v["answer"] == v["N"] for v in run["clean_QF"]),
            clean_T=sum(v["answer"] == v["T"] for v in run["clean_QF"]),
            clean_other=sum(v["answer"] not in [v["N"], v["T"]] for v in run["clean_QF"]),
            extra_controls=[{key: value for key, value in entry.items()
                             if key not in ["outputs", "donor_order"]}
                            for entry in run["extra_controls"]],
            wall_seconds=run["wall_seconds"]))
    mean = sum(entry["flip_rate"] for entry in records) / len(records)
    out = dict(freeze_sha256=freeze_hash, h7_summary_sha256=h7_hash,
               source_commit=args.expected_commit, per_seed=records,
               mean_flip_to_N=mean,
               passes=mean >= .5 and all(entry["above_all_random"] for entry in records),
               rule="Mean flip to N >= 0.50; every seed strictly above its 100 random sets",
               validity_note="Reported separately; H9 specifies no validity pass threshold")
    dump(Path(args.output) / "qf_summary.json", out)
    print(json.dumps(out, indent=2))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["run", "aggregate"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--h7-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen, freeze_hash, h7_hash = verify(args)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.phase == "run":
        run_seed(args, frozen, freeze_hash, h7_hash)
    else:
        aggregate(args, frozen, freeze_hash, h7_hash)


if __name__ == "__main__":
    main()
