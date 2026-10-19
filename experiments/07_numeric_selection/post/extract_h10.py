"""Deterministic train-only, verbatim prose extraction for H10; no model calls."""
import argparse, collections, hashlib, json, re, time
from decimal import Decimal
from pathlib import Path

SEED = 20261018
YEAR = r"(?:19|20)\d{2}"
MONEY = re.compile(r"\$\s*([+-]?\d[\d,.]*)")
PAIR = re.compile(r"\$\s*(?P<value>(?:0|[1-9]\d*(?:,\d{3})*)(?:\.\d+)?)\s+(?P<unit>million|billion|thousand)\s+(?:in|for|during)\s+(?:the\s+)?(?:fiscal\s+)?(?:year\s+)?(?P<year>"+YEAR+r")\b", re.I)
ITEM = re.compile(r"\b(?:revenues?|sales|income|earnings|profits?|expenses?|costs?|cash|capital expenditure|capital spending|depreciation|amortization|interest|dividends|assets|liabilities|debt|backlog|tax|contributions|pension|benefits)\b", re.I)
PRED = re.compile(r"\s+(?:was|were|totaled|totalled|amounted to|increased|decreased|rose|fell)\b", re.I)

def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()

def dump(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True)+"\n").encode()
    Path(path).write_bytes(raw)
    return sha_bytes(raw)

def records(path, allow_partial=False):
    decoder = json.JSONDecoder()
    with Path(path).open() as f:
        buffer, end = "", False
        while True:
            if not end:
                chunk = f.read(4 * 1024 * 1024)
                if not chunk: end = True
                buffer += chunk
            buffer = buffer.lstrip()
            while buffer.startswith(("[", ",")):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]") or (not buffer and end):
                return
            try:
                value, pos = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if end:
                    if allow_partial: return
                    raise
                continue
            yield value
            buffer = buffer[pos:]
            # Consume already buffered records before another chunk.
            while True:
                buffer = buffer.lstrip()
                if buffer.startswith(","): buffer = buffer[1:].lstrip()
                if buffer.startswith("]"): return
                try: value, pos = decoder.raw_decode(buffer)
                except json.JSONDecodeError: break
                yield value
                buffer = buffer[pos:]

def company_name(raw):
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw[:15000].splitlines()]
    for idx, line in enumerate(lines):
        if re.search(r"exact name of (?:the )?registrant", line, re.I):
            prior = [x.strip("# ").strip() for x in lines[:idx] if x.strip("# ").strip()]
            if prior:
                prefix = line[:re.search(r"exact name of (?:the )?registrant", line, re.I).start()].strip("# (")
                options = ([prefix] if prefix else []) + list(reversed(prior[-5:]))
                for name in options:
                    if len(name) <= 180 and not re.search(r"\b(?:transition|exchange|report|section|pursuant|commission|file number|united states|mark one)\b", name, re.I):
                        if re.search(r"\b(?:inc|corporation|company|corp|ltd|limited|plc|group|holdings)\b", name, re.I):
                            return name
    return None

def candidates(raw, company, doc):
    normalized = re.sub(r"\s+", " ", raw).strip()
    number = r"(?:0|[1-9]\d*(?:,\d{3})*)(?:\.\d+)?"
    respective = re.compile(
        r"^(?P<item>.+?)\s+(?:was|were|totaled|totalled|amounted to)\s+(?:approximately\s+|about\s+)?"
        r"\$\s*(?P<a>"+number+r")\s*(?:(?P<unitA>million|billion|thousand)\s*)?,?\s+and\s+"
        r"\$\s*(?P<b>"+number+r")\s+(?P<unitB>million|billion|thousand)\s+"
        r"(?:in|for|during)\s+(?:the\s+)?(?:fiscal\s+)?(?:years?\s+)?"
        r"(?P<yearA>"+YEAR+r")\s+and\s+(?:fiscal\s+)?(?P<yearB>"+YEAR+r"),?\s+respectively\b", re.I)
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        if not sentence.endswith((".", "!", "?")) or "|" in sentence or "#" in sentence:
            continue
        money = list(MONEY.finditer(sentence))
        years = list(re.finditer(r"\b"+YEAR+r"\b", sentence))
        if len(money) != 2 or len(years) != 2:
            continue
        pairs = list(PAIR.finditer(sentence))
        if len(pairs) == 2 and {p.start() for p in pairs} == {p.start() for p in money}:
            if pairs[0]["unit"].lower() != pairs[1]["unit"].lower():
                continue
            values = [p["value"] for p in pairs]
            fiscal = [int(p["year"]) for p in pairs]
            prefix = sentence[:pairs[0].start()]
            predicate = PRED.search(prefix)
            if predicate is None:
                continue
            connector = prefix[predicate.start():].strip()
            if re.fullmatch(r"(?:(?:was|were|totaled|totalled)|amounted to|(?:increased|decreased|rose|fell) to)\s*(?:(?:approximately|about)\s*)?", connector, re.I) is None:
                continue
            item = prefix[:predicate.start()].strip()
            item = re.sub(r"^\([a-z]\)\s+", "", item)
            bridge = sentence[pairs[0].end():pairs[1].start()].strip()
            bridge = re.sub(r"^[,;]\s*", "", bridge)
            bridge = re.sub(r"^(?:as )?compared (?:with|to)(?:\s+|$)|^(?:and|versus|from|up from|down from)(?:\s+|$)", "", bridge, flags=re.I)
            bridge = re.sub(r"^(?:approximately|about)\s*", "", bridge, flags=re.I).strip()
            if bridge and bridge.casefold() not in {item.casefold()+" of", item.casefold()+" was", item.casefold()+" were"}:
                continue
            spans = [[p.start("value"), p.end("value")] for p in pairs]
            unit = pairs[0]["unit"].lower()
            rule = "two explicit amount-unit-in/for/during-year pairs"
        else:
            match = respective.search(sentence)
            if match is None:
                continue
            if match["unitA"] and match["unitA"].lower() != match["unitB"].lower():
                continue
            item = match["item"].strip()
            values = [match["a"], match["b"]]
            fiscal = [int(match["yearA"]), int(match["yearB"])]
            spans = [[match.start(k), match.end(k)] for k in ["a","b"]]
            unit = match["unitB"].lower()
            rule = "two amounts and two years aligned by explicit respectively"
        digits = [re.sub(r"\D", "", v) for v in values]
        if fiscal[0] == fiscal[1] or len(digits[0]) != len(digits[1]) or digits[0][0] == digits[1][0]:
            continue
        item = item.strip(" ,")
        if len(item) > 120 or not ITEM.search(item) or re.search(r"\d|\$|\b(?:We|It|They|This|These)\b", item):
            continue
        if re.search(r"\b(?:quarter|months|percent|percentage|per share)\b", sentence, re.I):
            continue
        group_key = f"{SEED}|{company.casefold()}|{doc}"
        group = int(sha_bytes(group_key.encode()), 16) % 4
        identity = sha_bytes((doc+"|"+sentence+"|"+item).encode())
        numeric = [Decimal(v.replace(",", "")) for v in values]
        numeric = [int(v) if v == v.to_integral_value() else float(v) for v in numeric]
        yield dict(id=identity, company=company, document_sha256=doc, group=group,
            context=sentence, item_name=item, N=numeric[0], T=numeric[1], display_values=values,
            numeric_format=("decimal" if any("." in v for v in values) else "comma integer" if any("," in v for v in values) else "plain integer"),
            yearA=fiscal[0], yearB=fiscal[1], unit=unit, amount_spans=spans,
            sentence_count=1, extraction=rule, source_context_contains_sentence=sentence in normalized)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inspect-partial", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    seen_docs, found, counts = set(), {}, collections.Counter()
    seen_sentences = set()
    for index, row in enumerate(records(args.source, args.inspect_partial)):
        counts["train_records_read"] += 1
        raw = row["Context"]
        normalized = re.sub(r"\s+", " ", raw).strip()
        doc = sha_bytes(normalized.encode())
        if doc in seen_docs:
            counts["duplicate_context_records"] += 1
            continue
        seen_docs.add(doc)
        company = company_name(raw)
        if company is None:
            counts["documents_without_company_header"] += 1
            continue
        for entry in candidates(raw, company, doc):
            key = (company.casefold(), entry["context"])
            if key in seen_sentences:
                counts["duplicate_candidate_sentences"] += 1
                continue
            seen_sentences.add(key)
            entry["source_train_record_index"] = index
            found.setdefault(entry["id"], entry)
    ordered = sorted(found.values(), key=lambda x: sha_bytes(f'{SEED}|{x["id"]}'.encode()))
    counts["unique_documents"] = len(seen_docs)
    counts["eligible_candidates"] = len(ordered)
    for form, n in collections.Counter(x["numeric_format"] for x in ordered).items(): counts["candidate_format_"+form] = n
    group_counts = dict(collections.Counter(x["group"] for x in ordered))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    result = dict(seed=SEED, counts=dict(counts), candidate_group_counts=group_counts,
        candidates=ordered, partial_inspection=args.inspect_partial,
        source=str(Path(args.source).resolve()), wall_seconds=time.perf_counter()-started,
        extraction_limitations="Positive dollar amounts with the same explicit million/billion/thousand units; annual prose only. Commas and decimals are retained verbatim. Digit count counts digits excluding separators. The frozen unique-integer scorer is unchanged. No model-dependent filtering.")
    if not args.inspect_partial:
        digest = hashlib.sha256()
        with open(args.source, "rb") as f:
            for block in iter(lambda: f.read(8*1024*1024), b""):
                digest.update(block)
        result["source_sha256"] = digest.hexdigest()
    path = output / ("candidate_inspection.json" if args.inspect_partial else "candidates.json")
    dump(path, result)
    print(json.dumps({k:v for k,v in result.items() if k!="candidates"}, indent=2))
    for v in ordered[:12]:
        print(json.dumps(v, ensure_ascii=False))

if __name__ == "__main__": main()
