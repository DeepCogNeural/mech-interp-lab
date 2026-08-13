#!/usr/bin/env python3
"""Package the exploratory Exp05 head-to-span bridge follow-up.

The model-backed bridge run is intentionally kept outside Git.  This small,
model-free packager reads its JSON result, re-derives the public scalar tables
and confidence intervals, and emits a compact evidence packet.  It never
copies the raw result into the repository, so checked-in tables can be
reaggregated without that file but the full packet cannot be regenerated from
the repository alone.

Example::

    python3 experiments/05_number_agreement_circuit/make_bridge_summary.py \
      --bridge-results /absolute/path/to/bridge_results.json

When the off-Git raw result is unavailable, the presentation layer can still
be regenerated without pretending to revalidate that raw artifact::

    python3 experiments/05_number_agreement_circuit/make_bridge_summary.py \
      --reaggregate-checked-in

That narrower mode accepts only the exact checked-in seed and matched-row CSV
bytes produced by the original raw-dependent packet. It re-derives statistics,
figures, indexes, and checksums while explicitly recording that the raw result
was not reopened in the current invocation.

The default output is the Exp05 ``results/`` directory.  ``--output`` is
useful for isolated semantic regeneration into a temporary directory under
the current hash-pinned data inputs and renderer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results"
EXPECTED_SCHEMA = "exp05-number-agreement-bridge-rescue-v1"
EXPECTED_RAW_SHA256 = "9d844605de4d20ec5638bf793d21e8750ea606984d7229531fdc9910aa1e45ef"
EXPECTED_SEEDS = tuple(range(20260814, 20260822))
EXPECTED_Q4_SEEDS = tuple(range(20260806, 20260814))
MATCHED_PER_SEED = 100
T_CRIT_8 = 2.365
EXPECTED_SEED_CSV_SHA256 = "0845c76545bdd96a2f2f0b0e68cb5c6726cab2309a9c9ca062372d999ee2e92c"
EXPECTED_MATCHED_CSV_SHA256 = "936f34b458f581330dca2b27dcf6039310ee55751949a6f50e580d19db6332e5"
EXPECTED_HISTORICAL_RECEIPT_SHA256 = "f3ff8fdc8352531e13be5b7babcefe9fe4033513c4c7e04ed77d1bfdc8f1f9f1"
HISTORICAL_RECEIPT_NAME = "bridge_historical_provenance_receipt.json"
BRIDGE_PUBLISH_FILES = (
    "bridge_seed_metrics.csv",
    "bridge_matched_ratios.csv",
    HISTORICAL_RECEIPT_NAME,
    "bridge_result_summary.json",
    "figure_bridge_rescue.svg",
    "figure_bridge_rescue.png",
    "bridge_figure_manifest.json",
    "RESULTS.md",
    "index.json",
    "artifact_index.json",
    "checksums.sha256",
)

METRIC_FIELDS = [
    "seed",
    "source_q4_seed",
    "l7_head",
    "reader_head",
    "target_latent_count",
    "target_projector_rank",
    "status",
    "R_target",
    "R_target_clamped",
    "R_complement",
    "R_matched_mean",
    "R_matched_max",
    "R_matched_second_largest",
    "target_exceeds_matched_max",
    "target_minus_matched_max",
    "target_clamped_minus_target",
    "target_reader_coefficient",
    "target_reader_cosine",
    "target_signed_delta_mean_vs_source_A",
    "full_signed_delta_mean_vs_source_A",
    "complement_signed_delta_mean_vs_source_A",
]
MATCHED_FIELDS = [
    "seed",
    "source_q4_seed",
    "draw_index",
    "latent_ids",
    "rank",
    "R_matched",
    "reader_coefficient",
    "reader_cosine",
    "signed_delta_mean_vs_source_A",
    "source_bridge_results_sha256",
]

BRIDGE_CLAIM = (
    "In this exploratory follow-up, the fixed 12-row decoder span carried a "
    "large fraction of the L7H4-induced directed-logit effect on fresh seeds "
    "and exceeded every matched span maximum in the 8-seed sample."
)
BRIDGE_NOT_CLAIMED = [
    "This is an exploratory follow-up with no preregistered verdict or threshold.",
    "It does not establish that the fixed span is a natural, necessary, or sufficient representation.",
    "It does not establish individual-latent causality, a complete circuit, or full mediation.",
    "The L8H5 hook_z@final clamp overwrites the complete per-head output at the final query position; it is not a value-only clamp.",
    "The clamp is a dependence control, not a proof of an L7H4-to-span-to-readout mediation path.",
    "It does not distinguish an all-position clamp from a parallel route under this final-only upstream intervention.",
    "R_target is a directed-logit effect ratio, not an activation-reconstruction percentage.",
]
BRIDGE_DESIGN = {
    "follow_up_type": "fresh out-of-sample exploratory bridge",
    "timing_decision": "L7_ONLY_RESID_PRE8",
    "upstream_head": "L7H4",
    "fixed_decoder_span": "12 layer-8 res-jb decoder rows from Q4",
    "reader_head": "L8H5",
    "matched_control": "100 rank-12 target-excluded spans per seed",
    "reader_clamp": {
        "hook": "blocks.8.attn.hook_z",
        "head": "L8H5",
        "query_position": "final",
        "replacement": "natural source-A L7H4-arm hook_z output",
        "semantics": "complete per-head z after attention-pattern-weighted value aggregation",
        "value_only": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _json_object_from_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {label}")
    return value


def _read_bound_json(path: Path, label: str) -> tuple[dict[str, Any], str, int]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    return _json_object_from_bytes(data, label), sha256_bytes(data), len(data)


def read_json(path: Path) -> dict[str, Any]:
    value, _sha, _size = _read_bound_json(path, str(path))
    return value


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        # Preserve enough precision that public CSV means recompute to the
        # JSON aggregate without relying on a loose rounding tolerance.
        return format(value, ".15g")
    return str(value)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("descriptive summary requires finite non-empty values")
    mean = statistics.fmean(values)
    if len(values) == 1:
        standard_error = 0.0
        low = high = mean
    else:
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        low = mean - T_CRIT_8 * standard_error
        high = mean + T_CRIT_8 * standard_error
    return {
        "finite_count": len(values),
        "mean": mean,
        "standard_error": standard_error,
        "degrees_of_freedom": len(values) - 1,
        "t_critical": T_CRIT_8,
        "ci95": {"low": low, "high": high},
        "status": "ESTIMABLE",
    }


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc


def _read_csv_exact_bytes(data: bytes, label: str, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(expected_fields):
        raise ValueError(f"{label} header differs from the frozen compact schema")
    return [dict(row) for row in reader]


def _checked_in_rows(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate and load the exact published compact rows without the off-Git raw.

    The fixed byte hashes are the authority for which compact payload may enter
    this narrower lifecycle. Structural checks below make the derivation readable
    and fail closed if the schema or scientific grid is ever changed deliberately.
    """

    seed_path = output / "bridge_seed_metrics.csv"
    matched_path = output / "bridge_matched_ratios.csv"
    seed_bytes = _read_bytes(seed_path, "checked-in bridge seed CSV")
    matched_bytes = _read_bytes(matched_path, "checked-in bridge matched CSV")
    observed_seed_hash = sha256_bytes(seed_bytes)
    observed_matched_hash = sha256_bytes(matched_bytes)
    if observed_seed_hash != EXPECTED_SEED_CSV_SHA256:
        raise ValueError(f"checked-in bridge seed CSV SHA mismatch: {observed_seed_hash}")
    if observed_matched_hash != EXPECTED_MATCHED_CSV_SHA256:
        raise ValueError(f"checked-in bridge matched CSV SHA mismatch: {observed_matched_hash}")

    raw_seed_rows = _read_csv_exact_bytes(seed_bytes, seed_path.name, METRIC_FIELDS)
    raw_matched_rows = _read_csv_exact_bytes(matched_bytes, matched_path.name, MATCHED_FIELDS)
    if len(raw_seed_rows) != len(EXPECTED_SEEDS) or len(raw_matched_rows) != len(EXPECTED_SEEDS) * MATCHED_PER_SEED:
        raise ValueError("checked-in bridge compact rows do not contain the frozen 8 + 800 grid")

    metric_rows: list[dict[str, Any]] = []
    for raw in raw_seed_rows:
        row: dict[str, Any] = {
            "seed": int(raw["seed"]),
            "source_q4_seed": int(raw["source_q4_seed"]),
            "l7_head": raw["l7_head"],
            "reader_head": raw["reader_head"],
            "target_latent_count": int(raw["target_latent_count"]),
            "target_projector_rank": int(raw["target_projector_rank"]),
            "status": raw["status"],
            "target_exceeds_matched_max": raw["target_exceeds_matched_max"].strip().lower() == "true",
        }
        for field in METRIC_FIELDS:
            if field not in row:
                row[field] = finite_float(raw[field], f"checked-in seed row {row['seed']} {field}")
        metric_rows.append(row)
    metric_rows.sort(key=lambda row: int(row["seed"]))
    if tuple(int(row["seed"]) for row in metric_rows) != EXPECTED_SEEDS:
        raise ValueError("checked-in bridge seed ids differ from the frozen fresh-seed set")
    if tuple(int(row["source_q4_seed"]) for row in metric_rows) != EXPECTED_Q4_SEEDS:
        raise ValueError("checked-in bridge Q4 ordinal ids differ from the frozen source-seed set")
    for row in metric_rows:
        seed = int(row["seed"])
        if (
            row["status"] != "COMPLETE"
            or row["l7_head"] != "L7H4"
            or row["reader_head"] != "L8H5"
            or int(row["target_latent_count"]) != 12
            or int(row["target_projector_rank"]) != 12
        ):
            raise ValueError(f"checked-in seed {seed} changed its frozen bridge object")

    compact_rows: list[dict[str, Any]] = []
    for raw in raw_matched_rows:
        try:
            latent_ids = json.loads(raw["latent_ids"])
        except json.JSONDecodeError as exc:
            raise ValueError("checked-in bridge matched row has invalid latent-id JSON") from exc
        seed = int(raw["seed"])
        draw = int(raw["draw_index"])
        if not isinstance(latent_ids, list) or len(latent_ids) != 12 or len(set(int(value) for value in latent_ids)) != 12:
            raise ValueError(f"checked-in matched row {seed}/{draw} is not rank-twelve by ids")
        compact_rows.append(
            {
                "seed": seed,
                "source_q4_seed": int(raw["source_q4_seed"]),
                "draw_index": draw,
                "latent_ids": json.dumps([int(value) for value in latent_ids], separators=(",", ":")),
                "rank": int(raw["rank"]),
                "R_matched": finite_float(raw["R_matched"], f"checked-in matched row {seed}/{draw} R"),
                "reader_coefficient": finite_float(raw["reader_coefficient"], f"checked-in matched row {seed}/{draw} coefficient"),
                "reader_cosine": finite_float(raw["reader_cosine"], f"checked-in matched row {seed}/{draw} cosine"),
                "signed_delta_mean_vs_source_A": finite_float(raw["signed_delta_mean_vs_source_A"], f"checked-in matched row {seed}/{draw} signed effect"),
                "source_bridge_results_sha256": raw["source_bridge_results_sha256"],
            }
        )
    compact_rows.sort(key=lambda row: (int(row["seed"]), int(row["draw_index"])))

    by_seed = {seed: [] for seed in EXPECTED_SEEDS}
    source_q4_by_seed = {int(row["seed"]): int(row["source_q4_seed"]) for row in metric_rows}
    for row in compact_rows:
        seed = int(row["seed"])
        if seed not in by_seed:
            raise ValueError(f"checked-in matched row has unknown seed {seed}")
        if int(row["source_q4_seed"]) != source_q4_by_seed[seed]:
            raise ValueError(f"checked-in matched row {seed}/{row['draw_index']} changed its Q4 ordinal")
        if int(row["rank"]) != 12 or row["source_bridge_results_sha256"] != EXPECTED_RAW_SHA256:
            raise ValueError(f"checked-in matched row {seed}/{row['draw_index']} changed its rank or raw receipt")
        by_seed[seed].append(row)

    metric_by_seed = {int(row["seed"]): row for row in metric_rows}
    for seed in EXPECTED_SEEDS:
        rows = by_seed[seed]
        if [int(row["draw_index"]) for row in rows] != list(range(MATCHED_PER_SEED)):
            raise ValueError(f"checked-in seed {seed} matched draw ids are not exactly 0..99")
        effects = sorted(float(row["R_matched"]) for row in rows)
        metric = metric_by_seed[seed]
        recomputed = {
            "R_matched_mean": statistics.fmean(effects),
            "R_matched_max": effects[-1],
            "R_matched_second_largest": effects[-2],
        }
        for field, expected in recomputed.items():
            if abs(float(metric[field]) - expected) > 1e-12:
                raise ValueError(f"checked-in seed {seed} {field} differs from its 100 matched rows")
        target_exceeds = float(metric["R_target"]) > effects[-1]
        if bool(metric["target_exceeds_matched_max"]) is not target_exceeds:
            raise ValueError(f"checked-in seed {seed} target-edge flag is inconsistent")
        if abs(float(metric["target_minus_matched_max"]) - (float(metric["R_target"]) - effects[-1])) > 1e-12:
            raise ValueError(f"checked-in seed {seed} target-minus-edge field is inconsistent")

    target_values = [float(row["R_target"]) for row in metric_rows]
    clamped_values = [float(row["R_target_clamped"]) for row in metric_rows]
    complement_values = [float(row["R_complement"]) for row in metric_rows]
    matched_mean_values = [float(row["R_matched_mean"]) for row in metric_rows]
    matched_max_values = [float(row["R_matched_max"]) for row in metric_rows]
    matched_second_values = [float(row["R_matched_second_largest"]) for row in metric_rows]
    reader_target_values = [float(row["target_reader_coefficient"]) for row in metric_rows]
    target_exceeds = [bool(row["target_exceeds_matched_max"]) for row in metric_rows]

    receipt_path = output / HISTORICAL_RECEIPT_NAME
    receipt, receipt_sha, receipt_bytes = _read_bound_json(
        receipt_path,
        "hash-pinned historical bridge provenance receipt",
    )
    if receipt_sha != EXPECTED_HISTORICAL_RECEIPT_SHA256:
        raise ValueError(f"historical bridge provenance receipt SHA mismatch: {receipt_sha}")
    expected_origin = {
        "source_commit": "10d5c05fc94cd8e0822de4842f391d48488e6061",
        "summary_path": "experiments/05_number_agreement_circuit/results/bridge_result_summary.json",
        "summary_file_sha256": "cc2957d22f813991e7b55cb535621446fbceccec395844c0eaa773a6cda7c7cf",
        "reported_fields_canonicalization": "SHA-256 of UTF-8 Python canonical JSON for the original object {source_receipt, integrity}: sort_keys=true, separators=(',', ':'), ensure_ascii=false",
        "reported_fields_canonical_sha256": "3043c8f9773d4924841b48df4616ff2ab7acc7b6742b1626a7352ccb094860a0",
    }
    raw_receipt = receipt.get("raw_result_receipt")
    reported_integrity = receipt.get("reported_integrity")
    if (
        receipt.get("schema") != "exp05-bridge-historical-provenance-receipt-v1"
        or receipt.get("status") != "HISTORICAL_REPORT_NOT_REVALIDATED"
        or receipt.get("origin") != expected_origin
        or not isinstance(raw_receipt, Mapping)
        or raw_receipt.get("basename") != "bridge_results.json"
        or raw_receipt.get("sha256") != EXPECTED_RAW_SHA256
        or int(raw_receipt.get("bytes", -1)) != 653_980
        or not isinstance(reported_integrity, Mapping)
        or reported_integrity.get("raw_result_sha256") != EXPECTED_RAW_SHA256
    ):
        raise ValueError("historical bridge provenance receipt changed its frozen authority fields")
    reaggregation = {
        "mode": "checked_in_hash_bound",
        "raw_result_revalidated_this_invocation": False,
        "seed_csv_sha256": observed_seed_hash,
        "matched_csv_sha256": observed_matched_hash,
        "historical_receipt_sha256": receipt_sha,
        "historical_gate_identity_git_fields": "reported by the hash-pinned historical receipt; not revalidated in this invocation",
    }
    integrity = {
        "historical_raw_dependent_report": {
            "status": "REPORTED_BY_ORIGINAL_PACKET_NOT_REVALIDATED",
            "authority_receipt": {
                "path": HISTORICAL_RECEIPT_NAME,
                "bytes": receipt_bytes,
                "sha256": receipt_sha,
            },
            "origin": dict(expected_origin),
            "reported_integrity": dict(reported_integrity),
        },
        "reaggregation": reaggregation,
    }
    summary = {
        "schema": "exp05-public-bridge-rescue-summary-v1",
        "status": "COMPLETE",
        "verdict": "EXPLORATORY_NO_PREREGISTERED_VERDICT",
        "source_receipt": {
            **dict(raw_receipt),
            "status": "HISTORICAL_REPORTED_NOT_REVALIDATED",
            "authority_receipt": HISTORICAL_RECEIPT_NAME,
            "current_invocation_revalidated": False,
        },
        "seed_count": len(metric_rows),
        "seeds": list(EXPECTED_SEEDS),
        "matched_rows": len(compact_rows),
        "matched_per_seed": MATCHED_PER_SEED,
        "target_exceeds_matched_max_count": sum(target_exceeds),
        "target_exceeds_matched_max_all_seeds": all(target_exceeds),
        "metrics": {
            "R_target": descriptive(target_values),
            "R_target_clamped": descriptive(clamped_values),
            "R_complement": descriptive(complement_values),
            "R_matched_mean": descriptive(matched_mean_values),
            "R_matched_max": descriptive(matched_max_values),
            "R_matched_second_largest": descriptive(matched_second_values),
            "target_reader_coefficient": descriptive(reader_target_values),
        },
        "design": BRIDGE_DESIGN,
        "integrity": integrity,
        "claim": BRIDGE_CLAIM,
        "claim_boundary": "The bridge packet is descriptive follow-up evidence, not a preregistered adjudication.",
        "not_claimed": BRIDGE_NOT_CLAIMED,
        "compact_artifacts": {
            "seed_metrics": "bridge_seed_metrics.csv",
            "matched_ratios": "bridge_matched_ratios.csv",
            "figure": "figure_bridge_rescue.svg",
            "historical_provenance_receipt": HISTORICAL_RECEIPT_NAME,
            "raw_result_copied": False,
        },
        "review_receipt": None,
        "regeneration": dict(reaggregation),
        "reproducibility": {
            "checked_in_reaggregation": "requires the exact hash-bound checked-in seed and matched-span CSV rows",
            "raw_dependent_packaging": "requires the off-Git bridge_results.json bound by source_receipt.sha256; it was not available in this invocation",
            "full_model_rerun": "requires the model, SAE, cached assets, and execution environment; not reproducible from Git alone",
        },
        "reproduce": {
            "script": "experiments/05_number_agreement_circuit/make_bridge_summary.py",
            "command": "python3 experiments/05_number_agreement_circuit/make_bridge_summary.py --reaggregate-checked-in",
            "raw_dependent_command": "python3 experiments/05_number_agreement_circuit/make_bridge_summary.py --bridge-results /absolute/path/to/bridge_results.json",
            "independent_checks": [
                "verify the fixed seed/matched CSV SHA-256 values before reaggregation",
                "recompute per-seed matched maxima from bridge_matched_ratios.csv",
                "recompute aggregate means and t(7) intervals from bridge_seed_metrics.csv",
                "sha256sum -c checksums.sha256",
            ],
        },
    }
    return metric_rows, compact_rows, summary


def _base_package(output: Path) -> None:
    """Seed a temporary output with the checked-in Q4 packet when needed.

    This keeps ``--output`` regeneration self-contained while preserving the
    existing Q4 artifacts and their public documentation.  No raw model
    outputs are introduced; the checked-in packet is already compact.
    """

    if output.resolve() == DEFAULT_OUTPUT.resolve():
        output.mkdir(parents=True, exist_ok=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        return
    for source in DEFAULT_OUTPUT.iterdir():
        target = output / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def validate_and_extract(result_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    result, raw_sha, raw_bytes = _read_bound_json(result_path, "raw bridge result")
    if raw_sha != EXPECTED_RAW_SHA256:
        raise ValueError(
            "bridge result SHA mismatch: "
            f"expected {EXPECTED_RAW_SHA256}, got {raw_sha}"
        )
    if result.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected bridge schema: {result.get('schema')!r}")
    if result.get("status") != "COMPLETE":
        raise ValueError(f"bridge result is not COMPLETE: {result.get('status')!r}")
    rows = result.get("seed_results")
    if not isinstance(rows, list):
        raise ValueError("bridge result has no seed_results list")
    ordered = sorted(rows, key=lambda row: int(row.get("seed", -1)))
    observed = tuple(int(row.get("seed", -1)) for row in ordered)
    if observed != EXPECTED_SEEDS:
        raise ValueError(f"bridge seed set mismatch: {observed!r}")
    if any(row.get("status") != "COMPLETE" for row in ordered):
        raise ValueError("all bridge seeds must be COMPLETE")

    matched_rows = result.get("matched_rows")
    if not isinstance(matched_rows, list) or len(matched_rows) != len(EXPECTED_SEEDS) * MATCHED_PER_SEED:
        raise ValueError("bridge result must contain exactly 800 matched rows")
    by_seed: dict[int, list[Mapping[str, Any]]] = {seed: [] for seed in EXPECTED_SEEDS}
    for row in matched_rows:
        if not isinstance(row, Mapping):
            raise ValueError("malformed matched row")
        seed = int(row.get("seed", -1))
        if seed not in by_seed:
            raise ValueError(f"matched row has unexpected seed: {seed}")
        by_seed[seed].append(row)
    if any(len(values) != MATCHED_PER_SEED for values in by_seed.values()):
        raise ValueError("each bridge seed must have exactly 100 matched rows")

    metric_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    target_values: list[float] = []
    clamped_values: list[float] = []
    complement_values: list[float] = []
    matched_mean_values: list[float] = []
    matched_max_values: list[float] = []
    matched_second_values: list[float] = []
    reader_target_values: list[float] = []
    target_exceeds_max: list[bool] = []
    gate_a_passes: list[bool] = []
    retained_pairs: list[int] = []
    evaluation_pair_counts: list[int] = []
    identity_non_final: list[float] = []
    identity_selected: list[float] = []
    identity_final: list[float] = []
    identity_final_tolerances: list[float] = []

    for seed_row in ordered:
        seed = int(seed_row["seed"])
        natural = seed_row.get("natural")
        clamped = seed_row.get("clamped")
        summary = seed_row.get("matched_summary")
        l7_head = seed_row.get("l7_head")
        reader_head = seed_row.get("reader_head")
        if not all(isinstance(item, Mapping) for item in (natural, clamped, summary, l7_head, reader_head)):
            raise ValueError(f"seed {seed} has incomplete bridge fields")
        natural_target = natural["target"]
        natural_full = natural["full"]
        natural_comp = natural["complement"]
        clamped_target = clamped["target"]
        if not all(isinstance(item, Mapping) for item in (natural_target, natural_full, natural_comp, clamped_target)):
            raise ValueError(f"seed {seed} has malformed natural/clamped metrics")
        matched_for_seed = by_seed[seed]
        matched_values = [finite_float(item.get("R_matched"), f"seed {seed} matched R") for item in matched_for_seed]
        matched_values_sorted = sorted(matched_values)
        target = finite_float(natural_target.get("R"), f"seed {seed} target R")
        clamped_r = finite_float(clamped_target.get("R"), f"seed {seed} clamped target R")
        complement = finite_float(natural_comp.get("R"), f"seed {seed} complement R")
        matched_mean = statistics.fmean(matched_values)
        matched_max = matched_values_sorted[-1]
        matched_second = matched_values_sorted[-2]
        declared_max = finite_float(summary.get("max"), f"seed {seed} matched max")
        declared_second = finite_float(summary.get("second_largest"), f"seed {seed} matched second largest")
        if abs(matched_max - declared_max) > 1e-12 or abs(matched_second - declared_second) > 1e-12:
            raise ValueError(f"seed {seed} matched summary disagrees with raw matched rows")
        pass_edge = target > matched_max
        target_exceeds_max.append(pass_edge)
        reader = natural_target.get("reader")
        if not isinstance(reader, Mapping):
            raise ValueError(f"seed {seed} target has no reader projection")
        reader_coefficient = finite_float(reader.get("coefficient"), f"seed {seed} reader coefficient")
        reader_cosine = finite_float(reader.get("cosine"), f"seed {seed} reader cosine")
        gate_a = seed_row.get("gate_a")
        identity = seed_row.get("identity_diagnostics")
        if not isinstance(gate_a, Mapping) or not isinstance(identity, Mapping):
            raise ValueError(f"seed {seed} is missing Gate-A or identity diagnostics")
        gate_passed = bool(gate_a.get("passed"))
        retained = int(gate_a.get("retained_pairs", -1))
        eval_pair_count = len(seed_row.get("evaluation_pair_ids", []))
        if not gate_passed or retained < 0 or eval_pair_count != 150:
            raise ValueError(f"seed {seed} failed the public Gate-A/evaluation integrity contract")
        if identity.get("status") != "PASS":
            raise ValueError(f"seed {seed} identity diagnostics are not PASS")
        gate_a_passes.append(gate_passed)
        retained_pairs.append(retained)
        evaluation_pair_counts.append(eval_pair_count)
        identity_non_final.append(finite_float(identity.get("non_final_positions_max_abs"), f"seed {seed} non-final identity"))
        identity_selected.append(finite_float(identity.get("selected_positions_max_abs"), f"seed {seed} selected identity"))
        identity_final.append(finite_float(identity.get("full_vs_true_final_logit_max_abs"), f"seed {seed} final identity"))
        identity_final_tolerances.append(finite_float(identity.get("full_vs_true_final_logit_tolerance"), f"seed {seed} final tolerance"))
        metric_rows.append(
            {
                "seed": seed,
                "source_q4_seed": int(seed_row.get("source_q4_seed", -1)),
                "l7_head": f"L{int(l7_head['layer'])}H{int(l7_head['head'])}",
                "reader_head": f"L{int(reader_head['layer'])}H{int(reader_head['head'])}",
                "target_latent_count": len(seed_row.get("target_latent_ids", [])),
                "target_projector_rank": int(seed_row.get("target_projector", {}).get("rank", -1)),
                "status": seed_row.get("status"),
                "R_target": target,
                "R_target_clamped": clamped_r,
                "R_complement": complement,
                "R_matched_mean": matched_mean,
                "R_matched_max": matched_max,
                "R_matched_second_largest": matched_second,
                "target_exceeds_matched_max": pass_edge,
                "target_minus_matched_max": target - matched_max,
                "target_clamped_minus_target": clamped_r - target,
                "target_reader_coefficient": reader_coefficient,
                "target_reader_cosine": reader_cosine,
                "target_signed_delta_mean_vs_source_A": finite_float(
                    natural_target.get("signed_delta_mean_vs_source_A"), f"seed {seed} target signed delta"
                ),
                "full_signed_delta_mean_vs_source_A": finite_float(
                    natural_full.get("signed_delta_mean_vs_source_A"), f"seed {seed} full signed delta"
                ),
                "complement_signed_delta_mean_vs_source_A": finite_float(
                    natural_comp.get("signed_delta_mean_vs_source_A"), f"seed {seed} complement signed delta"
                ),
            }
        )
        for item in sorted(matched_for_seed, key=lambda row: int(row.get("draw_index", -1))):
            latent_ids = item.get("latent_ids")
            if not isinstance(latent_ids, list) or len(latent_ids) != 12 or len(set(latent_ids)) != 12:
                raise ValueError(f"seed {seed} matched row has malformed latent ids")
            compact_rows.append(
                {
                    "seed": seed,
                    "source_q4_seed": int(item.get("source_q4_seed", seed_row.get("source_q4_seed", -1))),
                    "draw_index": int(item.get("draw_index", -1)),
                    "latent_ids": json.dumps([int(value) for value in latent_ids], separators=(",", ":")),
                    "rank": int(item.get("projector", {}).get("rank", 12)),
                    "R_matched": finite_float(item.get("R_matched"), f"seed {seed} matched R"),
                    "reader_coefficient": finite_float(item.get("reader", {}).get("coefficient"), f"seed {seed} matched reader coefficient"),
                    "reader_cosine": finite_float(item.get("reader", {}).get("cosine"), f"seed {seed} matched reader cosine"),
                    "signed_delta_mean_vs_source_A": finite_float(
                        item.get("signed_delta_mean_vs_source_A"), f"seed {seed} matched signed delta"
                    ),
                    "source_bridge_results_sha256": raw_sha,
                }
            )
        target_values.append(target)
        clamped_values.append(clamped_r)
        complement_values.append(complement)
        matched_mean_values.append(matched_mean)
        matched_max_values.append(matched_max)
        matched_second_values.append(matched_second)
        reader_target_values.append(reader_coefficient)

    metric_rows.sort(key=lambda row: int(row["seed"]))
    compact_rows.sort(key=lambda row: (int(row["seed"]), int(row["draw_index"])))
    if len(compact_rows) != 800:
        raise ValueError(f"expected 800 compact matched rows, found {len(compact_rows)}")
    if not all(target_exceeds_max):
        raise ValueError("target does not exceed matched maximum on every seed")
    if sum(gate_a_passes) != len(EXPECTED_SEEDS) or min(retained_pairs) != 230 or max(retained_pairs) != 237:
        raise ValueError("unexpected Gate-A pass count or retained-pair range")
    if set(evaluation_pair_counts) != {150}:
        raise ValueError("unexpected evaluation pair count")
    if max(identity_non_final) != 0.0 or max(identity_selected) != 0.0:
        raise ValueError("identity diagnostics have non-zero non-final or selected positions")
    if max(identity_final) > 1e-5 or set(identity_final_tolerances) != {1e-5}:
        raise ValueError("identity final-logit tolerance contract failed")

    aggregate = {
        "R_target": descriptive(target_values),
        "R_target_clamped": descriptive(clamped_values),
        "R_complement": descriptive(complement_values),
        "R_matched_mean": descriptive(matched_mean_values),
        "R_matched_max": descriptive(matched_max_values),
        "R_matched_second_largest": descriptive(matched_second_values),
        "target_reader_coefficient": descriptive(reader_target_values),
    }
    summary = {
        "schema": "exp05-public-bridge-rescue-summary-v1",
        "status": "COMPLETE",
        "verdict": "EXPLORATORY_NO_PREREGISTERED_VERDICT",
        "source_receipt": {
            "basename": result_path.name,
            "bytes": raw_bytes,
            "sha256": raw_sha,
        },
        "seed_count": len(metric_rows),
        "seeds": list(EXPECTED_SEEDS),
        "matched_rows": len(compact_rows),
        "matched_per_seed": MATCHED_PER_SEED,
        "target_exceeds_matched_max_count": sum(1 for value in target_exceeds_max if value),
        "target_exceeds_matched_max_all_seeds": all(target_exceeds_max),
        "metrics": aggregate,
        "design": BRIDGE_DESIGN,
        "integrity": {
            "raw_result_sha256": raw_sha,
            "gate_a": {
                "passed_count": sum(gate_a_passes),
                "seed_count": len(EXPECTED_SEEDS),
                "all_passed": all(gate_a_passes),
                "retained_pairs_per_seed_range": [min(retained_pairs), max(retained_pairs)],
                "evaluation_pairs_per_seed": sorted(set(evaluation_pair_counts)),
            },
            "identity": {
                "status": "PASS",
                "non_final_positions_max_abs": max(identity_non_final),
                "selected_positions_max_abs": max(identity_selected),
                "full_vs_true_final_logit_max_abs": max(identity_final),
                "full_vs_true_final_logit_tolerance": max(identity_final_tolerances),
                "full_vs_true_final_logit_within_tolerance": max(identity_final) < max(identity_final_tolerances),
            },
            "git": {
                "start_commit": result.get("git", {}).get("commit"),
                "final_commit": result.get("git_final", {}).get("commit"),
                "start_status_porcelain": result.get("git", {}).get("status_porcelain", ""),
                "final_status_porcelain": result.get("git_final", {}).get("status_porcelain", ""),
                "clean_start": result.get("git", {}).get("status_porcelain", "") == "",
                "clean_final": result.get("git_final", {}).get("status_porcelain", "") == "",
            },
        },
        "claim": BRIDGE_CLAIM,
        "claim_boundary": "The bridge packet is descriptive follow-up evidence, not a preregistered adjudication.",
        "not_claimed": BRIDGE_NOT_CLAIMED,
        "compact_artifacts": {
            "seed_metrics": "bridge_seed_metrics.csv",
            "matched_ratios": "bridge_matched_ratios.csv",
            "figure": "figure_bridge_rescue.svg",
            "raw_result_copied": False,
        },
        "review_receipt": None,
        "regeneration": {
            "mode": "raw_dependent_packaging",
            "raw_result_revalidated_this_invocation": True,
            "source_raw_sha256": raw_sha,
        },
        "reproducibility": {
            "checked_in_reaggregation": "seed and matched-span summaries can be recomputed from the checked-in CSV/JSON packet",
            "raw_dependent_packaging": "requires the off-Git bridge_results.json bound by source_receipt.sha256",
            "full_model_rerun": "requires the model, SAE, cached assets, and execution environment; not reproducible from Git alone",
        },
        "reproduce": {
            "script": "experiments/05_number_agreement_circuit/make_bridge_summary.py",
            "command": "python3 experiments/05_number_agreement_circuit/make_bridge_summary.py --reaggregate-checked-in",
            "raw_dependent_command": "python3 experiments/05_number_agreement_circuit/make_bridge_summary.py --bridge-results /absolute/path/to/bridge_results.json",
            "independent_checks": [
                "recompute per-seed matched maxima from bridge_matched_ratios.csv",
                "recompute aggregate means and t(7) intervals from bridge_seed_metrics.csv",
                "sha256sum -c checksums.sha256",
            ],
        },
    }
    return metric_rows, compact_rows, summary


def write_figure(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    regeneration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = [int(row["seed"]) for row in rows]
    x = list(range(len(rows)))
    short = [str(seed)[-2:] for seed in seeds]
    target = [float(row["R_target"]) for row in rows]
    matched_max = [float(row["R_matched_max"]) for row in rows]
    clamped = [float(row["R_target_clamped"]) for row in rows]
    coeff = [float(row["target_reader_coefficient"]) for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.24, top=0.80, wspace=0.20)

    ax = axes[0]
    ax.plot(x, target, "o-", color="#b91c1c", lw=1.7, ms=4, label="target span")
    ax.plot(x, matched_max, "s--", color="#2563eb", lw=1.3, ms=3.5, label="matched max")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, short)
    ax.set_ylabel("directed-logit ratio")
    ax.set_title("A  target vs matched maximum", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.text(
        0.03,
        0.96,
        "8/8 target > matched max",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#166534",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f0fdf4", "edgecolor": "#86efac"},
    )

    ax = axes[1]
    ax.plot(x, target, "o-", color="#b91c1c", lw=1.7, ms=4, label="natural target")
    ax.plot(x, clamped, "D--", color="#7c3aed", lw=1.3, ms=3.5,
            label="L8H5 hook_z@final clamped")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, short)
    ax.set_ylabel("directed-logit ratio")
    ax.set_title("B  natural vs final-position L8H5-z clamp", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.text(
        0.03,
        0.96,
        "complete final-position head output replaced",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#4b5563",
    )

    ax = axes[2]
    ax.plot(x, coeff, "o-", color="#0f766e", lw=1.7, ms=4)
    ax.set_ylim(0, max(0.2, max(coeff) * 1.25))
    ax.set_xticks(x, short)
    ax.set_ylabel("coefficient")
    ax.set_title("C  descriptive L8 reader projection", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.text(
        0.03,
        0.96,
        "target span vs full activation delta",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#4b5563",
    )

    fig.suptitle(
        "Exp05 exploratory bridge: fixed-span rescue of an L7H4-induced effect",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Fresh seeds; directed-logit ratios. hook_z@final is the complete head output, not a value-only path.",
        ha="center",
        fontsize=7.5,
        color="#4b5563",
    )
    svg_path = output / "figure_bridge_rescue.svg"
    png_path = output / "figure_bridge_rescue.png"
    fig.savefig(svg_path, format="svg", metadata={"Date": None})
    fig.savefig(png_path, format="png", dpi=180, metadata={"Date": None})
    plt.close(fig)

    # Canonicalize the insignificant serialization details that otherwise vary
    # with Matplotlib's process-local clip-path ids and float formatting.
    svg_text = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n"
    clip_ids: list[str] = []
    for old_id in re.findall(r'id="([mp][0-9a-f]+)"', svg_text):
        if old_id not in clip_ids:
            clip_ids.append(old_id)
    for index, old_id in enumerate(clip_ids, start=1):
        svg_text = svg_text.replace(old_id, f"clip_path_{index}")

    def stable_float(match: re.Match[str]) -> str:
        value = float(match.group(0))
        rounded = f"{value:.5f}".rstrip("0").rstrip(".")
        return "0" if rounded in {"-0", ""} else rounded

    svg_text = re.sub(r"(?<![A-Za-z0-9_.-])-?\d+\.\d+(?:e[+-]?\d+)?", stable_float, svg_text)
    svg_path.write_text(svg_text, encoding="utf-8")
    manifest = {
        "schema": "exp05-public-bridge-figure-manifest-v1",
        "figure": "figure_bridge_rescue.svg",
        "png": "figure_bridge_rescue.png",
        "panels": {
            "A": "per-seed natural target R versus maximum matched-span R",
            "B": "per-seed natural target R versus complete L8H5 hook_z@final-clamped target R",
            "C": "per-seed descriptive L8 reader projection coefficient",
        },
        "source": "bridge_seed_metrics.csv",
        "claim_boundary": "Descriptive exploratory bridge evidence; no mediation claim.",
    }
    if regeneration is not None:
        manifest["regeneration"] = dict(regeneration)
    return manifest


def update_results_md(path: Path) -> None:
    marker = "## Exploratory bridge follow-up"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Exp05 public results\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    section = f"""
{marker}

The single follow-up is a fresh out-of-sample, exploratory bridge around the
L7H4 intervention.  It asks whether the fixed 12-row layer-8 decoder span
also carries that upstream head-induced effect, while replacing the complete
L8H5 `hook_z` output at the final query position as a dependence control.  This
is not a value-only clamp.  The compact evidence is in
[`bridge_seed_metrics.csv`](bridge_seed_metrics.csv),
[`bridge_matched_ratios.csv`](bridge_matched_ratios.csv), and
[`bridge_result_summary.json`](bridge_result_summary.json); the raw 639 KB
model result stays outside Git. Its previously reported raw-dependent
integrity fields are preserved in the separately hash-pinned
[`bridge_historical_provenance_receipt.json`](bridge_historical_provenance_receipt.json).
The current presentation can be regenerated
from the two exact hash-bound checked-in CSVs without reopening the raw result;
that narrower mode does not revalidate the historical Gate-A, identity, Git,
or raw-receipt fields. Regenerating the full raw-dependent packet still
requires that off-Git raw result. No
bridge-specific independent review receipt is present in this packet.

**Exploratory finding.** On eight fresh seeds, the target span had mean
`R_target=0.6786` (95% t(7) CI `[0.6738, 0.6834]`) versus mean matched-span
maximum `0.4110`, and exceeded the matched maximum on all 8 seeds.  The
complement ratio was `0.3053`.  With L8H5 `hook_z@final` clamped, the target remained large:
mean `R_target_clamped=0.6740` (95% t(7) CI `[0.6696, 0.6785]`).  This does
not support dominant dependence on L8H5's complete tested final-position head output or a mediation claim; it only says that this
fixed span carries the tested L7H4-induced effect better than the matched
controls in this exploratory sample.

**Historical run-integrity report.** The original raw-dependent packet reported
Gate A passing on 8/8 seeds, 230–237 retained pairs per seed, 150 evaluation
pairs per seed, zero non-final and selected-position identity maxima, a
`9.536743e-06 < 1e-5` full-vs-true final-logit maximum, and a clean start and
finish at commit `0d7c4db`. The hash-pinned historical receipt binds those
reported fields to raw SHA-256
`9d844605de4d20ec5638bf793d21e8750ea606984d7229531fdc9910aa1e45ef`;
this checked-in reaggregation did not reopen or revalidate the raw result.

This follow-up has no preregistered verdict or threshold.  It does not
establish naturalness, necessity, sufficiency, individual-latent causality,
or full mediation.  The three-panel overview is
[`figure_bridge_rescue.svg`](figure_bridge_rescue.svg) (and its PNG).

Reproduce the compact packet with:

```bash
python3 experiments/05_number_agreement_circuit/make_bridge_summary.py \\
  --reaggregate-checked-in
```
"""
    path.write_text(existing.rstrip() + "\n" + section.lstrip(), encoding="utf-8")


def update_indexes(output: Path, summary: Mapping[str, Any]) -> None:
    generated = [
        "bridge_seed_metrics.csv",
        "bridge_matched_ratios.csv",
        HISTORICAL_RECEIPT_NAME,
        "bridge_result_summary.json",
        "figure_bridge_rescue.svg",
        "figure_bridge_rescue.png",
        "bridge_figure_manifest.json",
    ]
    for name in ("index.json", "artifact_index.json"):
        path = output / name
        index = read_json(path) if path.exists() else {
            "schema": "exp05-public-evidence-index-v1",
            "title": "Exp05 public evidence packet",
            "artifacts": [],
            "checksums": "checksums.sha256",
        }
        artifacts = list(index.get("artifacts", []))
        for artifact in generated:
            if artifact not in artifacts:
                artifacts.append(artifact)
        index["artifacts"] = artifacts
        previous = str(index.get("claim_boundary", ""))
        bridge_boundary = "The exploratory bridge packet is descriptive and carries no preregistered verdict or mediation claim."
        base_boundary = previous.replace(bridge_boundary, "").strip()
        index["claim_boundary"] = (base_boundary + " " + bridge_boundary).strip()
        index["bridge_followup"] = {
            "summary": "bridge_result_summary.json",
            "raw_result_copied": False,
            "source_sha256": summary["source_receipt"]["sha256"],
            "historical_provenance_receipt": {
                "path": HISTORICAL_RECEIPT_NAME,
                "sha256": summary["regeneration"].get("historical_receipt_sha256"),
                "raw_result_revalidated_this_invocation": False,
            },
            "status": "COMPLETE_EXPLORATORY",
            "target_exceeds_matched_max_all_seeds": summary["target_exceeds_matched_max_all_seeds"],
            "review_receipt": None,
            "reader_clamp": "complete L8H5 hook_z@final output; not value-only",
            "regeneration": summary.get("regeneration"),
        }
        policy = dict(index.get("package_policy", {}))
        policy["raw_bridge_results_copied"] = False
        policy["bridge_exploratory_verdict_emitted"] = False
        index["package_policy"] = policy
        json_dump(path, index)


def write_checksums(output: Path) -> None:
    checksum_path = output / "checksums.sha256"
    generated = sorted(path for path in output.rglob("*") if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in generated),
        encoding="utf-8",
    )


def _stage_packet(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    source = output if output.exists() else DEFAULT_OUTPUT
    if source.exists():
        for item in source.iterdir():
            target = stage / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    return stage


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_staged_packet(stage: Path, output: Path) -> None:
    """Publish one complete directory generation, restoring the prior one on failure."""

    lock = output.with_name(f".{output.name}.bridge-publish.lock")
    transaction: Path | None = None
    previous: Path | None = None
    previous_moved = False
    new_moved = False
    committed = False
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"bridge publication lock already exists: {lock}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        for name in BRIDGE_PUBLISH_FILES:
            if not (stage / name).is_file():
                raise ValueError(f"staged bridge packet lacks required publication file {name}")
            _fsync_path(stage / name)
        _fsync_path(stage)
        if not output.exists():
            os.rename(stage, output)
            new_moved = True
        else:
            os.chmod(stage, output.stat().st_mode & 0o7777)
            transaction = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.previous.", dir=output.parent)
            )
            previous = transaction / "packet"
            os.rename(output, previous)
            previous_moved = True
            os.rename(stage, output)
            new_moved = True
        _fsync_path(output)
        _fsync_path(output.parent)
        committed = True
        if transaction is not None:
            cleanup = transaction
            transaction = None
            try:
                shutil.rmtree(cleanup)
            except OSError:
                # The new generation is already durable. A partial failure while
                # deleting the obsolete backup must never roll back to that now
                # potentially damaged old generation.
                pass
            else:
                _fsync_path(output.parent)
    except BaseException:
        if not committed and previous_moved and previous is not None and previous.exists():
            failed = transaction / "failed-packet" if transaction is not None else None
            if new_moved and output.exists() and failed is not None:
                os.rename(output, failed)
                new_moved = False
            os.rename(previous, output)
            previous_moved = False
            _fsync_path(output)
            _fsync_path(output.parent)
            if transaction is not None:
                shutil.rmtree(transaction)
                transaction = None
                _fsync_path(output.parent)
        raise
    finally:
        if transaction is not None and transaction.exists() and not previous_moved:
            shutil.rmtree(transaction)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def build(bridge_results: Path, output: Path) -> Path:
    output = output.resolve()
    stage = _stage_packet(output)
    try:
        metric_rows, compact_rows, summary = validate_and_extract(bridge_results)
        write_csv(stage / "bridge_seed_metrics.csv", METRIC_FIELDS, metric_rows)
        write_csv(stage / "bridge_matched_ratios.csv", MATCHED_FIELDS, compact_rows)
        json_dump(stage / "bridge_result_summary.json", summary)
        figure_manifest = write_figure(stage, metric_rows, summary.get("regeneration"))
        json_dump(stage / "bridge_figure_manifest.json", figure_manifest)
        update_results_md(stage / "RESULTS.md")
        update_indexes(stage, summary)
        write_checksums(stage)
        _publish_staged_packet(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output


def build_checked_in(output: Path) -> Path:
    """Refresh only the presentation derived from exact published compact rows."""

    output = output.resolve()
    stage = _stage_packet(output)
    try:
        metric_rows, _compact_rows, summary = _checked_in_rows(stage)
        json_dump(stage / "bridge_result_summary.json", summary)
        figure_manifest = write_figure(stage, metric_rows, summary.get("regeneration"))
        json_dump(stage / "bridge_figure_manifest.json", figure_manifest)
        update_results_md(stage / "RESULTS.md")
        update_indexes(stage, summary)
        write_checksums(stage)
        _publish_staged_packet(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return output


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bridge-results", help="completed raw bridge result JSON (kept outside Git)")
    source.add_argument(
        "--reaggregate-checked-in",
        action="store_true",
        help="refresh derived presentation from the exact hash-bound checked-in compact CSVs",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="public results directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        output_path = Path(args.output).resolve()
        if args.reaggregate_checked_in:
            output = build_checked_in(output_path)
        else:
            output = build(Path(args.bridge_results).resolve(), output_path)
    except Exception as exc:
        print(f"make_bridge_summary: {exc}", file=sys.stderr)
        return 2
    print(f"wrote bridge evidence packet: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
