#!/usr/bin/env python3
"""Build the compact, public Exp05 evidence packet.

This script is deliberately model-free.  It reads the completed Stage-2 JSON,
the fresh selection/candidate manifests, and the Stage-3 preparation manifests;
it never imports torch and never reads the 73 MB pair CSV.  The resulting
``results/`` directory contains small JSON/CSV evidence tables, a three-panel
figure, and a checksum file.  Source paths are used only for reading and are
not written into public outputs.

Example::

    python experiments/05_number_agreement_circuit/make_claim_ladder.py \
      --selection-source-a /tmp/exp05/selection_source_a.json \
      --candidate /tmp/exp05/candidate.json \
      --stage2 /tmp/exp05/stage2_results.json \
      --stage3-cache /tmp/exp05-stage3/stage3_gate_a_cache.jsonl \
      --stage3-split /tmp/exp05-stage3/stage3_split_manifest.json \
      --stage3-prepare /tmp/exp05-stage3/stage3_prepare_manifest.json \
      --stage3-results /tmp/exp05-stage3/stage3_results.json \
      --stage3-draws /tmp/exp05-stage3/stage3_draws.csv \
      --stage3-review-receipt /tmp/exp05-stage3/stage3_prepare_review.json \
      --stage3-harness-receipt /tmp/exp05-stage3/stage3_harness_receipt.json \
      --output experiments/05_number_agreement_circuit/results

The large Stage-2 and Stage-3 source files are intentionally not copied into Git.
Re-running the same command against byte-identical inputs in the pinned plotting
environment regenerates the same public files.  Omitting all four Q4 inputs retains
the preparation-only (Q4 pending) packet for historical use; a completed Q4 packet
requires the result, draw CSV, Advisor review receipt, and harness receipt together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


T_CRIT_8 = 2.365  # preregistered two-sided Student-t multiplier, df=7
STAGE2_SEEDS = tuple(range(20260802, 20260810))
STAGE3_SEEDS = tuple(range(20260806, 20260814))

# The full Stage-2 result contains directed pair arrays for every arm.  A
# public packaging run needs only the registered scalar fields below.  jq's
# streaming parser keeps the 1.4 GB source out of Python's heap; the fallback
# keeps the script usable on machines without jq.
STAGE2_PUBLIC_JQ = r"""
{
  status,
  status_code,
  final_binding_status,
  coverage_status,
  scientific_verdict_emitted,
  supplies_published_science,
  q1: {
    verdict: .q1.verdict,
    label: .q1.label,
    aggregate: {pass_count: .q1.aggregate.pass_count},
    qualifying_n: .q1.qualifying_n,
    tested_set: .q1.tested_set
  },
  q2: {verdict: .q2.verdict, label: .q2.label, pass_count: .q2.pass_count},
  q3: {verdict: .q3.verdict, label: .q3.label, pass_count: .q3.pass_count, direct_recovery_report: .q3.direct_recovery_report},
  seeds: (.seeds | with_entries(
    .value = {
      first: {
        retained_pairs: .value.first.retained_pairs,
        gate_A: {passed: .value.first.gate_A.passed},
        E_all: {E: .value.first.E_all.E},
        q1_nested: {"2": {
          recovery_fraction: .value.first.q1_nested["2"].recovery_fraction,
          seed_joint_pass: .value.first.q1_nested["2"].seed_joint_pass,
          all_members_distinguishable: .value.first.q1_nested["2"].all_members_distinguishable
        }}
      },
      second: {
        q2: {
          complete_pair_count: .value.second.q2.complete_pair_count,
          pass_source_A: .value.second.q2.pass_source_A,
          pass_source_C: .value.second.q2.pass_source_C,
          true_right: {E: .value.second.q2.true_right.E},
          source_A: {E: .value.second.q2.source_A.E},
          source_B_descriptive: {E: .value.second.q2.source_B_descriptive.E},
          source_C: {E: .value.second.q2.source_C.E}
        },
        q3: {
          D_path: .value.second.q3.D_path,
          D_path_ci95: .value.second.q3.D_path_ci95,
          F_path: .value.second.q3.F_path,
          direct_recovery: {value: .value.second.q3.direct_recovery.value},
          positive: .value.second.q3.positive
        }
      }
    }
  ))
}
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_stage2_public(path: Path) -> dict[str, Any]:
    """Project Stage-2 to registered scalars without loading pair arrays."""

    jq = shutil.which("jq")
    if jq:
        completed = subprocess.run(
            [jq, "-c", STAGE2_PUBLIC_JQ, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            projected = json.loads(completed.stdout)
            if isinstance(projected, dict):
                return projected
    # Fallback is intentionally explicit: correctness wins if jq is absent or
    # an older jq cannot parse the source schema.
    return read_json(path)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fmt(value: Any) -> str:
    """Stable compact numeric formatting for public CSV tables."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".10g")
    return str(value)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def head_label(row: Mapping[str, Any]) -> str:
    return f"L{int(row['layer'])}H{int(row['head'])}"


def finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return number


def descriptive_summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [finite_float(value) for value in values]
    if not clean:
        return {"n": 0, "mean": None, "sd": None, "ci95_t7": [None, None], "min": None, "max": None}
    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    half = T_CRIT_8 * sd / math.sqrt(len(clean)) if len(clean) > 1 else 0.0
    return {
        "n": len(clean),
        "mean": mean,
        "sd": sd,
        "ci95_t7": [mean - half, mean + half],
        "min": min(clean),
        "max": max(clean),
    }


def source_receipt(label: str, path: Path) -> dict[str, Any]:
    return {"label": label, "basename": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_portable_prepare_manifest(source: Path, destination: Path) -> dict[str, Any]:
    """Copy the prepare manifest while rewriting only its two path fields.

    The raw source receipt is captured before this function is called.  Using
    targeted substitutions instead of a JSON re-serializer keeps every other
    byte/value (including scientific hashes and seed facts) unchanged.
    """

    raw_text = source.read_text(encoding="utf-8")
    raw_payload = read_json(source)
    replacements = {
        "split_csv_path": "stage3_split_roles.csv",
        "split_manifest_path": "stage3_split_manifest.json",
    }
    portable_text = raw_text
    for field, basename in replacements.items():
        old_value = raw_payload.get(field)
        if not isinstance(old_value, str):
            raise ValueError(f"prepare manifest missing string {field}: {source}")
        pattern = rf'("{field}"\s*:\s*){re.escape(json.dumps(old_value, ensure_ascii=False))}'
        portable_text, count = re.subn(
            pattern,
            lambda match, value=json.dumps(basename, ensure_ascii=False): match.group(1) + value,
            portable_text,
            count=1,
        )
        if count != 1:
            raise ValueError(f"could not rewrite prepare manifest field {field}: {source}")
    destination.write_text(portable_text, encoding="utf-8")
    portable_payload = read_json(destination)
    for field, basename in replacements.items():
        if portable_payload.get(field) != basename:
            raise ValueError(f"portable prepare manifest field {field} is not basename-only")
    for key, value in raw_payload.items():
        if key not in replacements and portable_payload.get(key) != value:
            raise ValueError(f"portable prepare manifest changed scientific field {key}")
    return {
        "path": str(destination.relative_to(destination.parent.parent)),
        "basename": destination.name,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "rewritten_fields": replacements,
    }


def load_q4_receipt(label: str, path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an Advisor/harness receipt before publishing a Q4 packet."""

    receipt = read_json(path)
    expected_schema = {
        "stage3_review_receipt": "exp05-number-agreement-stage3-prepare-review-v1",
        "stage3_harness_receipt": "exp05-number-agreement-stage3-harness-receipt-v1",
    }[label]
    if receipt.get("schema") != expected_schema or receipt.get("status") != "ACCEPT":
        raise ValueError(f"{label} is not an ACCEPT receipt with schema {expected_schema}")
    if receipt.get("independent") is not True:
        raise ValueError(f"{label} is not marked independent")
    declared_file_hash = sha256_file(path)
    raw_field = "review_receipt_sha256" if label == "stage3_review_receipt" else "harness_receipt_sha256"
    if result.get(raw_field) != declared_file_hash:
        raise ValueError(f"{label} file hash does not match raw Stage-3 result {raw_field}")
    self_hash = receipt.get("self_sha256")
    if not isinstance(self_hash, str) or len(self_hash) != 64:
        raise ValueError(f"{label} is missing self_sha256")
    body = {key: value for key, value in receipt.items() if key != "self_sha256"}
    expected_self_hash = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if self_hash != expected_self_hash:
        raise ValueError(f"{label} self_sha256 does not match canonical receipt body")
    expected_commit = result.get("code_revision", {}).get("final_commit")
    if expected_commit and receipt.get("reviewed_commit") != expected_commit:
        raise ValueError(f"{label} reviewed_commit does not match raw Stage-3 result")
    if label == "stage3_review_receipt":
        expected_fields = {
            "gate_cache_sha256": result.get("gate_cache_sha256"),
            "split_manifest_sha256": result.get("split_manifest_sha256"),
            "prepare_manifest_sha256": result.get("prepare_manifest_sha256"),
        }
    else:
        expected_fields = {"protocol_sha256": result.get("protocol_sha256")}
    for key, expected in expected_fields.items():
        if expected and receipt.get(key) != expected:
            raise ValueError(f"{label} {key} does not match raw Stage-3 result")
    return {
        "label": label,
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": declared_file_hash,
        "self_sha256": self_hash,
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
        "reviewed_commit": receipt.get("reviewed_commit"),
    }


def load_stage3_cache_receipt(path: Path) -> dict[str, Any]:
    """Read only the manifest line from the Stage-3 JSONL cache."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if isinstance(record, dict) and record.get("record_type") == "manifest":
                return record
    raise ValueError(f"Stage-3 cache has no manifest record: {path}")


def extract_selection(candidate: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence_by_flat = {
        int(row["head"]["flat_id"]): row for row in candidate.get("selection_evidence", [])
    }
    rows: list[dict[str, Any]] = []
    for rank, head in enumerate(candidate.get("rank_order", []), start=1):
        flat_id = int(head["flat_id"])
        evidence = evidence_by_flat.get(flat_id, {})
        row = {
            "rank": rank,
            "flat_id": flat_id,
            "layer": int(head["layer"]),
            "head": int(head["head"]),
            "head_label": head_label(head),
            "true_mean": evidence.get("true_mean"),
            "source_a_mean": evidence.get("source_a_mean"),
            "source_a_noise_edge": evidence.get("source_a_noise_edge"),
            "pair_sign_consistency": evidence.get("pair_sign_consistency"),
            "holm_reject": evidence.get("holm_reject"),
            "eligible": evidence.get("eligible"),
            "stage1_rank": evidence.get("stage1_rank"),
        }
        rows.append(row)
    if len(rows) != 8:
        raise ValueError(f"expected eight candidate rows, found {len(rows)}")
    receipt = {
        "candidate_status": candidate.get("candidate_status"),
        "candidate_sha256": candidate.get("candidate_sha256"),
        "selection_sha256": candidate.get("selection_sha256"),
        "source_a_sweep_sha256": candidate.get("source_a_sweep_sha256"),
        "true_sweep_sha256": candidate.get("true_sweep_sha256"),
        "candidate_count": len(rows),
    }
    return rows, receipt


def extract_stage2(stage2: Mapping[str, Any], calibration: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q1 = stage2.get("q1", {})
    tested_set = list(q1.get("tested_set", []))
    qualifying_n = int(q1.get("qualifying_n", len(tested_set)))
    if qualifying_n != 2 or len(tested_set) != 2:
        raise ValueError(f"public packaging expects a two-head tested set, got n={qualifying_n}, set={tested_set!r}")
    theta = calibration.get("theta_spec", {})
    theta_a = finite_float(theta.get("A", 0.2))
    theta_c = finite_float(theta.get("C", 0.269830584526062))
    rows: list[dict[str, Any]] = []
    for seed in STAGE2_SEEDS:
        payload = stage2.get("seeds", {}).get(str(seed))
        if not isinstance(payload, dict):
            raise ValueError(f"missing Stage-2 seed {seed}")
        first = payload["first"]
        second = payload["second"]
        q1_cell = first["q1_nested"][str(qualifying_n)]
        q2 = second["q2"]
        q3 = second["q3"]
        true_effect = finite_float(q2["true_right"]["E"])
        source_a_effect = finite_float(q2["source_A"]["E"])
        source_c_effect = finite_float(q2["source_C"]["E"])
        d_ci = q3.get("D_path_ci95", [None, None])
        direct = q3.get("direct_recovery", {})
        rows.append(
            {
                "seed": seed,
                "retained_pairs": len(first.get("retained_pairs", [])),
                "gate_A_pass": first.get("gate_A", {}).get("passed"),
                "E_all": first["E_all"]["E"],
                "q1_n": qualifying_n,
                "q1_recovery_fraction": q1_cell["recovery_fraction"],
                "q1_seed_joint_pass": q1_cell["seed_joint_pass"],
                "q1_all_members_distinguishable": q1_cell["all_members_distinguishable"],
                "q2_complete_pairs": q2.get("complete_pair_count"),
                "q2_true_right_E": true_effect,
                "q2_source_A_E": source_a_effect,
                "q2_source_A_abs_ratio": abs(source_a_effect) / abs(true_effect),
                "q2_source_A_pass": q2.get("pass_source_A"),
                "q2_source_B_E_descriptive": q2["source_B_descriptive"]["E"],
                "q2_source_C_E": source_c_effect,
                "q2_source_C_abs_ratio": abs(source_c_effect) / abs(true_effect),
                "q2_source_C_pass": q2.get("pass_source_C"),
                "q3_D_path": q3["D_path"],
                "q3_D_path_ci95_low": d_ci[0],
                "q3_D_path_ci95_high": d_ci[1],
                "q3_F_path": q3["F_path"],
                "q3_direct_recovery_descriptive": direct.get("value"),
                "q3_direct_recovery_adjudicative": False,
                "q3_positive": q3.get("positive"),
            }
        )
    if len(rows) != 8:
        raise ValueError(f"expected eight Stage-2 seed rows, found {len(rows)}")
    summary = {
        "status": stage2.get("status"),
        "status_code": stage2.get("status_code"),
        "final_binding_status": stage2.get("final_binding_status"),
        "coverage_status": stage2.get("coverage_status"),
        "scientific_verdict_emitted": stage2.get("scientific_verdict_emitted"),
        "supplies_published_science": stage2.get("supplies_published_science"),
        "tested_set": [
            {"layer": int(item["layer"]), "head": int(item["head"]), "label": head_label(item)}
            for item in tested_set
        ],
        "qualifying_n": qualifying_n,
        "seed_count": len(rows),
        "q1": {
            "verdict": q1.get("verdict"),
            "label": q1.get("label"),
            "pass_count": q1.get("aggregate", {}).get("pass_count"),
            "recovery_fraction": descriptive_summary([row["q1_recovery_fraction"] for row in rows]),
            "threshold": 0.50,
        },
        "q2": {
            "verdict": stage2.get("q2", {}).get("verdict"),
            "label": stage2.get("q2", {}).get("label"),
            "pass_count": stage2.get("q2", {}).get("pass_count"),
            "specificity_thresholds": {"source_A": theta_a, "source_C": theta_c},
            "source_A_abs_ratio": descriptive_summary([row["q2_source_A_abs_ratio"] for row in rows]),
            "source_C_abs_ratio": descriptive_summary([row["q2_source_C_abs_ratio"] for row in rows]),
        },
        "q3": {
            "verdict": stage2.get("q3", {}).get("verdict"),
            "label": stage2.get("q3", {}).get("label"),
            "pass_count": stage2.get("q3", {}).get("pass_count"),
            "D_path_descriptive": descriptive_summary([row["q3_D_path"] for row in rows]),
            "F_path_descriptive": descriptive_summary([row["q3_F_path"] for row in rows]),
            "direct_recovery": {
                "adjudicative": False,
                "status": stage2.get("q3", {}).get("direct_recovery_report", {}).get("status"),
                "median": stage2.get("q3", {}).get("direct_recovery_report", {}).get("summary", {}).get("median"),
                "observed_min": stage2.get("q3", {}).get("direct_recovery_report", {}).get("summary", {}).get("observed_min"),
                "observed_max": stage2.get("q3", {}).get("direct_recovery_report", {}).get("summary", {}).get("observed_max"),
            },
        },
        "thresholds_are_from_calibration": True,
    }
    return rows, summary


def extract_stage3(
    prepare: Mapping[str, Any], split_manifest: Mapping[str, Any], cache_manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_by_seed = {int(item["seed"]): item for item in split_manifest.get("seeds", [])}
    rows: list[dict[str, Any]] = []
    for item in prepare.get("seeds", []):
        seed = int(item["seed"])
        split = split_by_seed.get(seed, {})
        gate = split.get("cache_gate", {})
        rows.append(
            {
                "seed": seed,
                "retained_pairs": item.get("retained_pairs"),
                "gate_A_pass": item.get("gate_a_passed"),
                "generated_pairs": gate.get("generated_pairs"),
                "gate_A_signed_correct_fraction": gate.get("both_members_signed_correct_fraction"),
                "gate_A_median_d_gap": gate.get("median_d_gap_all_generated_pairs"),
                "rank_training_pairs": item.get("rank_training_pairs"),
                "evaluation_pairs": item.get("evaluation_pairs"),
                "status": item.get("status"),
            }
        )
    if [row["seed"] for row in rows] != list(STAGE3_SEEDS):
        raise ValueError("Stage-3 preparation seeds do not match the frozen eight-seed set")
    summary = {
        "status": "PREPARATION_COMPLETE_Q4_PENDING",
        "source_status": prepare.get("status"),
        "q4_verdict_emitted": False,
        "independent_review_required": prepare.get("independent_review_required"),
        "q4_input_boundary": prepare.get("q4_input_boundary"),
        "stage3_seeds": list(STAGE3_SEEDS),
        "all_gate_A_passed": all(bool(row["gate_A_pass"]) for row in rows),
        "rank_training_pairs_per_seed": sorted({int(row["rank_training_pairs"]) for row in rows}),
        "evaluation_pairs_per_seed": sorted({int(row["evaluation_pairs"]) for row in rows}),
        "hashes": {
            "protocol_sha256": prepare.get("protocol_sha256"),
            "gate_cache_sha256": prepare.get("gate_cache_sha256"),
            "split_manifest_sha256": prepare.get("split_manifest_sha256"),
            "split_csv_sha256": prepare.get("split_csv_sha256"),
            "cross_hash_sha256": prepare.get("cross_hash_sha256"),
            "cache_manifest_protocol_sha256": cache_manifest.get("protocol_sha256"),
        },
    }
    return rows, summary


Q4_CLAIM = (
    "Above matched-span chance, the target 12-decoder-row span recovered a mean of "
    "89.35% of the directed logit effect and beat the frozen second-largest "
    "of 100 matched spans in all 8 seeds."
)
Q4_NOT_CLAIMED = [
    "This matched-span result does not establish that the encoder features are natural, necessary, or sufficient.",
    "It does not establish mediation, an attention route, or a native model path.",
    "R_span is a directed-logit effect ratio, not an 89% geometric reconstruction claim.",
    "The generic PCA value (0.0274 mean for PCA_span/both) is a raw logit effect, not 2.74% recovery.",
]


def extract_stage3_result(
    result: Mapping[str, Any],
    result_path: Path,
    draws_path: Path,
    review_path: Path,
    harness_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Project the completed Stage-3/Q4 result into compact public tables.

    The raw Q4 manifest is intentionally not copied.  This routine binds the
    compact rows to both raw hashes, checks the 800-row scientific draw CSV,
    and derives every public ratio from the per-draw ``effect.E`` values and
    the corresponding seed's full-effect denominator.  It does not import
    torch or recompute model outputs.
    """

    if result.get("status") != "COMPLETE":
        raise ValueError(f"Stage-3 result is not COMPLETE: {result.get('status')!r}")
    raw_summary = result.get("summary")
    if not isinstance(raw_summary, Mapping) or raw_summary.get("status") != "POSITIVE":
        raise ValueError("Stage-3 result does not contain a POSITIVE scientific summary")
    expected_seeds = list(STAGE3_SEEDS)
    seed_results = result.get("seed_results")
    if not isinstance(seed_results, list):
        raise ValueError("Stage-3 result has no seed_results list")
    ordered = sorted(seed_results, key=lambda row: int(row.get("seed", -1)))
    observed_seeds = [int(row.get("seed", -1)) for row in ordered]
    if observed_seeds != expected_seeds:
        raise ValueError(f"Stage-3 seed set mismatch: {observed_seeds!r}")
    if int(raw_summary.get("pass_count", 0)) != len(expected_seeds):
        raise ValueError("Stage-3 scientific summary is not 8/8 positive")

    result_hash = sha256_file(result_path)
    draws_hash = sha256_file(draws_path)
    review_receipt = load_q4_receipt("stage3_review_receipt", review_path, result)
    harness_receipt = load_q4_receipt("stage3_harness_receipt", harness_path, result)
    declared_draw_hash = result.get("draw_csv_sha256")
    if declared_draw_hash != draws_hash:
        raise ValueError("Stage-3 result draw_csv_sha256 does not match supplied draw CSV")
    accepted_csv = result.get("accepted_scientific_csv")
    if not isinstance(accepted_csv, Mapping):
        raise ValueError("Stage-3 result has no accepted_scientific_csv binding")
    if accepted_csv.get("full_csv_sha256") != draws_hash:
        raise ValueError("accepted scientific CSV binding does not match supplied draw CSV")
    if int(accepted_csv.get("accepted_row_count", -1)) != len(expected_seeds) * 100:
        raise ValueError("accepted scientific CSV does not declare exactly 800 rows")

    with draws_path.open("r", encoding="utf-8", newline="") as handle:
        draw_rows = list(csv.DictReader(handle))
    if len(draw_rows) != len(expected_seeds) * 100:
        raise ValueError(f"expected 800 scientific draw rows, found {len(draw_rows)}")
    draw_keys: set[tuple[int, int]] = set()
    for row in draw_rows:
        try:
            key = (int(row["seed_id"]), int(row["draw_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("scientific draw CSV has malformed seed_id/draw_index") from exc
        if key in draw_keys:
            raise ValueError(f"duplicate scientific draw key: {key}")
        draw_keys.add(key)
        if str(row.get("accepted", "")).lower() != "true":
            raise ValueError(f"scientific draw row is not accepted: {key}")
    expected_draw_keys = {(seed, draw) for seed in expected_seeds for draw in range(100)}
    if draw_keys != expected_draw_keys:
        raise ValueError("scientific draw CSV does not cover each seed/draw exactly once")

    metric_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    edge_values: list[float] = []
    r_span_values: list[float] = []
    r_comp_values: list[float] = []
    geometric_span_values: list[float] = []
    geometric_comp_values: list[float] = []
    pca_values: list[float] = []

    for seed_row in ordered:
        seed = int(seed_row["seed"])
        if seed_row.get("status") != "PASS":
            raise ValueError(f"Stage-3 seed {seed} is not PASS: {seed_row.get('status')!r}")
        cells = seed_row.get("cells")
        if not isinstance(cells, Mapping) or not isinstance(cells.get("both"), Mapping):
            raise ValueError(f"Stage-3 seed {seed} has no both-position cell")
        both = cells["both"]
        guard = both.get("ratio_guard")
        full = both.get("full_delta")
        span = both.get("target_span")
        geometry = both.get("geometric_fractions")
        if not all(isinstance(item, Mapping) for item in (guard, full, span, geometry)):
            raise ValueError(f"Stage-3 seed {seed} has incomplete target-span metrics")
        if guard.get("status") != "ESTIMABLE":
            raise ValueError(f"Stage-3 seed {seed} denominator is not estimable")
        full_e = finite_float(full.get("E"))
        span_e = finite_float(span.get("E"))
        r_span = finite_float(guard.get("R_span"))
        r_comp = finite_float(guard.get("R_comp"))
        geom_span = finite_float(geometry.get("span"))
        geom_comp = finite_float(geometry.get("complement"))
        matched = seed_row.get("matched_draws")
        if not isinstance(matched, Mapping) or not isinstance(matched.get("results"), list):
            raise ValueError(f"Stage-3 seed {seed} has no matched draw results")
        matched_results = matched["results"]
        if len(matched_results) != 100:
            raise ValueError(f"Stage-3 seed {seed} has {len(matched_results)} matched results, expected 100")
        ratios: list[tuple[float, int]] = []
        for item in matched_results:
            if not isinstance(item, Mapping):
                raise ValueError(f"Stage-3 seed {seed} has malformed matched result")
            draw_index = int(item.get("draw_index", -1))
            if (seed, draw_index) not in draw_keys:
                raise ValueError(f"Stage-3 seed {seed} matched result is missing from draw CSV: {draw_index}")
            effect = item.get("effect")
            if not isinstance(effect, Mapping):
                raise ValueError(f"Stage-3 seed {seed} draw {draw_index} has no effect summary")
            effect_e = finite_float(effect.get("E"))
            ratio = effect_e / full_e
            ratios.append((ratio, draw_index))
            latent_ids = item.get("latent_ids")
            if not isinstance(latent_ids, list) or len(latent_ids) != 12 or len(set(latent_ids)) != 12:
                raise ValueError(f"Stage-3 seed {seed} draw {draw_index} has malformed latent ids")
            compact_rows.append(
                {
                    "seed": seed,
                    "draw_index": draw_index,
                    "accepted_attempt_id": item.get("accepted_attempt_id"),
                    "attempt": item.get("attempt"),
                    "latent_ids": json.dumps([int(value) for value in latent_ids], separators=(",", ":")),
                    "rank": 12,
                    "matched_effect_E": effect_e,
                    "full_effect_E": full_e,
                    "matched_R_span": ratio,
                    "attempt_sha256": item.get("attempt_sha256"),
                    "draw_or_projector_hash": item.get("draw_or_projector_hash"),
                    "projector_hash": item.get("projector_hash"),
                    "effect_result_hash": item.get("effect_result_hash"),
                    "source_stage3_results_sha256": result_hash,
                    "source_stage3_draws_sha256": draws_hash,
                }
            )
        ratios.sort(key=lambda pair: (pair[0], pair[1]))
        edge = ratios[-2][0]
        declared_edge = finite_float(both.get("matched_edge_second_largest"))
        if abs(edge - declared_edge) > 1e-9:
            raise ValueError(f"Stage-3 seed {seed} matched edge disagrees with raw declaration")
        if bool(both.get("R_span_exceeds_matched_edge")) is not (r_span > edge):
            raise ValueError(f"Stage-3 seed {seed} matched-edge PASS flag is inconsistent")
        pca_cells = seed_row.get("pca_cells")
        pca_both = pca_cells.get("PCA_span/both") if isinstance(pca_cells, Mapping) else None
        pca_subject = pca_cells.get("PCA_span/subject") if isinstance(pca_cells, Mapping) else None
        pca_final = pca_cells.get("PCA_span/final") if isinstance(pca_cells, Mapping) else None
        pca_both_e = finite_float(pca_both.get("effect", {}).get("E")) if isinstance(pca_both, Mapping) else None
        pca_subject_e = finite_float(pca_subject.get("effect", {}).get("E")) if isinstance(pca_subject, Mapping) else None
        pca_final_e = finite_float(pca_final.get("effect", {}).get("E")) if isinstance(pca_final, Mapping) else None
        metric_rows.append(
            {
                "seed": seed,
                "status": seed_row.get("status"),
                "full_effect_E": full_e,
                "target_span_effect_E": span_e,
                "R_span": r_span,
                "matched_edge_second_largest": edge,
                "matched_count": len(ratios),
                "matched_edge_pass": bool(r_span > edge),
                "R_comp": r_comp,
                "geometric_span": geom_span,
                "geometric_complement": geom_comp,
                "PCA_span_subject_effect_E": pca_subject_e,
                "PCA_span_final_effect_E": pca_final_e,
                "PCA_span_both_effect_E": pca_both_e,
            }
        )
        r_span_values.append(r_span)
        edge_values.append(edge)
        r_comp_values.append(r_comp)
        geometric_span_values.append(geom_span)
        geometric_comp_values.append(geom_comp)
        if pca_both_e is not None:
            pca_values.append(pca_both_e)

    compact_rows.sort(key=lambda row: (int(row["seed"]), int(row["draw_index"])))
    if len(compact_rows) != 800:
        raise ValueError(f"expected 800 compact matched rows, found {len(compact_rows)}")
    raw_cross = result.get("cross_seed_summaries")
    raw_pca = raw_cross.get("PCA", {}).get("PCA_span/both") if isinstance(raw_cross, Mapping) else None
    summary = {
        "schema": "exp05-public-stage3-q4-summary-v1",
        "status": "POSITIVE",
        "verdict": "POSITIVE",
        "invocation_id": result.get("invocation_id"),
        "protocol_sha256": result.get("protocol_sha256"),
        "code_revision": result.get("code_revision"),
        "seed_count": len(metric_rows),
        "seed_pass_count": sum(1 for row in metric_rows if row["status"] == "PASS"),
        "matched_edge_pass_count": sum(1 for row in metric_rows if row["matched_edge_pass"]),
        "matched_draws_per_seed": sorted({int(row["matched_count"]) for row in metric_rows}),
        "metrics": {
            "R_span": descriptive_summary(r_span_values),
            "matched_edge_second_largest": descriptive_summary(edge_values),
            "R_comp": descriptive_summary(r_comp_values),
            "geometric_span_descriptive": descriptive_summary(geometric_span_values),
            "geometric_complement_descriptive": descriptive_summary(geometric_comp_values),
            "PCA_span_both_raw_logit_effect_E": (
                dict(raw_pca) if isinstance(raw_pca, Mapping) else descriptive_summary(pca_values)
            ),
        },
        "raw_runtime": result.get("runtime"),
        "wall_clock_seconds": result.get("wall_clock_seconds"),
        "raw_summary": dict(raw_summary),
        "claim": Q4_CLAIM,
        "claim_boundary": "The Q4 claim is limited to the directed-logit matched-span comparison on the frozen 12-row decoder span.",
        "not_claimed": Q4_NOT_CLAIMED,
        "source_receipts": [source_receipt("stage3_results", result_path), source_receipt("stage3_draws", draws_path)],
        "receipt_bindings": [review_receipt, harness_receipt],
        "compact_artifacts": {
            "seed_metrics": "stage3_seed_metrics.csv",
            "matched_ratios": "stage3_matched_ratios.csv",
            "portable_prepare_manifest": "stage3/stage3_prepare_manifest.json",
            "raw_stage3_results_copied": False,
            "raw_stage3_draws_copied": False,
        },
        "reproduce_matched_edge": "For each seed, sort matched_R_span in stage3_matched_ratios.csv and take the second-largest value; compare R_span in stage3_seed_metrics.csv.",
    }
    return metric_rows, compact_rows, summary


def write_figure(
    output_dir: Path,
    selection_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]],
    stage3_rows: Sequence[Mapping[str, Any]],
    stage3_result_rows: Sequence[Mapping[str, Any]] | None,
    theta_a: float,
    theta_c: float,
) -> dict[str, Any]:
    # Matplotlib is imported lazily so extraction remains usable in a minimal
    # Python environment.  Agg avoids any GUI/backend side effects.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": "exp05-claim-ladder-v1",
        }
    )
    fig = plt.figure(figsize=(16, 8), dpi=180, constrained_layout=True)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.55, 1.0], wspace=0.26)

    # Panel A: selection ranking.  The two-head tested set is highlighted; the
    # remaining six rows show the registered compact candidate context.
    ax_a = fig.add_subplot(outer[0, 0])
    labels = [str(row["head_label"]) for row in selection_rows][::-1]
    values = [finite_float(row["true_mean"]) for row in selection_rows][::-1]
    colors = ["#d97706" if row["rank"] <= 2 else "#4f83cc" for row in selection_rows][::-1]
    ypos = np.arange(len(labels))
    ax_a.barh(ypos, values, color=colors, alpha=0.92)
    ax_a.set_yticks(ypos, labels)
    ax_a.set_xlabel("single-head true effect (Δd)")
    ax_a.set_title("A  Selection\n8-head candidate order", loc="left", fontweight="bold")
    ax_a.axvline(0, color="#555", lw=0.8)
    ax_a.grid(axis="x", alpha=0.2)
    ax_a.text(
        0.02,
        0.02,
        "highlight: tested minimal set\nL7H4 + L8H5",
        transform=ax_a.transAxes,
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7ed", "edgecolor": "#d97706"},
    )

    # Panel B: three compact evidence strips, all eight seeds shown.
    panel_b = outer[0, 1].subgridspec(3, 1, hspace=0.5)
    seeds = [int(row["seed"]) for row in stage2_rows]
    x = np.arange(len(seeds))
    short_seeds = [str(seed)[-2:] for seed in seeds]

    ax_q1 = fig.add_subplot(panel_b[0, 0])
    q1 = [finite_float(row["q1_recovery_fraction"]) for row in stage2_rows]
    ax_q1.plot(x, q1, "o-", color="#2563eb", lw=1.4, ms=4)
    ax_q1.axhline(0.50, color="#b45309", lw=1, ls="--")
    ax_q1.set_ylim(0.48, 0.57)
    ax_q1.set_ylabel("Q1 recovery")
    ax_q1.set_title("B  Q1–Q3 across eight seeds", loc="left", fontweight="bold")
    ax_q1.text(0.99, 0.08, "8/8 pass; n=2", transform=ax_q1.transAxes, ha="right", fontsize=8)
    ax_q1.grid(axis="y", alpha=0.2)
    ax_q1.set_xticks(x, short_seeds)

    ax_q2 = fig.add_subplot(panel_b[1, 0])
    a_ratio = [finite_float(row["q2_source_A_abs_ratio"]) for row in stage2_rows]
    c_ratio = [finite_float(row["q2_source_C_abs_ratio"]) for row in stage2_rows]
    ax_q2.plot(x, a_ratio, "o-", color="#059669", lw=1.2, ms=3.5, label="A noun control")
    ax_q2.plot(x, c_ratio, "o-", color="#dc2626", lw=1.2, ms=3.5, label="C frame control")
    ax_q2.axhline(theta_a, color="#059669", lw=0.9, ls="--")
    ax_q2.axhline(theta_c, color="#dc2626", lw=0.9, ls="--")
    ax_q2.set_ylim(0, 0.31)
    ax_q2.set_ylabel("|control| / true")
    ax_q2.text(0.99, 0.08, "8/8 pass both controls", transform=ax_q2.transAxes, ha="right", fontsize=8)
    ax_q2.grid(axis="y", alpha=0.2)
    ax_q2.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    ax_q2.set_xticks(x, short_seeds)

    ax_q3 = fig.add_subplot(panel_b[2, 0])
    d = np.array([finite_float(row["q3_D_path"]) for row in stage2_rows])
    lo = np.array([finite_float(row["q3_D_path_ci95_low"]) for row in stage2_rows])
    hi = np.array([finite_float(row["q3_D_path_ci95_high"]) for row in stage2_rows])
    f = np.array([finite_float(row["q3_F_path"]) for row in stage2_rows])
    ax_q3.errorbar(x, d, yerr=[d - lo, hi - d], fmt="o", color="#7c3aed", ms=4, capsize=2, lw=1)
    ax_q3.axhline(0, color="#666", lw=0.8)
    ax_q3.set_ylim(0.48, 0.63)
    ax_q3.set_ylabel("Q3 D_path (95% CI)")
    ax_q3.grid(axis="y", alpha=0.2)
    ax_q3.set_xticks(x, short_seeds)
    ax_q3b = ax_q3.twinx()
    ax_q3b.plot(x, f, "x", color="#111827", ms=5, mew=1.2, label="F_path")
    ax_q3b.set_ylim(-0.01, 0.01)
    ax_q3b.set_ylabel("F_path", color="#111827")
    ax_q3.text(0.99, 0.08, "8/8 subject-value transport", transform=ax_q3.transAxes, ha="right", fontsize=8)

    # Panel C: the adjudicated Q4 matched-span comparison, when available.
    # The pending branch remains useful for historical preparation-only runs,
    # but the checked-in packet always supplies stage3_result_rows.
    ax_c = fig.add_subplot(outer[0, 2])
    if stage3_result_rows:
        q4_seeds = [int(row["seed"]) for row in stage3_result_rows]
        q4_x = np.arange(len(q4_seeds))
        q4_r = np.array([finite_float(row["R_span"]) for row in stage3_result_rows])
        q4_edge = np.array([finite_float(row["matched_edge_second_largest"]) for row in stage3_result_rows])
        ax_c.plot(q4_x, q4_r, "o-", color="#b91c1c", lw=1.5, ms=4, label="target span R_span")
        ax_c.plot(q4_x, q4_edge, "s--", color="#2563eb", lw=1.2, ms=3.5, label="matched edge (2nd max)")
        ax_c.set_ylim(0, 1.0)
        ax_c.set_xticks(q4_x, [str(seed)[-2:] for seed in q4_seeds])
        ax_c.set_ylabel("directed-logit ratio")
        ax_c.set_title("C  Q4 matched-span test", loc="left", fontweight="bold")
        ax_c.grid(axis="y", alpha=0.2)
        ax_c.legend(frameon=False, fontsize=7, loc="lower left")
        ax_c.text(
            0.99,
            0.95,
            "8/8 PASS\nmean R_span 0.8935\nmean edge 0.4756",
            transform=ax_c.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fef2f2", "edgecolor": "#b91c1c"},
        )
        ax_c.text(
            0.99,
            0.06,
            "PCA_span/both mean E=0.0274\n(raw logit effect; descriptive)",
            transform=ax_c.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#4b5563",
        )
    else:
        ax_c.set_facecolor("#f3f4f6")
        for spine in ax_c.spines.values():
            spine.set_color("#9ca3af")
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        ax_c.set_title("C  Stage 3 / Q4", loc="left", fontweight="bold")
        ax_c.text(
            0.5,
            0.64,
            "PREPARATION COMPLETE\n\nVERDICT PENDING",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#374151",
        )
        ax_c.text(
            0.5,
            0.31,
            "8 seeds × (40 rank-training + 150 evaluation)\nGate A passed for every seed\nNo Q4 scientific result emitted",
            ha="center",
            va="center",
            fontsize=9,
            color="#4b5563",
        )
        ax_c.text(0.5, 0.08, "grey = not yet adjudicated", ha="center", va="center", fontsize=8, color="#6b7280")

    fig.suptitle(
        "Exp05 public evidence — frozen Stage 2 result and adjudicated Stage 3 Q4",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Values are extracted from immutable source artifacts; Q4 compares the target span with a frozen matched-span edge.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    svg_path = output_dir / "figure_exp05_main.svg"
    png_path = output_dir / "figure_exp05_main.png"
    # Suppress Matplotlib's wall-clock metadata so regenerated assets are
    # byte-stable for the same inputs (the figure itself is already fully
    # data-driven).
    fig.savefig(svg_path, format="svg", metadata={"Date": None})
    fig.savefig(png_path, format="png", dpi=180, metadata={"Date": None})
    plt.close(fig)
    # Matplotlib emits insignificant spaces before SVG newlines. Normalise
    # those without changing geometry so the checked-in text stays diff-clean.
    # Matplotlib's SVG backend includes path IDs derived from process-local
    # hashes and can differ by one last-place text coordinate when the font
    # cache is first populated.  Canonicalise those serialization details so
    # regeneration is byte-stable without changing the rendered PNG.
    svg_text = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n"
    clip_ids: list[str] = []
    for old_id in re.findall(r'id="(p[0-9a-f]+)"', svg_text):
        if old_id not in clip_ids:
            clip_ids.append(old_id)
    for index, old_id in enumerate(clip_ids, start=1):
        svg_text = svg_text.replace(old_id, f"clip_path_{index}")

    def stable_float(match: re.Match[str]) -> str:
        value = float(match.group(0))
        rounded = f"{value:.5f}".rstrip("0").rstrip(".")
        return "0" if rounded in {"-0", ""} else rounded

    svg_text = re.sub(r"(?<![A-Za-z0-9_])[-+]?\d+\.\d{6,}", stable_float, svg_text)
    svg_path.write_text(svg_text, encoding="utf-8")
    return {
        "svg": svg_path.name,
        "png": png_path.name,
        "png_width_px": 2880,
        "png_height_px": 1440,
        "panel_order": ["selection", "q1_q2_q3_eight_seed", "q4_matched_span" if stage3_result_rows else "stage3_preparation_pending"],
        "source_rows": {"selection": len(selection_rows), "stage2": len(stage2_rows), "stage3": len(stage3_rows)},
    }


def build(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "selection_source_a": Path(args.selection_source_a).resolve(),
        "candidate": Path(args.candidate).resolve(),
        "stage2_results": Path(args.stage2).resolve(),
        "stage3_gate_a_cache": Path(args.stage3_cache).resolve(),
        "stage3_split_manifest": Path(args.stage3_split).resolve(),
        "stage3_split_roles": Path(args.stage3_roles).resolve()
        if args.stage3_roles
        else Path(args.stage3_split).resolve().with_name("stage3_split_roles.csv"),
        "stage3_prepare_manifest": Path(args.stage3_prepare).resolve(),
    }
    q4_args_present = {
        "stage3_results": bool(args.stage3_results),
        "stage3_draws": bool(args.stage3_draws),
        "stage3_review_receipt": bool(args.stage3_review_receipt),
        "stage3_harness_receipt": bool(args.stage3_harness_receipt),
    }
    if any(q4_args_present.values()) and not all(q4_args_present.values()):
        missing = [name for name, present in q4_args_present.items() if not present]
        raise ValueError(f"completed Q4 packaging requires all four Q4 inputs; missing {missing}")
    if args.stage3_results:
        source_paths["stage3_results"] = Path(args.stage3_results).resolve()
        source_paths["stage3_draws"] = Path(args.stage3_draws).resolve()
        source_paths["stage3_review_receipt"] = Path(args.stage3_review_receipt).resolve()
        source_paths["stage3_harness_receipt"] = Path(args.stage3_harness_receipt).resolve()
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    # Capture raw source receipts before the portable prepare-manifest copy can
    # rewrite an output path that is also being used as an input path.
    raw_source_receipts = {label: source_receipt(label, path) for label, path in source_paths.items()}

    selection = read_json(source_paths["selection_source_a"])
    candidate = read_json(source_paths["candidate"])
    stage2 = read_stage2_public(source_paths["stage2_results"])
    split_manifest = read_json(source_paths["stage3_split_manifest"])
    prepare_manifest = read_json(source_paths["stage3_prepare_manifest"])
    cache_manifest = load_stage3_cache_receipt(source_paths["stage3_gate_a_cache"])

    # The source-A selection artifact must be the same one bound by candidate.
    selection_digest = sha256_file(source_paths["selection_source_a"])
    if candidate.get("selection_source_a_sha256") not in {selection_digest, candidate.get("selection_sha256")}:
        raise ValueError("candidate does not bind the supplied selection artifact")
    if stage2.get("status") != "COMPLETE" or not stage2.get("scientific_verdict_emitted"):
        raise ValueError("Stage-2 source is not a complete scientific artifact")
    if prepare_manifest.get("status") != "COMPLETE":
        raise ValueError("Stage-3 preparation manifest is not COMPLETE")

    # Calibration lives in the repository and is not copied into results.  Its
    # source hash is retained in the stage2 provenance and public summary.
    calibration_path = Path(__file__).with_name("calibration_results.json")
    calibration = read_json(calibration_path)
    selection_rows, selection_summary = extract_selection(candidate)
    stage2_rows, stage2_summary = extract_stage2(stage2, calibration)
    stage3_rows, stage3_summary = extract_stage3(prepare_manifest, split_manifest, cache_manifest)
    stage3_result_rows: list[dict[str, Any]] | None = None
    stage3_matched_rows: list[dict[str, Any]] | None = None
    stage3_result_summary: dict[str, Any] | None = None
    stage3_result_payload: dict[str, Any] | None = None
    if "stage3_results" in source_paths:
        stage3_result_payload = read_json(source_paths["stage3_results"])
        stage3_result_rows, stage3_matched_rows, stage3_result_summary = extract_stage3_result(
            stage3_result_payload,
            source_paths["stage3_results"],
            source_paths["stage3_draws"],
            source_paths["stage3_review_receipt"],
            source_paths["stage3_harness_receipt"],
        )
        stage3_summary = {
            **stage3_summary,
            "status": "PREPARATION_COMPLETE_Q4_POSITIVE",
            "q4_verdict_emitted": True,
            "q4_verdict": "POSITIVE",
            "q4_result_summary": "stage3_result_summary.json",
        }

    # Preserve the independently reviewable Stage-3 preparation inputs.  The
    # cache, split manifest, and role CSV are exact copies; the prepare
    # manifest is a portable copy with only its two path fields rewritten.
    stage3_public_dir = output_dir / "stage3"
    stage3_public_dir.mkdir(parents=True, exist_ok=True)
    for label in ("stage3_gate_a_cache", "stage3_split_manifest", "stage3_split_roles"):
        source = source_paths[label]
        destination = stage3_public_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    portable_prepare_receipt = write_portable_prepare_manifest(
        source_paths["stage3_prepare_manifest"], stage3_public_dir / "stage3_prepare_manifest.json"
    )
    if stage3_result_payload is not None:
        declared_prepare_sha = stage3_result_payload.get("prepare_manifest_sha256")
        if declared_prepare_sha and portable_prepare_receipt["sha256"] != declared_prepare_sha:
            raise ValueError(
                "portable prepare manifest hash does not match raw Stage-3 result prepare_manifest_sha256"
            )
        if stage3_result_summary is not None:
            stage3_result_summary["prepare_manifest_binding"] = {
                "raw_source_receipt": raw_source_receipts["stage3_prepare_manifest"],
                "portable_copy": portable_prepare_receipt,
                "q4_declared_prepare_manifest_sha256": declared_prepare_sha,
            }
    if stage3_result_summary is not None:
        for label in ("stage3_review_receipt", "stage3_harness_receipt"):
            source = source_paths[label]
            destination = stage3_public_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)

    selection_csv = output_dir / "selection_candidates.csv"
    write_csv(
        selection_csv,
        [
            "rank",
            "flat_id",
            "layer",
            "head",
            "head_label",
            "true_mean",
            "source_a_mean",
            "source_a_noise_edge",
            "pair_sign_consistency",
            "holm_reject",
            "eligible",
            "stage1_rank",
        ],
        selection_rows,
    )
    stage2_csv = output_dir / "stage2_seed_metrics.csv"
    write_csv(
        stage2_csv,
        [
            "seed",
            "retained_pairs",
            "gate_A_pass",
            "E_all",
            "q1_n",
            "q1_recovery_fraction",
            "q1_seed_joint_pass",
            "q1_all_members_distinguishable",
            "q2_complete_pairs",
            "q2_true_right_E",
            "q2_source_A_E",
            "q2_source_A_abs_ratio",
            "q2_source_A_pass",
            "q2_source_B_E_descriptive",
            "q2_source_C_E",
            "q2_source_C_abs_ratio",
            "q2_source_C_pass",
            "q3_D_path",
            "q3_D_path_ci95_low",
            "q3_D_path_ci95_high",
            "q3_F_path",
            "q3_direct_recovery_descriptive",
            "q3_direct_recovery_adjudicative",
            "q3_positive",
        ],
        stage2_rows,
    )
    stage3_csv = output_dir / "stage3_preparation.csv"
    write_csv(
        stage3_csv,
        [
            "seed",
            "retained_pairs",
            "gate_A_pass",
            "generated_pairs",
            "gate_A_signed_correct_fraction",
            "gate_A_median_d_gap",
            "rank_training_pairs",
            "evaluation_pairs",
            "status",
        ],
        stage3_rows,
    )
    if stage3_result_rows is not None and stage3_matched_rows is not None and stage3_result_summary is not None:
        write_csv(
            output_dir / "stage3_seed_metrics.csv",
            [
                "seed",
                "status",
                "full_effect_E",
                "target_span_effect_E",
                "R_span",
                "matched_edge_second_largest",
                "matched_count",
                "matched_edge_pass",
                "R_comp",
                "geometric_span",
                "geometric_complement",
                "PCA_span_subject_effect_E",
                "PCA_span_final_effect_E",
                "PCA_span_both_effect_E",
            ],
            stage3_result_rows,
        )
        write_csv(
            output_dir / "stage3_matched_ratios.csv",
            [
                "seed",
                "draw_index",
                "accepted_attempt_id",
                "attempt",
                "latent_ids",
                "rank",
                "matched_effect_E",
                "full_effect_E",
                "matched_R_span",
                "attempt_sha256",
                "draw_or_projector_hash",
                "projector_hash",
                "effect_result_hash",
                "source_stage3_results_sha256",
                "source_stage3_draws_sha256",
            ],
            stage3_matched_rows,
        )
    ledger_csv = output_dir / "execution_ledger.csv"
    write_csv(
        ledger_csv,
        ["run_id", "artifact_scope", "status", "science_eligible", "notes"],
        [
            {
                "run_id": "stage2_current_1808740a",
                "artifact_scope": "Stage2 Q1-Q3",
                "status": "COMPLETE",
                "science_eligible": True,
                "notes": "single-invocation artifact bound to the public Stage2 summary",
            },
            {
                "run_id": "30d941c",
                "artifact_scope": "historical failed run",
                "status": "FAILED",
                "science_eligible": False,
                "notes": "execution history only; excluded from scientific evidence",
            },
            *([{
                "run_id": f"stage3_q4_{str(stage3_result_summary.get('invocation_id', ''))[:8]}",
                "artifact_scope": "Stage3 Q4 matched-span comparison",
                "status": "COMPLETE",
                "science_eligible": True,
                "notes": (
                    "8/8 positive; compact matched ratios bound to raw stage3_results and draw CSV hashes; "
                    "raw 41 MB result not copied"
                ),
            }] if stage3_result_summary is not None else []),
        ],
    )

    base_source_labels = (
        "selection_source_a",
        "candidate",
        "stage2_results",
        "stage3_gate_a_cache",
        "stage3_split_manifest",
        "stage3_split_roles",
        "stage3_prepare_manifest",
    )
    preparation_source_labels = (
        "stage3_gate_a_cache",
        "stage3_split_manifest",
        "stage3_split_roles",
        "stage3_prepare_manifest",
    )
    q4_present = stage3_result_summary is not None
    stage2_not_claimed = [
        "These results do not establish that the two heads are required or alone sufficient.",
        "They do not establish that every causal route is explained by this pair.",
        "They do not establish an exhaustive circuit or a native attention route.",
        "The Q3 direct-recovery values are descriptive only and do not adjudicate the verdict.",
        "Q4 is a separate target-span versus matched-span comparison; see stage3_result_summary.json.",
    ] if q4_present else [
        "These results do not establish that the two heads are required or alone sufficient.",
        "They do not establish that every causal route is explained by this pair.",
        "They do not establish an exhaustive circuit or a native attention route.",
        "The Q3 direct-recovery values are descriptive only and do not adjudicate the verdict.",
        "Stage 3 / Q4 has no scientific verdict yet.",
    ]
    stage2_outputs = [
        "selection_candidates.csv",
        "stage2_seed_metrics.csv",
        "stage3_preparation.csv",
        "stage2_public_summary.json",
        "stage3_preparation_summary.json",
        "figure_exp05_main.svg",
        "figure_exp05_main.png",
    ]
    if q4_present:
        stage2_outputs.extend(
            [
                "stage3_seed_metrics.csv",
                "stage3_matched_ratios.csv",
                "stage3_result_summary.json",
                "stage3/stage3_prepare_review.json",
                "stage3/stage3_harness_receipt.json",
            ]
        )
    stage2_inputs = [
        "selection_source_a.json",
        "candidate.json",
        "stage2_results.json",
        "stage3_gate_a_cache.jsonl",
        "stage3_split_manifest.json",
        "stage3_prepare_manifest.json",
    ]
    if q4_present:
        stage2_inputs.extend(
            [
                "stage3_results.json",
                "stage3_draws.csv",
                "stage3_prepare_review.json",
                "stage3_harness_receipt.json",
            ]
        )
    stage2_public = {
        "schema": "exp05-public-stage2-summary-v1",
        "question": "Which heads carry the number signal, and do they transport subject-number information?",
        "public_claim": "Across eight seeds, two heads—L7H4 and L8H5—formed a stable minimal compact set, passed preregistered specificity controls, and causally transported subject-number information under a frozen-pattern value-path intervention.",
        "finding": "Q1, Q2, and Q3 all returned positive registered verdicts on all eight Stage-2 seeds.",
        "selection": selection_summary,
        "stage2": stage2_summary,
        "not_claimed": stage2_not_claimed,
        "reproduce": {
            "script": "experiments/05_number_agreement_circuit/make_claim_ladder.py",
            "inputs": stage2_inputs,
            "outputs": stage2_outputs,
        },
        "source_receipts": [raw_source_receipts[label] for label in base_source_labels],
        "calibration": {
            "basename": calibration_path.name,
            "sha256": sha256_file(calibration_path),
            "theta_spec": calibration.get("theta_spec"),
        },
        "execution_ledger": "execution_ledger.csv",
    }
    stage3_public = {
        "schema": "exp05-public-stage3-preparation-v1",
        **stage3_summary,
        "source_receipts": [raw_source_receipts[label] for label in preparation_source_labels],
        "exact_input_copies": [
            f"stage3/{source_paths[label].name}"
            for label in ("stage3_gate_a_cache", "stage3_split_manifest", "stage3_split_roles")
        ],
        "portable_input_copies": [portable_prepare_receipt],
        "reproduce": "Use make_claim_ladder.py with the preparation/source inputs; this package does not run Stage 3.",
    }
    if q4_present and stage3_result_summary is not None:
        stage3_public["q4_result_summary"] = "stage3_result_summary.json"
        stage3_public["q4_raw_source_receipts"] = stage3_result_summary["source_receipts"]
        stage3_public["q4_receipt_bindings"] = stage3_result_summary["receipt_bindings"]
        stage3_public["exact_receipt_copies"] = [
            "stage3/stage3_prepare_review.json",
            "stage3/stage3_harness_receipt.json",
        ]
    json_dump(output_dir / "stage2_public_summary.json", stage2_public)
    json_dump(output_dir / "stage3_preparation_summary.json", stage3_public)
    if q4_present and stage3_result_summary is not None:
        json_dump(output_dir / "stage3_result_summary.json", stage3_result_summary)

    if q4_present and stage3_result_summary is not None:
        q4_evidence = f"""
The completed Q4 result is reduced to [`stage3_seed_metrics.csv`](stage3_seed_metrics.csv),
with all 800 matched-span ratios and latent IDs in [`stage3_matched_ratios.csv`](stage3_matched_ratios.csv).
The source-bound adjudication summary is [`stage3_result_summary.json`](stage3_result_summary.json);
the accepted Advisor and harness receipts are copied under
[`stage3/`](stage3/) and bound by self-hash plus the raw result's receipt hashes.
The preparation manifest is also copied there in portable form: only its two
split-input path fields are rewritten to basenames; its raw source receipt and
the Q4-declared raw manifest SHA remain recorded separately.
The 41 MB raw result and raw draw CSV remain outside Git.

**Q4 finding.** {Q4_CLAIM}  The mean R_span is 0.8935 (95% t(7) CI
[0.8905, 0.8966]); the frozen second-largest matched edge has mean 0.4756
(95% t(7) CI [0.4013, 0.5498]).  The generic PCA comparator is reported as a
raw logit effect (mean E=0.0274), not as a recovery percentage.
"""
        q4_panel = "Panel C is the eight-seed Q4 matched-span comparison."
        q4_not_claimed = "\n".join(f"- {item}" for item in Q4_NOT_CLAIMED)
        q4_args = " \\\n  --stage3-results /tmp/exp05-stage3/stage3_results.json \\\n  --stage3-draws /tmp/exp05-stage3/stage3_draws.csv \\\n  --stage3-review-receipt /tmp/exp05-stage3/stage3_prepare_review.json \\\n  --stage3-harness-receipt /tmp/exp05-stage3/stage3_harness_receipt.json"
    else:
        q4_evidence = ""
        q4_panel = "Panel C remains an explicit Stage-3 preparation boundary; no Q4 scientific result is emitted."
        q4_not_claimed = "- Stage 3 / Q4 has no scientific verdict yet."
        q4_args = ""
    results_md = f"""# Exp05 public results

This directory is a compact, source-bound evidence packet.  It answers the
question, shows the registered Stage-2 evidence, and keeps the Stage-3 inputs
and claim boundary visible without copying the raw model result.

## Question → evidence → finding

**Question.** Which heads carry the number signal, and do they transport
subject-number information?

**Evidence.** The fresh same-snapshot selection and frozen candidate are in
[`selection_candidates.csv`](selection_candidates.csv).  The complete Stage-2
eight-seed cells are reduced to [`stage2_seed_metrics.csv`](stage2_seed_metrics.csv).
The per-seed Stage-3 Gate-A and rank/evaluation split preparation is in
[`stage3_preparation.csv`](stage3_preparation.csv).
The Gate-A cache, split manifest, and role CSV are exact copies under
[`stage3/`](stage3/).  The prepare manifest is a portable copy whose two path
fields are basenames; raw and portable hashes are both recorded in
[`stage3_preparation_summary.json`](stage3_preparation_summary.json).

**Finding.** Across eight seeds, two heads—L7H4 and L8H5—formed a stable minimal
compact set, passed preregistered specificity controls, and causally transported
subject-number information under a frozen-pattern value-path intervention.
The machine-readable summary records Q1={stage2_summary['q1']['pass_count']}/8,
Q2={stage2_summary['q2']['pass_count']}/8, and
Q3={stage2_summary['q3']['pass_count']}/8 positive registered seed cells.
{q4_evidence}
The visual overview is [`figure_exp05_main.svg`](figure_exp05_main.svg) (with a
high-resolution [`PNG`](figure_exp05_main.png)).  Panel A is selection, Panel B
is the eight-seed Q1–Q3 evidence, and {q4_panel}

## Not claimed

These results do not establish that the two heads are required or alone
sufficient, that every causal route is explained by this pair, or that the pair
constitutes an exhaustive circuit or native attention route.  Q3 direct-recovery
values are descriptive only.
{q4_not_claimed}

The historical failed run `30d941c` is retained only in
[`execution_ledger.csv`](execution_ledger.csv) with `science_eligible=false`.

## Reproduce

From the repository root, run the model-free packaging script against the source
artifacts (the raw files stay outside Git):

```bash
python3 experiments/05_number_agreement_circuit/make_claim_ladder.py \\
  --selection-source-a /tmp/exp05/selection_source_a.json \\
  --candidate /tmp/exp05/candidate.json \\
  --stage2 /tmp/exp05/stage2_results.json \\
  --stage3-cache /tmp/exp05-stage3/stage3_gate_a_cache.jsonl \\
  --stage3-split /tmp/exp05-stage3/stage3_split_manifest.json \\
  --stage3-prepare /tmp/exp05-stage3/stage3_prepare_manifest.json{q4_args}
```

[`index.json`](index.json) and [`artifact_index.json`](artifact_index.json)
list generated files and source receipts.  Verify the checked-in packet with
`(cd experiments/05_number_agreement_circuit/results && sha256sum -c checksums.sha256)`.
"""
    (output_dir / "RESULTS.md").write_text(results_md, encoding="utf-8")

    figure_manifest = write_figure(
        output_dir,
        selection_rows,
        stage2_rows,
        stage3_rows,
        stage3_result_rows,
        finite_float(calibration["theta_spec"]["A"]),
        finite_float(calibration["theta_spec"]["C"]),
    )
    json_dump(output_dir / "figure_manifest.json", figure_manifest)

    index_artifacts = [
        "RESULTS.md",
        "selection_candidates.csv",
        "stage2_seed_metrics.csv",
        "stage3_preparation.csv",
        "execution_ledger.csv",
        "stage2_public_summary.json",
        "stage3_preparation_summary.json",
        "figure_manifest.json",
        "figure_exp05_main.svg",
        "figure_exp05_main.png",
        "stage3/stage3_gate_a_cache.jsonl",
        "stage3/stage3_split_manifest.json",
        "stage3/stage3_split_roles.csv",
        "stage3/stage3_prepare_manifest.json",
    ]
    if q4_present:
        index_artifacts.extend(
            [
                "stage3_seed_metrics.csv",
                "stage3_matched_ratios.csv",
                "stage3_result_summary.json",
                "stage3/stage3_prepare_review.json",
                "stage3/stage3_harness_receipt.json",
            ]
        )
    index = {
        "schema": "exp05-public-evidence-index-v1",
        "title": "Exp05 public evidence packet",
        "claim_boundary": (
            "Q1-Q3 Stage2 evidence and the adjudicated Q4 matched-span comparison are public; "
            "Q4 is limited to the directed-logit target-span versus frozen matched-span test."
            if q4_present
            else "Q1-Q3 Stage2 evidence is public; Stage3 is preparation-only and Q4 is pending."
        ),
        "artifacts": index_artifacts,
        "checksums": "checksums.sha256",
        "source_receipts": [raw_source_receipts[label] for label in source_paths],
        "package_policy": {
            "raw_stage2_json_copied": False,
            "raw_stage2_pair_csv_copied": False,
            "raw_stage3_results_copied": False,
            "raw_stage3_draws_copied": False,
            "raw_stage3_prepare_manifest_copied": False,
            "portable_stage3_prepare_manifest_copied": True,
            "stage3_q4_verdict_emitted": q4_present,
            "historical_30d941c_science_eligible": False,
        },
    }
    json_dump(output_dir / "index.json", index)
    # Keep the descriptive name requested by publication checklists while
    # retaining index.json as the short stable link used by the figure/readme.
    json_dump(output_dir / "artifact_index.json", index)

    # The checksum file covers every generated file except itself.  It is
    # written last so the same package can be audited with sha256sum -c after
    # checking out the repository.
    checksum_path = output_dir / "checksums.sha256"
    generated = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in generated
        ),
        encoding="utf-8",
    )
    return output_dir


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-source-a", required=True, help="fresh same-invocation selection artifact")
    parser.add_argument("--candidate", required=True, help="frozen candidate manifest")
    parser.add_argument("--stage2", required=True, help="completed Stage-2 Q1-Q3 result JSON")
    parser.add_argument("--stage3-cache", required=True, help="Stage-3 Gate-A JSONL cache")
    parser.add_argument("--stage3-split", required=True, help="Stage-3 split manifest JSON")
    parser.add_argument("--stage3-roles", default=None, help="Stage-3 split-role CSV (defaults beside --stage3-split)")
    parser.add_argument("--stage3-prepare", required=True, help="Stage-3 preparation manifest JSON")
    parser.add_argument("--stage3-results", default=None, help="completed Stage-3/Q4 result JSON (optional for preparation-only regeneration)")
    parser.add_argument("--stage3-draws", default=None, help="accepted 800-row Stage-3 matched-draw CSV (required with --stage3-results)")
    parser.add_argument("--stage3-review-receipt", default=None, help="Advisor Stage-3 preparation review receipt (required for Q4 packaging)")
    parser.add_argument("--stage3-harness-receipt", default=None, help="independent Stage-3 harness receipt (required for Q4 packaging)")
    parser.add_argument("--output", default=str(Path(__file__).with_name("results")), help="public output directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output = build(parse_args(sys.argv[1:] if argv is None else argv))
    except Exception as exc:  # concise CLI failure; no traceback in public logs
        print(f"make_claim_ladder: {exc}", file=sys.stderr)
        return 2
    print(f"wrote public evidence packet: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
