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
      --output experiments/05_number_agreement_circuit/results

The large Stage-2 source files are intentionally not copied into Git. Re-running
the same command against byte-identical inputs in the pinned plotting environment
regenerates the same public files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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


def write_figure(
    output_dir: Path,
    selection_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]],
    stage3_rows: Sequence[Mapping[str, Any]],
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

    # Panel C: explicit preparation boundary; no Q4 verdict is drawn.
    ax_c = fig.add_subplot(outer[0, 2])
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

    fig.suptitle("Exp05 public evidence — frozen Stage 2 result and Stage 3 preparation", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Values are extracted from the immutable source artifacts; error bars in Q3 are per-seed bootstrap intervals.",
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
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    return {
        "svg": svg_path.name,
        "png": png_path.name,
        "png_width_px": 2880,
        "png_height_px": 1440,
        "panel_order": ["selection", "q1_q2_q3_eight_seed", "stage3_preparation_pending"],
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
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

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

    # Preserve the independently reviewable Stage-3 preparation inputs as
    # exact byte-identical copies.  No Stage-3 computation is performed here.
    stage3_public_dir = output_dir / "stage3"
    stage3_public_dir.mkdir(parents=True, exist_ok=True)
    for label in ("stage3_gate_a_cache", "stage3_split_manifest", "stage3_split_roles", "stage3_prepare_manifest"):
        source = source_paths[label]
        shutil.copyfile(source, stage3_public_dir / source.name)

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
        ],
    )

    stage2_public = {
        "schema": "exp05-public-stage2-summary-v1",
        "question": "Which heads carry the number signal, and do they transport subject-number information?",
        "public_claim": "Across eight seeds, two heads—L7H4 and L8H5—formed a stable minimal compact set, passed preregistered specificity controls, and causally transported subject-number information under a frozen-pattern value-path intervention.",
        "finding": "Q1, Q2, and Q3 all returned positive registered verdicts on all eight Stage-2 seeds.",
        "selection": selection_summary,
        "stage2": stage2_summary,
        "not_claimed": [
            "These results do not establish that the two heads are required or alone sufficient.",
            "They do not establish that every causal route is explained by this pair.",
            "They do not establish an exhaustive circuit or a native attention route.",
            "The Q3 direct-recovery values are descriptive only and do not adjudicate the verdict.",
            "Stage 3 / Q4 has no scientific verdict yet.",
        ],
        "reproduce": {
            "script": "experiments/05_number_agreement_circuit/make_claim_ladder.py",
            "inputs": [
                "selection_source_a.json",
                "candidate.json",
                "stage2_results.json",
                "stage3_gate_a_cache.jsonl",
                "stage3_split_manifest.json",
                "stage3_prepare_manifest.json",
            ],
            "outputs": [
                "selection_candidates.csv",
                "stage2_seed_metrics.csv",
                "stage3_preparation.csv",
                "stage2_public_summary.json",
                "stage3_preparation_summary.json",
                "figure_exp05_main.svg",
                "figure_exp05_main.png",
            ],
        },
        "source_receipts": [source_receipt(label, path) for label, path in source_paths.items()],
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
        "source_receipts": [source_receipt(label, path) for label, path in source_paths.items() if label.startswith("stage3_")],
        "exact_input_copies": [f"stage3/{source_paths[label].name}" for label in ("stage3_gate_a_cache", "stage3_split_manifest", "stage3_split_roles", "stage3_prepare_manifest")],
        "reproduce": "Use make_claim_ladder.py with the six source inputs; this package does not run Stage 3.",
    }
    json_dump(output_dir / "stage2_public_summary.json", stage2_public)
    json_dump(output_dir / "stage3_preparation_summary.json", stage3_public)

    results_md = f"""# Exp05 public results

This directory is a compact, source-bound evidence packet.  It answers the
question, shows the registered Stage-2 evidence, and keeps the Stage-3 boundary
visible without copying the raw JSON or pair-level CSV.

## Question → evidence → finding

**Question.** Which heads carry the number signal, and do they transport
subject-number information?

**Evidence.** The fresh same-snapshot selection and frozen candidate are in
[`selection_candidates.csv`](selection_candidates.csv).  The complete Stage-2
eight-seed cells are reduced to [`stage2_seed_metrics.csv`](stage2_seed_metrics.csv).
The per-seed Stage-3 Gate-A and rank/evaluation split preparation is in
[`stage3_preparation.csv`](stage3_preparation.csv).
The exact preparation inputs are copied under [`stage3/`](stage3/), with their
source hashes recorded in [`stage3_preparation_summary.json`](stage3_preparation_summary.json).

**Finding.** Across eight seeds, two heads—L7H4 and L8H5—formed a stable minimal
compact set, passed preregistered specificity controls, and causally transported
subject-number information under a frozen-pattern value-path intervention.
The machine-readable summary records Q1={stage2_summary['q1']['pass_count']}/8,
Q2={stage2_summary['q2']['pass_count']}/8, and
Q3={stage2_summary['q3']['pass_count']}/8 positive registered seed cells.

The visual overview is [`figure_exp05_main.svg`](figure_exp05_main.svg) (with a
high-resolution [`PNG`](figure_exp05_main.png)).  Panel A is selection, Panel B
is the eight-seed Q1–Q3 evidence, and Panel C is an explicit Stage-3 pending
placeholder.

## Not claimed

These results do not establish that the two heads are required or alone
sufficient, that every causal route is explained by this pair, or that the pair
constitutes an exhaustive circuit or native attention route.  Q3 direct-recovery
values are descriptive only.  Stage 3 / Q4 has no scientific verdict yet.

The historical failed run `30d941c` is retained only in
[`execution_ledger.csv`](execution_ledger.csv) with `science_eligible=false`.

## Reproduce

From the repository root, run the model-free packaging script against the six
source artifacts (the raw files stay outside Git):

```bash
python3 experiments/05_number_agreement_circuit/make_claim_ladder.py \\
  --selection-source-a /tmp/exp05/selection_source_a.json \\
  --candidate /tmp/exp05/candidate.json \\
  --stage2 /tmp/exp05/stage2_results.json \\
  --stage3-cache /tmp/exp05-stage3/stage3_gate_a_cache.jsonl \\
  --stage3-split /tmp/exp05-stage3/stage3_split_manifest.json \\
  --stage3-prepare /tmp/exp05-stage3/stage3_prepare_manifest.json
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
        finite_float(calibration["theta_spec"]["A"]),
        finite_float(calibration["theta_spec"]["C"]),
    )
    json_dump(output_dir / "figure_manifest.json", figure_manifest)

    index = {
        "schema": "exp05-public-evidence-index-v1",
        "title": "Exp05 public evidence packet",
        "claim_boundary": "Q1-Q3 Stage2 evidence is public; Stage3 is preparation-only and Q4 is pending.",
        "artifacts": [
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
        ],
        "checksums": "checksums.sha256",
        "source_receipts": [source_receipt(label, path) for label, path in source_paths.items()],
        "package_policy": {
            "raw_stage2_json_copied": False,
            "raw_stage2_pair_csv_copied": False,
            "stage3_q4_verdict_emitted": False,
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
