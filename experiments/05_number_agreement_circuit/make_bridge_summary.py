#!/usr/bin/env python3
"""Package the exploratory Exp05 head-to-span bridge follow-up.

The model-backed bridge run is intentionally kept outside Git.  This small,
model-free packager reads its JSON result, re-derives the public scalar tables
and confidence intervals, and emits a compact evidence packet.  It never
copies the raw result into the repository.

Example::

    python3 experiments/05_number_agreement_circuit/make_bridge_summary.py \
      --bridge-results /private/tmp/mech-interp-exp05-bridge.MUlks5/bridge_results.json

The default output is the Exp05 ``results/`` directory.  ``--output`` is
useful for deterministic regeneration into a temporary directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results"
EXPECTED_SCHEMA = "exp05-number-agreement-bridge-rescue-v1"
EXPECTED_RAW_SHA256 = "9d844605de4d20ec5638bf793d21e8750ea606984d7229531fdc9910aa1e45ef"
EXPECTED_SEEDS = tuple(range(20260814, 20260822))
MATCHED_PER_SEED = 100
T_CRIT_8 = 2.365

BRIDGE_CLAIM = (
    "In this exploratory follow-up, the fixed 12-row decoder span carried a "
    "large fraction of the L7H4-induced directed-logit effect on fresh seeds "
    "and exceeded every matched span maximum in the 8-seed sample."
)
BRIDGE_NOT_CLAIMED = [
    "This is an exploratory follow-up with no preregistered verdict or threshold.",
    "It does not establish that the fixed span is a natural, necessary, or sufficient representation.",
    "It does not establish individual-latent causality, a complete circuit, or full mediation.",
    "The L8H5 reader clamp is a dependence control, not a proof of an L7H4-to-span-to-readout mediation path.",
    "R_target is a directed-logit effect ratio, not an activation-reconstruction percentage.",
]


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    raw_sha = sha256_file(result_path)
    if raw_sha != EXPECTED_RAW_SHA256:
        raise ValueError(
            "bridge result SHA mismatch: "
            f"expected {EXPECTED_RAW_SHA256}, got {raw_sha}"
        )
    result = read_json(result_path)
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
            "bytes": result_path.stat().st_size,
            "sha256": raw_sha,
        },
        "seed_count": len(metric_rows),
        "seeds": list(EXPECTED_SEEDS),
        "matched_rows": len(compact_rows),
        "matched_per_seed": MATCHED_PER_SEED,
        "target_exceeds_matched_max_count": sum(1 for value in target_exceeds_max if value),
        "target_exceeds_matched_max_all_seeds": all(target_exceeds_max),
        "metrics": aggregate,
        "design": {
            "follow_up_type": "fresh out-of-sample exploratory bridge",
            "timing_decision": "L7_ONLY_RESID_PRE8",
            "upstream_head": "L7H4",
            "fixed_decoder_span": "12 layer-8 res-jb decoder rows from Q4",
            "reader_head": "L8H5",
            "matched_control": "100 rank-12 target-excluded spans per seed",
            "reader_clamp": "L8H5.final fixed to natural source-A L7H4 arm value",
        },
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
        "reproduce": {
            "script": "experiments/05_number_agreement_circuit/make_bridge_summary.py",
            "command": "python3 experiments/05_number_agreement_circuit/make_bridge_summary.py --bridge-results /path/to/bridge_results.json",
            "independent_checks": [
                "recompute per-seed matched maxima from bridge_matched_ratios.csv",
                "recompute aggregate means and t(7) intervals from bridge_seed_metrics.csv",
                "sha256sum -c checksums.sha256",
            ],
        },
    }
    return metric_rows, compact_rows, summary


def write_figure(output: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
    ax.plot(x, clamped, "D--", color="#7c3aed", lw=1.3, ms=3.5, label="L8H5 clamped")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, short)
    ax.set_ylabel("directed-logit ratio")
    ax.set_title("B  natural vs L8H5-clamped target", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.text(
        0.03,
        0.96,
        "clamp leaves the target effect large",
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

    fig.suptitle("Exp05 exploratory bridge: L7H4 → fixed SAE span → readout", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Fresh seeds; ratios are directed-logit effects. The clamp is a dependence control, not a mediation test.",
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
    return {
        "schema": "exp05-public-bridge-figure-manifest-v1",
        "figure": "figure_bridge_rescue.svg",
        "png": "figure_bridge_rescue.png",
        "panels": {
            "A": "per-seed natural target R versus maximum matched-span R",
            "B": "per-seed natural target R versus L8H5-clamped target R",
            "C": "per-seed descriptive L8 reader projection coefficient",
        },
        "source": "bridge_seed_metrics.csv",
        "claim_boundary": "Descriptive exploratory bridge evidence; no mediation claim.",
    }


def update_results_md(path: Path) -> None:
    marker = "## Exploratory bridge follow-up"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Exp05 public results\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    section = f"""
{marker}

The single follow-up is a fresh out-of-sample, exploratory bridge around the
L7H4 intervention.  It asks whether the fixed 12-row layer-8 decoder span
also carries that upstream head-induced effect, while clamping the L8H5 reader
as a dependence control.  The compact evidence is in
[`bridge_seed_metrics.csv`](bridge_seed_metrics.csv),
[`bridge_matched_ratios.csv`](bridge_matched_ratios.csv), and
[`bridge_result_summary.json`](bridge_result_summary.json); the raw 639 KB
model result stays outside Git.

**Exploratory finding.** On eight fresh seeds, the target span had mean
`R_target=0.6786` (95% t(7) CI `[0.6738, 0.6834]`) versus mean matched-span
maximum `0.4110`, and exceeded the matched maximum on all 8 seeds.  The
complement ratio was `0.3053`.  With L8H5 clamped, the target remained large:
mean `R_target_clamped=0.6740` (95% t(7) CI `[0.6696, 0.6785]`).  This does
not support dominant dependence on L8H5's tested final-value path or a mediation claim; it only says that this
fixed span carries the tested L7H4-induced effect better than the matched
controls in this exploratory sample.

**Run integrity.** Gate A passed on 8/8 seeds, retaining 230–237 pairs per
seed; the evaluation split contains 150 pairs per seed.  The timing identity
checks have non-final and selected-position maxima of 0, and the full-vs-true
final-logit maximum is `9.536743e-06 < 1e-5`.  The run started and finished
with a clean worktree at commit `0d7c4db`; the raw result is bound by SHA-256
`9d844605de4d20ec5638bf793d21e8750ea606984d7229531fdc9910aa1e45ef`.

This follow-up has no preregistered verdict or threshold.  It does not
establish naturalness, necessity, sufficiency, individual-latent causality,
or full mediation.  The three-panel overview is
[`figure_bridge_rescue.svg`](figure_bridge_rescue.svg) (and its PNG).

Reproduce the compact packet with:

```bash
python3 experiments/05_number_agreement_circuit/make_bridge_summary.py \\
  --bridge-results /path/to/bridge_results.json
```
"""
    path.write_text(existing.rstrip() + "\n" + section.lstrip(), encoding="utf-8")


def update_indexes(output: Path, summary: Mapping[str, Any]) -> None:
    generated = [
        "bridge_seed_metrics.csv",
        "bridge_matched_ratios.csv",
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
        index["claim_boundary"] = (
            previous if bridge_boundary in previous else (previous + " " + bridge_boundary).strip()
        )
        index["bridge_followup"] = {
            "summary": "bridge_result_summary.json",
            "raw_result_copied": False,
            "source_sha256": summary["source_receipt"]["sha256"],
            "status": "COMPLETE_EXPLORATORY",
            "target_exceeds_matched_max_all_seeds": summary["target_exceeds_matched_max_all_seeds"],
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


def build(bridge_results: Path, output: Path) -> Path:
    _base_package(output)
    metric_rows, compact_rows, summary = validate_and_extract(bridge_results)
    metric_fields = [
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
    matched_fields = [
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
    write_csv(output / "bridge_seed_metrics.csv", metric_fields, metric_rows)
    write_csv(output / "bridge_matched_ratios.csv", matched_fields, compact_rows)
    json_dump(output / "bridge_result_summary.json", summary)
    figure_manifest = write_figure(output, metric_rows)
    json_dump(output / "bridge_figure_manifest.json", figure_manifest)
    update_results_md(output / "RESULTS.md")
    update_indexes(output, summary)
    write_checksums(output)
    return output


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-results", required=True, help="completed raw bridge result JSON (kept outside Git)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="public results directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        output = build(Path(args.bridge_results).resolve(), Path(args.output).resolve())
    except Exception as exc:
        print(f"make_bridge_summary: {exc}", file=sys.stderr)
        return 2
    print(f"wrote bridge evidence packet: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
