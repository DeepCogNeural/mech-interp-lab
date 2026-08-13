"""Build a compact public packet from one hash-bound Experiment 06 result.

This script is model-free. It never recomputes an intervention or changes the
registered verdict; it only projects the compact raw result into reviewable JSON,
CSV, Markdown, and checksum files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


RAW_SCHEMA = "exp06-cross-template-bridge-result-v1"
PUBLIC_SCHEMA = "exp06-cross-template-bridge-public-v1"
VERDICTS = {"POSITIVE", "MECHANISM_NEGATIVE", "SPAN_NEGATIVE", "NON_ESTIMABLE"}
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_experiment as runner  # noqa: E402


class PacketStop(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _json_object_from_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PacketStop(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PacketStop(f"{label} must be a JSON object")
    return value


def _read_bound_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PacketStop(f"cannot read {label}: {exc}") from exc
    return _json_object_from_bytes(data, label), _sha256_bytes(data)


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buffer.getvalue()


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise PacketStop("public packet cannot contain a non-finite metric")
    return number


def _finite(value: Any, label: str) -> float:
    number = _finite_or_none(value)
    if number is None:
        raise PacketStop(f"{label} is missing")
    return number


def _close(actual: Any, expected: Any, label: str, *, tolerance: float = 1e-12) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise PacketStop(f"{label} nullability differs from the recomputed value")
        return
    left, right = float(actual), float(expected)
    if not (math.isfinite(left) and math.isfinite(right)) or not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise PacketStop(f"{label}={left!r} differs from recomputed {right!r}")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _t_summary8(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if len(finite) != 8 or not all(math.isfinite(value) for value in finite):
        raise PacketStop("registered intervals require eight finite seed values")
    mean = math.fsum(finite) / 8.0
    variance = math.fsum((value - mean) ** 2 for value in finite) / 7.0
    standard_error = math.sqrt(variance / 8.0)
    half_width = runner.T_CRITICAL_DF7 * standard_error
    return {
        "n": 8,
        "mean": mean,
        "standard_error": standard_error,
        "t_critical": runner.T_CRITICAL_DF7,
        "degrees_of_freedom": 7,
        "ci95": {"low": mean - half_width, "high": mean + half_width},
    }


def _reaggregate_adjudication(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Independent model-free implementation of the frozen four-branch rule."""

    observed = [int(row["seed"]) for row in seed_results]
    if observed != list(runner.FRESH_SEEDS):
        raise PacketStop("adjudication rows do not match the registered seed order")
    valid = [row for row in seed_results if row.get("status") == "ADJUDICABLE"]
    invalid = [int(row["seed"]) for row in seed_results if row.get("status") == "GATE_A_POPULATION_INVALID"]
    if len(valid) + len(invalid) != 8:
        raise PacketStop("adjudication rows contain an unknown status")
    coverage = {
        "registered_seed_count": 8,
        "observed_seed_count": 8,
        "gate_a_valid_seed_count": len(valid),
        "gate_a_population_invalid_seeds": invalid,
        "missing_registered_seeds": [],
        "minimum_required": 8,
        "passed": len(valid) == 8,
    }
    if len(valid) != 8:
        return {
            "verdict": "NON_ESTIMABLE",
            "coverage": coverage,
            "mechanism_transfer": None,
            "span_transfer": None,
            "statistics": None,
        }
    direct = [float(row["estimands"]["D_s"]) for row in valid]
    target = [float(row["estimands"]["T_s"]) for row in valid]
    advantage = [float(row["estimands"]["A_s"]) for row in valid]
    summaries = {"D_s": _t_summary8(direct), "T_s": _t_summary8(target), "A_s": _t_summary8(advantage)}
    floor_count = sum(value >= runner.MECHANISM_EFFECT_FLOOR for value in direct)
    mechanism_conditions = {
        "direct_lower_bound_above_zero": summaries["D_s"]["ci95"]["low"] > 0.0,
        "registered_seed_floor_count": floor_count,
        "registered_seed_floor_count_at_least_six": floor_count >= 6,
        "absolute_floor": runner.MECHANISM_EFFECT_FLOOR,
    }
    mechanism_passed = bool(
        mechanism_conditions["direct_lower_bound_above_zero"]
        and mechanism_conditions["registered_seed_floor_count_at_least_six"]
    )
    if not mechanism_passed:
        return {
            "verdict": "MECHANISM_NEGATIVE",
            "coverage": coverage,
            "mechanism_transfer": {"passed": False, "conditions": mechanism_conditions},
            "span_transfer": {"adjudicated": False, "reason": "mechanism transfer gate failed"},
            "statistics": summaries,
        }
    joint_count = sum(t_value > 0.0 and a_value > 0.0 for t_value, a_value in zip(target, advantage))
    span_conditions = {
        "target_lower_bound_above_zero": summaries["T_s"]["ci95"]["low"] > 0.0,
        "advantage_lower_bound_above_zero": summaries["A_s"]["ci95"]["low"] > 0.0,
        "joint_target_and_advantage_positive_registered_seed_count": joint_count,
        "joint_target_and_advantage_positive_registered_seed_count_at_least_six": joint_count >= 6,
    }
    span_passed = bool(
        span_conditions["target_lower_bound_above_zero"]
        and span_conditions["advantage_lower_bound_above_zero"]
        and span_conditions["joint_target_and_advantage_positive_registered_seed_count_at_least_six"]
    )
    return {
        "verdict": "POSITIVE" if span_passed else "SPAN_NEGATIVE",
        "coverage": coverage,
        "mechanism_transfer": {"passed": True, "conditions": mechanism_conditions},
        "span_transfer": {"adjudicated": True, "passed": span_passed, "conditions": span_conditions},
        "statistics": summaries,
    }


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_asset_provenance(raw: Mapping[str, Any]) -> None:
    inputs = raw.get("inputs")
    provenance = raw.get("asset_provenance")
    if not isinstance(inputs, Mapping) or not isinstance(provenance, Mapping):
        raise PacketStop("raw result lacks exact model/SAE asset provenance")
    expected_contract_sha = _sha256_bytes(_canonical(runner.ASSET_CONTRACT).encode("utf-8"))
    if inputs.get("asset_contract_sha256") != expected_contract_sha:
        raise PacketStop("raw result does not bind the frozen asset contract")
    if provenance.get("contract_sha256") != expected_contract_sha:
        raise PacketStop("asset receipt does not bind the frozen asset contract")
    expected_loader_boundary = (
        "loaders consumed private copies written from the exact hash-validated cache byte buffers; "
        "TransformerLens config and weights were constructed from the staged local directory "
        "without a model-name lookup"
    )
    if provenance.get("loader_boundary") != expected_loader_boundary:
        raise PacketStop("asset receipt does not bind hashing to the loaded byte copies")
    repositories = provenance.get("repositories")
    if not isinstance(repositories, Mapping):
        raise PacketStop("asset receipt lacks repository snapshots")
    for group in ("gpt2", "sae"):
        expected = runner.ASSET_CONTRACT[group]
        observed = repositories.get(group)
        if not isinstance(observed, Mapping):
            raise PacketStop(f"asset receipt lacks {group} snapshot")
        if observed.get("repo_id") != expected["repo_id"] or observed.get("revision") != expected["revision"]:
            raise PacketStop(f"asset receipt changed the frozen {group} repository/revision")
        if observed.get("files") != expected["files"]:
            raise PacketStop(f"asset receipt changed the frozen {group} file hashes")
    if provenance.get("runtime_versions") != runner.ASSET_CONTRACT["runtime_versions"]:
        raise PacketStop("asset receipt runtime versions differ from the frozen contract")

    model_state = provenance.get("model_state")
    if not isinstance(model_state, Mapping) or model_state.get("exact_match") is not True:
        raise PacketStop("asset receipt lacks an unchanged before/after model state")
    model_before, model_after = model_state.get("before"), model_state.get("after")
    if not isinstance(model_before, Mapping) or dict(model_before) != dict(model_after or {}):
        raise PacketStop("model before/after fingerprints differ")
    if (
        model_before.get("schema") != "exp06-model-state-fingerprint-v1"
        or not _is_sha256(model_before.get("sha256"))
        or int(model_before.get("key_count", 0)) <= 0
        or model_before.get("scheme") != runner.ASSET_CONTRACT["state_fingerprint"]["model_scheme"]
    ):
        raise PacketStop("model state fingerprint is malformed")

    sae_decoder = provenance.get("sae_decoder")
    if not isinstance(sae_decoder, Mapping) or sae_decoder.get("exact_match") is not True:
        raise PacketStop("asset receipt lacks an unchanged before/after SAE decoder")
    sae_before, sae_after = sae_decoder.get("before"), sae_decoder.get("after")
    if not isinstance(sae_before, Mapping) or dict(sae_before) != dict(sae_after or {}):
        raise PacketStop("SAE decoder before/after fingerprints differ")
    if (
        sae_before.get("schema") != "exp06-sae-decoder-fingerprint-v1"
        or not _is_sha256(sae_before.get("sha256"))
        or sae_before.get("dtype") != "torch.float32"
        or sae_before.get("shape") != [24_576, runner.RESIDUAL_WIDTH]
        or sae_before.get("scheme") != runner.ASSET_CONTRACT["state_fingerprint"]["sae_decoder_scheme"]
    ):
        raise PacketStop("SAE decoder fingerprint is malformed")


def _validate(
    raw: Mapping[str, Any],
    expected_raw_sha: str,
    actual_raw_sha: str,
    frozen_q4: Mapping[str, Any],
) -> None:
    if actual_raw_sha != expected_raw_sha:
        raise PacketStop(f"raw SHA-256 {actual_raw_sha} differs from expected {expected_raw_sha}")
    if raw.get("schema") != RAW_SCHEMA or raw.get("status") != "COMPLETE":
        raise PacketStop("raw result must be COMPLETE with the Experiment 06 schema")
    verdict = raw.get("verdict")
    if verdict not in VERDICTS or raw.get("scientific_verdict_emitted") is not True:
        raise PacketStop("raw result lacks a registered Experiment 06 verdict")
    seed_results = raw.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != 8:
        raise PacketStop("raw result must record all eight registered seed rows")
    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping):
        raise PacketStop("raw result lacks frozen input provenance")
    registered = inputs.get("registered_seeds")
    observed = [int(row.get("seed", -1)) for row in seed_results if isinstance(row, Mapping)]
    if not isinstance(registered, list) or tuple(int(item) for item in registered) != runner.FRESH_SEEDS or observed != list(runner.FRESH_SEEDS):
        raise PacketStop("seed rows do not preserve the exact registered Experiment 06 order")
    if inputs.get("q4_raw_sha256") != runner.Q4_RAW_SHA256:
        raise PacketStop("raw result does not bind the frozen Q4 byte hash")
    protocol_sha = str(inputs.get("protocol_sha256", ""))
    if len(protocol_sha) != 64 or any(character not in "0123456789abcdef" for character in protocol_sha):
        raise PacketStop("raw result lacks a canonical protocol SHA-256")
    expected_protocol_sha = _sha256_bytes(
        json.dumps(runner.FROZEN_PROTOCOL, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    if protocol_sha != expected_protocol_sha:
        raise PacketStop("raw result does not bind the in-code frozen protocol contract")
    q4_ordinals = inputs.get("q4_seed_ordinals")
    if not isinstance(q4_ordinals, list) or len(q4_ordinals) != 8:
        raise PacketStop("raw result lacks eight Q4 ordinal bindings")
    q4_ordinals = [int(item) for item in q4_ordinals]
    if q4_ordinals != sorted(set(q4_ordinals)):
        raise PacketStop("Q4 ordinal bindings are not eight unique sorted source seeds")
    target_ids = inputs.get("target_latent_ids")
    if not isinstance(target_ids, list) or len(target_ids) != runner.TARGET_COUNT or len(set(int(item) for item in target_ids)) != runner.TARGET_COUNT:
        raise PacketStop("raw result lacks the fixed twelve unique target ids")
    target_id_set = {int(item) for item in target_ids}
    if any(item < 0 or item >= 24_576 for item in target_id_set):
        raise PacketStop("target latent id is outside the registered SAE dictionary")
    if [int(item) for item in frozen_q4.get("target_latent_ids", ())] != [int(item) for item in target_ids]:
        raise PacketStop("compact result target ids differ from the hash-bound Q4 artifact")
    ordinal_sets = frozen_q4.get("ordinal_sets")
    if not isinstance(ordinal_sets, list) or len(ordinal_sets) != 8:
        raise PacketStop("hash-bound Q4 artifact lacks eight ordinal sets")
    if [int(row.get("source_q4_seed", -1)) for row in ordinal_sets] != q4_ordinals:
        raise PacketStop("compact result Q4 ordinals differ from the hash-bound Q4 artifact")
    matched = raw.get("matched_rows")
    if not isinstance(matched, list):
        raise PacketStop("raw result lacks matched rows")
    if verdict != "NON_ESTIMABLE" and len(matched) != 800:
        raise PacketStop("a directional verdict requires exactly 800 matched rows")
    adjudication = raw.get("adjudication")
    if not isinstance(adjudication, Mapping) or adjudication.get("verdict") != verdict:
        raise PacketStop("top-level and adjudication verdicts differ")
    if raw.get("claim_boundary") != runner.FROZEN_PROTOCOL["claim_boundary"]:
        raise PacketStop("raw result claim boundary differs from the frozen protocol")
    _validate_asset_provenance(raw)
    expected_design = {
        "template_family": "source_C_relative_clause_with_adverb",
        "public_label": "mechanism-held-out evaluation on a calibration-exposed template family",
        "l7_intervention": "frozen base-pattern L7H4 subject-value replacement at final query",
        "capture_hook": "blocks.8.hook_resid_pre",
        "source_A": "same template and number, different matrix-subject lemma",
        "source_A_mapping": "fixed cyclic next lemma in the registered 20-lemma order; no fixed points",
        "fresh_selection": "Gate A then first at most 150 retained pair ids; no object reselection",
        "reader_clamp": None,
    }
    if raw.get("design") != expected_design:
        raise PacketStop("raw result design metadata differs from the frozen runner contract")
    git = raw.get("git")
    git_final = raw.get("git_final")
    if not isinstance(git, Mapping) or not isinstance(git_final, Mapping) or dict(git) != dict(git_final):
        raise PacketStop("source git provenance changed or is incomplete")
    if not git.get("commit") or git.get("expected_commit") != git.get("commit") or git.get("require_clean_tree") is not True or git.get("status_porcelain") != "":
        raise PacketStop("raw result does not prove one exact clean source commit")
    if not git.get("source_tree_sha256"):
        raise PacketStop("raw result lacks the source-tree digest")
    offline = raw.get("offline_env")
    if not isinstance(offline, Mapping) or any(offline.get(key) != "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise PacketStop("raw result lacks offline model/SAE provenance")

    valid_seeds: set[int] = set()
    gate_invalid_seeds: set[int] = set()
    seed_by_id: dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(seed_results):
        if not isinstance(row, Mapping):
            raise PacketStop("seed result is not an object")
        seed = int(row["seed"])
        seed_by_id[seed] = row
        if int(row.get("source_q4_seed", -1)) != q4_ordinals[index]:
            raise PacketStop(f"seed {seed} changed its Q4 ordinal binding")
        status = row.get("status")
        gate = row.get("gate_a")
        if not isinstance(gate, Mapping):
            raise PacketStop(f"seed {seed} lacks Gate A evidence")
        if int(gate.get("generated_pairs", -1)) != runner.REQUESTED_PAIRS:
            raise PacketStop(f"seed {seed} Gate A does not cover 240 pairs")
        retained_ids = gate.get("retained_pair_ids")
        if not isinstance(retained_ids, list) or [int(item) for item in retained_ids] != sorted(set(int(item) for item in retained_ids)):
            raise PacketStop(f"seed {seed} Gate A retained ids are not unique and sorted")
        if any(int(item) < 0 or int(item) >= runner.REQUESTED_PAIRS for item in retained_ids):
            raise PacketStop(f"seed {seed} Gate A retained id is out of range")
        retained_count = int(gate.get("retained_pairs", -1))
        if retained_count != len(retained_ids):
            raise PacketStop(f"seed {seed} Gate A retained count differs from its ids")
        fraction = _finite(gate.get("fraction"), f"seed {seed} Gate A fraction")
        median_gap = _finite(gate.get("median_gap"), f"seed {seed} Gate A median gap")
        _close(fraction, retained_count / runner.REQUESTED_PAIRS, f"seed {seed} Gate A fraction")
        expected_thresholds = {"fraction_at_least": 0.6, "retained_at_least": 140, "median_gap_at_least": 1.0}
        if gate.get("thresholds") != expected_thresholds:
            raise PacketStop(f"seed {seed} Gate A thresholds differ from the frozen protocol")
        expected_gate_pass = bool(fraction >= 0.6 and retained_count >= 140 and median_gap >= 1.0)
        if gate.get("passed") is not expected_gate_pass:
            raise PacketStop(f"seed {seed} Gate A decision disagrees with its compact metrics")
        if status == "ADJUDICABLE":
            valid_seeds.add(seed)
            if gate.get("passed") is not True:
                raise PacketStop(f"adjudicable seed {seed} lacks a passing Gate A")
            estimands = row.get("estimands")
            if not isinstance(estimands, Mapping):
                raise PacketStop(f"adjudicable seed {seed} lacks estimands")
            for key in (
                "D_s",
                "full_signed_effect",
                "T_s",
                "E_s_second_largest_matched",
                "M_s_max_descriptive",
                "A_s",
                "complement_signed_effect",
                "target_plus_complement_signed_effect_sum",
                "target_plus_complement_minus_full_signed_effect",
            ):
                _finite(estimands.get(key), f"seed {seed} {key}")
            if [int(item) for item in row.get("target_latent_ids", ())] != [int(item) for item in target_ids]:
                raise PacketStop(f"seed {seed} changed the target ids")
            if int(row.get("target_projector", {}).get("rank", -1)) != runner.TARGET_COUNT:
                raise PacketStop(f"seed {seed} target projector is not rank twelve")
            expected_eval_ids = [int(item) for item in retained_ids[: runner.MAX_EVAL_PAIRS]]
            if [int(item) for item in row.get("evaluation_pair_ids", ())] != expected_eval_ids or int(row.get("evaluation_item_count", -1)) != 2 * len(expected_eval_ids):
                raise PacketStop(f"seed {seed} evaluation population differs from the registered Gate-A slice")
            diagnostics = row.get("identity_diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise PacketStop(f"seed {seed} lacks identity diagnostics")
            selected_error = _finite(
                diagnostics.get("selected_positions_max_abs"),
                f"seed {seed} selected identity error",
            )
            non_final_tolerance = _finite(
                diagnostics.get("non_final_tolerance"),
                f"seed {seed} non-final tolerance",
            )
            non_final_error = _finite(
                diagnostics.get("non_final_positions_max_abs"),
                f"seed {seed} non-final identity error",
            )
            full_tolerance = _finite(
                diagnostics.get("full_vs_true_final_logit_tolerance"),
                f"seed {seed} full-rescue tolerance",
            )
            full_error = _finite(
                diagnostics.get("full_vs_true_final_logit_max_abs"),
                f"seed {seed} full-rescue identity error",
            )
            if non_final_tolerance != runner.NON_FINAL_TOLERANCE or non_final_error > runner.NON_FINAL_TOLERANCE:
                raise PacketStop(f"seed {seed} violates the non-final timing identity")
            if full_tolerance != runner.FULL_LOGIT_TOLERANCE or full_error > runner.FULL_LOGIT_TOLERANCE:
                raise PacketStop(f"seed {seed} violates the full-rescue logit identity")
            if selected_error < 0.0 or non_final_error < 0.0 or full_error < 0.0:
                raise PacketStop(f"seed {seed} identity errors must be non-negative")
        elif status == "GATE_A_POPULATION_INVALID":
            gate_invalid_seeds.add(seed)
            if row.get("reason") != "GATE_A_FAIL" or gate.get("passed") is not False:
                raise PacketStop(f"seed {seed} has an invalid NON_ESTIMABLE reason")
        else:
            raise PacketStop(f"seed {seed} has unknown status {status!r}")

    grid: dict[int, dict[int, Mapping[str, Any]]] = {seed: {} for seed in runner.FRESH_SEEDS}
    for row in matched:
        if not isinstance(row, Mapping):
            raise PacketStop("matched row is not an object")
        seed = int(row.get("seed", -1))
        draw = int(row.get("draw_index", -1))
        if seed not in grid or draw in grid[seed]:
            raise PacketStop("matched grid has an unknown or duplicate seed/draw key")
        if int(row.get("source_q4_seed", -1)) != q4_ordinals[list(runner.FRESH_SEEDS).index(seed)]:
            raise PacketStop(f"matched row for seed {seed} changed its Q4 ordinal binding")
        latent_ids = [int(item) for item in row.get("latent_ids", ())]
        if len(latent_ids) != runner.TARGET_COUNT or len(set(latent_ids)) != runner.TARGET_COUNT or target_id_set.intersection(latent_ids):
            raise PacketStop(f"matched row {seed}/{draw} violates rank-12 target exclusion")
        if any(item < 0 or item >= 24_576 for item in latent_ids):
            raise PacketStop(f"matched row {seed}/{draw} has an out-of-range latent id")
        projector = row.get("projector")
        if not isinstance(projector, Mapping) or int(projector.get("rank", -1)) != runner.TARGET_COUNT or projector.get("arithmetic") != "float64":
            raise PacketStop(f"matched row {seed}/{draw} has invalid projector metadata")
        _finite(row.get("M_sj"), f"matched row {seed}/{draw} M_sj")
        if seed in valid_seeds:
            direct_for_ratio = float(seed_by_id[seed]["estimands"]["D_s"])
            expected_ratio = _safe_ratio(float(row["M_sj"]), direct_for_ratio)
            _close(row.get("descriptive_ratio"), expected_ratio, f"matched row {seed}/{draw} descriptive ratio")
        grid[seed][draw] = row

    for seed in runner.FRESH_SEEDS:
        draws = grid[seed]
        if seed in valid_seeds:
            if set(draws) != set(range(runner.MATCHED_COUNT)):
                raise PacketStop(f"adjudicable seed {seed} lacks the exact 0..99 matched grid")
            effects = sorted(float(draws[index]["M_sj"]) for index in range(runner.MATCHED_COUNT))
            frozen_draws = ordinal_sets[list(runner.FRESH_SEEDS).index(seed)].get("matched")
            if not isinstance(frozen_draws, list) or len(frozen_draws) != runner.MATCHED_COUNT:
                raise PacketStop(f"hash-bound Q4 ordinal for seed {seed} lacks 100 matched sets")
            frozen_by_draw = {int(row["draw_index"]): [int(item) for item in row["latent_ids"]] for row in frozen_draws}
            for draw_index in range(runner.MATCHED_COUNT):
                if [int(item) for item in draws[draw_index]["latent_ids"]] != frozen_by_draw.get(draw_index):
                    raise PacketStop(f"matched row {seed}/{draw_index} differs from the hash-bound Q4 latent set")
            estimands = seed_by_id[seed]["estimands"]
            edge = effects[-2]
            maximum = effects[-1]
            target = float(estimands["T_s"])
            full = float(estimands["full_signed_effect"])
            complement = float(estimands["complement_signed_effect"])
            target_plus_complement = target + complement
            _close(estimands["E_s_second_largest_matched"], edge, f"seed {seed} second-largest edge")
            _close(estimands["M_s_max_descriptive"], maximum, f"seed {seed} matched maximum")
            _close(estimands["A_s"], target - edge, f"seed {seed} target advantage")
            _close(estimands["D_s"], full, f"seed {seed} direct/full identity", tolerance=1e-5)
            _close(estimands["target_plus_complement_signed_effect_sum"], target_plus_complement, f"seed {seed} closure sum")
            _close(estimands["target_plus_complement_minus_full_signed_effect"], target_plus_complement - full, f"seed {seed} closure difference")
            expected_closure_ratio = _safe_ratio(target_plus_complement, full)
            _close(estimands.get("target_plus_complement_over_full_ratio_descriptive"), expected_closure_ratio, f"seed {seed} closure ratio")
            _close(estimands.get("target_over_direct_ratio_descriptive"), _safe_ratio(target, float(estimands["D_s"])), f"seed {seed} target/direct ratio")
            _close(estimands.get("complement_over_direct_ratio_descriptive"), _safe_ratio(complement, float(estimands["D_s"])), f"seed {seed} complement/direct ratio")
        elif draws:
            raise PacketStop(f"Gate-A-invalid seed {seed} unexpectedly has matched rows")

    expected_matched_count = len(valid_seeds) * runner.MATCHED_COUNT
    if len(matched) != expected_matched_count:
        raise PacketStop("matched row count differs from the complete valid-seed grids")
    runtime = raw.get("runtime")
    if not isinstance(runtime, Mapping) or int(runtime.get("adjudicable_seed_count", -1)) != len(valid_seeds) or int(runtime.get("matched_row_count", -1)) != len(matched):
        raise PacketStop("runtime row counts disagree with the compact payload")
    recomputed = _reaggregate_adjudication(seed_results)
    if _canonical(recomputed) != _canonical(raw.get("adjudication")):
        raise PacketStop("stored adjudication differs from independent compact-row reaggregation")


def _seed_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw["seed_results"]:
        gate = item.get("gate_a") if isinstance(item.get("gate_a"), Mapping) else {}
        estimands = item.get("estimands") if isinstance(item.get("estimands"), Mapping) else {}
        reason = item.get("reason")
        rows.append(
            {
                "seed": int(item["seed"]),
                "source_q4_seed": int(item.get("source_q4_seed", -1)),
                "status": str(item.get("status", "")),
                "reason": json.dumps(reason, sort_keys=True, ensure_ascii=False) if reason is not None else "",
                "gate_a_fraction": _finite_or_none(gate.get("fraction")),
                "gate_a_retained_pairs": gate.get("retained_pairs"),
                "gate_a_median_gap": _finite_or_none(gate.get("median_gap")),
                "evaluation_item_count": item.get("evaluation_item_count"),
                "D_s": _finite_or_none(estimands.get("D_s")),
                "full_signed_effect": _finite_or_none(estimands.get("full_signed_effect")),
                "T_s": _finite_or_none(estimands.get("T_s")),
                "E_s_second_largest_matched": _finite_or_none(estimands.get("E_s_second_largest_matched")),
                "A_s": _finite_or_none(estimands.get("A_s")),
                "M_s_max_descriptive": _finite_or_none(estimands.get("M_s_max_descriptive")),
                "complement_signed_effect_descriptive": _finite_or_none(estimands.get("complement_signed_effect")),
                "complement_over_direct_ratio_descriptive": _finite_or_none(estimands.get("complement_over_direct_ratio_descriptive")),
                "target_plus_complement_signed_effect_sum_descriptive": _finite_or_none(estimands.get("target_plus_complement_signed_effect_sum")),
                "target_plus_complement_minus_full_signed_effect_descriptive": _finite_or_none(estimands.get("target_plus_complement_minus_full_signed_effect")),
                "target_plus_complement_over_full_ratio_descriptive": _finite_or_none(estimands.get("target_plus_complement_over_full_ratio_descriptive")),
                "target_over_direct_ratio_descriptive": _finite_or_none(estimands.get("target_over_direct_ratio_descriptive")),
            }
        )
    return rows


def _matched_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw["matched_rows"]:
        rows.append(
            {
                "seed": int(item["seed"]),
                "source_q4_seed": int(item["source_q4_seed"]),
                "draw_index": int(item["draw_index"]),
                "latent_ids": json.dumps([int(value) for value in item["latent_ids"]], separators=(",", ":")),
                "M_sj": _finite_or_none(item["M_sj"]),
                "descriptive_ratio": _finite_or_none(item.get("descriptive_ratio")),
                "projector_rank": int(item["projector"]["rank"]),
            }
        )
    return rows


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _interval_line(label: str, summary: Mapping[str, Any] | None) -> str:
    if not isinstance(summary, Mapping):
        return f"- `{label}`: not estimable"
    ci = summary.get("ci95", {})
    return f"- `{label}`: mean `{_fmt(summary.get('mean'))}`, 95% t(7) CI `[{_fmt(ci.get('low'))}, {_fmt(ci.get('high'))}]`"


def _claim_for(verdict: str) -> str:
    claims = {
        "POSITIVE": (
            "On a relative-clause family exposed during calibration but not previously used for bridge "
            "adjudication, the fixed L7H4 intervention retained a nontrivial true-vs-fixed-source-A effect across "
            "fresh lexical seeds, and the fixed 12-dimensional SAE span stably exceeded the "
            "second-largest raw-effect edge among the 100 frozen Q4 matched latent sets; this supports "
            "reappearance of the intervention-defined head-to-subspace relation under a second prompt/control "
            "construction, not a one-factor estimate of template-family transfer."
        ),
        "MECHANISM_NEGATIVE": (
            "The fixed L7H4 true-vs-fixed-source-A handle did not meet the preregistered mechanism "
            "criterion, so span transfer is not adjudicated; this does not show that L7H4 is uninvolved "
            "in number agreement or that no other transport mechanism exists."
        ),
        "SPAN_NEGATIVE": (
            "The fixed L7H4 handle met the preregistered mechanism criterion on this family, but the fixed 12-dimensional SAE span did "
            "not meet the registered positive-effect and frozen-tail-edge criterion on this family."
        ),
        "NON_ESTIMABLE": (
            "The registered seeds did not form a complete estimable eight-seed evidence set, so the run "
            "provides no directional conclusion about mechanism or span transfer."
        ),
    }
    return claims[verdict]


def _results_markdown(summary: Mapping[str, Any], seed_count: int, matched_count: int) -> str:
    adjudication = summary["adjudication"]
    statistics = adjudication.get("statistics") or {}
    verdict = str(summary["verdict"])
    assets = summary["asset_provenance"]
    lines = [
        "# Experiment 06 results",
        "",
        f"**Registered verdict: `{verdict}`.**",
        "",
        _claim_for(verdict),
        "",
        "## Registered estimands",
        "",
        _interval_line("D_s", statistics.get("D_s")),
        _interval_line("T_s", statistics.get("T_s")),
        _interval_line("A_s = T_s - second-largest(M_sj)", statistics.get("A_s")),
        "",
        "Ratios, complement, closure, and the maximum matched draw are descriptive only and did not "
        "enter the verdict.",
        "Exp06 reuses Q4's fixed latent sets and second-largest tail-order statistic, not Q4's "
        "normalized `R` estimand.",
        "",
        "## Evidence and provenance",
        "",
        f"- registered seed rows: `{seed_count}`; matched rows: `{matched_count}`;",
        f"- raw result SHA-256: `{summary['source']['raw_result_sha256']}`;",
        f"- source commit: `{summary['source']['git_commit']}`;",
        f"- Q4 raw SHA-256: `{summary['inputs']['q4_raw_sha256']}`;",
        f"- protocol canonical SHA-256: `{summary['inputs']['protocol_sha256']}`;",
        f"- GPT-2 revision: `{assets['repositories']['gpt2']['revision']}`;",
        f"- SAE revision: `{assets['repositories']['sae']['revision']}`;",
        f"- unchanged model-state fingerprint: `{assets['model_state']['before']['sha256']}`;",
        f"- unchanged SAE-decoder fingerprint: `{assets['sae_decoder']['before']['sha256']}`;",
        "- bridge-specific independent review receipt: none in this packet.",
        "",
        "## Scope",
        "",
        "This is a mechanism-held-out evaluation on a calibration-exposed template family. It is not a "
        "fully unseen-template test, independent external validation, broad syntactic generalisation, "
        "natural or monosemantic latent semantics, individual-latent causality, necessity, sufficiency, "
        "mediation, a complete circuit, or evidence beyond the tested model, family, and intervention. "
        "Because the source-A construction also differs from the Experiment 05 bridge, this is not a "
        "one-factor estimate of template-family change.",
        "",
    ]
    return "\n".join(lines)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_staged_directory(stage: Path, output_dir: Path) -> None:
    """Publish one complete packet under a cooperative exclusive lock."""

    lock = output_dir.with_name(f".{output_dir.name}.publish.lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PacketStop(f"packet publication lock already exists: {lock}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        if output_dir.exists():
            raise PacketStop(f"refusing to overwrite existing packet directory: {output_dir}")
        os.rename(stage, output_dir)
        _fsync_directory(output_dir.parent)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def build_packet(raw_path: Path, q4_path: Path, output_dir: Path, expected_raw_sha: str) -> dict[str, Any]:
    raw_path = raw_path.expanduser().resolve()
    q4_path = q4_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    raw, actual_raw_sha = _read_bound_json(raw_path, "Experiment 06 raw result")
    q4, actual_q4_sha = _read_bound_json(q4_path, "Q4 raw result")
    if actual_q4_sha != runner.Q4_RAW_SHA256:
        raise PacketStop(f"Q4 SHA-256 {actual_q4_sha} differs from the frozen input")
    try:
        frozen_q4 = runner.bridge.parse_q4_frozen_sets(q4)
    except Exception as exc:
        raise PacketStop(f"cannot parse the hash-bound Q4 frozen sets: {exc}") from exc
    _validate(raw, expected_raw_sha, actual_raw_sha, frozen_q4)
    if output_dir.exists():
        raise PacketStop(f"refusing to overwrite existing packet directory: {output_dir}")
    seeds = _seed_rows(raw)
    matched = _matched_rows(raw)
    summary = {
        "schema": PUBLIC_SCHEMA,
        "status": "COMPLETE",
        "verdict": raw["verdict"],
        "review_receipt": None,
        "source": {
            "raw_result_external": True,
            "raw_result_sha256": actual_raw_sha,
            "git_commit": raw.get("git", {}).get("commit"),
            "source_tree_sha256": raw.get("git", {}).get("source_tree_sha256"),
        },
        "inputs": {
            "protocol_sha256": raw["inputs"]["protocol_sha256"],
            "q4_raw_sha256": raw["inputs"]["q4_raw_sha256"],
            "registered_seeds": raw["inputs"]["registered_seeds"],
            "q4_seed_ordinals": raw["inputs"]["q4_seed_ordinals"],
            "target_latent_ids": raw["inputs"]["target_latent_ids"],
        },
        "design": raw["design"],
        "asset_provenance": raw["asset_provenance"],
        "adjudication": raw["adjudication"],
        "claim_boundary": raw["claim_boundary"],
        "row_counts": {"seed_rows": len(seeds), "matched_rows": len(matched)},
        "public_claim": _claim_for(str(raw["verdict"])),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage.", dir=output_dir.parent))
    try:
        artifacts = {
            "result_summary.json": _json_text(summary),
            "seed_metrics.csv": _csv_text(list(seeds[0].keys()), seeds),
            "matched_effects.csv": _csv_text(
                ["seed", "source_q4_seed", "draw_index", "latent_ids", "M_sj", "descriptive_ratio", "projector_rank"],
                matched,
            ),
            "RESULTS.md": _results_markdown(summary, len(seeds), len(matched)),
        }
        artifact_index = {
            "schema": "exp06-cross-template-bridge-artifact-index-v1",
            "status": "COMPLETE",
            "source_raw_sha256": actual_raw_sha,
            "verdict": raw["verdict"],
            "review_receipt": None,
            "artifacts": {
                name: {"sha256": _sha256_bytes(text.encode("utf-8")), "bytes": len(text.encode("utf-8"))}
                for name, text in artifacts.items()
            },
        }
        artifacts["artifact_index.json"] = _json_text(artifact_index)
        for name, text_value in artifacts.items():
            _atomic_text(stage / name, text_value)
        checksums = "".join(
            f"{_sha256_bytes(text_value.encode('utf-8'))}  {name}\n"
            for name, text_value in sorted(artifacts.items())
        )
        _atomic_text(stage / "checksums.sha256", checksums)
        _fsync_directory(stage)
        _publish_staged_directory(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the model-free Experiment 06 public packet")
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--q4-results", required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_packet(Path(args.raw_result), Path(args.q4_results), Path(args.output_dir), str(args.expected_raw_sha256))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
