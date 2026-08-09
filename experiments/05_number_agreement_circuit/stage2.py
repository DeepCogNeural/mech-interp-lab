"""Experiment 05 Stage 2 (Q1--Q3), frozen fail-closed adjudication runner.

This module is intentionally an executable *protocol implementation*, rather
than a notebook.  It never loads a model at import time.  A run must provide
the frozen protocol, calibration, fresh paired-sweep selection artifact, and an
immutable candidate artifact produced from that selection.  Every
seed is checkpointed atomically; a runtime-cap stop writes only a resumable
checkpoint and a non-verdict manifest.

The numerical estimators are delegated to ``exp05_core``.  The adapter below
requires the declared core API and refuses to substitute local estimators when
the API is absent or incompatible: silent changes to the bootstrap unit,
percentile interpolation, Holm family, or deterministic RNG would invalidate
the pre-registration.

Run (only after the repository is committed and the user has authorised the
model/experiment gate):

.. code-block:: console

   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg \
     ../../.venv/bin/python stage2.py \
     --protocol protocol_v1.json --calibration calibration_results.json \
     --selection selection_source_a.json \
     --candidate candidate.json \
     --expected-git-commit <commit> --require-clean-tree \
     --max-wall-seconds 10800 --output stage2_results.json \
     --pair-output stage2_pair_stats.csv \
     --checkpoint stage2_checkpoint.json

No model, experiment, or test runs on import.  ``main``/``run`` are the only
execution entry points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# These imports are deliberately the same reusable primitives used by the
# shipped Stage-1 runner.  Importing them does not load GPT-2.
from calibrate import (  # noqa: E402
    gate_a,
    make_source_a,
    make_source_b,
    make_source_c_relative_clause,
)
from pilot import (  # noqa: E402
    CleanPass,
    build_stimuli,
    directed_indices,
    load_model,
    positions_for_kind,
    require_one_token,
    set_determinism,
)
from stage1 import (  # noqa: E402
    HEADS_PER_LAYER,
    HOOK_Z,
    LAYER_COUNT,
    PATCH_BATCH_SIZE,
    AttentionPatchRunner,
    _patch_hook,
    _source_values,
    assert_hook_z_layout,
    cached_stage1_clean_pass,
    clean_readout_microbatched,
)


PROTOCOL_SCHEMA = "exp05-number-agreement-protocol"
CALIBRATION_SCHEMA_PREFIX = "exp05-number-agreement-calibration-v1"
CORE_API_VERSION = "exp05-core-v1"
STAGE2_SCHEMA = "exp05-number-agreement-stage2-v1; frozen Q1-Q3; CPU float32; Amendments 1-8"
CHECKPOINT_SCHEMA = "exp05-number-agreement-stage2.checkpoint.v1; Amendments 1-8"
PAIR_SCHEMA = "exp05-number-agreement-stage2.pairs.v1"
CANDIDATE_SCHEMA = "exp05.candidate.v1"

EXPECTED_SEEDS = tuple(range(20_260_802, 20_260_810))
SELECTION_SEED = 20_260_801
REQUESTED_PAIRS = 240
EXPECTED_HEADS = 144
Q2_MIN_PAIRS = 40
Q2_UNRESOLVED_CODE = "SCIENTIFIC_UNRESOLVED_Q2_LT40_COMPLETE_PAIRS"
Q1_MAX = 8
BOOTSTRAP_HEAD = 100_000
BOOTSTRAP_INTERVAL = 10_000
HOLM_ALPHA = 0.05
Q3_TEST_ID = 301
HEAD_TEST_ID_OFFSET = 1
THETA_A = 0.20
THETA_C = 0.26983
DIRECTIONS = ("singular_to_plural", "plural_to_singular")


class Stage2Stop(RuntimeError):
    """Fail-closed stop; status is never interpreted as a scientific verdict."""

    def __init__(self, gate: str, message: str, *, status: str = "STOPPED"):
        super().__init__(message)
        self.gate = gate
        self.status = status


class RuntimeCapStop(Stage2Stop):
    def __init__(self, message: str):
        super().__init__("max_wall_seconds", message, status="INCOMPLETE_RUNTIME_CAP")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite artifact value: {value}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported artifact value: {type(value)!r}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_git_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and value == value.lower() and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Stage2Stop("input_read", f"Unable to hash {path}: {exc}") from exc
    return digest.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return _sha256_bytes(
        _json_bytes(
            {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes_sha256": _sha256_bytes(value.numpy().tobytes()),
            }
        )
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = _csv_bytes(rows, fieldnames)
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _jsonable(row.get(name)) for name in fieldnames})
    return buffer.getvalue().encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2Stop("missing_input", f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2Stop("invalid_input_json", f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Stage2Stop("invalid_input_schema", f"{label} must be a JSON object: {path}")
    return value


def _resolve_path(value: str | Path, *, base: Path = HERE) -> Path:
    path = Path(value).expanduser()
    return path.resolve(strict=False) if path.is_absolute() else (base / path).resolve(strict=False)


def _git(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=HERE, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise Stage2Stop("git_unavailable", f"Cannot inspect git provenance: {exc}") from exc
    if result.returncode != 0:
        raise Stage2Stop("git_provenance", f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _require_clean_tree_except_runtime_artifacts(paths: Sequence[Path]) -> list[str]:
    repo_root = Path(_git(["rev-parse", "--show-toplevel"])).resolve()
    allowed: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(repo_root)
        except ValueError:
            continue
        allowed.add(resolved)
    dirty = _git(["status", "--porcelain=v1", "--untracked-files=all"])
    allowed_seen: list[str] = []
    disallowed: list[str] = []
    for line in dirty.splitlines():
        if not line:
            continue
        raw = line[3:]
        if " -> " in raw or (raw.startswith('"') and raw.endswith('"')):
            disallowed.append(line)
            continue
        resolved = (repo_root / raw).resolve()
        if resolved in allowed:
            allowed_seen.append(raw)
        else:
            disallowed.append(line)
    if disallowed:
        raise Stage2Stop("dirty_tree", "Repository has changes outside the exact declared runtime artifacts: " + "; ".join(disallowed[:8]))
    return sorted(allowed_seen)


def _canonical_heads() -> list[dict[str, int]]:
    return [{"layer": layer, "head": head} for layer in range(LAYER_COUNT) for head in range(HEADS_PER_LAYER)]


def _head_key(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def _flat_test_id(layer: int, head: int) -> int:
    return 12 * int(layer) + int(head) + HEAD_TEST_ID_OFFSET


def _check_runtime(started: float, cap: float | None, where: str) -> None:
    if cap is not None and (time.perf_counter() - started) >= cap:
        raise RuntimeCapStop(f"Declared runtime cap {cap:.3f}s reached at {where}.")


def _lookup(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _require_equal(value: Any, expected: Any, gate: str, label: str) -> None:
    if value != expected:
        raise Stage2Stop(gate, f"{label}={value!r} differs from frozen {expected!r}.")


def _validate_protocol(protocol: Mapping[str, Any], path: Path) -> None:
    _require_equal(protocol.get("schema"), PROTOCOL_SCHEMA, "protocol_schema", "protocol.schema")
    _require_equal(int(protocol.get("version", -1)), 1, "protocol_version", "protocol.version")
    design_freeze = protocol.get("design_freeze")
    if not isinstance(design_freeze, Mapping):
        raise Stage2Stop("protocol_version", "protocol.design_freeze must be an object containing latest_amendment.")
    # Amendment 9 is Q4-only; Stage-2 still validates and executes under its
    # frozen Amendment-8 single-invocation contract while accepting the
    # protocol's global latest amendment.
    _require_equal(int(design_freeze.get("latest_amendment", -1)), 9, "protocol_version", "protocol.design_freeze.latest_amendment")
    _require_equal(design_freeze.get("preserved_amendments"), list(range(1, 10)), "protocol_version", "protocol.design_freeze.preserved_amendments")
    if protocol.get("status") not in {"designed_not_executed", "frozen"}:
        raise Stage2Stop("protocol_status", f"Unexpected protocol status {protocol.get('status')!r} in {path}.")
    model = protocol.get("model")
    if not isinstance(model, Mapping):
        raise Stage2Stop("protocol_model", "protocol.model must be an object.")
    for key, expected in (("name", "gpt2-small"), ("mechanism_library", "TransformerLens"), ("activation_dtype", "float32"), ("residual_width", 768)):
        _require_equal(model.get(key), expected, "protocol_model", f"protocol.model.{key}")
    heads = protocol.get("head_universe")
    if not isinstance(heads, Mapping):
        raise Stage2Stop("protocol_head_universe", "protocol.head_universe must be an object.")
    for key, expected in (("layer_count", 12), ("heads_per_layer", 12), ("total_heads", 144)):
        _require_equal(int(heads.get(key, -1)), expected, "protocol_head_universe", f"protocol.head_universe.{key}")
    gate_a_protocol = protocol.get("gate_a")
    conditions = gate_a_protocol.get("conditions") if isinstance(gate_a_protocol, Mapping) else None
    if not isinstance(conditions, Mapping):
        raise Stage2Stop("protocol_gate_a", "protocol.gate_a.conditions must be an object.")
    for key, expected in (("both_members_signed_correct_fraction_at_least", 0.6), ("minimum_retained_pairs", 140), ("median_clean_d_gap_at_least", 1.0)):
        _require_equal(conditions.get(key), expected, "protocol_gate_a", f"gate_a.conditions.{key}")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, Mapping):
        raise Stage2Stop("protocol_seeds", "protocol.seeds must be an object.")
    _require_equal(tuple(int(seed) for seed in seeds.get("stage2_adjudication", ())), EXPECTED_SEEDS, "protocol_seeds", "stage2 seeds")
    stage2 = protocol.get("stage2")
    if not isinstance(stage2, Mapping):
        raise Stage2Stop("protocol_stage2", "protocol.stage2 must be an object.")
    input_artifacts = stage2.get("input_artifacts")
    if not isinstance(input_artifacts, Mapping):
        raise Stage2Stop("protocol_stage2", "stage2.input_artifacts must be an object.")
    _require_equal(input_artifacts.get("required"), ["protocol", "calibration", "selection", "candidate"], "protocol_stage2", "Stage-2 required artifacts")
    _require_equal(input_artifacts.get("stage1_artifact_cli_input"), False, "protocol_stage2", "Stage-1 CLI input boundary")
    _require_equal(stage2.get("adjudication_seeds"), list(EXPECTED_SEEDS), "protocol_stage2", "stage2.adjudication_seeds")
    rule = stage2.get("candidate_rule")
    if not isinstance(rule, Mapping):
        raise Stage2Stop("protocol_candidate_rule", "protocol.stage2.candidate_rule must be an object.")
    _require_equal(rule.get("top_rank_count"), 10, "protocol_candidate_rule", "top_rank_count")
    _require_equal(rule.get("tie_break"), "layer then head ascending", "protocol_candidate_rule", "tie_break")
    _require_equal(rule.get("pool"), "C = fresh same-snapshot true/source-A top-10 heads satisfying all registered requirements", "protocol_candidate_rule", "pool")
    _require_equal(rule.get("source_seed"), SELECTION_SEED, "protocol_candidate_rule", "source_seed")
    _require_equal(rule.get("rank_source"), "fresh_true_source signed mean delta_d descending from same-snapshot selection invocation", "protocol_candidate_rule", "rank_source")
    _require_equal(rule.get("requires_source_a_noise_edge"), True, "protocol_candidate_rule", "requires_source_a_noise_edge")
    _require_equal(rule.get("requires_true_source_holm_distinguishability"), True, "protocol_candidate_rule", "requires_true_source_holm_distinguishability")
    if not math.isclose(float(rule.get("requires_pair_level_sign_consistency_at_least", float("nan"))), 0.9, abs_tol=0.0):
        raise Stage2Stop("protocol_candidate_rule", "Candidate pair-level sign-consistency floor must equal 0.9.")
    single_invocation = stage2.get("single_invocation_provenance")
    if not isinstance(single_invocation, Mapping):
        raise Stage2Stop("protocol_single_invocation", "protocol.stage2.single_invocation_provenance is required for Amendment 8.")
    _require_equal(single_invocation.get("amendment"), 8, "protocol_single_invocation", "single_invocation_provenance.amendment")
    for key, expected in (("all_eight_seeds_and_adjudicating_cells_one_fresh_invocation", True), ("cross_invocation_scientific_reuse_allowed", False), ("prior_checkpoint_role", "diagnostic_integrity_inspection_only"), ("prior_checkpoint_rows_enter_fresh_adjudicating_dataset", False), ("restart_from_zero_after_incomplete_attempt", True), ("only_final_complete_invocation_supplies_science", True)):
        _require_equal(single_invocation.get(key), expected, "protocol_single_invocation", f"single_invocation_provenance.{key}")
    incomplete_artifact = single_invocation.get("incomplete_artifact")
    if not isinstance(incomplete_artifact, Mapping):
        raise Stage2Stop("protocol_single_invocation", "single_invocation_provenance.incomplete_artifact is required.")
    for key, expected in (("execution_status", "EXECUTION_INCOMPLETE"), ("scientific_reuse_allowed", False), ("resumable_for_adjudication", False), ("scientific_verdict_emitted", False)):
        _require_equal(incomplete_artifact.get(key), expected, "protocol_single_invocation", f"incomplete_artifact.{key}")
    _require_equal(single_invocation.get("complete_requires"), ["all registered adjudicating cells for all eight seeds in one invocation", "complete pair-level CSV from that invocation", "final checkpoint from that invocation", "pair CSV and checkpoint exact seed and execution-cell coverage match", "configuration code revision model-state fingerprint input hashes and artifact content hashes bound in final manifest"], "protocol_single_invocation", "single_invocation_provenance.complete_requires")
    _require_equal(single_invocation.get("complete_status_codes"), ["STAGE2_COMPLETE_SINGLE_INVOCATION", "STAGE2_FINAL_ARTIFACTS_BOUND", "STAGE2_PAIR_CSV_CHECKPOINT_COVERAGE_MATCH"], "protocol_single_invocation", "single_invocation_provenance.complete_status_codes")
    _require_equal(single_invocation.get("incomplete_status_codes"), ["STAGE2_EXECUTION_INCOMPLETE_RUNTIME_CAP", "STAGE2_EXECUTION_INCOMPLETE_INTERRUPTED", "STAGE2_EXECUTION_INCOMPLETE_REQUIRED_CELL_MISSING", "STAGE2_EXECUTION_INCOMPLETE_SEED_COVERAGE", "STAGE2_EXECUTION_INCOMPLETE_FINAL_BINDING", "STAGE2_EXECUTION_INCOMPLETE_ARTIFACT_MISMATCH", "STAGE2_NONVERDICT_NONRESUMABLE", "STAGE2_RESTART_FROM_ZERO_REQUIRED", "STAGE2_CROSS_INVOCATION_REUSE_REJECTED"], "protocol_single_invocation", "single_invocation_provenance.incomplete_status_codes")
    _require_equal(single_invocation.get("compute_accounting_fields"), ["final_complete_invocation_logical_fe", "prior_incomplete_attempt_logical_fe", "cumulative_logical_fe_all_attempts", "final_complete_invocation_wall_time", "cumulative_wall_time_all_attempts"], "protocol_single_invocation", "single_invocation_provenance.compute_accounting_fields")
    edge_rule = rule.get("source_a_noise_edge")
    if not isinstance(edge_rule, Mapping):
        raise Stage2Stop("protocol_candidate_rule", "Candidate source-A noise-edge rule is missing.")
    for key, expected in (("values", "absolute source-A mean delta_d for all 144 heads"), ("quantile", 0.99), ("method", 'numpy.percentile(values, 99, method="linear")')):
        _require_equal(edge_rule.get(key), expected, "protocol_candidate_rule", f"source_a_noise_edge.{key}")
    bootstrap = protocol.get("bootstrap_and_multiple_testing")
    if not isinstance(bootstrap, Mapping):
        raise Stage2Stop("protocol_bootstrap", "protocol.bootstrap_and_multiple_testing must be an object.")
    _require_equal(bootstrap.get("per_head_resamples"), BOOTSTRAP_HEAD, "protocol_bootstrap", "per_head_resamples")
    _require_equal(bootstrap.get("other_stage2_interval_resamples"), BOOTSTRAP_INTERVAL, "protocol_bootstrap", "other_stage2_interval_resamples")
    _require_equal(bootstrap.get("percentile_method"), 'numpy.percentile(method="linear")', "protocol_bootstrap", "percentile_method")
    _require_equal(bootstrap.get("resampling_unit"), "retained_minimal_pair_cluster", "protocol_bootstrap", "resampling_unit")
    _require_equal(bootstrap.get("directed_edits_and_arms_resampled_together"), True, "protocol_bootstrap", "paired arm resampling")
    _require_equal(bootstrap.get("per_head_p_value"), "min(1, 2 * min((n_mu_bootstrap_le_0 + 1)/(B + 1), (n_mu_bootstrap_ge_0 + 1)/(B + 1)))", "protocol_bootstrap", "per-head p-value formula")
    holm = bootstrap.get("holm")
    if not isinstance(holm, Mapping):
        raise Stage2Stop("protocol_bootstrap", "Holm family declaration is missing.")
    for key, expected in (("alpha", HOLM_ALPHA), ("method", "Holm step-down"), ("family", "144 heads within each seed"), ("raw_statistic", "head raw mean delta_d; E_all division is effect-size reporting only")):
        _require_equal(holm.get(key), expected, "protocol_bootstrap", f"holm.{key}")
    rng = protocol.get("rng")
    if not isinstance(rng, Mapping) or rng.get("generator") != "numpy.random.default_rng" or rng.get("seed_formula") != "experiment_seed * 1000 + test_id" or rng.get("iteration_order_derived_seed_forbidden") is not True:
        raise Stage2Stop("protocol_rng", "Frozen deterministic RNG formula is missing or changed.")
    test_ids = protocol.get("test_ids")
    if not isinstance(test_ids, Mapping) or test_ids.get("q3_true_minus_source_a_interval") != Q3_TEST_ID:
        raise Stage2Stop("protocol_test_ids", "Frozen Q3 test id 301 is missing or changed.")
    q1 = protocol.get("q1")
    if not isinstance(q1, Mapping):
        raise Stage2Stop("protocol_q1", "protocol.q1 must be an object.")
    q1_adjudication = q1.get("adjudication")
    if not isinstance(q1_adjudication, Mapping):
        raise Stage2Stop("protocol_q1", "q1.adjudication must be an object.")
    for key, expected in (("max_heads", Q1_MAX), ("recovery_fraction_min", 0.5), ("pass_seed_count", 6), ("seed_count", 8), ("all_members_individually_distinguishable", True)):
        _require_equal(q1_adjudication.get(key), expected, "protocol_q1", f"q1.adjudication.{key}")
    guard = q1.get("e_all_guard")
    if not isinstance(guard, Mapping):
        raise Stage2Stop("protocol_q1", "q1.e_all_guard must be an object.")
    for key, expected in (("vector", "retained complete-pair E_all effects e"), ("arithmetic_dtype", "float64"), ("float64_eps", "numpy.finfo(numpy.float64).eps"), ("unresolved_status", "SCIENTIFIC_UNRESOLVED")):
        _require_equal(guard.get(key), expected, "protocol_q1", f"q1.e_all_guard.{key}")
    formula = guard.get("formula")
    if not isinstance(formula, Mapping):
        raise Stage2Stop("protocol_q1", "q1.e_all_guard.formula must be an object.")
    for key, expected in (("E", "mean(e)"), ("A", "mean(abs(e))"), ("g", "abs(E) / A"), ("gamma", "sqrt(float64_eps)")):
        _require_equal(formula.get(key), expected, "protocol_q1", f"q1.e_all_guard.formula.{key}")
    if "1e-12" not in list(guard.get("forbidden_denominator_guards", [])):
        raise Stage2Stop("protocol_q1", "q1 denominator guard must explicitly forbid 1e-12.")
    gate_failure = q1.get("base_global_gate_a_failure")
    if not isinstance(gate_failure, Mapping):
        raise Stage2Stop("protocol_q1", "q1.base_global_gate_a_failure must be an object.")
    for key, expected in (("execution_status", "COMPLETE"), ("axis_class", "SCIENTIFIC_UNRESOLVED"), ("cell_status", "SKIPPED_BY_PREREGISTERED_SCIENTIFIC_GATE"), ("reason_code", "Q1_BASE_GLOBAL_GATE_A_FAIL"), ("shared_q2_q3_cache_is_not_q1_execution", True), ("does_not_clear_q2_item_ids", True)):
        _require_equal(gate_failure.get(key), expected, "protocol_q1", f"q1.base_global_gate_a_failure.{key}")
    _require_equal(gate_failure.get("skip_cells"), ["true_source_sweep", "source_a_sweep", "E_all", "E(S_n)", "Q1_recovery_ratio"], "protocol_q1", "q1 base-gate skip cells")
    tested_selection = q1.get("tested_set_selection")
    if not isinstance(tested_selection, Mapping):
        raise Stage2Stop("protocol_q1", "q1.tested_set_selection must be an object.")
    for key, expected in (("scientific_unresolved_counts_as_pass", False), ("select_minimum_n_if", "observed_pass_lower_count >= 6"), ("selected_min_n_status", "Q1_TESTED_SET_SELECTED_MIN_N_PASS_GE6"), ("fallback_set", "S_min(8, |C|)"), ("fallback_status", "Q1_TESTED_SET_SELECTED_FALLBACK_MIN8C"), ("fallback_changes_q1_axis_verdict", False), ("blocked_status", "Q1_TESTED_SET_BLOCKED_EXECUTION_INCOMPLETE")):
        _require_equal(tested_selection.get(key), expected, "protocol_q1", f"q1.tested_set_selection.{key}")
    q2 = protocol.get("q2")
    if not isinstance(q2, Mapping):
        raise Stage2Stop("protocol_q2", "protocol.q2 must be an object.")
    _require_equal(q2.get("block_if_complete_pairs_below"), Q2_MIN_PAIRS, "protocol_q2", "q2 block threshold")
    _require_equal(q2.get("block_code"), "Q2_BLOCKED_LT40_COMPLETE_PAIRS", "protocol_q2", "q2 legacy block-code alias")
    eligibility = q2.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise Stage2Stop("protocol_q2", "q2.eligibility must be an object.")
    _require_equal(eligibility.get("population"), "complete_pair_ids(base_gate_a_retained intersect source_c_gate_a_retained)", "protocol_q2", "q2 eligibility population")
    _require_equal(eligibility.get("both_directed_edits_required"), True, "protocol_q2", "q2 both-directed requirement")
    _require_equal(eligibility.get("minimum_complete_pairs"), Q2_MIN_PAIRS, "protocol_q2", "q2 eligibility minimum")
    source_c_global = eligibility.get("global_source_c_gate_a_passed")
    if not isinstance(source_c_global, Mapping):
        raise Stage2Stop("protocol_q2", "q2 global source-C diagnostic contract is missing.")
    for key, expected in (("role", "diagnostic_only"), ("never_zeroes_intersection", True), ("never_replaces_item_intersection", True), ("never_creates_q2_block", True)):
        _require_equal(source_c_global.get(key), expected, "protocol_q2", f"q2 global source-C {key}")
    lt40 = eligibility.get("lt40")
    if not isinstance(lt40, Mapping):
        raise Stage2Stop("protocol_q2", "q2.eligibility.lt40 must be an object.")
    _require_equal(lt40.get("axis_class"), "SCIENTIFIC_UNRESOLVED", "protocol_q2", "q2 lt40 axis class")
    _require_equal(lt40.get("status"), Q2_UNRESOLVED_CODE, "protocol_q2", "q2 lt40 operative status")
    thresholds = q2.get("thresholds")
    if not isinstance(thresholds, Mapping) or not math.isclose(float(thresholds.get("source_A", float("nan"))), THETA_A, abs_tol=1e-12) or not math.isclose(float(thresholds.get("source_C", float("nan"))), THETA_C, abs_tol=5e-6):
        raise Stage2Stop("protocol_q2_thresholds", "Q2 source-A/source-C thresholds differ from frozen constants.")
    q2_adjudication = q2.get("adjudication")
    if not isinstance(q2_adjudication, Mapping):
        raise Stage2Stop("protocol_q2", "q2.adjudication must be an object.")
    for key, expected in (("pass_seed_count", 6), ("seed_count", 8), ("source_A_abs_ratio_max", THETA_A), ("source_C_abs_ratio_max", THETA_C)):
        if isinstance(expected, float):
            if not math.isclose(float(q2_adjudication.get(key, float("nan"))), expected, abs_tol=5e-6):
                raise Stage2Stop("protocol_q2", f"q2.adjudication.{key} differs from {expected}.")
        else:
            _require_equal(q2_adjudication.get(key), expected, "protocol_q2", f"q2.adjudication.{key}")
    q3 = protocol.get("q3")
    if not isinstance(q3, Mapping):
        raise Stage2Stop("protocol_q3", "protocol.q3 must be an object.")
    if q3.get("kernel_name") != "simultaneous_clamped_z_intervention_under_frozen_base_attention_weights":
        raise Stage2Stop("protocol_q3_kernel", "Q3 clamped-z kernel declaration is missing or changed.")
    if q3.get("formula") != "z_star = z_base + P_base(final,subject) * (V_source(subject) - V_base(subject))":
        raise Stage2Stop("protocol_q3_formula", "Q3 z-star formula is missing or changed.")
    retained_gate = q3.get("retained_item_gate")
    if not isinstance(retained_gate, Mapping):
        raise Stage2Stop("protocol_q3", "q3.retained_item_gate must be an object.")
    for key, expected in (("requires_tested_set", True), ("zero_retained_item_execution_status", "COMPLETE"), ("zero_retained_item_axis_class", "SCIENTIFIC_UNRESOLVED"), ("zero_retained_item_cell_status", "SKIPPED_NO_RETAINED_ITEMS"), ("zero_retained_item_reason_code", "Q3_ZERO_RETAINED_ITEMS"), ("no_tested_set_status", "NOT_INSTANTIATED_NO_TESTED_SET"), ("base_global_q1_gate_does_not_suppress_q3", True)):
        _require_equal(retained_gate.get(key), expected, "protocol_q3", f"q3.retained_item_gate.{key}")
    decision = q3.get("decision_statistic")
    if not isinstance(decision, Mapping):
        raise Stage2Stop("protocol_q3", "q3.decision_statistic must be an object.")
    for key, expected in (("resamples", BOOTSTRAP_INTERVAL), ("test_id", Q3_TEST_ID), ("pass_seed_count", 6), ("seed_count", 8), ("requires_D_path_gt_zero", True), ("requires_ci_lower_gt_zero", True)):
        _require_equal(decision.get(key), expected, "protocol_q3", f"q3.decision_statistic.{key}")
    recovery = decision.get("recovery_fraction")
    if not isinstance(recovery, Mapping):
        raise Stage2Stop("protocol_q3", "q3 direct-recovery reporting contract is missing.")
    for key, expected in (("estimator", "E(delta_true_path) / E(delta_direct(S*))"), ("scope", "per_seed_descriptive_only"), ("across_seed_inferential_interval", "withdrawn"), ("used_for_adjudication", False), ("used_for_partial_identification", False), ("no_pass_fail_threshold", True)):
        _require_equal(recovery.get(key), expected, "protocol_q3", f"q3 recovery_fraction.{key}")
    slots = recovery.get("seed_slots")
    if not isinstance(slots, Mapping):
        raise Stage2Stop("protocol_q3", "q3 recovery seed_slots must be an object.")
    _require_equal(slots.get("count"), 8, "protocol_q3", "q3 recovery slot count")
    _require_equal(slots.get("seed_ids"), list(EXPECTED_SEEDS), "protocol_q3", "q3 recovery slot ids")
    aggregation = protocol.get("seed_aggregation")
    if not isinstance(aggregation, Mapping):
        raise Stage2Stop("protocol_seed_aggregation", "seed_aggregation must be an object.")
    _require_equal(aggregation.get("per_seed_classes"), ["PASS", "COMPLETED_FAIL", "SCIENTIFIC_UNRESOLVED", "EXECUTION_INCOMPLETE"], "protocol_seed_aggregation", "seed classes")
    _require_equal(aggregation.get("class_counts_must_sum_to"), 8, "protocol_seed_aggregation", "class count")
    incomplete_rule = aggregation.get("execution_incomplete_rule")
    if not isinstance(incomplete_rule, Mapping) or incomplete_rule.get("axis_status") != "BLOCKED_EXECUTION_INCOMPLETE" or incomplete_rule.get("scientific_verdict_emitted") is not False:
        raise Stage2Stop("protocol_seed_aggregation", "execution-incomplete aggregation rule is missing or changed.")
    partial = aggregation.get("partial_identification")
    if not isinstance(partial, Mapping):
        raise Stage2Stop("protocol_seed_aggregation", "partial-identification rule is missing.")
    for key, expected in (("lower_bound", "PASS count"), ("upper_bound", "PASS count + SCIENTIFIC_UNRESOLVED count"), ("positive", "lower_bound >= 6"), ("negative", "upper_bound < 6"), ("otherwise", "INCONCLUSIVE_UNRESOLVED_SEEDS")):
        _require_equal(partial.get(key), expected, "protocol_seed_aggregation", f"partial_identification.{key}")


def _validate_calibration(calibration: Mapping[str, Any], path: Path) -> dict[str, float]:
    if not str(calibration.get("schema", "")).startswith(CALIBRATION_SCHEMA_PREFIX):
        raise Stage2Stop("calibration_schema", f"Unexpected calibration schema in {path}.")
    if calibration.get("status") not in {"completed_with_lower_bound_runtime_projection", "completed"}:
        raise Stage2Stop("calibration_status", f"Calibration is not complete: {calibration.get('status')!r}.")
    theta = calibration.get("theta_spec")
    if not isinstance(theta, Mapping):
        raise Stage2Stop("calibration_theta", "Calibration theta_spec is missing.")
    if not math.isclose(float(theta.get("A", float("nan"))), THETA_A, abs_tol=1e-12) or not math.isclose(float(theta.get("C", float("nan"))), THETA_C, abs_tol=5e-6):
        raise Stage2Stop("calibration_theta", "Calibration theta constants differ from protocol_v1.")
    return {"A": float(theta["A"]), "C": float(theta["C"])}


def _candidate_records(candidate: Mapping[str, Any], path: Path) -> tuple[str, list[dict[str, int]], dict[str, Any]]:
    """Validate the exact immutable ``freeze_candidate.py`` output contract."""
    _require_equal(candidate.get("schema"), CANDIDATE_SCHEMA, "candidate_schema", "candidate.schema")
    status = str(candidate.get("status", ""))
    candidate_status = str(candidate.get("candidate_status", ""))
    allowed_statuses = {
        ("COMPLETE", "NONEMPTY"),
        ("COMPLETE_NO_CANDIDATES", "EMPTY_UNDER_FROZEN_RULE"),
    }
    if (status, candidate_status) not in allowed_statuses:
        raise Stage2Stop("candidate_status", f"Candidate status pair {status!r}/{candidate_status!r} is not an exact frozen completion state.")
    completed = status == "COMPLETE"
    if candidate.get("immutable") is not True or candidate.get("manual_override") is not False:
        raise Stage2Stop("candidate_immutability", "Candidate artifact must declare immutable=true and manual_override=false.")
    candidate_heads = candidate.get("candidate_heads")
    rank_rows = candidate.get("rank_order")
    candidate_c = candidate.get("candidate_C")
    if not isinstance(candidate_heads, list) or not isinstance(rank_rows, list) or not isinstance(candidate_c, list):
        raise Stage2Stop("candidate_schema", "candidate_heads, rank_order, and candidate_C must all be explicit JSON lists.")
    rows = rank_rows
    selection_hash = candidate.get("selection_source_a_sha256")
    if not _is_sha256(selection_hash):
        raise Stage2Stop("candidate_provenance", "Candidate artifact lacks an exact lowercase-hex selection_source_a_sha256.")
    protocol_hash = candidate.get("protocol_sha256")
    if not _is_sha256(protocol_hash):
        raise Stage2Stop("candidate_provenance", "Candidate artifact lacks an exact lowercase-hex protocol_sha256.")
    for key in ("true_sweep_sha256", "source_a_sweep_sha256"):
        value = candidate.get(key)
        if not _is_sha256(value):
            raise Stage2Stop("candidate_provenance", f"Candidate artifact lacks an exact lowercase-hex {key}.")
    if candidate.get("selection_sha256") != selection_hash:
        raise Stage2Stop("candidate_provenance", "selection_sha256 and selection_source_a_sha256 must be identical.")
    declared_hash = candidate.get("candidate_sha256")
    if not _is_sha256(declared_hash):
        raise Stage2Stop("candidate_hash", "Candidate artifact must contain a lowercase-hex candidate_sha256.")
    material = {key: value for key, value in candidate.items() if key != "candidate_sha256"}
    if declared_hash != _sha256_bytes(_json_bytes(material)):
        raise Stage2Stop("candidate_hash", "Candidate artifact hash does not match its immutable contents.")

    def parse_rows(values: Sequence[Any], label: str) -> list[dict[str, int]]:
        parsed: list[dict[str, int]] = []
        seen_local: set[tuple[int, int]] = set()
        for index, row in enumerate(values):
            if not isinstance(row, Mapping):
                raise Stage2Stop("candidate_row", f"{label} row {index} is not an object.")
            try:
                layer, head = int(row["layer"]), int(row["head"])
            except (KeyError, TypeError, ValueError) as exc:
                raise Stage2Stop("candidate_row", f"{label} row {index} lacks integer layer/head.") from exc
            if not (0 <= layer < LAYER_COUNT and 0 <= head < HEADS_PER_LAYER):
                raise Stage2Stop("candidate_row", f"{label} row {index} is outside the frozen 12x12 universe.")
            key = (layer, head)
            if key in seen_local:
                raise Stage2Stop("candidate_duplicate", f"{label} repeats L{layer}H{head}.")
            seen_local.add(key)
            if int(row.get("flat_id", -1)) != _flat_test_id(layer, head):
                raise Stage2Stop("candidate_flat_id", f"{label} row {index} flat_id disagrees with 12*layer+head+1.")
            parsed.append({"layer": layer, "head": head})
        return parsed

    out = parse_rows(rows, "rank_order")
    flat_heads = parse_rows(candidate_heads, "candidate_heads")
    flat_c = parse_rows(candidate_c, "candidate_C")
    member_set = {(row["layer"], row["head"]) for row in out}
    if {(row["layer"], row["head"]) for row in flat_heads} != member_set or {(row["layer"], row["head"]) for row in flat_c} != member_set:
        raise Stage2Stop("candidate_membership", "candidate_heads, rank_order, and candidate_C disagree on C membership.")
    expected_flat = sorted(flat_heads, key=lambda row: (row["layer"], row["head"]))
    if flat_heads != expected_flat or flat_c != expected_flat:
        raise Stage2Stop("candidate_order", "candidate_heads and candidate_C must be in canonical flat-head order.")
    if not completed and out:
        raise Stage2Stop("candidate_empty_status", "COMPLETE_NO_CANDIDATES cannot contain rows.")
    if completed and not out:
        raise Stage2Stop("candidate_empty_status", "Empty C must use EMPTY_UNDER_FROZEN_RULE/COMPLETE_NO_CANDIDATES.")
    return ("COMPLETE" if completed else "COMPLETE_NO_CANDIDATES"), out, {"selection_sha256": selection_hash, "protocol_sha256": protocol_hash, "true_sweep_sha256": candidate.get("true_sweep_sha256"), "source_a_sweep_sha256": candidate.get("source_a_sweep_sha256"), "candidate_sha256": declared_hash}


def _validate_selection_state_fingerprint(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping) or value.get("schema") != "exp05.model_state_fingerprint.v1" or value.get("scheme") != "lexicographic state_dict keys; key/dtype/shape JSON plus uncast contiguous tensor bytes; uint64 length framing" or value.get("encoding_detail") != "canonical JSON metadata; unsigned uint64 big-endian metadata and raw-byte lengths" or not _is_sha256(value.get("sha256")):
        raise Stage2Stop("selection_state_fingerprint", f"Selection {label} fingerprint schema/hash is incomplete.")
    entries = value.get("entries")
    if not isinstance(entries, list) or type(value.get("key_count")) is not int or value.get("key_count") != len(entries) or not entries:
        raise Stage2Stop("selection_state_fingerprint", f"Selection {label} fingerprint entry registry is incomplete.")
    keys: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("key"), str) or not isinstance(entry.get("dtype"), str) or not isinstance(entry.get("shape"), list) or any(type(size) is not int or size < 0 for size in entry.get("shape", [])) or type(entry.get("byte_length")) is not int or int(entry.get("byte_length")) < 0 or not _is_sha256(entry.get("bytes_sha256")):
            raise Stage2Stop("selection_state_fingerprint", f"Selection {label} contains a malformed state entry.")
        keys.append(str(entry["key"]))
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise Stage2Stop("selection_state_fingerprint", f"Selection {label} state keys are not unique lexicographic order.")
    return str(value["sha256"])


def _validate_selection_a7_provenance(selection: Mapping[str, Any], protocol: Mapping[str, Any], *, selection_file_hash: str, expected_commit: str) -> dict[str, Any]:
    """Consume the frozen A7 producer contract before Stage-2 model work."""

    if selection.get("schema") != "exp05.selection.v1" or selection.get("status") != "COMPLETE" or selection.get("dirty") is not False or selection.get("snapshot_provenance_status") != "READY":
        raise Stage2Stop("selection_provenance", "Selection must be a clean COMPLETE READY A7 artifact.")
    provenance = selection.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage2Stop("selection_provenance", "Selection dependency lacks provenance.")
    if provenance.get("dirty") is not False or provenance.get("git_status") != "clean" or provenance.get("require_clean_tree") is not True or provenance.get("clean_tree_scope") != "repository excluding exact declared runtime artifacts" or provenance.get("snapshot_provenance_status") != "READY":
        raise Stage2Stop("selection_provenance", "Selection clean-tree/snapshot provenance is not the frozen A7 contract.")
    completion_clean = provenance.get("completion_clean_tree_check")
    if not isinstance(completion_clean, Mapping) or completion_clean.get("dirty") is not False or completion_clean.get("git_status") != "clean" or completion_clean.get("clean_tree_scope") != "repository excluding exact declared runtime artifacts" or completion_clean.get("commit") != provenance.get("commit"):
        raise Stage2Stop("selection_provenance", "Selection lacks the exact final clean-tree/commit check.")
    commit = provenance.get("commit")
    if not _is_git_commit(commit) or commit != expected_commit or not _is_sha256(provenance.get("invocation_id")):
        raise Stage2Stop("selection_provenance", "Selection invocation/commit provenance is not bound to this Stage-2 commit.")
    correctness = selection.get("correctness")
    required_correctness = {
        "snapshot_provenance_status": "READY",
        "fresh_true_and_source_a_same_invocation": True,
        "same_in_memory_model": True,
        "model_state_before_after_both_sweeps_exact": True,
        "normalized_config_unchanged": True,
        "tokenizer_assets_unchanged": True,
        "pinned_gpt2_and_sae_revisions_unchanged": True,
        "sae_loaded": False,
        "sae_revision_fingerprint_present": True,
        "immutable_clean_base_cache_shared_and_unchanged": True,
    }
    if not isinstance(correctness, Mapping) or any(correctness.get(key) != value for key, value in required_correctness.items()):
        raise Stage2Stop("selection_provenance", "Selection correctness does not prove same-invocation A7 execution.")
    fingerprints = provenance.get("model_state_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise Stage2Stop("selection_state_fingerprint", "Selection model_state_fingerprints is missing.")
    before = fingerprints.get("before_sweeps")
    after_true = fingerprints.get("after_true_sweep")
    after_source_a = fingerprints.get("after_source_a_sweep")
    before_hash = _validate_selection_state_fingerprint(before, label="before_sweeps")
    true_hash = _validate_selection_state_fingerprint(after_true, label="after_true_sweep")
    source_a_hash = _validate_selection_state_fingerprint(after_source_a, label="after_source_a_sweep")
    if true_hash != before_hash or source_a_hash != before_hash or fingerprints.get("all_exact_match") is not True or not isinstance(after_true, Mapping) or after_true.get("exact_match_before") is not True or not isinstance(after_source_a, Mapping) or after_source_a.get("exact_match_before") is not True:
        raise Stage2Stop("selection_state_fingerprint", "Selection before/after model-state fingerprints are not identical.")
    config = provenance.get("normalized_model_config")
    config_checks = provenance.get("normalized_model_config_checks")
    if not isinstance(config, Mapping) or not _is_sha256(config.get("sha256")) or not isinstance(config_checks, Mapping) or config_checks.get("before_sweeps_sha256") != config.get("sha256") or config_checks.get("after_true_sweep_sha256") != config.get("sha256") or config_checks.get("after_source_a_sweep_sha256") != config.get("sha256") or config_checks.get("all_exact_match") is not True:
        raise Stage2Stop("selection_provenance", "Selection normalized model config is not stable across both sweeps.")
    tokenizer = provenance.get("tokenizer_assets")
    tokenizer_checks = provenance.get("tokenizer_asset_checks")
    if not isinstance(tokenizer, Mapping) or tokenizer.get("schema") != "exp05.tokenizer_assets.v1" or not _is_sha256(tokenizer.get("aggregate_sha256")) or not _is_sha256(tokenizer.get("vocab_sha256")) or not _is_sha256(tokenizer.get("backend_tokenizer_sha256")) or not _is_sha256(tokenizer.get("special_tokens_map_sha256")) or type(tokenizer.get("vocab_size")) is not int or not isinstance(tokenizer.get("local_files"), list) or not isinstance(tokenizer_checks, Mapping) or tokenizer_checks.get("before_sweeps_sha256") != tokenizer.get("aggregate_sha256") or tokenizer_checks.get("after_true_sweep_sha256") != tokenizer.get("aggregate_sha256") or tokenizer_checks.get("after_source_a_sweep_sha256") != tokenizer.get("aggregate_sha256") or tokenizer_checks.get("all_exact_match") is not True:
        raise Stage2Stop("selection_provenance", "Selection tokenizer assets are not stable across both sweeps.")
    environment = provenance.get("environment")
    if not isinstance(environment, Mapping) or not _is_sha256(environment.get("sha256")):
        raise Stage2Stop("selection_provenance", "Selection runtime environment fingerprint is missing.")
    env_material = {key: value for key, value in environment.items() if key != "sha256"}
    if _sha256_bytes(_json_bytes(env_material)) != environment.get("sha256"):
        raise Stage2Stop("selection_provenance", "Selection runtime environment hash does not match its fields.")
    clean_cache = provenance.get("immutable_clean_base_cache")
    cache_before = clean_cache.get("before_sweeps") if isinstance(clean_cache, Mapping) else None
    if not isinstance(cache_before, Mapping) or cache_before.get("schema") != "exp05.immutable_clean_base_cache.v1" or not _is_sha256(cache_before.get("sha256")) or not isinstance(cache_before.get("tensor_hashes"), Mapping) or set(cache_before.get("tensor_hashes")) != {"base_tokens", "base_lengths", "base_final_positions", "clean_base_d", "sign_alignment", "true_source_z_by_layer"} or not all(_is_sha256(value) for key, value in cache_before.get("tensor_hashes", {}).items() if key != "true_source_z_by_layer") or not isinstance(cache_before.get("tensor_hashes", {}).get("true_source_z_by_layer"), Mapping) or set(cache_before["tensor_hashes"]["true_source_z_by_layer"]) != {str(layer) for layer in range(LAYER_COUNT)} or not all(_is_sha256(value) for value in cache_before["tensor_hashes"]["true_source_z_by_layer"].values()) or _sha256_bytes(_json_bytes(cache_before["tensor_hashes"])) != cache_before.get("sha256") or not isinstance(clean_cache, Mapping) or clean_cache.get("after_true_sweep_sha256") != cache_before.get("sha256") or clean_cache.get("after_source_a_sweep_sha256") != cache_before.get("sha256") or clean_cache.get("all_exact_match") is not True:
        raise Stage2Stop("selection_provenance", "Selection immutable clean-base cache fingerprint is incomplete or changed.")
    model = provenance.get("model")
    revisions = protocol.get("model", {}).get("expected_local_snapshot_revisions", {}) if isinstance(protocol.get("model"), Mapping) else {}
    local_snapshots = model.get("local_snapshot_revisions") if isinstance(model, Mapping) else None
    if not isinstance(model, Mapping) or not isinstance(revisions, Mapping) or not isinstance(local_snapshots, Mapping) or set(local_snapshots) != {"gpt2", "sae"} or not _is_sha256(model.get("local_snapshot_revisions_sha256")) or _sha256_bytes(_json_bytes(dict(local_snapshots))) != model.get("local_snapshot_revisions_sha256") or model.get("sae_loaded") is not False or model.get("sae_revision_fingerprint_present") is not True or model.get("architecture") != {"n_layers": 12, "n_heads": 12, "d_model": 768, "d_vocab": 50_257} or model.get("device") != "cpu" or model.get("dtype") != "float32" or model.get("offline") is not True or model.get("snapshot_revision_check") != "exact local refs/main match":
        raise Stage2Stop("selection_provenance", "Selection GPT-2/SAE revision fingerprint is incomplete.")
    for name in ("gpt2", "sae"):
        entry = local_snapshots.get(name)
        expected_revision = revisions.get(name)
        if not isinstance(entry, Mapping) or not _is_git_commit(expected_revision) or entry.get("expected_revision") != expected_revision or entry.get("observed_revision") != expected_revision or entry.get("snapshot_present") is not True or entry.get("revision_check") != "exact local refs/main and snapshots/<revision> match" or not _is_sha256(entry.get("refs_main_sha256")) or not isinstance(entry.get("refs_main_path"), str) or not Path(entry["refs_main_path"]).is_absolute() or not isinstance(entry.get("snapshot_path"), str) or not Path(entry["snapshot_path"]).is_absolute():
            raise Stage2Stop("selection_provenance", f"Selection {name} revision metadata is incomplete.")
        ref_raw = Path(entry["refs_main_path"]).expanduser()
        if ref_raw.is_symlink() or not ref_raw.is_file():
            raise Stage2Stop("selection_provenance", f"Selection {name} refs/main is not a regular file.")
        ref_bytes = ref_raw.read_bytes()
        if _sha256_bytes(ref_bytes) != entry.get("refs_main_sha256") or ref_bytes.decode("utf-8").strip() != expected_revision or not Path(entry["snapshot_path"]).expanduser().is_dir():
            raise Stage2Stop("selection_provenance", f"Selection {name} local revision ref no longer matches protocol.")
    if model.get("snapshot_revision_expected") != local_snapshots["gpt2"].get("observed_revision") or model.get("snapshot_revision_observed") != local_snapshots["gpt2"].get("observed_revision") or model.get("snapshot_ref_path") != str(Path(local_snapshots["gpt2"]["refs_main_path"]).expanduser().resolve(strict=False)) or model.get("local_model_revision") != revisions.get("gpt2") or model.get("local_sae_revision") != revisions.get("sae") or provenance.get("local_model_revision") != revisions.get("gpt2") or provenance.get("local_sae_revision") != revisions.get("sae") or provenance.get("activation_dtype") != "float32" or provenance.get("runtime_environment_fingerprint") != environment.get("sha256"):
        raise Stage2Stop("selection_provenance", "Selection direct GPT-2/SAE provenance fields disagree with pinned revisions.")
    runtime = selection.get("runtime")
    expected_breakdown = {"clean": 1, "fresh_true_source_cache": 1, "fresh_source_A_cache": 1, "fresh_true_source_144_heads": 144, "fresh_source_A_144_heads": 144, "historical_stage1_crosscheck": 0, "total": 291}
    observed_breakdown = runtime.get("logical_forward_equivalents_breakdown") if isinstance(runtime, Mapping) else None
    if not isinstance(runtime, Mapping) or type(runtime.get("logical_forward_equivalents")) is not int or runtime.get("logical_forward_equivalents") != 291 or not isinstance(observed_breakdown, Mapping) or set(observed_breakdown) != set(expected_breakdown) or any(type(observed_breakdown.get(key)) is not int or observed_breakdown.get(key) != value for key, value in expected_breakdown.items()) or runtime.get("logical_forward_equivalents_definition") != "1 clean + 1 true cache + 1 source-A cache + 144 fresh true heads + 144 fresh source-A heads":
        raise Stage2Stop("selection_provenance", "Selection A7 logical forward-equivalent accounting is not exact 291.")
    pair_path = provenance.get("pair_output")
    if not isinstance(pair_path, str) or not Path(pair_path).is_absolute() or not _is_sha256(provenance.get("pair_output_sha256")) or Path(pair_path).is_symlink() or not Path(pair_path).is_file() or _sha256_file(Path(pair_path)) != provenance.get("pair_output_sha256"):
        raise Stage2Stop("selection_provenance", "Selection paired-sweep output path/hash is not bound to the final artifact.")
    return {"selection_provenance_sha256": _sha256_bytes(_json_bytes(dict(provenance))), "state_dict_sha256_before": before_hash, "normalized_model_config_sha256": config["sha256"], "tokenizer_assets_sha256": tokenizer["aggregate_sha256"], "clean_base_cache_sha256": cache_before["sha256"], "environment_sha256": environment["sha256"], "local_model_revision": revisions["gpt2"], "local_sae_revision": revisions["sae"], "pair_output_sha256": provenance["pair_output_sha256"]}


def _validate_selection_dependency(
    selection: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
    core: "CoreAdapter",
    *,
    selection_file_hash: str,
    expected_commit: str,
) -> dict[str, Any]:
    selection_summary = _validate_selection_a7_provenance(selection, protocol, selection_file_hash=selection_file_hash, expected_commit=expected_commit)
    if candidate.get("selection_source_a_sha256") != selection_file_hash or candidate.get("selection_sha256") != selection_file_hash:
        raise Stage2Stop("selection_hash", "Candidate selection hash does not match the supplied selection artifact.")
    true_sweep = selection.get("true_sweep")
    source_a_sweep = selection.get("source_a_sweep")
    if not isinstance(true_sweep, Mapping) or not isinstance(source_a_sweep, Mapping):
        raise Stage2Stop("selection_sweeps", "Selection dependency lacks canonical true/source-A sweeps.")
    if candidate.get("true_sweep_sha256") != _sha256_bytes(_json_bytes(true_sweep)) or candidate.get("source_a_sweep_sha256") != _sha256_bytes(_json_bytes(source_a_sweep)):
        raise Stage2Stop("selection_hash", "Candidate fresh sweep hashes do not match supplied selection sweeps.")
    for sweep_name, sweep, source_label in (("true", true_sweep, "true"), ("source_a", source_a_sweep, "source_a")):
        if sweep.get("schema") != "exp05.stage_sweep.v1" or sweep.get("status") != "COMPLETE" or sweep.get("dirty") is not False or sweep.get("seed") != SELECTION_SEED or sweep.get("source") != source_label or sweep.get("head_count") != EXPECTED_HEADS or sweep.get("directions") != list(DIRECTIONS) or not isinstance(sweep.get("heads"), list) or len(sweep.get("heads")) != EXPECTED_HEADS:
            raise Stage2Stop("selection_sweeps", f"Selection {sweep_name} sweep does not satisfy the frozen A7 producer schema.")
    rebuilt = core.reconstruct_candidate(true_sweep, source_a_sweep, protocol)
    expected_status = "COMPLETE" if rebuilt.get("status") == "NONEMPTY" else "COMPLETE_NO_CANDIDATES"
    for key in ("candidate_status", "candidate_heads", "rank_order", "nested_sets", "selection_evidence"):
        if candidate.get(key) != rebuilt.get(key):
            raise Stage2Stop("candidate_reconstruction", f"Candidate {key} differs from reconstruction over supplied selection sweeps.")
    if candidate.get("status") != expected_status or candidate.get("candidate_C") != rebuilt.get("candidate_heads"):
        raise Stage2Stop("candidate_reconstruction", "Candidate completion status or candidate_C differs from reconstruction.")
    selection_summary["candidate_manifest_sha256"] = _sha256_bytes(_json_bytes(dict(candidate)))
    selection_summary["candidate_count"] = len(candidate.get("candidate_C", [])) if isinstance(candidate.get("candidate_C"), list) else -1
    return selection_summary


class CoreAdapter:
    """Strict adapter for the frozen pure-statistics core.

    ``exp05_core`` owns the exact bootstrap/Holm/RNG conventions.  This class
    intentionally exposes only a narrow contract and raises on any missing or
    incompatible export.  It never silently falls back to a local estimator.
    """

    def __init__(self) -> None:
        try:
            import exp05_core as core
        except ImportError as exc:
            raise Stage2Stop("core_missing", "exp05_core.py is required for Stage-2; refusing local math fallback.") from exc
        self.core = core
        if getattr(core, "CORE_API_VERSION", None) != CORE_API_VERSION:
            raise Stage2Stop("core_api", f"Expected exp05_core.CORE_API_VERSION={CORE_API_VERSION!r}.")
        required = (
            "pair_clusters",
            "bootstrap_p_value",
            "percentile_ci",
            "holm_step_down",
            "linear_percentile",
            "rng_for",
            "q3_true_minus_a",
            "aggregate_eight_seed_statuses",
            "construct_candidate",
        )
        missing = [name for name in required if not callable(getattr(core, name, None))]
        if missing:
            raise Stage2Stop("core_api", f"exp05_core is missing required exports: {', '.join(missing)}")

    def pair_means(self, values: Sequence[float] | torch.Tensor) -> Any:
        # ``pair_clusters`` is the canonical conversion from the 2N directed
        # records to an N x 2 complete-minimal-pair matrix.  It refuses odd or
        # reordered inputs in the core; no direction-level fallback is allowed.
        tensor = values.detach().cpu() if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            if tensor.numel() % 2:
                raise Stage2Stop("pair_cluster_shape", "Directed effects must contain both directions of every pair.")
            tensor = tensor.reshape(-1, 2)
        return self.core.pair_clusters(tensor)

    def bootstrap_p(self, pair_values: Any, *, seed: int, test_id: int, resamples: int) -> float:
        return float(self.core.bootstrap_p_value(pair_values, experiment_seed=seed, test_id=test_id, resamples=resamples))

    def bootstrap_ci(self, pair_values: Any, *, seed: int, test_id: int, resamples: int) -> tuple[float, float]:
        result = self.core.percentile_ci(pair_values, seed=seed, test_id=test_id, resamples=resamples)
        if not isinstance(result, Sequence) or len(result) != 2:
            raise Stage2Stop("core_result", "exp05_core.bootstrap_mean_ci must return (low, high).")
        return float(result[0]), float(result[1])

    def holm(self, p_values: Sequence[float], alpha: float) -> list[bool]:
        result = self.core.holm_step_down(p_values, alpha=alpha)
        if not isinstance(result, Sequence) or len(result) != len(p_values):
            raise Stage2Stop("core_result", "exp05_core.holm_step_down must return one boolean per p-value.")
        return [bool(value) for value in result]

    def percentile(self, values: Sequence[float] | torch.Tensor, q: float) -> float:
        return float(self.core.linear_percentile(values, q=q))

    def rng(self, seed: int, test_id: int) -> Any:
        return self.core.rng_for(seed, test_id)

    def q3(self, true_values: Any, source_a_values: Any, *, seed: int) -> Any:
        return self.core.q3_true_minus_a(true_values, source_a_values, seed=seed, test_id=Q3_TEST_ID, resamples=BOOTSTRAP_INTERVAL)

    def aggregate(self, statuses: Sequence[str]) -> Mapping[str, Any]:
        result = self.core.aggregate_eight_seed_statuses(statuses)
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        if not isinstance(payload, Mapping):
            raise Stage2Stop("core_result", "exp05_core.aggregate_eight_seed_statuses returned a non-mapping result.")
        return payload

    def reconstruct_candidate(self, true_sweep: Mapping[str, Any], source_a_sweep: Mapping[str, Any], protocol: Mapping[str, Any]) -> Mapping[str, Any]:
        result = self.core.construct_candidate(true_sweep, source_a_sweep, protocol)
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        if not isinstance(payload, Mapping):
            raise Stage2Stop("core_result", "exp05_core.construct_candidate returned a non-mapping result.")
        return payload


def _compact_clean(model: Any, stimuli: Any, d: torch.Tensor, is_id: int, are_id: int) -> CleanPass:
    width = max(is_id, are_id) + 1
    logits = torch.zeros((stimuli.tokens.shape[0], stimuli.tokens.shape[1], width), dtype=torch.float32)
    rows = torch.arange(stimuli.tokens.shape[0])
    finals = stimuli.lengths - 1
    logits[rows, finals, are_id] = d
    return CleanPass(logits=logits, residuals={})


def _clean_d(model: Any, stimuli: Any, is_id: int, are_id: int, *, started: float | None = None, cap: float | None = None, label: str = "clean") -> torch.Tensor:
    if started is not None:
        _check_runtime(started, cap, f"before {label}")
    result = clean_readout_microbatched(model, stimuli.tokens, stimuli.lengths, is_id, are_id)
    if started is not None:
        _check_runtime(started, cap, f"after {label}")
    return result


def _gate_and_indices(model: Any, stimuli: Any, is_id: int, are_id: int) -> tuple[dict[str, Any], list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_d = _clean_d(model, stimuli, is_id, are_id)
    clean = _compact_clean(model, stimuli, clean_d, is_id, are_id)
    gate, retained, _ = gate_a(stimuli, clean, is_id, are_id)
    if not gate.get("passed"):
        return gate, retained, *directed_indices(REQUESTED_PAIRS, retained), clean_d
    base_indices, source_indices, signs = directed_indices(REQUESTED_PAIRS, retained)
    return gate, retained, base_indices, source_indices, signs, clean_d


def _pair_records_from_directed(values: torch.Tensor, pair_ids: Sequence[int], directions: Sequence[str]) -> list[dict[str, Any]]:
    flat = values.detach().float().cpu().tolist()
    if len(flat) != len(pair_ids) or len(flat) != len(directions):
        raise Stage2Stop("pair_record_length", "Directed value count does not match canonical pair/direction arrays.")
    return [{"pair_id": int(pair), "direction": str(direction), "effect": float(value)} for pair, direction, value in zip(pair_ids, directions, flat)]


def _paired_effect_row(values: torch.Tensor, signs: torch.Tensor, pair_ids: Sequence[int], directions: Sequence[str]) -> dict[str, Any]:
    aligned = values.detach().float().cpu() * signs.detach().float().cpu()
    pair_values = aligned.reshape(-1, 2)
    mean_effect = float(aligned.mean())
    return {
        "directed_raw": [float(value) for value in values.detach().float().cpu().tolist()],
        "directed_sign_aligned": [float(value) for value in aligned.tolist()],
        "pair_means": [float(value) for value in pair_values.mean(dim=1).tolist()],
        "pair_ids": [int(value) for value in pair_ids[::2]],
        "directions": list(directions),
        "E": mean_effect,
        "E_delta_d": mean_effect,
        "pair_sign_consistency": float((pair_values > 0).all(dim=1).float().mean()),
        "directed_sign_consistency": float((aligned > 0).float().mean()),
        "pair_records": _pair_records_from_directed(aligned, pair_ids, directions),
    }


def _mean_float64(values: Sequence[Any], label: str) -> float:
    numeric = [float(value) for value in values]
    if not numeric:
        raise Stage2Stop("empty_evaluation_units", f"{label} has no frozen evaluation units.")
    if not all(math.isfinite(value) for value in numeric):
        raise Stage2Stop("non_finite_effect", f"{label} contains a non-finite required effect.")
    try:
        result = math.fsum(numeric) / len(numeric)
    except (OverflowError, ValueError) as exc:
        raise Stage2Stop("non_finite_effect", f"{label} cannot be accumulated in float64: {exc}") from exc
    return float(result)


def _q1_e_all_guard(e_all: Mapping[str, Any]) -> dict[str, Any]:
    values = e_all.get("pair_means")
    if not isinstance(values, list) or not values:
        return {"estimable": False, "E_all_float64": None, "A_all_float64": None, "g": None, "gamma": math.sqrt(2.0**-52), "reason_code": "Q1_SCIENTIFIC_UNRESOLVED_EALL_ZERO_SCALE"}
    numeric = [float(value) for value in values]
    gamma = math.sqrt(2.0**-52)
    if not all(math.isfinite(value) for value in numeric):
        return {"estimable": False, "E_all_float64": None, "A_all_float64": None, "g": None, "gamma": gamma, "reason_code": "Q1_SCIENTIFIC_UNRESOLVED_EALL_NONFINITE"}
    e_mean = math.fsum(numeric) / len(numeric)
    a_mean = math.fsum(abs(value) for value in numeric) / len(numeric)
    if not math.isfinite(e_mean) or not math.isfinite(a_mean):
        return {"estimable": False, "E_all_float64": None, "A_all_float64": None, "g": None, "gamma": gamma, "reason_code": "Q1_SCIENTIFIC_UNRESOLVED_EALL_NONFINITE"}
    if a_mean == 0.0:
        return {"estimable": False, "E_all_float64": e_mean, "A_all_float64": a_mean, "g": None, "gamma": gamma, "reason_code": "Q1_SCIENTIFIC_UNRESOLVED_EALL_ZERO_SCALE"}
    g = abs(e_mean) / a_mean
    if g <= gamma:
        return {"estimable": False, "E_all_float64": e_mean, "A_all_float64": a_mean, "g": g, "gamma": gamma, "reason_code": "Q1_SCIENTIFIC_UNRESOLVED_EALL_NEAR_ZERO"}
    return {"estimable": True, "E_all_float64": e_mean, "A_all_float64": a_mean, "g": g, "gamma": gamma, "reason_code": None}


def _nullable_nonfinite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _nullable_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nullable_nonfinite(item) for item in value]
    return value


def _head_effect_row(patched_d: torch.Tensor, clean_d: torch.Tensor, signs: torch.Tensor, pair_ids: Sequence[int], directions: Sequence[str], layer: int, head: int) -> dict[str, Any]:
    return {"layer": int(layer), "head": int(head), **_paired_effect_row(patched_d - clean_d, signs, pair_ids, directions)}


def _source_head_sweep(model: Any, base_tokens: torch.Tensor, base_lengths: torch.Tensor, base_final: torch.Tensor, source_z: Mapping[int, torch.Tensor], source_final: torch.Tensor, clean_base_d: torch.Tensor, signs: torch.Tensor, pair_ids: Sequence[int], directions: Sequence[str], source_label: str, *, started: float, cap: float | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runner = AttentionPatchRunner(model)
    local = torch.arange(base_tokens.shape[0])
    is_id = require_one_token(model.tokenizer, " is")
    are_id = require_one_token(model.tokenizer, " are")
    rows: list[dict[str, Any]] = []
    for item in _canonical_heads():
        _check_runtime(started, cap, f"before {source_label} {_head_key(item['layer'], item['head'])}")
        layer, head = item["layer"], item["head"]
        replacement = _source_values(source_z[layer], local, source_final, head)
        patched = runner.run_one(
            hook_kind=HOOK_Z,
            layer=layer,
            head=head,
            base_tokens=base_tokens,
            base_positions=base_final,
            replacement=replacement,
            label=f"stage2_{source_label}_z_final",
            lengths=base_lengths,
            is_id=is_id,
            are_id=are_id,
        )
        rows.append(_head_effect_row(patched, clean_base_d, signs, pair_ids, directions, layer, head))
        _check_runtime(started, cap, f"after {source_label} {_head_key(layer, head)}")
    return rows, runner.records


def _run_all_z(model: Any, base_tokens: torch.Tensor, base_lengths: torch.Tensor, base_final: torch.Tensor, replacements_by_layer: Mapping[int, torch.Tensor], clean_base_d: torch.Tensor, signs: torch.Tensor, label: str, *, started: float | None = None, cap: float | None = None) -> torch.Tensor:
    if started is not None:
        _check_runtime(started, cap, f"before {label}")
    runner = AttentionPatchRunner(model)
    patched = runner.run_all_z(
        base_tokens=base_tokens,
        base_positions=base_final,
        replacements_by_layer=dict(replacements_by_layer),
        label=label,
        lengths=base_lengths,
        is_id=require_one_token(model.tokenizer, " is"),
        are_id=require_one_token(model.tokenizer, " are"),
    )
    del clean_base_d, signs
    if started is not None:
        _check_runtime(started, cap, f"after {label}")
    return patched


def _run_selected_z(model: Any, base_tokens: torch.Tensor, base_lengths: torch.Tensor, base_final: torch.Tensor, replacements: Mapping[tuple[int, int], torch.Tensor], clean_base_d: torch.Tensor, signs: torch.Tensor, selected: Sequence[Mapping[str, int]], label: str, *, started: float, cap: float | None) -> torch.Tensor:
    """Jointly patch selected heads only at final ``hook_z``."""
    values: list[torch.Tensor] = []
    is_id = require_one_token(model.tokenizer, " is")
    are_id = require_one_token(model.tokenizer, " are")
    for start in range(0, base_tokens.shape[0], PATCH_BATCH_SIZE):
        _check_runtime(started, cap, f"before {label} microbatch {start}")
        stop = min(start + PATCH_BATCH_SIZE, base_tokens.shape[0])
        chunk_positions = base_final[start:stop]
        hooks: list[tuple[str, Any]] = []
        for row in selected:
            layer, head = int(row["layer"]), int(row["head"])
            replacement = replacements[(layer, head)][start:stop]

            def hook(activation: torch.Tensor, hook: Any, *, replacement: torch.Tensor = replacement, head: int = head) -> torch.Tensor:
                del hook
                return _patch_hook(
                    activation,
                    base_positions=chunk_positions,
                    replacement=replacement,
                    head=head,
                    expected_heads=int(model.cfg.n_heads),
                    expected_d_head=int(model.cfg.d_head),
                )

            hooks.append((f"blocks.{layer}.attn.{HOOK_Z}", hook))
        with torch.no_grad():
            logits = model.run_with_hooks(base_tokens[start:stop], fwd_hooks=hooks, return_type="logits")
        # Local import keeps this module aligned with the shipped readout.
        from pilot import logit_difference

        values.append(logit_difference(logits, base_lengths[start:stop], is_id, are_id).detach().float().cpu())
        del logits
        _check_runtime(started, cap, f"after {label} microbatch {start}")
    del clean_base_d, signs
    return torch.cat(values)


def _z_replacements(z_cache: Mapping[int, torch.Tensor], source_cache: Mapping[int, torch.Tensor], local_indices: torch.Tensor, source_positions: torch.Tensor, selected: Sequence[Mapping[str, int]]) -> dict[tuple[int, int], torch.Tensor]:
    return {
        (int(row["layer"]), int(row["head"])): _source_values(source_cache[int(row["layer"])], local_indices, source_positions, int(row["head"]))
        for row in selected
    }


def _subset_rows(rows: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, int]]) -> list[dict[str, Any]]:
    wanted = {(int(row["layer"]), int(row["head"])) for row in selected}
    return [dict(row) for row in rows if (int(row["layer"]), int(row["head"])) in wanted]


def _source_a_edge(rows: Sequence[Mapping[str, Any]], core: CoreAdapter) -> float:
    return core.percentile([abs(float(row["E"])) for row in rows], q=99.0)


def _source_gate(model: Any, stimuli: Any, is_id: int, are_id: int, *, started: float | None = None, cap: float | None = None) -> tuple[dict[str, Any], list[int]]:
    clean_d = _clean_d(model, stimuli, is_id, are_id, started=started, cap=cap, label="source-C Gate-A clean")
    clean = _compact_clean(model, stimuli, clean_d, is_id, are_id)
    gate, retained, _ = gate_a(stimuli, clean, is_id, are_id)
    return gate, retained


def _q1_head_distinguishability(true_rows: Sequence[Mapping[str, Any]], source_a_rows: Sequence[Mapping[str, Any]], *, seed: int, core: CoreAdapter, edge: float, started: float, cap: float | None) -> tuple[list[dict[str, Any]], list[bool]]:
    if len(true_rows) != EXPECTED_HEADS or len(source_a_rows) != EXPECTED_HEADS:
        raise Stage2Stop("sweep_rows", "Stage-2 head sweeps must each contain all 144 rows.")
    p_values: list[float] = []
    records: list[dict[str, Any]] = []
    for index, (true_row, a_row) in enumerate(zip(true_rows, source_a_rows)):
        layer, head = int(true_row["layer"]), int(true_row["head"])
        if (layer, head) != (int(a_row["layer"]), int(a_row["head"])):
            raise Stage2Stop("sweep_order", "True/source-A head orders differ.")
        _check_runtime(started, cap, f"before Q1 bootstrap {_head_key(layer, head)}")
        pair_means = core.pair_means(true_row["directed_sign_aligned"])
        p = core.bootstrap_p(pair_means, seed=seed, test_id=_flat_test_id(layer, head), resamples=BOOTSTRAP_HEAD)
        _check_runtime(started, cap, f"after Q1 bootstrap {_head_key(layer, head)}")
        p_values.append(p)
        records.append({"layer": layer, "head": head, "E": float(true_row["E"]), "source_a_E": float(a_row["E"]), "source_a_abs_edge": float(edge), "pair_sign_consistency": float(true_row["pair_sign_consistency"]), "raw_p": p, "test_id": _flat_test_id(layer, head), "rng_seed": seed * 1000 + _flat_test_id(layer, head), "bootstrap_resamples": BOOTSTRAP_HEAD})
    rejects = core.holm(p_values, alpha=HOLM_ALPHA)
    for record, reject in zip(records, rejects):
        record["holm_reject"] = bool(reject)
        record["distinguishable"] = bool(reject and abs(float(record["E"])) > edge)
    return records, [bool(record["distinguishable"]) for record in records]


def _joint_e(rows: Mapping[tuple[int, int], torch.Tensor], selected: Sequence[Mapping[str, int]], signs: torch.Tensor) -> torch.Tensor:
    values = [rows[(int(row["layer"]), int(row["head"]))] for row in selected]
    if not values:
        return torch.zeros(signs.numel(), dtype=torch.float32)
    return torch.stack(values, dim=0).sum(dim=0) * 0.0


def _q1_decision(candidate: Sequence[Mapping[str, int]], per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidate:
        return {"status": "COMPLETE_NO_CANDIDATES", "outer_code": "Q1_COMPLETE_NO_CANDIDATES", "subtype": "EMPTY_UNDER_FROZEN_RULE", "label": None, "tested_set": [], "qualifying_n": None, "curve": []}
    max_n = min(Q1_MAX, len(candidate))
    curve: list[dict[str, Any]] = []
    for n in range(1, max_n + 1):
        n_ok = 0
        for seed in per_seed:
            curve_row = seed.get("q1_nested", {}).get(str(n), {})
            if curve_row.get("seed_joint_pass") is True:
                n_ok += 1
        curve.append({"n": n, "seed_count_joint_pass": n_ok, "aggregate_pass": n_ok >= 6})
    qualifying = next((int(row["n"]) for row in curve if row["aggregate_pass"]), None)
    if qualifying is None:
        tested = list(candidate[:max_n])
        return {"status": "COMPLETE", "outer_code": "Q1_COMPLETE", "subtype": "Q1_TESTED_SET_SELECTED_FALLBACK_MIN8C", "selection_status": "Q1_TESTED_SET_SELECTED_FALLBACK_MIN8C", "label": None, "tested_set": tested, "qualifying_n": None, "curve": curve}
    return {"status": "COMPLETE", "outer_code": "Q1_COMPLETE", "subtype": "Q1_TESTED_SET_SELECTED_MIN_N_PASS_GE6", "selection_status": "Q1_TESTED_SET_SELECTED_MIN_N_PASS_GE6", "label": None, "tested_set": list(candidate[:qualifying]), "qualifying_n": qualifying, "curve": curve}


def _summarize_status(per_seed: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    complete = [row for row in per_seed if row.get("status") == "COMPLETE"]
    passed = [row for row in complete if row.get(key) is True]
    return {"complete_seed_count": len(complete), "positive_seed_count": len(passed), "aggregate_positive": len(passed) >= 6}


def _sum_forward_equivalents(value: Any) -> int:
    """Sum the actual recorded forward events, including nested arm maps."""

    if isinstance(value, Mapping):
        return sum(_sum_forward_equivalents(item) for item in value.values())
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return 0


def _cache_activations(model: Any, tokens: torch.Tensor, *, need_v: bool = False, need_pattern: bool = False, started: float, cap: float | None, label: str) -> dict[str, Any]:
    """Cache immutable clean activations in deterministic microbatches.

    ``hook_z`` is always retained.  Q3 additionally requests ``hook_v`` and
    ``hook_pattern``; neither is ever read from an intervention forward.  The
    cache is detached to CPU before the next microbatch so a runtime-cap stop
    cannot leave a live model hook or a partial verdict.
    """
    names: set[str] = set()
    for layer in range(LAYER_COUNT):
        names.add(f"blocks.{layer}.attn.hook_z")
        if need_v:
            names.add(f"blocks.{layer}.attn.hook_v")
        if need_pattern:
            names.add(f"blocks.{layer}.attn.hook_pattern")
    z_parts: dict[int, list[torch.Tensor]] = {layer: [] for layer in range(LAYER_COUNT)}
    v_parts: dict[int, list[torch.Tensor]] | None = {layer: [] for layer in range(LAYER_COUNT)} if need_v else None
    p_parts: dict[int, list[torch.Tensor]] | None = {layer: [] for layer in range(LAYER_COUNT)} if need_pattern else None
    for start in range(0, tokens.shape[0], PATCH_BATCH_SIZE):
        _check_runtime(started, cap, f"before {label} cache microbatch {start}")
        stop = min(start + PATCH_BATCH_SIZE, tokens.shape[0])
        with torch.no_grad():
            result, cache = model.run_with_cache(tokens[start:stop], names_filter=lambda name: name in names, return_type=None)
        if result is not None:
            raise Stage2Stop("cache_return_type", "Activation cache requested return_type=None but received logits.")
        for layer in range(LAYER_COUNT):
            z_parts[layer].append(cache[f"blocks.{layer}.attn.hook_z"].detach().float().cpu().clone())
            if need_v and v_parts is not None:
                v_parts[layer].append(cache[f"blocks.{layer}.attn.hook_v"].detach().float().cpu().clone())
            if need_pattern and p_parts is not None:
                p_parts[layer].append(cache[f"blocks.{layer}.attn.hook_pattern"].detach().float().cpu().clone())
        del cache
        _check_runtime(started, cap, f"after {label} cache microbatch {start}")
    out: dict[str, Any] = {"z": {layer: torch.cat(z_parts[layer]) for layer in range(LAYER_COUNT)}}
    if need_v and v_parts is not None:
        out["v"] = {layer: torch.cat(v_parts[layer]) for layer in range(LAYER_COUNT)}
    if need_pattern and p_parts is not None:
        out["pattern"] = {layer: torch.cat(p_parts[layer]) for layer in range(LAYER_COUNT)}
    return out


def _head_replacements(source_cache: Mapping[int, torch.Tensor], source_positions: torch.Tensor, selected: Sequence[Mapping[str, int]]) -> dict[tuple[int, int], torch.Tensor]:
    local = torch.arange(source_positions.numel())
    return {
        (int(row["layer"]), int(row["head"])): _source_values(source_cache[int(row["layer"])], local, source_positions, int(row["head"]))
        for row in selected
    }


def _direct_effect(model: Any, *, base_tokens: torch.Tensor, base_lengths: torch.Tensor, base_final: torch.Tensor, source_cache: Mapping[int, torch.Tensor], source_positions: torch.Tensor, clean_base_d: torch.Tensor, signs: torch.Tensor, selected: Sequence[Mapping[str, int]], label: str, pair_ids: Sequence[int], directions: Sequence[str], started: float, cap: float | None) -> dict[str, Any]:
    replacements = _head_replacements(source_cache, source_positions, selected)
    patched = _run_selected_z(model, base_tokens, base_lengths, base_final, replacements, clean_base_d, signs, selected, label, started=started, cap=cap)
    return _paired_effect_row(patched - clean_base_d, signs, pair_ids, directions)


def _all_head_effect(model: Any, *, base_tokens: torch.Tensor, base_lengths: torch.Tensor, base_final: torch.Tensor, source_cache: Mapping[int, torch.Tensor], source_positions: torch.Tensor, clean_base_d: torch.Tensor, signs: torch.Tensor, label: str, pair_ids: Sequence[int], directions: Sequence[str], started: float | None = None, cap: float | None = None) -> dict[str, Any]:
    local = torch.arange(source_positions.numel())
    replacements = {layer: source_cache[layer][local, source_positions, :, :].clone() for layer in range(LAYER_COUNT)}
    patched = _run_all_z(model, base_tokens, base_lengths, base_final, replacements, clean_base_d, signs, label, started=started, cap=cap)
    return _paired_effect_row(patched - clean_base_d, signs, pair_ids, directions)


def _q3_z_star(*, base_bundle: Mapping[str, Any], source_bundle: Mapping[str, Any], base_final: torch.Tensor, base_subject: torch.Tensor, source_subject: torch.Tensor, selected: Sequence[Mapping[str, int]]) -> dict[tuple[int, int], torch.Tensor]:
    rows = torch.arange(base_final.numel())
    out: dict[tuple[int, int], torch.Tensor] = {}
    for row in selected:
        layer, head = int(row["layer"]), int(row["head"])
        base_z = base_bundle["z"][layer][rows, base_final, head, :].detach().float().cpu()
        pattern = base_bundle["pattern"][layer][rows, head, base_final, base_subject].detach().float().cpu()
        base_v = base_bundle["v"][layer][rows, base_subject, head, :].detach().float().cpu()
        source_v = source_bundle["v"][layer][rows, source_subject, head, :].detach().float().cpu()
        out[(layer, head)] = base_z + pattern.unsqueeze(-1) * (source_v - base_v)
    return out


def _q3_effect(model: Any, *, base_tokens: torch.Tensor, base_lengths: torch.Tensor, base_final: torch.Tensor, clean_base_d: torch.Tensor, signs: torch.Tensor, base_bundle: Mapping[str, Any], source_bundle: Mapping[str, Any], base_subject: torch.Tensor, source_subject: torch.Tensor, selected: Sequence[Mapping[str, int]], label: str, pair_ids: Sequence[int], directions: Sequence[str], started: float, cap: float | None) -> dict[str, Any]:
    replacements = _q3_z_star(base_bundle=base_bundle, source_bundle=source_bundle, base_final=base_final, base_subject=base_subject, source_subject=source_subject, selected=selected)
    patched = _run_selected_z(model, base_tokens, base_lengths, base_final, replacements, clean_base_d, signs, selected, label, started=started, cap=cap)
    return _paired_effect_row(patched - clean_base_d, signs, pair_ids, directions)


def _seed_first_phase(model: Any, *, seed: int, candidate: Sequence[Mapping[str, int]], started: float, cap: float | None, core: CoreAdapter) -> dict[str, Any]:
    """Run true/source-A sweeps plus all Q1 quantities for one seed."""
    is_id = require_one_token(model.tokenizer, " is")
    are_id = require_one_token(model.tokenizer, " are")
    base = build_stimuli(model.tokenizer, REQUESTED_PAIRS, seed)
    clean_d_all = _clean_d(model, base, is_id, are_id, started=started, cap=cap, label=f"seed {seed} Gate-A clean")
    clean = _compact_clean(model, base, clean_d_all, is_id, are_id)
    gate, retained, _ = gate_a(base, clean, is_id, are_id)
    base_indices, source_indices, signs = directed_indices(REQUESTED_PAIRS, retained)
    pair_ids = [int(pair) for pair in retained for _ in (0, 1)]
    directions = list(DIRECTIONS) * len(retained)
    if not gate.get("passed"):
        return {
            "seed": seed,
            "status": "FIRST_COMPLETE",
            "execution_status": "COMPLETE",
            "gate_A": gate,
            "retained_pairs": [int(value) for value in retained],
            "retained_directed_edits": {"pair_ids": pair_ids, "directions": directions, "base_indices": [int(value) for value in base_indices.tolist()], "source_indices": [int(value) for value in source_indices.tolist()], "signs": [float(value) for value in signs.tolist()]},
            "q1_cell_status": "SKIPPED_BY_PREREGISTERED_SCIENTIFIC_GATE",
            "q1_scientific_unresolved_reason": "Q1_BASE_GLOBAL_GATE_A_FAIL",
            "q1_nested": {},
            "true_heads": [],
            "source_a_heads": [],
            "logical_forward_equivalents": {"gate_a_clean": 1, "conditional_q1_skip": 0},
            "input_hashes": {"base_tokens": _tensor_hash(base.tokens)},
        }
    base_final = positions_for_kind(base, base_indices, "final").squeeze(1)
    source_final = positions_for_kind(base, source_indices, "final").squeeze(1)
    base_subject = positions_for_kind(base, base_indices, "subject").squeeze(1)
    source_subject = positions_for_kind(base, source_indices, "subject").squeeze(1)
    if not torch.equal(base_final, source_final) or not torch.equal(base.lengths[base_indices], base.lengths[source_indices]):
        raise Stage2Stop("true_positions", f"Seed {seed} source/base final positions or lengths differ under single-flip construction.")
    base_tokens = base.tokens[base_indices]
    clean_base_d = clean_d_all[base_indices].detach().float().cpu()
    true_tokens = base.tokens[source_indices]
    source_a = make_source_a(model.tokenizer, base, seed)
    source_a_tokens = source_a.tokens[base_indices]
    source_a_final = positions_for_kind(source_a, base_indices, "final").squeeze(1)
    if not torch.equal(base.lengths[base_indices], source_a.lengths[base_indices]):
        raise Stage2Stop("source_a_lengths", f"Seed {seed} Source-A changed retained sequence lengths.")

    true_bundle = _cache_activations(model, true_tokens, started=started, cap=cap, label=f"seed {seed} true")
    source_a_bundle = _cache_activations(model, source_a_tokens, started=started, cap=cap, label=f"seed {seed} source-A")
    base_bundle = _cache_activations(model, base_tokens, started=started, cap=cap, label=f"seed {seed} base")
    for layer, values in base_bundle["z"].items():
        if values.ndim != 4 or values.shape[2] != int(model.cfg.n_heads) or values.shape[3] != int(model.cfg.d_head):
            raise Stage2Stop("hook_z_layout", f"Seed {seed} layer {layer} has unexpected hook_z shape {tuple(values.shape)}.")
    true_rows, true_records = _source_head_sweep(model, base_tokens, base.lengths[base_indices], base_final, true_bundle["z"], source_final, clean_base_d, signs, pair_ids, directions, "true", started=started, cap=cap)
    source_a_rows, source_a_records = _source_head_sweep(model, base_tokens, base.lengths[base_indices], base_final, source_a_bundle["z"], source_a_final, clean_base_d, signs, pair_ids, directions, "source_a", started=started, cap=cap)
    edge = _source_a_edge(source_a_rows, core)
    distinguish, distinguish_flags = _q1_head_distinguishability(true_rows, source_a_rows, seed=seed, core=core, edge=edge, started=started, cap=cap)
    rank_order = sorted(true_rows, key=lambda row: (-float(row["E"]), int(row["layer"]), int(row["head"])))
    # ``C`` is fixed from the selection artifact; Q1 below only uses the
    # candidate rows passed by the caller.  Persist all 144 flags now.
    e_all = _all_head_effect(model, base_tokens=base_tokens, base_lengths=base.lengths[base_indices], base_final=base_final, source_cache=true_bundle["z"], source_positions=source_final, clean_base_d=clean_base_d, signs=signs, label="stage2_true_E_all_z_final", pair_ids=pair_ids, directions=directions, started=started, cap=cap)
    e_all_guard = _q1_e_all_guard(e_all)
    if e_all_guard.get("reason_code") == "Q1_SCIENTIFIC_UNRESOLVED_EALL_NONFINITE":
        e_all = _nullable_nonfinite(e_all)
    q1_nested: dict[str, Any] = {}
    by_head = {(int(row["layer"]), int(row["head"])): row for row in distinguish}
    for n in range(1, min(Q1_MAX, len(candidate)) + 1):
        subset = list(candidate[:n])
        direct = _direct_effect(model, base_tokens=base_tokens, base_lengths=base.lengths[base_indices], base_final=base_final, source_cache=true_bundle["z"], source_positions=source_final, clean_base_d=clean_base_d, signs=signs, selected=subset, label=f"stage2_true_nested_S{n}", pair_ids=pair_ids, directions=directions, started=started, cap=cap)
        numerator = _mean_float64(direct["pair_means"], f"seed {seed} E(S_{n})")
        denominator = e_all_guard.get("E_all_float64")
        recovery = float(numerator / float(denominator)) if e_all_guard.get("estimable") is True else None
        all_distinguishable = all(bool(by_head[(int(row["layer"]), int(row["head"]))]["distinguishable"]) for row in subset)
        q1_nested[str(n)] = {"set": subset, "direct": direct, "numerator_float64": numerator, "recovery_fraction": recovery, "denominator_guard": dict(e_all_guard), "all_members_distinguishable": all_distinguishable, "seed_joint_pass": bool(recovery is not None and recovery >= 0.50 and all_distinguishable)}
    return {
        "seed": seed,
        "status": "FIRST_COMPLETE",
        "execution_status": "COMPLETE",
        "gate_A": gate,
        "retained_pairs": [int(value) for value in retained],
        "retained_directed_edits": {"pair_ids": pair_ids, "directions": directions, "base_indices": [int(value) for value in base_indices.tolist()], "source_indices": [int(value) for value in source_indices.tolist()], "signs": [float(value) for value in signs.tolist()], "base_final": [int(value) for value in base_final.tolist()], "source_final": [int(value) for value in source_final.tolist()]},
        "true_heads": true_rows,
        "source_a_heads": source_a_rows,
        "true_sweep": {"schema": "exp05.stage_sweep.v1", "status": "COMPLETE", "dirty": False, "seed": seed, "source": "true", "head_count": EXPECTED_HEADS, "directions": list(DIRECTIONS), "heads": true_rows},
        "source_a_sweep": {"schema": "exp05.stage_sweep.v1", "status": "COMPLETE", "dirty": False, "seed": seed, "source": "source_a", "head_count": EXPECTED_HEADS, "directions": list(DIRECTIONS), "heads": source_a_rows},
        "distinguishability": distinguish,
        "source_a_edge": edge,
        "E_all": e_all,
        "E_all_denominator_guard": e_all_guard,
        "q1_cell_status": "COMPLETE",
        "q1_scientific_unresolved_reason": e_all_guard.get("reason_code"),
        "q1_nested": q1_nested,
        "runtime_records": {"true": true_records, "source_a": source_a_records},
        "logical_forward_equivalents": {
            "gate_a_clean": 1,
            "true_cache": 1,
            "source_a_cache": 1,
            "base_cache": 1,
            "true_singleton_sweep": EXPECTED_HEADS,
            "source_a_singleton_sweep": EXPECTED_HEADS,
            "E_all_joint": 1,
            "q1_nested_sets": {str(n): 1 for n in range(1, min(Q1_MAX, len(candidate)) + 1)},
        },
        "input_hashes": {"base_tokens": _tensor_hash(base_tokens), "true_tokens": _tensor_hash(true_tokens), "source_a_tokens": _tensor_hash(source_a_tokens)},
    }


def _slice_cache(cache: Mapping[int, torch.Tensor], indices: torch.Tensor) -> dict[int, torch.Tensor]:
    return {int(layer): values[indices].contiguous() for layer, values in cache.items()}


def _seed_second_phase(model: Any, *, seed: int, first: Mapping[str, Any], tested_set: Sequence[Mapping[str, int]], started: float, cap: float | None, core: CoreAdapter, theta: Mapping[str, float]) -> dict[str, Any]:
    """Rebuild one seed after Q1 fixes S*, then run Q2 and Q3 only."""
    _check_runtime(started, cap, f"before second-phase seed {seed}")
    is_id = require_one_token(model.tokenizer, " is")
    are_id = require_one_token(model.tokenizer, " are")
    base = build_stimuli(model.tokenizer, REQUESTED_PAIRS, seed)
    clean_d_all = _clean_d(model, base, is_id, are_id, started=started, cap=cap, label=f"seed {seed} second-phase Gate-A clean")
    clean = _compact_clean(model, base, clean_d_all, is_id, are_id)
    gate, retained, _ = gate_a(base, clean, is_id, are_id)
    if [int(value) for value in retained] != [int(value) for value in first.get("retained_pairs", [])]:
        raise Stage2Stop("resume_input", f"Seed {seed} retained-pair list changed between Stage-2 phases.")
    if gate != first.get("gate_A"):
        raise Stage2Stop("resume_input", f"Seed {seed} Gate-A diagnostics changed between Stage-2 phases.")
    base_indices, source_indices, signs = directed_indices(REQUESTED_PAIRS, retained)
    pair_ids_all = [int(pair) for pair in retained for _ in (0, 1)]
    directions_all = list(DIRECTIONS) * len(retained)
    base_final = positions_for_kind(base, base_indices, "final").squeeze(1)
    source_final = positions_for_kind(base, source_indices, "final").squeeze(1)
    base_subject = positions_for_kind(base, base_indices, "subject").squeeze(1)
    source_subject = positions_for_kind(base, source_indices, "subject").squeeze(1)
    base_tokens = base.tokens[base_indices]
    clean_base_d = clean_d_all[base_indices].detach().float().cpu()
    true_tokens = base.tokens[source_indices]
    source_a = make_source_a(model.tokenizer, base, seed)
    source_a_tokens = source_a.tokens[base_indices]
    source_a_final = positions_for_kind(source_a, base_indices, "final").squeeze(1)
    source_a_subject = positions_for_kind(source_a, base_indices, "subject").squeeze(1)
    source_b = make_source_b(model.tokenizer, base)
    source_b_tokens = source_b.tokens[base_indices]
    source_b_final = positions_for_kind(source_b, base_indices, "final").squeeze(1)
    source_c = make_source_c_relative_clause(model.tokenizer, REQUESTED_PAIRS, seed, with_adverb=True)
    source_c_gate, source_c_retained = _source_gate(model, source_c, is_id, are_id, started=started, cap=cap)
    eligible = sorted(set(int(value) for value in retained).intersection(int(value) for value in source_c_retained))
    q2: dict[str, Any] = {
        "status": "COMPLETE",
        "eligible_pairs": eligible,
        "source_c_retained_pairs": [int(value) for value in source_c_retained],
        "source_c_gate_A": source_c_gate,
        "source_c_gate_passed": bool(source_c_gate.get("passed")),
        "tested_set": list(tested_set),
        "logical_forward_equivalents": {"gate_a_clean": 1, "source_c_gate_a_clean": 1},
    }
    if len(eligible) < Q2_MIN_PAIRS:
        q2.update({"status": Q2_UNRESOLVED_CODE, "label": None, "positive": None, "complete_pair_count": len(eligible), "source_c_gate_passed": bool(source_c_gate.get("passed"))})
    else:
        eligible_set = set(eligible)
        local_q2 = torch.tensor(
            [index for index, pair_id in enumerate(pair_ids_all) if int(pair_id) in eligible_set],
            dtype=torch.long,
        )
        q2_base_tokens = base_tokens[local_q2]
        q2_lengths = base.lengths[base_indices][local_q2]
        q2_base_final = base_final[local_q2]
        q2_clean_d = clean_base_d[local_q2]
        q2_signs = signs[local_q2]
        q2_pair_ids = [pair_ids_all[index] for index in local_q2.tolist()]
        q2_directions = [directions_all[index] for index in local_q2.tolist()]
        true_z = _cache_activations(model, true_tokens, started=started, cap=cap, label=f"seed {seed} Q2 true")["z"]
        a_z = _cache_activations(model, source_a_tokens, started=started, cap=cap, label=f"seed {seed} Q2 source-A")["z"]
        c_base_indices, _, c_signs = directed_indices(REQUESTED_PAIRS, eligible)
        if not torch.equal(q2_signs, c_signs):
            raise Stage2Stop("q2_signs", f"Seed {seed} source-C sign alignment differs from base intersection.")
        # Source C is the cross-template, number-matched control.  Using the
        # opposite member would silently turn this into another number flip.
        source_c_tokens = source_c.tokens[c_base_indices]
        source_c_final = positions_for_kind(source_c, c_base_indices, "final").squeeze(1)
        c_z = _cache_activations(model, source_c_tokens, started=started, cap=cap, label=f"seed {seed} Q2 source-C")["z"]
        b_z = _cache_activations(model, source_b_tokens[local_q2], started=started, cap=cap, label=f"seed {seed} Q2 source-B")["z"]
        q2_true = _direct_effect(model, base_tokens=q2_base_tokens, base_lengths=q2_lengths, base_final=q2_base_final, source_cache=_slice_cache(true_z, local_q2), source_positions=source_final[local_q2], clean_base_d=q2_clean_d, signs=q2_signs, selected=tested_set, label="stage2_q2_true_right", pair_ids=q2_pair_ids, directions=q2_directions, started=started, cap=cap)
        q2_a = _direct_effect(model, base_tokens=q2_base_tokens, base_lengths=q2_lengths, base_final=q2_base_final, source_cache=_slice_cache(a_z, local_q2), source_positions=source_a_final[local_q2], clean_base_d=q2_clean_d, signs=q2_signs, selected=tested_set, label="stage2_q2_source_A", pair_ids=q2_pair_ids, directions=q2_directions, started=started, cap=cap)
        q2_c = _direct_effect(model, base_tokens=q2_base_tokens, base_lengths=q2_lengths, base_final=q2_base_final, source_cache=c_z, source_positions=source_c_final, clean_base_d=q2_clean_d, signs=q2_signs, selected=tested_set, label="stage2_q2_source_C", pair_ids=q2_pair_ids, directions=q2_directions, started=started, cap=cap)
        q2_b = _direct_effect(model, base_tokens=q2_base_tokens, base_lengths=q2_lengths, base_final=q2_base_final, source_cache=b_z, source_positions=source_b_final[local_q2], clean_base_d=q2_clean_d, signs=q2_signs, selected=tested_set, label="stage2_q2_source_B_descriptive", pair_ids=q2_pair_ids, directions=q2_directions, started=started, cap=cap)
        e_right, e_a, e_c = float(q2_true["E"]), float(q2_a["E"]), float(q2_c["E"])
        if not all(math.isfinite(value) for value in (e_right, e_a, e_c)):
            raise Stage2Stop("q2_non_finite_effect", f"Seed {seed} produced a non-finite registered Q2 effect.")
        pass_a = bool(abs(e_a) <= float(theta["A"]) * abs(e_right))
        pass_c = bool(abs(e_c) <= float(theta["C"]) * abs(e_right))
        q2.update({"complete_pair_count": len(eligible), "true_right": q2_true, "source_A": q2_a, "source_C": q2_c, "source_B_descriptive": q2_b, "pass_source_A": pass_a, "pass_source_C": pass_c, "positive": bool(pass_a and pass_c), "label": "Number-specific under registered controls" if pass_a and pass_c else "Specificity bound not met", "status": "COMPLETE", "logical_forward_equivalents": {"gate_a_clean": 1, "source_c_gate_a_clean": 1, "true_cache": 1, "source_A_cache": 1, "source_C_cache": 1, "source_B_cache": 1, "true_arm": 1, "source_A_arm": 1, "source_C_arm": 1, "source_B_arm": 1}})
    q2["axis_status"] = "SCIENTIFIC_UNRESOLVED" if q2.get("status") == Q2_UNRESOLVED_CODE else ("PASS" if q2.get("positive") is True else "COMPLETED_FAIL")
    if not retained:
        q3 = {
            "status": "COMPLETE",
            "execution_status": "COMPLETE",
            "axis_status": "SCIENTIFIC_UNRESOLVED",
            "cell_status": "SKIPPED_NO_RETAINED_ITEMS",
            "reason_code": "Q3_ZERO_RETAINED_ITEMS",
            "positive": None,
            "label": None,
            "direct_recovery": {"seed": seed, "status": "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE", "value": None, "reason": {"code": "Q3_ZERO_RETAINED_ITEMS", "detail": "No retained base items were available for the registered Q3 descriptive ratio."}},
            "direct_recovery_inference": "Q3_DRF_NO_INFERENTIAL_INTERVAL",
            "q3_drf_affects_verdict": False,
            "q3_descriptive_direct_reference_logical_forward_equivalents": 0,
            "logical_forward_equivalents": {},
        }
        _check_runtime(started, cap, f"after scientific zero-retained skip seed {seed}")
        return {"seed": seed, "execution_status": "COMPLETE", "base_gate_A": gate, "q2": q2, "q3": q3, "status": "COMPLETE"}
    # Q3 uses the full base Gate-A population, independently of Q2's C
    # intersection.  The source-A and true arms share immutable base caches.
    base_bundle = _cache_activations(model, base_tokens, need_v=True, need_pattern=True, started=started, cap=cap, label=f"seed {seed} Q3 base")
    true_bundle = _cache_activations(model, true_tokens, need_v=True, started=started, cap=cap, label=f"seed {seed} Q3 true")
    a_bundle = _cache_activations(model, source_a_tokens, need_v=True, started=started, cap=cap, label=f"seed {seed} Q3 source-A")
    q3_true = _q3_effect(model, base_tokens=base_tokens, base_lengths=base.lengths[base_indices], base_final=base_final, clean_base_d=clean_base_d, signs=signs, base_bundle=base_bundle, source_bundle=true_bundle, base_subject=base_subject, source_subject=source_subject, selected=tested_set, label="stage2_q3_Bprime_true", pair_ids=pair_ids_all, directions=directions_all, started=started, cap=cap)
    q3_a = _q3_effect(model, base_tokens=base_tokens, base_lengths=base.lengths[base_indices], base_final=base_final, clean_base_d=clean_base_d, signs=signs, base_bundle=base_bundle, source_bundle=a_bundle, base_subject=base_subject, source_subject=source_a_subject, selected=tested_set, label="stage2_q3_Bprime_source_A", pair_ids=pair_ids_all, directions=directions_all, started=started, cap=cap)
    direct_payload = first.get("q1_nested", {}).get(str(len(tested_set)), {}).get("direct", {})
    direct_values = direct_payload.get("pair_means") if isinstance(direct_payload, Mapping) else None
    d_direct = _mean_float64(direct_values, f"seed {seed} direct-effect denominator") if isinstance(direct_values, list) and direct_values else float("nan")
    true_aligned = torch.tensor(q3_true["directed_sign_aligned"], dtype=torch.float32)
    a_aligned = torch.tensor(q3_a["directed_sign_aligned"], dtype=torch.float32)
    _check_runtime(started, cap, f"before Q3 bootstrap seed {seed}")
    q3_stat = core.q3(core.pair_means(true_aligned), core.pair_means(a_aligned), seed=seed)
    _check_runtime(started, cap, f"after Q3 bootstrap seed {seed}")
    q3_payload = q3_stat.as_dict() if hasattr(q3_stat, "as_dict") else dict(q3_stat)
    d_path = float(q3_payload["D_path"])
    d_ci = list(q3_payload["D_path_percentile_95_ci"])
    q3_true_mean = _mean_float64(q3_true["pair_means"], f"seed {seed} Q3 true-path effect")
    if first.get("q1_cell_status") == "SKIPPED_BY_PREREGISTERED_SCIENTIFIC_GATE":
        direct_recovery = {"seed": seed, "status": "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE", "value": None, "reason": {"code": "Q3_DRF_REFERENCE_SKIPPED_BY_Q1_BASE_GLOBAL_GATE", "detail": "The direct reference was conditionally skipped by the preregistered Q1 base-global Gate-A rule."}}
    elif math.isfinite(d_direct) and d_direct != 0.0:
        direct_recovery = {"seed": seed, "status": "Q3_DRF_DESCRIPTIVE_VALUE_FINITE", "value": float(q3_true_mean / d_direct)}
    else:
        direct_recovery = {"seed": seed, "status": "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE", "value": None, "reason": {"code": "NONFINITE_VALUE", "detail": "The registered direct-effect denominator was non-finite or exactly zero."}}
    q3 = {"status": "COMPLETE", "execution_status": "COMPLETE", "cell_status": "COMPLETE", "axis_status": "PASS" if q3_payload["subject_value_transport_shown"] else "COMPLETED_FAIL", "true_path": q3_true, "source_A_path": q3_a, "F_path": float(q3_a["E"]), "D_path": d_path, "D_path_ci95": d_ci, "positive": bool(q3_payload["subject_value_transport_shown"]), "label": "Subject-value transport shown" if q3_payload["subject_value_transport_shown"] else "Subject-value transport not shown", "direct_recovery": direct_recovery, "direct_recovery_inference": "Q3_DRF_NO_INFERENTIAL_INTERVAL", "q3_drf_affects_verdict": False, "q3_descriptive_direct_reference_logical_forward_equivalents": 0, "test_id": Q3_TEST_ID, "rng_seed": seed * 1000 + Q3_TEST_ID, "bootstrap_resamples": BOOTSTRAP_INTERVAL, "logical_forward_equivalents": {"q3_base_cache": 1, "q3_true_cache": 1, "q3_source_A_cache": 1, "q3_true_arm": 1, "q3_source_A_arm": 1}}
    _check_runtime(started, cap, f"after second-phase seed {seed}")
    return {"seed": seed, "execution_status": "COMPLETE", "base_gate_A": gate, "q2": q2, "q3": q3, "status": "COMPLETE"}


def _validate_model_provenance(model: Any, protocol_model: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise Stage2Stop("offline_provenance", f"{key}=1 is required for a model run.")
    cfg = model.cfg
    expected = {"n_layers": 12, "n_heads": 12, "d_model": 768, "d_vocab": 50_257}
    observed = {name: int(getattr(cfg, name, -1)) for name in expected}
    if observed != expected:
        raise Stage2Stop("model_architecture", f"Expected GPT-2-small architecture {expected}, observed {observed}.")
    model_name = str(getattr(cfg, "model_name", ""))
    if model_name and model_name.lower() not in {"gpt2-small", "gpt2"}:
        raise Stage2Stop("model_revision", f"Unexpected TransformerLens model name {model_name!r}.")
    revisions = protocol_model.get("expected_local_snapshot_revisions") if isinstance(protocol_model, Mapping) else None
    expected_revision = revisions.get("gpt2") if isinstance(revisions, Mapping) else None
    cache_candidates: list[Path] = []
    hub_cache = os.environ.get("HF_HUB_CACHE")
    hf_home = os.environ.get("HF_HOME")
    if hub_cache:
        cache_candidates.append(Path(hub_cache).expanduser())
    if hf_home:
        cache_candidates.append(Path(hf_home).expanduser() / "hub")
    cache_candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    observed_revision: str | None = None
    observed_ref: Path | None = None
    for root in cache_candidates:
        ref = root / "models--gpt2" / "refs" / "main"
        if ref.is_file():
            observed_ref = ref
            observed_revision = ref.read_text(encoding="utf-8").strip()
            break
    if not isinstance(expected_revision, str) or not expected_revision:
        raise Stage2Stop("model_revision", "protocol.model.expected_local_snapshot_revisions.gpt2 is missing.")
    if observed_revision != expected_revision:
        raise Stage2Stop("model_revision", f"GPT-2 local snapshot ref {observed_revision!r} at {observed_ref} differs from expected {expected_revision!r}.")
    transformer_lens_module = sys.modules.get("transformer_lens")
    transformer_lens_version = getattr(transformer_lens_module, "__version__", "unknown")
    return {"model_name": model_name or "gpt2-small (architecture-pinned)", "architecture": observed, "device": "cpu", "dtype": "float32", "offline": True, "torch_version": str(getattr(torch, "__version__", "unknown")), "transformerlens_version": str(transformer_lens_version), "gpt2_snapshot_revision": observed_revision, "gpt2_snapshot_ref": str(observed_ref)}


def _model_state_fingerprint(model: Any, *, started: float, cap: float | None, label: str) -> dict[str, Any]:
    """Use the exact A7 selection fingerprint (uncast bytes + length framing)."""

    digest = hashlib.sha256()
    digest.update(b"exp05.model_state_fingerprint.v1\0")
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise Stage2Stop("model_state_fingerprint", "Model state_dict is empty or not a mapping.")
    entries: list[dict[str, Any]] = []
    for key in sorted(state):
        _check_runtime(started, cap, f"{label} state key {key}")
        value = state[key]
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise Stage2Stop("model_state_fingerprint", f"Model state entry {key!r} is invalid.")
        tensor = value.detach().cpu().contiguous()
        raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        metadata = {"key": key, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        encoded_metadata = _json_bytes(metadata)
        digest.update(len(encoded_metadata).to_bytes(8, "big"))
        digest.update(encoded_metadata)
        digest.update(len(raw_bytes).to_bytes(8, "big"))
        digest.update(raw_bytes)
        entries.append({**metadata, "byte_length": len(raw_bytes), "bytes_sha256": _sha256_bytes(raw_bytes)})
    _check_runtime(started, cap, f"after {label} state fingerprint")
    return {"schema": "exp05.model_state_fingerprint.v1", "sha256": digest.hexdigest(), "key_count": len(entries), "entries": entries, "scheme": "lexicographic state_dict keys; key/dtype/shape JSON plus uncast contiguous tensor bytes; uint64 length framing", "encoding_detail": "canonical JSON metadata; unsigned uint64 big-endian metadata and raw-byte lengths"}


def _public_seed(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value_item for key, value_item in value.items() if not str(key).startswith("_")}


def _checkpoint_metadata(*, commit: str, protocol_hash: str, calibration_hash: str, selection_hash: str, candidate_hash: str, candidate: Sequence[Mapping[str, int]]) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "core_api": CORE_API_VERSION, "commit": commit, "protocol_sha256": protocol_hash, "calibration_sha256": calibration_hash, "selection_sha256": selection_hash, "candidate_file_sha256": candidate_hash, "seeds": list(EXPECTED_SEEDS), "candidate_C": [dict(row) for row in candidate]}


def _seed_payload_hash(value: Mapping[str, Any]) -> str:
    """Hash one seed payload without trusting a derived verdict field.

    The per-seed hash is deliberately separate from the outer checkpoint hash.
    Resume validation below recomputes registered statistics from this payload;
    the hash only binds the payload that was actually validated.
    """

    material = {key: item for key, item in value.items() if key != "seed_payload_sha256"}
    return _sha256_bytes(_json_bytes(material))


def _require_finite_number(value: Any, gate: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Stage2Stop(gate, f"{label} must be a finite number, got {value!r}.")
    return float(value)


def _require_same_number(value: Any, expected: float, gate: str, label: str, *, tolerance: float = 1e-7) -> None:
    observed = _require_finite_number(value, gate, label)
    if not math.isclose(observed, float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise Stage2Stop(gate, f"{label}={observed!r} differs from recomputed {expected!r}.")


def _validate_effect_row_for_resume(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_pair_ids: Sequence[int],
    core: "CoreAdapter",
    expected_signs: Sequence[float] | None = None,
) -> None:
    """Re-derive every registered scalar from a persisted paired effect row.

    A checkpoint self-hash alone is not evidence: an attacker could rewrite a
    verdict and re-hash the file.  This validator therefore checks the complete
    directed/pair-record binding and all scalar summaries before a row may be
    reused.  It is model-free and deliberately rejects any shape it cannot
    reconstruct exactly.
    """

    if not isinstance(row, Mapping):
        raise Stage2Stop("checkpoint_effect_row", f"{label} must be an object.")
    directed_raw = row.get("directed_raw")
    aligned = row.get("directed_sign_aligned")
    pair_means = row.get("pair_means")
    pair_ids = row.get("pair_ids")
    directions = row.get("directions")
    pair_records = row.get("pair_records")
    if not all(isinstance(value, list) for value in (directed_raw, aligned, pair_means, pair_ids, directions, pair_records)):
        raise Stage2Stop("checkpoint_effect_row", f"{label} lacks explicit directed/pair-record arrays.")
    expected_ids = [int(value) for value in expected_pair_ids]
    expected_directions = list(DIRECTIONS) * len(expected_ids)
    if [int(value) for value in pair_ids] != expected_ids:
        raise Stage2Stop("checkpoint_effect_row", f"{label}.pair_ids differs from the frozen eligible pair set.")
    if list(directions) != expected_directions:
        raise Stage2Stop("checkpoint_effect_row", f"{label}.directions differs from the frozen singular/plural order.")
    if len(directed_raw) != 2 * len(expected_ids) or len(aligned) != len(directed_raw) or len(pair_records) != len(aligned) or len(pair_means) != len(expected_ids):
        raise Stage2Stop("checkpoint_effect_row", f"{label} has an inconsistent directed/pair array length.")
    raw_values = [_require_finite_number(value, "checkpoint_effect_row", f"{label}.directed_raw[{index}]") for index, value in enumerate(directed_raw)]
    aligned_values = [_require_finite_number(value, "checkpoint_effect_row", f"{label}.directed_sign_aligned[{index}]") for index, value in enumerate(aligned)]
    if expected_signs is not None:
        if len(expected_signs) != len(raw_values):
            raise Stage2Stop("checkpoint_effect_row", f"{label} sign payload length differs from directed effects.")
        for index, (raw, aligned_value, sign) in enumerate(zip(raw_values, aligned_values, expected_signs)):
            sign_value = _require_finite_number(sign, "checkpoint_effect_row", f"{label}.expected_signs[{index}]")
            if sign_value not in {-1.0, 1.0} or not math.isclose(aligned_value, raw * sign_value, rel_tol=1e-6, abs_tol=1e-7):
                raise Stage2Stop("checkpoint_effect_row", f"{label} directed sign alignment disagrees with the retained sign registry at {index}.")
    records = pair_records
    derived_means: list[float] = []
    for pair_index, pair_id in enumerate(expected_ids):
        pair_values = aligned_values[2 * pair_index : 2 * pair_index + 2]
        derived_means.append((pair_values[0] + pair_values[1]) / 2.0)
        for direction_index, direction in enumerate(DIRECTIONS):
            record = records[2 * pair_index + direction_index]
            if not isinstance(record, Mapping):
                raise Stage2Stop("checkpoint_effect_row", f"{label}.pair_records contains a non-object row.")
            if int(record.get("pair_id", -1)) != pair_id or record.get("direction") != direction:
                raise Stage2Stop("checkpoint_effect_row", f"{label}.pair_records order does not match pair_ids/directions.")
            _require_same_number(record.get("effect"), aligned_values[2 * pair_index + direction_index], "checkpoint_effect_row", f"{label}.pair_records[{2 * pair_index + direction_index}].effect")
    for index, (observed, expected) in enumerate(zip(pair_means, derived_means)):
        _require_same_number(observed, expected, "checkpoint_effect_row", f"{label}.pair_means[{index}]")
    expected_e = math.fsum(aligned_values) / len(aligned_values)
    _require_same_number(row.get("E"), expected_e, "checkpoint_effect_row", f"{label}.E")
    _require_same_number(row.get("E_delta_d"), expected_e, "checkpoint_effect_row", f"{label}.E_delta_d")
    expected_pair_consistency = sum(all(value > 0.0 for value in aligned_values[index : index + 2]) for index in range(0, len(aligned_values), 2)) / len(expected_ids)
    expected_directed_consistency = sum(value > 0.0 for value in aligned_values) / len(aligned_values)
    _require_same_number(row.get("pair_sign_consistency"), expected_pair_consistency, "checkpoint_effect_row", f"{label}.pair_sign_consistency")
    _require_same_number(row.get("directed_sign_consistency"), expected_directed_consistency, "checkpoint_effect_row", f"{label}.directed_sign_consistency")
    # ``directed_raw`` is not used for the sign-aligned estimator, but it must
    # remain finite and preserve the recorded shape.  Sign values themselves
    # are bound by the retained-directed-edit payload validator.
    del raw_values
    recomputed_pair_means = core.pair_means(aligned_values)
    if len(recomputed_pair_means) != len(pair_means):
        raise Stage2Stop("checkpoint_effect_row", f"{label} could not be reconstructed as complete pair clusters.")


def _validate_first_seed_for_resume(
    row: Mapping[str, Any],
    *,
    seed: int,
    candidate: Sequence[Mapping[str, int]],
    core: "CoreAdapter",
    started: float,
    cap: float | None,
) -> None:
    if not isinstance(row, Mapping) or int(row.get("seed", -1)) != seed or row.get("status") != "FIRST_COMPLETE" or row.get("execution_status") != "COMPLETE":
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} has invalid status/seed identity.")
    retained = row.get("retained_pairs")
    edits = row.get("retained_directed_edits")
    if not isinstance(retained, list) or len({int(value) for value in retained}) != len(retained) or not isinstance(edits, Mapping):
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} lacks a complete retained-pair/edit payload.")
    retained_ids = [int(value) for value in retained] if isinstance(retained, list) else []
    if retained_ids != sorted(retained_ids) or any(value < 0 or value >= REQUESTED_PAIRS for value in retained_ids):
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} retained pair IDs are outside canonical order/range.")
    expected_directed_ids = [pair for pair in retained_ids for _ in (0, 1)]
    expected_directions = list(DIRECTIONS) * len(retained_ids)
    if isinstance(edits, Mapping):
        if [int(value) for value in edits.get("pair_ids", [])] != expected_directed_ids or list(edits.get("directions", [])) != expected_directions:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} directed pair identity changed.")
        signs = edits.get("signs")
        base_indices = edits.get("base_indices")
        source_indices = edits.get("source_indices")
        if not all(isinstance(value, list) and len(value) == len(expected_directed_ids) for value in (signs, base_indices, source_indices)):
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} directed edit arrays are incomplete.")
        expected_base = [index for pair in retained_ids for index in (2 * pair, 2 * pair + 1)]
        expected_source = [index for pair in retained_ids for index in (2 * pair + 1, 2 * pair)]
        expected_sign = [sign for _pair in retained_ids for sign in (1.0, -1.0)]
        if [int(value) for value in base_indices] != expected_base or [int(value) for value in source_indices] != expected_source:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} directed pair index construction changed.")
        for index, sign in enumerate(signs):
            sign_value = _require_finite_number(sign, "checkpoint_seed", f"seed {seed} signs[{index}]")
            if sign_value != expected_sign[index]:
                raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} has a non-binary sign at {index}.")
            if not isinstance(base_indices[index], int) or not isinstance(source_indices[index], int):
                raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} has non-integer directed indices.")
    skipped = row.get("q1_cell_status") == "SKIPPED_BY_PREREGISTERED_SCIENTIFIC_GATE"
    if skipped:
        gate = row.get("gate_A")
        if not isinstance(gate, Mapping) or gate.get("passed") is not False or int(gate.get("generated_pairs", -1)) != REQUESTED_PAIRS or int(gate.get("retained_pairs", -1)) != len(retained_ids):
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} conditional Q1 skip lacks exact failed Gate-A evidence.")
        input_hashes = row.get("input_hashes")
        if not isinstance(input_hashes, Mapping) or not _is_sha256(input_hashes.get("base_tokens")) or set(input_hashes) != {"base_tokens"}:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} conditional Q1 skip lacks complete base input identity.")
        skip_fe = row.get("logical_forward_equivalents")
        if row.get("q1_scientific_unresolved_reason") != "Q1_BASE_GLOBAL_GATE_A_FAIL" or row.get("true_heads") != [] or row.get("source_a_heads") != [] or row.get("q1_nested") != {} or not isinstance(skip_fe, Mapping) or dict(skip_fe) != {"gate_a_clean": 1, "conditional_q1_skip": 0}:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} has an invalid conditional Q1 skip payload.")
        return
    if row.get("q1_cell_status") != "COMPLETE" or not isinstance(row.get("gate_A"), Mapping) or row.get("gate_A", {}).get("passed") is not True:
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} is neither a complete Q1 cell nor a registered Gate-A skip.")
    true_rows = row.get("true_heads")
    source_rows = row.get("source_a_heads")
    input_hashes = row.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != {"base_tokens", "true_tokens", "source_a_tokens"} or not all(_is_sha256(input_hashes.get(key)) for key in ("base_tokens", "true_tokens", "source_a_tokens")):
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} lacks complete base/true/source-A input identity.")
    if not isinstance(true_rows, list) or not isinstance(source_rows, list) or len(true_rows) != EXPECTED_HEADS or len(source_rows) != EXPECTED_HEADS:
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} does not contain exactly 144 true/source-A head rows.")
    canonical_heads = [(item["layer"], item["head"]) for item in _canonical_heads()]
    if [(int(item.get("layer", -1)), int(item.get("head", -1))) for item in true_rows] != canonical_heads or [(int(item.get("layer", -1)), int(item.get("head", -1))) for item in source_rows] != canonical_heads:
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} head order is not canonical 12x12 order.")
    for index, (true_row, source_row) in enumerate(zip(true_rows, source_rows)):
        _validate_effect_row_for_resume(true_row, label=f"seed {seed} true_heads[{index}]", expected_pair_ids=retained_ids, core=core, expected_signs=edits["signs"])
        _validate_effect_row_for_resume(source_row, label=f"seed {seed} source_a_heads[{index}]", expected_pair_ids=retained_ids, core=core, expected_signs=edits["signs"])
    edge = _require_finite_number(row.get("source_a_edge"), "checkpoint_seed", f"seed {seed}.source_a_edge")
    recomputed_edge = core.percentile([abs(float(item["E"])) for item in source_rows], q=99.0)
    _require_same_number(edge, recomputed_edge, "checkpoint_seed", f"seed {seed}.source_a_edge", tolerance=1e-9)
    distinguishability = row.get("distinguishability")
    if not isinstance(distinguishability, list) or len(distinguishability) != EXPECTED_HEADS:
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} lacks all 144 distinguishability records.")
    p_values: list[float] = []
    for index, (record, true_row, source_row) in enumerate(zip(distinguishability, true_rows, source_rows)):
        if not isinstance(record, Mapping) or (int(record.get("layer", -1)), int(record.get("head", -1))) != canonical_heads[index]:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} distinguishability order changed.")
        _require_same_number(record.get("E"), float(true_row["E"]), "checkpoint_seed", f"seed {seed}.distinguishability[{index}].E")
        _require_same_number(record.get("source_a_E"), float(source_row["E"]), "checkpoint_seed", f"seed {seed}.distinguishability[{index}].source_a_E")
        _require_same_number(record.get("source_a_abs_edge"), edge, "checkpoint_seed", f"seed {seed}.distinguishability[{index}].source_a_abs_edge")
        _require_same_number(record.get("pair_sign_consistency"), float(true_row["pair_sign_consistency"]), "checkpoint_seed", f"seed {seed}.distinguishability[{index}].pair_sign_consistency")
        p = _require_finite_number(record.get("raw_p"), "checkpoint_seed", f"seed {seed}.distinguishability[{index}].raw_p")
        if int(record.get("test_id", -1)) != _flat_test_id(*canonical_heads[index]) or int(record.get("rng_seed", -1)) != seed * 1000 + _flat_test_id(*canonical_heads[index]) or int(record.get("bootstrap_resamples", -1)) != BOOTSTRAP_HEAD:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} distinguishability RNG/test binding changed.")
        _check_runtime(started, cap, f"resume Q1 bootstrap {seed} {_head_key(*canonical_heads[index])}")
        recomputed_p = core.bootstrap_p(core.pair_means(true_row["directed_sign_aligned"]), seed=seed, test_id=_flat_test_id(*canonical_heads[index]), resamples=BOOTSTRAP_HEAD)
        _check_runtime(started, cap, f"after resume Q1 bootstrap {seed} {_head_key(*canonical_heads[index])}")
        _require_same_number(p, recomputed_p, "checkpoint_seed", f"seed {seed}.distinguishability[{index}].raw_p", tolerance=1e-12)
        p_values.append(p)
    expected_rejects = core.holm(p_values, alpha=HOLM_ALPHA)
    for index, record in enumerate(distinguishability):
        if "holm_reject" not in record or "distinguishable" not in record:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} distinguishability[{index}] lacks Holm/distinguishable fields.")
        reject = bool(expected_rejects[index])
        if bool(record.get("holm_reject")) != reject or bool(record.get("distinguishable")) != bool(reject and abs(float(record["E"])) > edge):
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} Holm/distinguishability result was edited.")
    e_all = row.get("E_all")
    if not isinstance(e_all, Mapping):
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} lacks E_all.")
    guard = row.get("E_all_denominator_guard")
    if not isinstance(guard, Mapping):
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} lacks E_all denominator guard.")
    e_all_has_null = any(
        any(value is None for value in e_all.get(field, []))
        for field in ("directed_raw", "directed_sign_aligned", "pair_means", "pair_records")
        if isinstance(e_all.get(field), list)
    )
    if e_all_has_null or guard.get("reason_code") == "Q1_SCIENTIFIC_UNRESOLVED_EALL_NONFINITE":
        # A non-finite E_all is a valid scientific unresolved outcome during
        # the original invocation, but its null sentinel is not a reusable
        # numerical row.  Force a fresh run instead of calling it corruption
        # or manufacturing a resumed Q1 result.
        raise Stage2Stop("checkpoint_resume_eall_nonfinite", f"Seed {seed} has a non-finite/null E_all sentinel; rerun the invocation from zero.")
    recomputed_guard = _q1_e_all_guard(e_all)
    _validate_effect_row_for_resume(e_all, label=f"seed {seed}.E_all", expected_pair_ids=retained_ids, core=core, expected_signs=edits["signs"])
    for key in ("estimable", "reason_code"):
        if guard.get(key) != recomputed_guard.get(key):
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} E_all guard {key} was edited.")
    for key in ("E_all_float64", "A_all_float64", "g", "gamma"):
        observed, expected = guard.get(key), recomputed_guard.get(key)
        if expected is None:
            if observed is not None:
                raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} E_all guard {key} should be null.")
        else:
            _require_same_number(observed, float(expected), "checkpoint_seed", f"seed {seed}.E_all_denominator_guard.{key}", tolerance=1e-12)
    nested = row.get("q1_nested")
    if not isinstance(nested, Mapping):
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} lacks q1_nested.")
    max_n = min(Q1_MAX, len(candidate))
    complete_fe = row.get("logical_forward_equivalents")
    if not isinstance(complete_fe, Mapping) or any(complete_fe.get(key) != 1 for key in ("gate_a_clean", "true_cache", "source_a_cache", "base_cache", "E_all_joint")) or complete_fe.get("true_singleton_sweep") != EXPECTED_HEADS or complete_fe.get("source_a_singleton_sweep") != EXPECTED_HEADS or not isinstance(complete_fe.get("q1_nested_sets"), Mapping) or set(complete_fe["q1_nested_sets"]) != {str(n) for n in range(1, max_n + 1)} or any(complete_fe["q1_nested_sets"].get(str(n)) != 1 for n in range(1, max_n + 1)):
        raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} logical forward-equivalent accounting is incomplete.")
    for n in range(1, max_n + 1):
        item = nested.get(str(n))
        if not isinstance(item, Mapping) or item.get("set") != list(candidate[:n]):
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} Q1 nested set S_{n} is not the frozen candidate prefix.")
        direct = item.get("direct")
        if not isinstance(direct, Mapping):
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} Q1 S_{n} lacks a direct effect row.")
        _validate_effect_row_for_resume(direct, label=f"seed {seed}.q1_nested[{n}].direct", expected_pair_ids=retained_ids, core=core, expected_signs=edits["signs"])
        numerator = math.fsum(float(value) for value in direct["pair_means"]) / len(direct["pair_means"])
        _require_same_number(item.get("numerator_float64"), numerator, "checkpoint_seed", f"seed {seed}.q1_nested[{n}].numerator_float64", tolerance=1e-12)
        denominator = guard.get("E_all_float64") if guard.get("estimable") is True else None
        observed_recovery = item.get("recovery_fraction")
        if denominator is None:
            if observed_recovery is not None:
                raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} Q1 S_{n} recovery must be null under E_all guard.")
        else:
            _require_same_number(observed_recovery, numerator / float(denominator), "checkpoint_seed", f"seed {seed}.q1_nested[{n}].recovery_fraction", tolerance=1e-12)
        expected_dist = all(bool(distinguishability[canonical_heads.index((int(item_row["layer"]), int(item_row["head"])))].get("distinguishable")) for item_row in candidate[:n])
        if "all_members_distinguishable" not in item or "seed_joint_pass" not in item:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} Q1 S_{n} lacks distinguishability/pass fields.")
        if bool(item.get("all_members_distinguishable")) != expected_dist:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} Q1 S_{n} distinguishability summary was edited.")
        expected_pass = bool(denominator is not None and numerator / float(denominator) >= 0.50 and expected_dist)
        if bool(item.get("seed_joint_pass")) != expected_pass:
            raise Stage2Stop("checkpoint_seed", f"First-phase seed {seed} Q1 S_{n} pass status was edited.")


def _validate_second_seed_for_resume(
    row: Mapping[str, Any],
    *,
    seed: int,
    first: Mapping[str, Any],
    tested_set: Sequence[Mapping[str, int]],
    core: "CoreAdapter",
    started: float,
    cap: float | None,
) -> None:
    if not isinstance(row, Mapping) or int(row.get("seed", -1)) != seed or row.get("status") != "COMPLETE" or row.get("execution_status") != "COMPLETE":
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} has invalid status/seed identity.")
    q2 = row.get("q2")
    q3 = row.get("q3")
    if not isinstance(q2, Mapping) or not isinstance(q3, Mapping):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} lacks Q2/Q3 payloads.")
    retained = [int(value) for value in first.get("retained_pairs", [])]
    source_c_retained = q2.get("source_c_retained_pairs")
    if not isinstance(source_c_retained, list) or len({int(value) for value in source_c_retained}) != len(source_c_retained):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} lacks source-C retained pair IDs.")
    source_c_ids = [int(value) for value in source_c_retained]
    if source_c_ids != sorted(source_c_ids) or any(value < 0 or value >= REQUESTED_PAIRS for value in source_c_ids):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} source-C retained pair IDs are not in canonical order.")
    source_c_gate = q2.get("source_c_gate_A")
    if not isinstance(source_c_gate, Mapping):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} lacks source-C Gate-A diagnostics.")
    if "source_c_gate_passed" in q2 and bool(q2.get("source_c_gate_passed")) != bool(source_c_gate.get("passed")):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} source-C global gate diagnostic was edited.")
    eligible = sorted(set(retained).intersection(source_c_ids))
    if [int(value) for value in q2.get("eligible_pairs", [])] != eligible:
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 eligible set is not base-retained intersect source-C-retained.")
    if q2.get("tested_set") != list(tested_set):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 tested set differs from the frozen Q1 set.")
    count = int(q2.get("complete_pair_count", -1))
    if count != len(eligible):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 complete-pair count was edited.")
    q2_fe = q2.get("logical_forward_equivalents")
    if not isinstance(q2_fe, Mapping) or q2_fe.get("gate_a_clean") != 1 or q2_fe.get("source_c_gate_a_clean") != 1:
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 Gate-A forward-equivalent accounting is incomplete.")
    if count >= Q2_MIN_PAIRS and any(q2_fe.get(key) != 1 for key in ("true_cache", "source_A_cache", "source_C_cache", "source_B_cache", "true_arm", "source_A_arm", "source_C_arm", "source_B_arm")):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 arm/cache forward-equivalent accounting is incomplete.")
    if "status" not in q2 or "axis_status" not in q2 or "positive" not in q2:
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 lacks status/axis/positive fields.")
    if count < Q2_MIN_PAIRS:
        if q2.get("status") != Q2_UNRESOLVED_CODE or q2.get("axis_status") != "SCIENTIFIC_UNRESOLVED" or q2.get("positive") is not None:
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 <40 status is not the frozen scientific-unresolved state.")
    else:
        if q2.get("status") != "COMPLETE" or q2.get("axis_status") not in {"PASS", "COMPLETED_FAIL"}:
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 completion status is invalid.")
        expected_ids = eligible
        for arm in ("true_right", "source_A", "source_C", "source_B_descriptive"):
            arm_row = q2.get(arm)
            _validate_effect_row_for_resume(arm_row, label=f"seed {seed}.q2.{arm}", expected_pair_ids=expected_ids, core=core, expected_signs=[sign for _pair in expected_ids for sign in (1.0, -1.0)])
        e_right = _require_finite_number(q2["true_right"]["E"], "checkpoint_seed", f"seed {seed}.q2.true_right.E")
        e_a = _require_finite_number(q2["source_A"]["E"], "checkpoint_seed", f"seed {seed}.q2.source_A.E")
        e_c = _require_finite_number(q2["source_C"]["E"], "checkpoint_seed", f"seed {seed}.q2.source_C.E")
        expected_a = abs(e_a) <= THETA_A * abs(e_right)
        expected_c = abs(e_c) <= THETA_C * abs(e_right)
        if "pass_source_A" not in q2 or "pass_source_C" not in q2:
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 lacks registered threshold fields.")
        if bool(q2.get("pass_source_A")) != expected_a or bool(q2.get("pass_source_C")) != expected_c or bool(q2.get("positive")) != bool(expected_a and expected_c):
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 arm threshold/status was edited.")
        expected_axis = "PASS" if expected_a and expected_c else "COMPLETED_FAIL"
        if q2.get("axis_status") != expected_axis:
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q2 axis status was edited.")
    if not retained:
        if q3.get("status") != "COMPLETE" or q3.get("execution_status") != "COMPLETE" or q3.get("cell_status") != "SKIPPED_NO_RETAINED_ITEMS" or q3.get("reason_code") != "Q3_ZERO_RETAINED_ITEMS" or q3.get("axis_status") != "SCIENTIFIC_UNRESOLVED" or q3.get("positive") is not None:
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} zero-retained Q3 skip is invalid.")
        if q3.get("logical_forward_equivalents") != {}:
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} zero-retained Q3 forward-equivalent accounting is non-empty.")
        recovery = q3.get("direct_recovery")
        if not isinstance(recovery, Mapping) or recovery.get("status") != "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE" or recovery.get("value") is not None or q3.get("direct_recovery_inference") != "Q3_DRF_NO_INFERENTIAL_INTERVAL" or q3.get("q3_drf_affects_verdict") is not False:
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} zero-retained Q3 direct-recovery skip is invalid.")
        return
    if q3.get("status") != "COMPLETE" or q3.get("execution_status") != "COMPLETE" or q3.get("cell_status") != "COMPLETE" or q3.get("axis_status") not in {"PASS", "COMPLETED_FAIL"} or "positive" not in q3:
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 completion status is invalid.")
    q3_fe = q3.get("logical_forward_equivalents")
    if not isinstance(q3_fe, Mapping) or any(q3_fe.get(key) != 1 for key in ("q3_base_cache", "q3_true_cache", "q3_source_A_cache", "q3_true_arm", "q3_source_A_arm")):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 forward-equivalent accounting is incomplete.")
    expected_q3_signs = [sign for _pair in retained for sign in (1.0, -1.0)]
    _validate_effect_row_for_resume(q3.get("true_path"), label=f"seed {seed}.q3.true_path", expected_pair_ids=retained, core=core, expected_signs=expected_q3_signs)
    _validate_effect_row_for_resume(q3.get("source_A_path"), label=f"seed {seed}.q3.source_A_path", expected_pair_ids=retained, core=core, expected_signs=expected_q3_signs)
    d_path = _require_finite_number(q3.get("D_path"), "checkpoint_seed", f"seed {seed}.q3.D_path")
    ci = q3.get("D_path_ci95")
    if not isinstance(ci, list) or len(ci) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in ci):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 D-path CI is invalid.")
    _require_same_number(q3.get("F_path"), float(q3["source_A_path"]["E"]), "checkpoint_seed", f"seed {seed}.q3.F_path")
    _check_runtime(started, cap, f"resume Q3 bootstrap {seed}")
    true_pair_matrix = core.pair_means(q3["true_path"]["directed_sign_aligned"])
    source_a_pair_matrix = core.pair_means(q3["source_A_path"]["directed_sign_aligned"])
    q3_recomputed = core.q3(true_pair_matrix, source_a_pair_matrix, seed=seed)
    _check_runtime(started, cap, f"after resume Q3 bootstrap {seed}")
    q3_expected = q3_recomputed.as_dict() if hasattr(q3_recomputed, "as_dict") else dict(q3_recomputed)
    _require_same_number(d_path, float(q3_expected["D_path"]), "checkpoint_seed", f"seed {seed}.q3.D_path", tolerance=1e-12)
    for index, (observed, expected) in enumerate(zip(ci, q3_expected["D_path_percentile_95_ci"])):
        _require_same_number(observed, float(expected), "checkpoint_seed", f"seed {seed}.q3.D_path_ci95[{index}]", tolerance=1e-12)
    expected_positive = bool(q3_expected["subject_value_transport_shown"])
    if bool(q3.get("positive")) != expected_positive or q3.get("axis_status") != ("PASS" if expected_positive else "COMPLETED_FAIL"):
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 D/CI/axis result was edited.")
    recovery = q3.get("direct_recovery")
    if not isinstance(recovery, Mapping) or int(recovery.get("seed", -1)) != seed or recovery.get("status") not in {"Q3_DRF_DESCRIPTIVE_VALUE_FINITE", "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE"}:
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 direct-recovery payload is invalid.")
    if recovery.get("value") is None:
        reason = recovery.get("reason")
        if not isinstance(reason, Mapping) or not str(reason.get("code", "")) or not str(reason.get("detail", "")):
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} null Q3 direct-recovery value lacks a structured reason.")
    else:
        _require_finite_number(recovery.get("value"), "checkpoint_seed", f"seed {seed}.q3.direct_recovery.value")
        if "reason" in recovery:
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} finite Q3 direct-recovery value must not contain a reason.")
    if q3.get("direct_recovery_inference") != "Q3_DRF_NO_INFERENTIAL_INTERVAL" or q3.get("q3_drf_affects_verdict") is not False:
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 direct-recovery inference contract was edited.")
    if first.get("q1_cell_status") == "SKIPPED_BY_PREREGISTERED_SCIENTIFIC_GATE":
        expected_recovery_status = "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE"
        expected_recovery_value = None
        expected_reason_code = "Q3_DRF_REFERENCE_SKIPPED_BY_Q1_BASE_GLOBAL_GATE"
    else:
        selected_n = len(tested_set)
        direct_nested = (first.get("q1_nested", {}) or {}).get(str(selected_n), {})
        direct_means = direct_nested.get("direct", {}).get("pair_means") if isinstance(direct_nested, Mapping) and isinstance(direct_nested.get("direct"), Mapping) else None
        d_direct = _mean_float64(direct_means, f"seed {seed} direct-effect denominator resume") if isinstance(direct_means, list) and direct_means else float("nan")
        if math.isfinite(d_direct) and d_direct != 0.0:
            expected_recovery_status = "Q3_DRF_DESCRIPTIVE_VALUE_FINITE"
            true_path_mean_float64 = _mean_float64(q3["true_path"]["pair_means"], f"seed {seed} Q3 true-path resume numerator")
            expected_recovery_value = float(true_path_mean_float64 / d_direct)
            expected_reason_code = None
        else:
            expected_recovery_status = "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE"
            expected_recovery_value = None
            expected_reason_code = "NONFINITE_VALUE"
    if recovery.get("status") != expected_recovery_status:
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 direct-recovery status differs from its validated inputs.")
    if expected_recovery_value is None:
        reason = recovery.get("reason")
        if recovery.get("value") is not None or not isinstance(reason, Mapping) or reason.get("code") != expected_reason_code or not str(reason.get("detail", "")):
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 direct-recovery unavailable reason differs from its validated inputs.")
    else:
        _require_same_number(recovery.get("value"), expected_recovery_value, "checkpoint_seed", f"seed {seed}.q3.direct_recovery.value", tolerance=1e-12)
    if first.get("q1_cell_status") == "SKIPPED_BY_PREREGISTERED_SCIENTIFIC_GATE":
        if recovery.get("status") != "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE" or recovery.get("value") is not None or recovery.get("reason", {}).get("code") != "Q3_DRF_REFERENCE_SKIPPED_BY_Q1_BASE_GLOBAL_GATE":
            raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 must preserve the Q1 Gate-A DRF interaction.")
    if q3.get("test_id") != Q3_TEST_ID or q3.get("rng_seed") != seed * 1000 + Q3_TEST_ID or q3.get("bootstrap_resamples") != BOOTSTRAP_INTERVAL:
        raise Stage2Stop("checkpoint_seed", f"Second-phase seed {seed} Q3 RNG/test binding changed.")
    del d_path


def _load_checkpoint(
    path: Path,
    metadata: Mapping[str, Any],
    *,
    core: "CoreAdapter" | None = None,
    started: float | None = None,
    cap: float | None = None,
) -> dict[str, Any]:
    checkpoint = _read_json(path, "checkpoint")
    declared_hash = checkpoint.get("checkpoint_sha256")
    if not _is_sha256(declared_hash):
        raise Stage2Stop("checkpoint_hash", "Checkpoint lacks a lowercase-hex checkpoint_sha256.")
    hash_material = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    if declared_hash != _sha256_bytes(_json_bytes(hash_material)):
        raise Stage2Stop("checkpoint_hash", "Checkpoint canonical self-hash does not match its contents.")
    if checkpoint.get("metadata") != dict(metadata):
        raise Stage2Stop("checkpoint_mismatch", "Checkpoint metadata differs from current commit/protocol/input hashes/candidate.")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise Stage2Stop("checkpoint_schema", "Checkpoint schema is not the frozen Stage-2 schema.")
    if checkpoint.get("scientific_reuse_allowed") is not False or checkpoint.get("resumable_for_adjudication") is not False:
        raise Stage2Stop("checkpoint_reuse_policy", "Checkpoint reuse policy must be explicitly diagnostic-only (false/false).")
    status = checkpoint.get("status")
    allowed_statuses = {"FIRST_PHASE_RUNNING", "FIRST_PHASE_COMPLETE", "SECOND_PHASE_RUNNING", "COMPLETE", "INCOMPLETE_RUNTIME_CAP"}
    if status not in allowed_statuses:
        raise Stage2Stop("checkpoint_schema", f"Checkpoint has unknown status {status!r}.")
    if checkpoint.get("scientific_verdict_emitted") is not (status == "COMPLETE"):
        raise Stage2Stop("checkpoint_reuse_policy", "Checkpoint scientific_verdict_emitted must be true only for a bound COMPLETE artifact.")
    if status == "COMPLETE":
        final_binding = checkpoint.get("final_binding")
        if not isinstance(final_binding, Mapping):
            raise Stage2Stop("checkpoint_final_binding", "COMPLETE checkpoint lacks final invocation binding.")
        for key in ("invocation_id", "invocation_config_sha256", "commit", "protocol_sha256", "protocol_file_sha256", "protocol_canonical_sha256", "calibration_sha256", "selection_sha256", "candidate_file_sha256", "state_dict_sha256_before", "state_dict_sha256_after", "state_dict_fingerprint_before", "state_dict_fingerprint_after", "execution_cell_registry", "pair_output", "pair_output_sha256", "coverage"):
            if key not in final_binding:
                raise Stage2Stop("checkpoint_final_binding", f"COMPLETE checkpoint final binding lacks {key}.")
        coverage = final_binding.get("coverage")
        registry = final_binding.get("execution_cell_registry")
        for hash_key in ("invocation_config_sha256", "protocol_sha256", "protocol_file_sha256", "protocol_canonical_sha256", "calibration_sha256", "selection_sha256", "candidate_file_sha256"):
            if not _is_sha256(final_binding.get(hash_key)):
                raise Stage2Stop("checkpoint_final_binding", f"COMPLETE checkpoint final binding hash {hash_key} is missing or malformed.")
        before_binding_hash = _validate_selection_state_fingerprint(final_binding.get("state_dict_fingerprint_before"), label="checkpoint before_sweeps")
        after_binding_hash = _validate_selection_state_fingerprint(final_binding.get("state_dict_fingerprint_after"), label="checkpoint after_sweeps")
        if before_binding_hash != final_binding.get("state_dict_sha256_before") or after_binding_hash != final_binding.get("state_dict_sha256_after") or before_binding_hash != after_binding_hash:
            raise Stage2Stop("checkpoint_final_binding", "COMPLETE checkpoint state fingerprints are not internally identical.")
        if not _is_sha256(final_binding.get("pair_output_sha256")) or not isinstance(coverage, Mapping) or coverage.get("coverage_mode") != "FULL_EXPECTED_STAGE2_CELLS" or coverage.get("expected_seeds") != list(EXPECTED_SEEDS) or not isinstance(registry, Mapping) or registry.get("coverage_mode") != "FULL_EXPECTED_STAGE2_CELLS" or registry.get("expected_seeds") != list(EXPECTED_SEEDS) or registry.get("coverage_consistent") is not True:
            raise Stage2Stop("checkpoint_final_binding", "COMPLETE checkpoint coverage is not the frozen eight-seed registry.")
        if registry.get("first_phase") != coverage.get("first_seed_ids") or registry.get("second_phase") != coverage.get("second_seed_ids") or registry.get("pair_csv_row_count") != coverage.get("pair_csv_row_count") or registry.get("expected_execution_cell_keys") != coverage.get("expected_execution_cell_keys") or registry.get("observed_execution_cell_keys") != coverage.get("observed_execution_cell_keys") or registry.get("expected_pair_csv_row_keys") != coverage.get("expected_pair_csv_row_keys") or registry.get("observed_pair_csv_row_keys") != coverage.get("observed_pair_csv_row_keys") or registry.get("registered_cell_count") != len(coverage.get("expected_execution_cell_keys", [])) or registry.get("executed_cell_count") != len(coverage.get("observed_execution_cell_keys", [])):
            raise Stage2Stop("checkpoint_final_binding", "COMPLETE checkpoint execution-cell registry and pair CSV coverage differ.")
        pair_output_value = final_binding.get("pair_output")
        if not isinstance(pair_output_value, str) or not pair_output_value:
            raise Stage2Stop("checkpoint_final_binding", "COMPLETE checkpoint pair_output path is missing.")
        pair_output_path = Path(pair_output_value).expanduser()
        if pair_output_path.is_symlink() or not pair_output_path.is_file():
            raise Stage2Stop("checkpoint_final_binding", "COMPLETE checkpoint pair_output is not a regular non-symlink file.")
        if started is not None:
            _check_runtime(started, cap, "before COMPLETE pair CSV integrity hash")
        if _sha256_file(pair_output_path) != final_binding.get("pair_output_sha256"):
            raise Stage2Stop("checkpoint_final_binding", "COMPLETE checkpoint pair_output hash does not match the bound CSV.")
        if started is not None:
            _check_runtime(started, cap, "after COMPLETE pair CSV integrity hash")
    first = checkpoint.get("first_seeds", {})
    second = checkpoint.get("second_seeds", {})
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise Stage2Stop("checkpoint_schema", "Checkpoint first_seeds/second_seeds must be objects.")
    allowed_keys = {str(seed) for seed in EXPECTED_SEEDS}
    first_keys = {str(key) for key in first}
    second_keys = {str(key) for key in second}
    if not first_keys.issubset(allowed_keys) or not second_keys.issubset(allowed_keys):
        raise Stage2Stop("checkpoint_seed", "Checkpoint contains a seed outside the frozen eight-seed registry.")
    if core is None and (first_keys or second_keys):
        # A caller that does not provide the frozen core must never turn a
        # persisted payload into a Q1/Q2/Q3 result merely from its outer hash.
        raise Stage2Stop("checkpoint_resume_core", "Resuming a non-empty checkpoint requires the frozen exp05_core validator.")
    if first_keys and sorted(first_keys, key=int) != [str(seed) for seed in EXPECTED_SEEDS[: len(first_keys)]]:
        raise Stage2Stop("checkpoint_seed", "First-phase checkpoint seeds are not the exact deterministic prefix already executed.")
    if second_keys and sorted(second_keys, key=int) != [str(seed) for seed in EXPECTED_SEEDS[: len(second_keys)]]:
        raise Stage2Stop("checkpoint_seed", "Second-phase checkpoint seeds are not the exact deterministic prefix already executed.")

    candidate = metadata.get("candidate_C")
    if not isinstance(candidate, list):
        raise Stage2Stop("checkpoint_metadata", "Checkpoint metadata candidate_C must be a list.")
    if status == "COMPLETE" and candidate:
        final_coverage = final_binding.get("coverage", {})
        expected_seed_ids = [str(seed) for seed in EXPECTED_SEEDS]
        if final_coverage.get("first_seed_ids") != expected_seed_ids or final_coverage.get("second_seed_ids") != expected_seed_ids:
            raise Stage2Stop("checkpoint_final_binding", "COMPLETE non-empty-candidate binding lacks all eight first/second seed cells.")
    if first_keys or second_keys:
        if core is None or started is None:
            raise Stage2Stop("checkpoint_resume_core", "Non-empty checkpoint validation requires core, started, and runtime cap context.")
        for key, row in first.items():
            if not isinstance(row, Mapping):
                raise Stage2Stop("checkpoint_seed", f"First-phase checkpoint row {key!r} is not an object.")
            declared_seed_hash = row.get("seed_payload_sha256")
            if not _is_sha256(declared_seed_hash) or declared_seed_hash != _seed_payload_hash(row):
                raise Stage2Stop("checkpoint_seed_hash", f"First-phase seed {key!r} lacks a valid canonical seed_payload_sha256.")
            _validate_first_seed_for_resume(row, seed=int(key), candidate=candidate, core=core, started=started, cap=cap)
        q1 = checkpoint.get("q1")
        tested_set = q1.get("tested_set") if isinstance(q1, Mapping) else None
        if second_keys and (not isinstance(q1, Mapping) or not isinstance(tested_set, list) or not tested_set):
            raise Stage2Stop("checkpoint_q1", "Second-phase checkpoint rows require a validated non-empty Q1 tested_set.")
        if isinstance(q1, Mapping) and first_keys == allowed_keys:
            expected_q1 = _q1_decision(candidate, [first[str(seed)] for seed in EXPECTED_SEEDS])
            for key in ("status", "selection_status", "tested_set", "qualifying_n", "curve"):
                if q1.get(key) != expected_q1.get(key):
                    raise Stage2Stop("checkpoint_q1", f"Checkpoint q1.{key} differs from recomputation over validated first-phase seeds.")
            selected_n = len(expected_q1.get("tested_set", []))
            q1_axis_statuses: list[str] = []
            for seed in EXPECTED_SEEDS:
                first_row = first[str(seed)]
                nested = (first_row.get("q1_nested", {}) or {}).get(str(selected_n), {})
                unresolved = first_row.get("q1_scientific_unresolved_reason") or (first_row.get("E_all_denominator_guard", {}) or {}).get("reason_code")
                expected_axis = "SCIENTIFIC_UNRESOLVED" if unresolved else ("PASS" if nested.get("seed_joint_pass") is True else "COMPLETED_FAIL")
                if first_row.get("q1_axis_status") != expected_axis or first_row.get("q1_axis_reason_code") != unresolved:
                    raise Stage2Stop("checkpoint_q1", f"Checkpoint seed {seed} q1 axis status differs from recomputation.")
                q1_axis_statuses.append(expected_axis)
            expected_aggregate = _aggregate_axis([{"axis_status": value} for value in q1_axis_statuses], axis="q1", core=core)
            if q1.get("aggregate") != expected_aggregate or q1.get("verdict") != expected_aggregate.get("status") or q1.get("label") != expected_aggregate.get("label"):
                raise Stage2Stop("checkpoint_q1", "Checkpoint q1 aggregate/verdict differs from recomputation over validated first-phase seeds.")
        for key, row in second.items():
            if not isinstance(row, Mapping):
                raise Stage2Stop("checkpoint_seed", f"Second-phase checkpoint row {key!r} is not an object.")
            declared_seed_hash = row.get("seed_payload_sha256")
            if not _is_sha256(declared_seed_hash) or declared_seed_hash != _seed_payload_hash(row):
                raise Stage2Stop("checkpoint_seed_hash", f"Second-phase seed {key!r} lacks a valid canonical seed_payload_sha256.")
            _validate_second_seed_for_resume(row, seed=int(key), first=first[str(key)], tested_set=tested_set, core=core, started=started, cap=cap)  # type: ignore[arg-type]
    candidate_nonempty = bool(candidate)
    if not candidate_nonempty:
        if status != "COMPLETE" or first_keys or second_keys:
            raise Stage2Stop("checkpoint_seed", "An empty frozen candidate may only have an empty COMPLETE checkpoint.")
    elif status == "FIRST_PHASE_COMPLETE" and (first_keys != allowed_keys or second_keys):
        raise Stage2Stop("checkpoint_seed", "FIRST_PHASE_COMPLETE requires all first-phase seeds and no second-phase seeds.")
    elif status == "SECOND_PHASE_RUNNING" and (first_keys != allowed_keys or not second_keys.issubset(allowed_keys)):
        raise Stage2Stop("checkpoint_seed", "SECOND_PHASE_RUNNING requires all first-phase seeds and a known second-phase subset.")
    elif status == "INCOMPLETE_RUNTIME_CAP" and second_keys and first_keys != allowed_keys:
        raise Stage2Stop("checkpoint_seed", "A runtime-cap checkpoint with second-phase seeds must contain all first-phase seeds.")
    elif status == "COMPLETE" and (first_keys != allowed_keys or second_keys != allowed_keys):
        raise Stage2Stop("checkpoint_seed", "COMPLETE nonempty-candidate checkpoint requires all eight first- and second-phase seeds.")
    elif status == "FIRST_PHASE_RUNNING" and second_keys:
        raise Stage2Stop("checkpoint_seed", "FIRST_PHASE_RUNNING cannot contain second-phase seeds.")
    if status == "COMPLETE":
        expected_pair_keys = _pair_csv_row_keys(_pair_csv_rows({"seeds": {str(seed): {"first": first.get(str(seed), {}), "second": second.get(str(seed), {})} for seed in EXPECTED_SEEDS}}))
        actual_pair_keys = _read_pair_csv_row_keys(Path(str(final_binding["pair_output"])).expanduser())
        if actual_pair_keys != expected_pair_keys or actual_pair_keys != list(final_binding.get("coverage", {}).get("observed_pair_csv_row_keys", [])):
            raise Stage2Stop("checkpoint_final_binding", "Bound pair CSV row-key set differs from validated checkpoint seed payloads.")
    prior_fe = sum(_sum_forward_equivalents(row.get("logical_forward_equivalents", {})) for row in first.values() if isinstance(row, Mapping))
    prior_fe += sum(
        _sum_forward_equivalents((row.get("q2", {}) or {}).get("logical_forward_equivalents", {}))
        + _sum_forward_equivalents((row.get("q3", {}) or {}).get("logical_forward_equivalents", {}))
        for row in second.values()
        if isinstance(row, Mapping)
    )
    attempt_runtime = checkpoint.get("attempt_runtime") if isinstance(checkpoint.get("attempt_runtime"), Mapping) else {}
    prior_wall = float(attempt_runtime.get("cumulative_wall_time_all_attempts", attempt_runtime.get("wall_clock_seconds", 0.0))) if isinstance(attempt_runtime.get("cumulative_wall_time_all_attempts", attempt_runtime.get("wall_clock_seconds", 0.0)), (int, float)) else 0.0
    return {
        "first_seeds": {str(key): value for key, value in first.items()},
        "second_seeds": {str(key): value for key, value in second.items()},
        "q1": checkpoint.get("q1"),
        "status": checkpoint.get("status", "RUNNING"),
        "diagnostic": {
            "checkpoint_status": status,
            "validated_first_seed_count": len(first),
            "validated_second_seed_count": len(second),
            "prior_incomplete_attempt_logical_fe": prior_fe if status != "COMPLETE" else 0,
            "prior_incomplete_attempt_wall_time": prior_wall if status != "COMPLETE" else 0.0,
            "scientific_reuse_allowed": False,
            "resumable_for_adjudication": False,
            "discarded_for_fresh_invocation": True,
        },
    }


def _write_checkpoint(path: Path, metadata: Mapping[str, Any], first: Mapping[str, Any], second: Mapping[str, Any], q1: Mapping[str, Any] | None, *, status: str, attempt_runtime: Mapping[str, Any] | None = None, final_binding: Mapping[str, Any] | None = None, started: float | None = None, cap: float | None = None) -> None:
    first_payloads: dict[str, Any] = {}
    second_payloads: dict[str, Any] = {}
    for key, value in first.items():
        item = _public_seed(value)
        item["seed_payload_sha256"] = _seed_payload_hash(item)
        first_payloads[str(key)] = item
    for key, value in second.items():
        item = _public_seed(value)
        item["seed_payload_sha256"] = _seed_payload_hash(item)
        second_payloads[str(key)] = item
    if started is not None:
        _check_runtime(started, cap, f"before checkpoint serialization {status}")
    payload = {"schema": CHECKPOINT_SCHEMA, "metadata": dict(metadata), "status": status, "scientific_reuse_allowed": False, "resumable_for_adjudication": False, "scientific_verdict_emitted": status == "COMPLETE", "first_seeds": first_payloads, "second_seeds": second_payloads, "q1": q1, "attempt_runtime": dict(attempt_runtime or {}), "final_binding": dict(final_binding or {})}
    payload["checkpoint_sha256"] = _sha256_bytes(_json_bytes(payload))
    _atomic_write_json(path, payload)
    if started is not None:
        _check_runtime(started, cap, f"after checkpoint serialization {status}")


def _aggregate_axis(rows: Sequence[Mapping[str, Any]], *, axis: str, core: CoreAdapter) -> dict[str, Any]:
    if len(rows) != len(EXPECTED_SEEDS):
        raise Stage2Stop("seed_aggregation", f"{axis} requires exactly eight per-seed rows, got {len(rows)}.")
    statuses = [str(row.get("axis_status", "")) for row in rows]
    try:
        payload = dict(core.aggregate(statuses))
    except Exception as exc:
        raise Stage2Stop("seed_aggregation", f"Frozen {axis} aggregation rejected its per-seed statuses: {exc}") from exc
    verdict = str(payload.get("verdict"))
    positive_label = {"q1": "Qualifying <=8-head set", "q2": "Number-specific under registered controls", "q3": "Subject-value transport shown"}.get(axis)
    negative_label = {"q1": "No qualifying <=8-head set", "q2": "Specificity bound not met", "q3": "Subject-value transport not shown"}.get(axis)
    payload.update({"axis": axis, "status": verdict, "label": positive_label if verdict == "POSITIVE" else negative_label if verdict == "NEGATIVE" else None, "per_seed_statuses": statuses})
    return payload


def _q3_direct_recovery_report(second: Mapping[str, Any]) -> dict[str, Any]:
    slots: list[dict[str, Any]] = []
    finite_values: list[float] = []
    for seed in EXPECTED_SEEDS:
        seed_payload = second.get(str(seed))
        q3 = seed_payload.get("q3") if isinstance(seed_payload, Mapping) else None
        entry = q3.get("direct_recovery") if isinstance(q3, Mapping) else None
        if isinstance(entry, Mapping):
            slot = {"seed": seed, "status": entry.get("status"), "value": entry.get("value")}
            if entry.get("value") is None:
                reason = entry.get("reason")
                if not isinstance(reason, Mapping) or not str(reason.get("code", "")) or not str(reason.get("detail", "")):
                    raise Stage2Stop("q3_direct_recovery", f"Seed {seed} null direct-recovery slot lacks a structured reason.")
                slot["reason"] = dict(reason)
            elif "reason" in entry:
                raise Stage2Stop("q3_direct_recovery", f"Seed {seed} finite direct-recovery slot must not contain a reason.")
        else:
            slot = {"seed": seed, "status": "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE", "value": None, "reason": {"code": "EXECUTION_INCOMPLETE", "detail": "The registered seed slot was not available in the Stage-2 result."}}
        value = slot.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            finite_values.append(float(value))
        else:
            slot["value"] = None
        slots.append(slot)
    ordered = sorted(finite_values)
    count = len(ordered)
    if count == 0:
        median = observed_min = observed_max = None
    else:
        middle = count // 2
        median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
        observed_min, observed_max = ordered[0], ordered[-1]
    return {"status": "Q3_DRF_NO_INFERENTIAL_INTERVAL", "seed_slots": slots, "summary": {"n_finite": count, "median": median, "observed_min": observed_min, "observed_max": observed_max}, "adjudicative": False}


def _q3_all_null_direct_recovery_report(reason_code: str, detail: str) -> dict[str, Any]:
    slots = [{"seed": seed, "status": "Q3_DRF_DESCRIPTIVE_VALUE_UNAVAILABLE", "value": None, "reason": {"code": reason_code, "detail": detail}} for seed in EXPECTED_SEEDS]
    return {"status": "Q3_DRF_NO_INFERENTIAL_INTERVAL", "seed_slots": slots, "summary": {"n_finite": 0, "median": None, "observed_min": None, "observed_max": None}, "adjudicative": False}


def _pair_csv_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_key, seed in (manifest.get("seeds") or {}).items():
        seed_int = int(seed_key)
        first = seed.get("first", {}) if isinstance(seed, Mapping) else {}
        for axis, arm_key in (("q1", "true_heads"), ("q1_source_a", "source_a_heads")):
            for head in first.get(arm_key, []):
                for record in head.get("pair_records", []):
                    rows.append({"seed": seed_int, "axis": axis, "arm": f"L{head.get('layer')}H{head.get('head')}", "pair_id": record.get("pair_id"), "direction": record.get("direction"), "value": record.get("effect", record.get("value"))})
        for axis, result_map in (("q2_true", (seed.get("second", {}).get("q2", {}) or {}).get("true_right")), ("q2_source_A", (seed.get("second", {}).get("q2", {}) or {}).get("source_A")), ("q2_source_C", (seed.get("second", {}).get("q2", {}) or {}).get("source_C")), ("q2_source_B", (seed.get("second", {}).get("q2", {}) or {}).get("source_B_descriptive")), ("q3_true", (seed.get("second", {}).get("q3", {}) or {}).get("true_path")), ("q3_source_A", (seed.get("second", {}).get("q3", {}) or {}).get("source_A_path"))):
            if not isinstance(result_map, Mapping):
                continue
            pair_ids = result_map.get("pair_ids", [])
            aligned = result_map.get("directed_sign_aligned", [])
            directions = result_map.get("directions", [])
            for index, value in enumerate(aligned):
                rows.append({"seed": seed_int, "axis": axis, "arm": "tested_set", "pair_id": pair_ids[index // 2] if index // 2 < len(pair_ids) else None, "direction": directions[index] if index < len(directions) else None, "value": value})
    return rows


def _pair_csv_row_keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        try:
            key = f"{int(row['seed'])}|{str(row['axis'])}|{str(row['arm'])}|{int(row['pair_id'])}|{str(row['direction'])}"
        except (KeyError, TypeError, ValueError) as exc:
            raise Stage2Stop("artifact_coverage", f"Pair CSV row lacks a canonical key: {row!r}") from exc
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise Stage2Stop("artifact_coverage", "Pair CSV contains duplicate execution-row keys.")
    return sorted(keys)


def _execution_cell_keys(seed_records: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for seed, record in seed_records.items():
        if isinstance(record, Mapping) and isinstance(record.get("first"), Mapping):
            keys.append(f"first:{seed}")
        if isinstance(record, Mapping) and isinstance(record.get("second"), Mapping):
            keys.append(f"second:{seed}")
    return sorted(keys)


def _read_pair_csv_row_keys(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["seed", "axis", "arm", "pair_id", "direction", "value"]:
                raise Stage2Stop("artifact_coverage", "Pair CSV header differs from the frozen row-key schema.")
            return _pair_csv_row_keys(list(reader))
    except OSError as exc:
        raise Stage2Stop("artifact_coverage", f"Cannot read bound pair CSV: {exc}") from exc


def _finalize_complete_artifacts(
    manifest: dict[str, Any],
    output_path: Path,
    pair_output_path: Path,
    checkpoint_path: Path | None,
    metadata: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    q1: Mapping[str, Any] | None,
    *,
    started: float,
    cap: float | None,
) -> None:
    """Commit complete artifacts in checkpoint -> CSV/hash -> manifest order."""

    _check_runtime(started, cap, "before final pair CSV planning")
    pair_rows = _pair_csv_rows(manifest)
    pair_csv_bytes = _csv_bytes(pair_rows, ("seed", "axis", "arm", "pair_id", "direction", "value"))
    pair_csv_hash = _sha256_bytes(pair_csv_bytes)
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise Stage2Stop("artifact_provenance", "Complete finalization lacks invocation provenance.")
    seed_records = manifest.get("seeds") or {}
    if not isinstance(seed_records, Mapping):
        raise Stage2Stop("artifact_coverage", "Complete finalization lacks a seed execution registry.")

    def _phase_seed_ids(phase: str) -> list[str]:
        return sorted(
            str(seed)
            for seed, record in seed_records.items()
            if isinstance(record, Mapping) and isinstance(record.get(phase), Mapping)
        )

    coverage_mode = "FULL_EXPECTED_STAGE2_CELLS"
    expected_execution_cell_keys = [f"{phase}:{seed}" for phase in ("first", "second") for seed in EXPECTED_SEEDS]
    observed_execution_cell_keys = _execution_cell_keys(seed_records)
    if observed_execution_cell_keys != sorted(expected_execution_cell_keys):
        raise Stage2Stop("artifact_coverage", "COMPLETE non-empty-candidate artifact does not contain exactly the expected first/second cells.")
    expected_pair_rows = _pair_csv_rows({"seeds": seed_records})
    expected_pair_row_keys = _pair_csv_row_keys(expected_pair_rows)
    observed_pair_row_keys = _pair_csv_row_keys(pair_rows)
    if observed_pair_row_keys != expected_pair_row_keys:
        raise Stage2Stop("artifact_coverage", "Pair CSV row-key set differs from the exact execution payload row-key set.")
    coverage = {
        "coverage_mode": coverage_mode,
        "expected_seeds": list(EXPECTED_SEEDS),
        "first_seed_ids": _phase_seed_ids("first"),
        "second_seed_ids": _phase_seed_ids("second"),
        "pair_csv_row_count": len(pair_rows),
        "expected_execution_cell_keys": sorted(expected_execution_cell_keys),
        "observed_execution_cell_keys": observed_execution_cell_keys,
        "expected_pair_csv_row_keys": expected_pair_row_keys,
        "observed_pair_csv_row_keys": observed_pair_row_keys,
    }
    registry = provenance.get("execution_cell_registry")
    if not isinstance(registry, dict):
        raise Stage2Stop("artifact_coverage", "Complete finalization lacks a mutable execution-cell registry.")
    registry["coverage_mode"] = coverage_mode
    registry["first_phase"] = list(coverage["first_seed_ids"])
    registry["second_phase"] = list(coverage["second_seed_ids"])
    registry["expected_execution_cell_keys"] = sorted(expected_execution_cell_keys)
    registry["observed_execution_cell_keys"] = observed_execution_cell_keys
    registry["pair_csv_row_count"] = len(pair_rows)
    registry["expected_pair_csv_row_keys"] = expected_pair_row_keys
    registry["observed_pair_csv_row_keys"] = observed_pair_row_keys
    registry["registered_cell_count"] = len(expected_execution_cell_keys)
    registry["executed_cell_count"] = len(observed_execution_cell_keys)
    registry["coverage_consistent"] = observed_execution_cell_keys == sorted(expected_execution_cell_keys) and observed_pair_row_keys == expected_pair_row_keys
    final_binding = {
        "invocation_id": provenance.get("invocation_id"),
        "invocation_config_sha256": provenance.get("invocation_config_sha256"),
        "commit": provenance.get("commit"),
        "protocol_sha256": provenance.get("protocol_sha256"),
        "calibration_sha256": provenance.get("calibration_sha256"),
        "selection_sha256": provenance.get("selection_sha256"),
        "candidate_file_sha256": provenance.get("candidate_file_sha256"),
        "protocol_file_sha256": provenance.get("protocol_file_sha256"),
        "protocol_canonical_sha256": provenance.get("protocol_canonical_sha256"),
        "state_dict_sha256_before": provenance.get("state_dict_sha256_before"),
        "state_dict_sha256_after": provenance.get("state_dict_sha256_after"),
        "state_dict_fingerprint_before": provenance.get("state_dict_fingerprint_before"),
        "state_dict_fingerprint_after": provenance.get("state_dict_fingerprint_after"),
        "execution_cell_registry": dict(registry) if isinstance(registry, Mapping) else None,
        "pair_output": str(pair_output_path),
        "pair_output_sha256": pair_csv_hash,
        "coverage": coverage,
    }
    provenance["pair_output"] = str(pair_output_path)
    provenance["pair_output_sha256"] = pair_csv_hash
    provenance["pair_output_row_count"] = len(pair_rows)
    provenance["final_binding"] = final_binding  # type: ignore[index]
    manifest["final_binding"] = final_binding
    manifest["execution_cell_registry"] = dict(registry) if isinstance(registry, Mapping) else None
    manifest["coverage_mode"] = coverage_mode
    manifest["pair_output"] = str(pair_output_path)
    manifest["pair_output_sha256"] = pair_csv_hash
    manifest["runtime"]["logical_forward_equivalents"] = int(manifest["runtime"].get("final_complete_invocation_logical_fe", 0))
    manifest["runtime"]["final_complete_invocation_wall_time"] = time.perf_counter() - started
    manifest["runtime"]["cumulative_logical_fe_all_attempts"] = int(manifest["runtime"].get("prior_incomplete_attempt_logical_fe", 0)) + int(manifest["runtime"].get("final_complete_invocation_logical_fe", 0))
    manifest["runtime"]["cumulative_wall_time_all_attempts"] = float(manifest["runtime"].get("prior_incomplete_attempt_wall_time", 0.0)) + float(manifest["runtime"]["final_complete_invocation_wall_time"])
    if checkpoint_path is None:
        raise Stage2Stop("checkpoint_required", "A non-empty Stage-2 COMPLETE artifact requires --checkpoint.")
    _write_checkpoint(
        checkpoint_path,
        metadata,
        first,
        second,
        q1,
        status="COMPLETE",
        attempt_runtime=manifest["runtime"],
        final_binding=final_binding,
        started=started,
        cap=cap,
    )
    checkpoint_resolved = str(checkpoint_path.expanduser().resolve(strict=False))
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    manifest["provenance"]["checkpoint"] = checkpoint_resolved
    manifest["provenance"]["checkpoint_sha256"] = checkpoint_sha256
    manifest["checkpoint"] = checkpoint_resolved
    manifest["checkpoint_sha256"] = checkpoint_sha256
    manifest["checkpoint_written"] = True
    _check_runtime(started, cap, "before final pair CSV serialization")
    _atomic_write_csv(pair_output_path, pair_rows, ("seed", "axis", "arm", "pair_id", "direction", "value"))
    _check_runtime(started, cap, "after final pair CSV serialization")
    _check_runtime(started, cap, "before final pair CSV hash")
    if _sha256_file(pair_output_path) != pair_csv_hash:
        raise Stage2Stop("artifact_pair_csv_hash", "Final pair CSV bytes differ from the pre-bound checkpoint hash.")
    _check_runtime(started, cap, "after final pair CSV hash")
    manifest["provenance"]["cross_invocation_scientific_reuse_allowed"] = False
    manifest["provenance"]["scientific_reuse_allowed"] = False
    manifest["provenance"]["resumable_for_adjudication"] = False
    manifest["provenance"]["scientific_verdict_emitted"] = True
    manifest["provenance"]["supplies_published_science"] = True
    manifest["cross_invocation_scientific_reuse_allowed"] = False
    manifest["scientific_reuse_allowed"] = False
    manifest["resumable_for_adjudication"] = False
    manifest["scientific_verdict_emitted"] = True
    manifest["supplies_published_science"] = True
    manifest["status_code"] = "STAGE2_COMPLETE_SINGLE_INVOCATION"
    manifest["final_binding_status"] = "STAGE2_FINAL_ARTIFACTS_BOUND"
    manifest["coverage_status"] = "STAGE2_PAIR_CSV_CHECKPOINT_COVERAGE_MATCH"
    _check_runtime(started, cap, "before final COMPLETE manifest serialization")
    _atomic_write_json(output_path, manifest)
    _check_runtime(started, cap, "after final COMPLETE manifest serialization")


def _base_manifest(*, args: argparse.Namespace, commit: str, protocol_path: Path, calibration_path: Path, selection_path: Path, candidate_path: Path, hashes: Mapping[str, str], candidate: Sequence[Mapping[str, int]], candidate_info: Mapping[str, Any], started: float, allowed_dirty_paths: Sequence[str]) -> dict[str, Any]:
    invocation_nonce = f"{time.time_ns()}:{os.getpid()}"
    invocation_id = _sha256_bytes(_json_bytes({"nonce": invocation_nonce, "commit": commit, "protocol_sha256": hashes["protocol"], "candidate_sha256": hashes["candidate"]}))
    config_hash = _sha256_bytes(_json_bytes({"commit": commit, "protocol_sha256": hashes["protocol"], "calibration_sha256": hashes["calibration"], "selection_sha256": hashes["selection"], "candidate_sha256": hashes["candidate"], "command": list(sys.argv), "max_wall_seconds": args.max_wall_seconds}))
    return {"schema": STAGE2_SCHEMA, "status": "RUNNING", "artifact_kind": "stage2_q1_q3", "candidate_C": [dict(row) for row in candidate], "tested_set": None, "q1": None, "q2": None, "q3": None, "seeds": {}, "scientific_verdict_emitted": False, "provenance": {"invocation_id": invocation_id, "invocation_config_sha256": config_hash, "cross_invocation_scientific_reuse_allowed": False, "scientific_reuse_allowed": False, "resumable_for_adjudication": False, "scientific_verdict_emitted": False, "commit": commit, "protocol": str(protocol_path), "protocol_sha256": hashes["protocol"], "protocol_file_sha256": hashes.get("protocol_file"), "protocol_canonical_sha256": hashes.get("protocol_canonical"), "calibration": str(calibration_path), "calibration_sha256": hashes["calibration"], "selection": str(selection_path), "selection_sha256": hashes["selection"], "candidate": str(candidate_path), "candidate_file_sha256": hashes["candidate"], "candidate_info": dict(candidate_info), "shipped_stage1_role": "descriptive historical cross-check inside selection only; not a Stage-2 input or gate", "expected_git_commit": args.expected_git_commit, "require_clean_tree": True, "dirty": False, "git_status": "clean" if not allowed_dirty_paths else "clean_except_declared_runtime_artifacts", "declared_runtime_artifacts_present": list(allowed_dirty_paths), "core_api": CORE_API_VERSION, "execution_cell_registry": {"expected_seeds": list(EXPECTED_SEEDS), "first_phase": [], "second_phase": [], "science_source": "this_invocation_only"}, "command": list(sys.argv), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__}, "runtime": {"started_unix": time.time(), "max_wall_seconds": args.max_wall_seconds, "device": "cpu", "dtype": "float32", "logical_forward_equivalents": 0, "final_complete_invocation_logical_fe": 0, "prior_incomplete_attempt_logical_fe": 0, "cumulative_logical_fe_all_attempts": 0, "final_complete_invocation_wall_time": 0.0, "prior_incomplete_attempt_wall_time": 0.0, "cumulative_wall_time_all_attempts": 0.0, "wall_clock_seconds": 0.0}, "error": None, "failed_gate": None}


def _startup_running_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": STAGE2_SCHEMA,
        "status": "RUNNING",
        "artifact_kind": "stage2_q1_q3",
        "cross_invocation_scientific_reuse_allowed": False,
        "scientific_reuse_allowed": False,
        "resumable_for_adjudication": False,
        "scientific_verdict_emitted": False,
        "q1": None,
        "q2": None,
        "q3": {"status": "EXECUTION_INCOMPLETE", "direct_recovery_report": _q3_all_null_direct_recovery_report("EXECUTION_INCOMPLETE", "Stage-2 invocation has not reached a final artifact.")},
        "tested_set": None,
        "seeds": {},
        "runtime": {"started_unix": time.time(), "max_wall_seconds": args.max_wall_seconds, "wall_clock_seconds": 0.0},
        "error": None,
        "failed_gate": None,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    protocol_path = _resolve_path(args.protocol)
    calibration_path = _resolve_path(args.calibration)
    selection_path = _resolve_path(args.selection)
    candidate_path = _resolve_path(args.candidate)
    output_path = _resolve_path(args.output)
    pair_output_path = _resolve_path(args.pair_output)
    checkpoint_path = _resolve_path(args.checkpoint) if args.checkpoint else None
    raw_runtime_values = [args.output, args.pair_output, *([args.checkpoint] if args.checkpoint else [])]
    raw_runtime_paths = [Path(value).expanduser() if Path(value).expanduser().is_absolute() else HERE / Path(value).expanduser() for value in raw_runtime_values]
    if any(path.is_symlink() for path in raw_runtime_paths):
        return {"schema": STAGE2_SCHEMA, "status": "STOPPED", "failed_gate": "runtime_path_symlink", "error": "Runtime artifact paths must not be symbolic links."}, 2
    input_paths = {protocol_path, calibration_path, selection_path, candidate_path}
    runtime_paths = [output_path, pair_output_path, *([checkpoint_path] if checkpoint_path else [])]
    if len(set(runtime_paths)) != len(runtime_paths) or any(path in input_paths for path in runtime_paths):
        return {"schema": STAGE2_SCHEMA, "status": "STOPPED", "failed_gate": "runtime_path_alias", "error": "Output, pair-output, and checkpoint must be mutually distinct and disjoint from all immutable inputs."}, 2
    invalid_existing = [str(path) for path in runtime_paths if path.is_symlink() or (path.exists() and not path.is_file())]
    if invalid_existing:
        return {"schema": STAGE2_SCHEMA, "status": "STOPPED", "failed_gate": "runtime_path_type", "error": "Existing runtime paths must be non-symlink regular files: " + ", ".join(invalid_existing)}, 2
    manifest: dict[str, Any] | None = None
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    q1: dict[str, Any] | None = None
    try:
        # Invalidate any old COMPLETE manifest before reading inputs or model
        # state.  Every later failure is therefore represented by a tombstone.
        manifest = _startup_running_manifest(args)
        _atomic_write_json(output_path, manifest)
        if not args.expected_git_commit:
            raise Stage2Stop("expected_git_commit_required", "--expected-git-commit is mandatory for a Stage-2 run.")
        if not args.require_clean_tree:
            raise Stage2Stop("require_clean_tree", "--require-clean-tree is mandatory; a dirty COMPLETE Stage-2 run is forbidden.")
        if args.max_wall_seconds is None or not math.isfinite(float(args.max_wall_seconds)) or float(args.max_wall_seconds) <= 0.0:
            raise Stage2Stop("max_wall_seconds", "--max-wall-seconds is mandatory and must be a finite positive number.")
        protocol = _read_json(protocol_path, "protocol")
        calibration = _read_json(calibration_path, "calibration")
        selection_payload = _read_json(selection_path, "selection")
        candidate_payload = _read_json(candidate_path, "candidate")
        _validate_protocol(protocol, protocol_path)
        theta = _validate_calibration(calibration, calibration_path)
        candidate_status, candidate_rows, candidate_info = _candidate_records(candidate_payload, candidate_path)
        if candidate_rows and checkpoint_path is None:
            raise Stage2Stop("checkpoint_required", "--checkpoint is mandatory for a non-empty Stage-2 adjudication run.")
        hashes = {"protocol": _sha256_file(protocol_path), "protocol_canonical": _sha256_bytes(_json_bytes(protocol)), "protocol_file": _sha256_file(protocol_path), "calibration": _sha256_file(calibration_path), "selection": _sha256_file(selection_path), "candidate": _sha256_file(candidate_path)}
        if candidate_info.get("protocol_sha256") != hashes["protocol"]:
            raise Stage2Stop("candidate_protocol_hash", "Candidate protocol_sha256 differs from current protocol file.")
        commit = _git(["rev-parse", "HEAD"])
        if str(args.expected_git_commit) != commit:
            raise Stage2Stop("git_revision", f"Expected commit {args.expected_git_commit}, running {commit}.")
        allowed_dirty_paths = _require_clean_tree_except_runtime_artifacts([output_path, pair_output_path, *([checkpoint_path] if checkpoint_path else [])])
        core = CoreAdapter()
        selection_dependency = _validate_selection_dependency(selection_payload, candidate_payload, protocol, core, selection_file_hash=hashes["selection"], expected_commit=commit)
        manifest = _base_manifest(args=args, commit=commit, protocol_path=protocol_path, calibration_path=calibration_path, selection_path=selection_path, candidate_path=candidate_path, hashes=hashes, candidate=candidate_rows, candidate_info=candidate_info, started=started, allowed_dirty_paths=allowed_dirty_paths)
        manifest["candidate_status"] = candidate_status
        selection_dependency["candidate_manifest_sha256"] = hashes["candidate"]
        selection_dependency["candidate_count"] = len(candidate_rows)
        manifest["provenance"]["selection_dependency"] = dict(selection_dependency)
        manifest["provenance"]["selection_provenance_sha256"] = selection_dependency.get("selection_provenance_sha256")
        manifest["provenance"]["candidate_manifest_sha256"] = hashes["candidate"]
        metadata = _checkpoint_metadata(commit=commit, protocol_hash=hashes["protocol"], calibration_hash=hashes["calibration"], selection_hash=hashes["selection"], candidate_hash=hashes["candidate"], candidate=candidate_rows)
        if candidate_rows and checkpoint_path and checkpoint_path.exists():
            try:
                loaded = _load_checkpoint(checkpoint_path, metadata, core=core, started=started, cap=args.max_wall_seconds)
            except RuntimeCapStop as exc:
                # Validation itself is not a resumable execution phase.  Do
                # not overwrite the last known-good checkpoint with an empty
                # tombstone if its deterministic re-derivation hits the cap.
                raise Stage2Stop("checkpoint_resume_cap", str(exc)) from exc
            diagnostic = loaded.get("diagnostic", {})
            manifest["provenance"]["prior_checkpoint_diagnostic"] = diagnostic
            manifest["runtime"]["prior_incomplete_attempt_logical_fe"] = int(diagnostic.get("prior_incomplete_attempt_logical_fe", 0))
            manifest["runtime"]["prior_incomplete_attempt_wall_time"] = float(diagnostic.get("prior_incomplete_attempt_wall_time", 0.0))
            # Amendment 8: a prior/interrupted checkpoint is integrity-only
            # evidence.  Never reuse or merge its scientific rows.
            first = {}
            second = {}
            q1 = None
        if not candidate_rows:
            empty_registry = {"coverage_mode": "VALID_COMPLETED_EMPTY_C_NO_STAGE2_CELLS", "expected_execution_cell_keys": [], "observed_execution_cell_keys": [], "registered_cell_count": 0, "executed_cell_count": 0, "coverage_consistent": True, "science_source": "selection_dependency_only"}
            manifest.update({"status": "NOT_INSTANTIATED_VALID_EMPTY_C", "status_code": "NOT_INSTANTIATED_VALID_EMPTY_C", "stage2_instantiation_status": "NOT_INSTANTIATED_VALID_EMPTY_C", "coverage_mode": "VALID_COMPLETED_EMPTY_C_NO_STAGE2_CELLS", "scientific_verdict": None, "scientific_verdict_emitted": False, "supplies_published_science": False, "scientific_csv_published": False, "checkpoint_written": False, "q1": {"status": "COMPLETE_NO_CANDIDATES", "outer_code": "Q1_COMPLETE_NO_CANDIDATES", "subtype": "EMPTY_UNDER_FROZEN_RULE", "label": None, "tested_set": [], "public_sentence": "No candidates were selected under the frozen source-A selection rule."}, "q2": {"status": "NOT_INSTANTIATED_NO_TESTED_SET", "label": None}, "q3": {"status": "NOT_INSTANTIATED_NO_TESTED_SET", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("NOT_INSTANTIATED_NO_TESTED_SET", "No tested head set exists because the frozen candidate selection was empty.")}, "tested_set": [], "stale_pair_output_is_not_bound": True, "execution_cell_registry": empty_registry, "runtime": {**manifest["runtime"], "logical_forward_equivalents": 0, "final_complete_invocation_logical_fe": 0, "prior_incomplete_attempt_logical_fe": 0, "cumulative_logical_fe_all_attempts": 0, "wall_clock_seconds": time.perf_counter() - started}})
            manifest["provenance"]["coverage_mode"] = "VALID_COMPLETED_EMPTY_C_NO_STAGE2_CELLS"
            manifest["provenance"]["stage2_instantiation_status"] = "NOT_INSTANTIATED_VALID_EMPTY_C"
            manifest["provenance"]["scientific_verdict"] = None
            manifest["provenance"]["scientific_csv_published"] = False
            manifest["provenance"]["checkpoint_written"] = False
            manifest["provenance"]["execution_cell_registry"] = empty_registry
            _atomic_write_json(output_path, manifest)
            return manifest, 0
        _check_runtime(started, args.max_wall_seconds, "before model determinism setup")
        set_determinism(SELECTION_SEED)
        _check_runtime(started, args.max_wall_seconds, "before model load")
        model = load_model()
        _check_runtime(started, args.max_wall_seconds, "after model load")
        _check_runtime(started, args.max_wall_seconds, "before model provenance validation")
        manifest["provenance"]["model"] = _validate_model_provenance(model, protocol.get("model", {}))
        _check_runtime(started, args.max_wall_seconds, "after model provenance validation")
        state_before_record = _model_state_fingerprint(model, started=started, cap=args.max_wall_seconds, label="before Stage-2")
        state_before = str(state_before_record["sha256"])
        manifest["provenance"]["state_dict_sha256_before"] = state_before
        manifest["provenance"]["state_dict_fingerprint_before"] = state_before_record
        selection_state_sha = manifest["provenance"].get("selection_dependency", {}).get("state_dict_sha256_before") if isinstance(manifest["provenance"].get("selection_dependency"), Mapping) else None
        if state_before != selection_state_sha:
            raise Stage2Stop("selection_state_fingerprint", "Stage-2 model state before sweeps differs from the validated selection snapshot.")
        for seed in EXPECTED_SEEDS:
            _check_runtime(started, args.max_wall_seconds, f"before first-phase seed {seed}")
            key = str(seed)
            if key not in first:
                first[key] = _seed_first_phase(model, seed=seed, candidate=candidate_rows, started=started, cap=args.max_wall_seconds, core=core)
                if checkpoint_path:
                    _write_checkpoint(checkpoint_path, metadata, first, second, q1, status="FIRST_PHASE_RUNNING", attempt_runtime=manifest.get("runtime", {}), started=started, cap=args.max_wall_seconds)
            manifest["seeds"][key] = {"first": _public_seed(first[key])}
        q1 = _q1_decision(candidate_rows, [first[str(seed)] for seed in EXPECTED_SEEDS])
        # Q1 status is assigned against the selected n (or the fallback set).
        if q1.get("status") == "COMPLETE_NO_CANDIDATES":
            q1_axis_rows = []
        else:
            selected_n = len(q1.get("tested_set", []))
            q1_axis_rows = []
            for seed in EXPECTED_SEEDS:
                row = first[str(seed)]
                nested = (row.get("q1_nested", {}) or {}).get(str(selected_n), {})
                unresolved_reason = row.get("q1_scientific_unresolved_reason") or (row.get("E_all_denominator_guard", {}) or {}).get("reason_code")
                row["q1_axis_status"] = "SCIENTIFIC_UNRESOLVED" if unresolved_reason else ("PASS" if nested.get("seed_joint_pass") else "COMPLETED_FAIL")
                row["q1_axis_reason_code"] = unresolved_reason
                q1_axis_rows.append({"axis_status": row["q1_axis_status"]})
                manifest["seeds"][str(seed)]["first"] = _public_seed(row)
            q1["aggregate"] = _aggregate_axis(q1_axis_rows, axis="q1", core=core)
            q1["verdict"] = q1["aggregate"]["status"]
            q1["label"] = q1["aggregate"]["label"]
        manifest["q1"] = q1
        manifest["tested_set"] = list(q1.get("tested_set", []))
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, metadata, first, second, q1, status="FIRST_PHASE_COMPLETE", attempt_runtime=manifest.get("runtime", {}), started=started, cap=args.max_wall_seconds)
        tested_set = list(q1.get("tested_set", []))
        if not tested_set:
            state_after_record = _model_state_fingerprint(model, started=started, cap=args.max_wall_seconds, label="after Stage-2")
            state_after = str(state_after_record["sha256"])
            manifest["provenance"]["state_dict_sha256_after"] = state_after
            manifest["provenance"]["state_dict_fingerprint_after"] = state_after_record
            if state_after != state_before:
                raise Stage2Stop("model_state_changed", "Model state fingerprint changed during the Stage-2 invocation.")
            manifest.update({"status": "COMPLETE", "q2": {"status": "NOT_INSTANTIATED_NO_TESTED_SET", "label": None}, "q3": {"status": "NOT_INSTANTIATED_NO_TESTED_SET", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("NOT_INSTANTIATED_NO_TESTED_SET", "No valid tested head set was instantiated for downstream Q3.")}, "runtime": {**manifest["runtime"], "wall_clock_seconds": time.perf_counter() - started}})
            manifest["runtime"]["final_complete_invocation_logical_fe"] = sum(_sum_forward_equivalents(first[str(seed)].get("logical_forward_equivalents", {})) for seed in EXPECTED_SEEDS)
            _finalize_complete_artifacts(manifest, output_path, pair_output_path, checkpoint_path, metadata, first, second, q1, started=started, cap=args.max_wall_seconds)
            return manifest, 0
        for seed in EXPECTED_SEEDS:
            _check_runtime(started, args.max_wall_seconds, f"before second-phase seed {seed}")
            key = str(seed)
            if key not in second:
                second[key] = _seed_second_phase(model, seed=seed, first=first[key], tested_set=tested_set, started=started, cap=args.max_wall_seconds, core=core, theta=theta)
                if checkpoint_path:
                    _write_checkpoint(checkpoint_path, metadata, first, second, q1, status="SECOND_PHASE_RUNNING", attempt_runtime=manifest.get("runtime", {}), started=started, cap=args.max_wall_seconds)
            manifest["seeds"][key] = {"first": _public_seed(first[key]), "second": _public_seed(second[key])}
        state_after_record = _model_state_fingerprint(model, started=started, cap=args.max_wall_seconds, label="after Stage-2")
        state_after = str(state_after_record["sha256"])
        manifest["provenance"]["state_dict_sha256_after"] = state_after
        manifest["provenance"]["state_dict_fingerprint_after"] = state_after_record
        if state_after != state_before:
            raise Stage2Stop("model_state_changed", "Model state fingerprint changed during the Stage-2 invocation.")
        q2_rows: list[dict[str, Any]] = []
        q3_rows: list[dict[str, Any]] = []
        for seed in EXPECTED_SEEDS:
            sec = second[str(seed)]
            q2_result = sec.get("q2", {})
            q3_result = sec.get("q3", {})
            q2_axis_status = str(q2_result.get("axis_status", ""))
            q3_axis_status = str(q3_result.get("axis_status", ""))
            q2_rows.append({"axis_status": q2_axis_status})
            q3_rows.append({"axis_status": q3_axis_status})
        manifest["q2"] = _aggregate_axis(q2_rows, axis="q2", core=core)
        manifest["q3"] = _aggregate_axis(q3_rows, axis="q3", core=core)
        manifest["q3"]["direct_recovery_report"] = _q3_direct_recovery_report(second)
        first_fe = sum(_sum_forward_equivalents(first[str(seed)].get("logical_forward_equivalents", {})) for seed in EXPECTED_SEEDS)
        second_fe = sum(
            _sum_forward_equivalents(second[str(seed)].get("q2", {}).get("logical_forward_equivalents", {}))
            + _sum_forward_equivalents(second[str(seed)].get("q3", {}).get("logical_forward_equivalents", {}))
            for seed in EXPECTED_SEEDS
        )
        manifest["runtime"]["final_complete_invocation_logical_fe"] = first_fe + second_fe
        manifest["runtime"]["logical_forward_equivalents"] = first_fe + second_fe
        manifest["runtime"]["wall_clock_seconds"] = time.perf_counter() - started
        manifest["status"] = "COMPLETE"
        _finalize_complete_artifacts(manifest, output_path, pair_output_path, checkpoint_path, metadata, first, second, q1, started=started, cap=args.max_wall_seconds)
        return manifest, 0
    except RuntimeCapStop as exc:
        if manifest is None:
            manifest = {"schema": STAGE2_SCHEMA, "status": exc.status, "status_code": "STAGE2_EXECUTION_INCOMPLETE_RUNTIME_CAP", "execution_status": "EXECUTION_INCOMPLETE", "cross_invocation_scientific_reuse_allowed": False, "scientific_reuse_allowed": False, "resumable_for_adjudication": False, "scientific_verdict_emitted": False, "supplies_published_science": False, "failed_gate": exc.gate, "error": str(exc), "q3": {"status": "EXECUTION_INCOMPLETE", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("EXECUTION_INCOMPLETE", "Stage-2 stopped before the Q3 seed slots could be executed.")}}
        else:
            manifest["status"] = exc.status
            manifest["status_code"] = "STAGE2_EXECUTION_INCOMPLETE_RUNTIME_CAP"
            manifest["execution_status"] = "EXECUTION_INCOMPLETE"
            manifest["cross_invocation_scientific_reuse_allowed"] = False
            manifest["scientific_reuse_allowed"] = False
            manifest["resumable_for_adjudication"] = False
            manifest["scientific_verdict_emitted"] = False
            manifest["supplies_published_science"] = False
            for stale_key in ("final_binding", "pair_output", "pair_output_sha256"):
                manifest.pop(stale_key, None)
            provenance = manifest.get("provenance")
            if isinstance(provenance, Mapping):
                provenance["cross_invocation_scientific_reuse_allowed"] = False
                provenance["scientific_reuse_allowed"] = False
                provenance["resumable_for_adjudication"] = False
                provenance["scientific_verdict_emitted"] = False
                provenance["supplies_published_science"] = False
                for stale_key in ("final_binding", "pair_output", "pair_output_sha256", "pair_output_row_count"):
                    provenance.pop(stale_key, None)
            manifest["failed_gate"] = exc.gate
            manifest["error"] = str(exc)
            manifest["q1"] = None
            manifest["q2"] = None
            manifest["q3"] = {"status": "EXECUTION_INCOMPLETE", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("EXECUTION_INCOMPLETE", "The authorized Stage-2 run stopped at its declared runtime cap.")}
            manifest["tested_set"] = None
            elapsed = time.perf_counter() - started
            partial_fe = sum(_sum_forward_equivalents(row.get("logical_forward_equivalents", {})) for row in first.values() if isinstance(row, Mapping))
            partial_fe += sum(_sum_forward_equivalents((row.get("q2", {}) or {}).get("logical_forward_equivalents", {})) + _sum_forward_equivalents((row.get("q3", {}) or {}).get("logical_forward_equivalents", {})) for row in second.values() if isinstance(row, Mapping))
            prior_fe = int(manifest["runtime"].get("prior_incomplete_attempt_logical_fe", 0))
            prior_wall = float(manifest["runtime"].get("prior_incomplete_attempt_wall_time", 0.0))
            manifest["runtime"]["logical_forward_equivalents"] = partial_fe
            manifest["runtime"]["current_incomplete_attempt_logical_fe"] = partial_fe
            manifest["runtime"]["current_incomplete_attempt_wall_time"] = elapsed
            manifest["runtime"]["final_complete_invocation_logical_fe"] = 0
            manifest["runtime"]["final_complete_invocation_wall_time"] = 0.0
            manifest["runtime"]["cumulative_logical_fe_all_attempts"] = prior_fe + partial_fe
            manifest["runtime"]["cumulative_wall_time_all_attempts"] = prior_wall + elapsed
            manifest["runtime"]["attempt_ledger"] = [{"invocation_id": manifest.get("provenance", {}).get("invocation_id") if isinstance(manifest.get("provenance"), Mapping) else None, "status": "INCOMPLETE_RUNTIME_CAP", "logical_forward_equivalents": partial_fe, "wall_time_seconds": elapsed, "first_seed_count": len(first), "second_seed_count": len(second)}]
            manifest["runtime"]["wall_clock_seconds"] = elapsed
        if checkpoint_path and manifest is not None:
            # A runtime-cap checkpoint is resumable state only; no partial
            # Q1/Q2/Q3 verdict is written to the result artifact.
            try:
                _write_checkpoint(checkpoint_path, metadata, first, second, q1, status=exc.status, attempt_runtime=manifest.get("runtime", {}))  # type: ignore[name-defined]
            except Exception as checkpoint_exc:
                manifest["checkpoint_error"] = f"Runtime-cap checkpoint write failed: {checkpoint_exc}"
        if manifest is not None:
            _atomic_write_json(output_path, manifest)
        return manifest or {"status": exc.status, "error": str(exc)}, 2
    except Stage2Stop as exc:
        if manifest is None:
            manifest = {"schema": STAGE2_SCHEMA, "status": exc.status, "status_code": "STAGE2_EXECUTION_INCOMPLETE_ARTIFACT_MISMATCH", "execution_status": "EXECUTION_INCOMPLETE", "cross_invocation_scientific_reuse_allowed": False, "scientific_reuse_allowed": False, "resumable_for_adjudication": False, "scientific_verdict_emitted": False, "supplies_published_science": False, "failed_gate": exc.gate, "error": str(exc), "q3": {"status": "EXECUTION_INCOMPLETE", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("EXECUTION_INCOMPLETE", "Stage-2 stopped before the Q3 seed slots could be executed.")}}
            if exc.gate.startswith("candidate") or exc.gate.startswith("selection"):
                manifest.update({"q1": {"status": "BLOCKED", "outer_code": "Q1_BLOCKED", "subtype": "UNRESOLVED_C", "label": None, "reason": "UNRESOLVED_C"}, "q2": {"status": "NOT_INSTANTIATED_UNRESOLVED_C", "label": None}, "q3": {"status": "NOT_INSTANTIATED_UNRESOLVED_C", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("EXECUTION_INCOMPLETE", "Candidate C was not validly frozen, so Q3 was not instantiated.")}, "tested_set": None})
        else:
            manifest["status"] = exc.status
            manifest["status_code"] = "STAGE2_EXECUTION_INCOMPLETE_ARTIFACT_MISMATCH"
            manifest["execution_status"] = "EXECUTION_INCOMPLETE"
            manifest["cross_invocation_scientific_reuse_allowed"] = False
            manifest["scientific_reuse_allowed"] = False
            manifest["resumable_for_adjudication"] = False
            manifest["scientific_verdict_emitted"] = False
            manifest["supplies_published_science"] = False
            for stale_key in ("final_binding", "pair_output", "pair_output_sha256"):
                manifest.pop(stale_key, None)
            provenance = manifest.get("provenance")
            if isinstance(provenance, Mapping):
                provenance["cross_invocation_scientific_reuse_allowed"] = False
                provenance["scientific_reuse_allowed"] = False
                provenance["resumable_for_adjudication"] = False
                provenance["scientific_verdict_emitted"] = False
                provenance["supplies_published_science"] = False
                for stale_key in ("final_binding", "pair_output", "pair_output_sha256", "pair_output_row_count"):
                    provenance.pop(stale_key, None)
            manifest["failed_gate"] = exc.gate
            manifest["error"] = str(exc)
            manifest["q1"] = None
            manifest["q2"] = None
            manifest["q3"] = {"status": "EXECUTION_INCOMPLETE", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("EXECUTION_INCOMPLETE", "Stage-2 failed an execution, artifact, hash, or provenance gate.")}
            manifest["tested_set"] = None
            manifest["runtime"]["wall_clock_seconds"] = time.perf_counter() - started
            if exc.gate.startswith("candidate") or exc.gate.startswith("selection"):
                manifest.update({"q1": {"status": "BLOCKED", "outer_code": "Q1_BLOCKED", "subtype": "UNRESOLVED_C", "label": None, "reason": "UNRESOLVED_C"}, "q2": {"status": "NOT_INSTANTIATED_UNRESOLVED_C", "label": None}, "q3": {"status": "NOT_INSTANTIATED_UNRESOLVED_C", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("EXECUTION_INCOMPLETE", "Candidate C was not validly frozen, so Q3 was not instantiated.")}, "tested_set": None})
        if manifest is not None:
            _atomic_write_json(output_path, manifest)
        return manifest, 2
    except Exception as exc:
        tombstone = {
            "schema": STAGE2_SCHEMA,
            "status": "STOPPED",
            "status_code": "STAGE2_EXECUTION_INCOMPLETE_INTERRUPTED",
            "execution_status": "EXECUTION_INCOMPLETE",
            "cross_invocation_scientific_reuse_allowed": False,
            "scientific_reuse_allowed": False,
            "resumable_for_adjudication": False,
            "scientific_verdict_emitted": False,
            "supplies_published_science": False,
            "failed_gate": "unexpected_execution_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "q1": None,
            "q2": None,
            "q3": {"status": "EXECUTION_INCOMPLETE", "label": None, "direct_recovery_report": _q3_all_null_direct_recovery_report("EXECUTION_INCOMPLETE", "An unexpected Stage-2 execution exception prevented completion.")},
            "tested_set": None,
            "stale_pair_output_is_not_bound": True,
            "runtime": {"wall_clock_seconds": time.perf_counter() - started, "max_wall_seconds": args.max_wall_seconds},
        }
        if isinstance(manifest, Mapping):
            provenance = manifest.get("provenance")
            if isinstance(provenance, Mapping):
                tombstone["provenance"] = {key: value for key, value in provenance.items() if key not in {"final_binding", "pair_output", "pair_output_sha256", "pair_output_row_count"}}
        _atomic_write_json(output_path, tombstone)
        return tombstone, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen Experiment 05 Stage 2 Q1-Q3.")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--require-clean-tree", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair-output", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, code = run(args)
    return int(code)


if __name__ == "__main__":  # pragma: no cover - execution is user-authorized
    raise SystemExit(main())
