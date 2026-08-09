"""Experiment 05 Stage 3 (Q4), frozen and fail-closed.

This file deliberately keeps the three execution gates separate:

``cache-gate-a``
    An explicitly model-backed, offline-only clean-logit run.  It writes one
    immutable JSONL row per Stage-3 seed and item.  It does not load the SAE,
    construct a candidate pool, or run an intervention.

``materialize-splits``
    A model-free operation.  It reads only the complete cache, recomputes the
    Gate-A bits from the cached logits, and delegates the Amendment-4 shuffle
    to ``exp05_core.amendment4_split``.  It cannot import the model stack.

``run``
    The Q4 adjudication run.  It accepts only the protocol, Gate-A cache,
    split/prepare artifacts, and an independent pushed review receipt.  It
    intentionally has no candidate-C, Stage-2, or attention-head input path.

No model is loaded at import time.  In particular, the split path imports no
third-party model or tensor package.  Every model-backed path is offline,
checkpointed at seed boundaries, and turns a runtime-cap stop into an
incomplete artifact rather than a partial scientific verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
EXP04 = HERE.parent / "04_causal_feature_interchange"
SOURCE_WORKTREE = HERE.parents[1]

PROTOCOL_SCHEMA = "exp05-number-agreement-protocol"
CACHE_SCHEMA = "exp05-number-agreement-stage3-gate-a-cache-v1"
SPLIT_SCHEMA = "exp05-number-agreement-stage3-split-manifest-v1"
PREPARE_SCHEMA = "exp05-number-agreement-stage3-prepare-manifest-v1"
REVIEW_SCHEMA = "exp05-number-agreement-stage3-prepare-review-v1"
HARNESS_SCHEMA = "exp05-number-agreement-stage3-harness-receipt-v1"
STAGE3_SCHEMA = "exp05-number-agreement-stage3-v1; frozen Q4; Amendments 3-9"
CHECKPOINT_SCHEMA = "exp05-number-agreement-stage3.checkpoint.v1"
DRAW_SCHEMA = "exp05-number-agreement-stage3.draws.v1"

STAGE3_SEEDS = tuple(range(20_260_806, 20_260_814))
REQUESTED_PAIRS = 240
LAYER = 8
TARGET_LATENTS = (8922, 8952, 13352, 13594, 15165, 17956, 19093, 19955, 21401, 21581, 21805, 23011)
TARGET_LATENT_COUNT = 12
SAE_WIDTH = 24_576
RESIDUAL_WIDTH = 768
TRAIN_PAIRS = 40
EVAL_CAP = 150
MATCHED_POOL_SIZE = 128
MATCHED_SUBSET_SIZE = 12
MATCHED_DRAW_COUNT = 100
MATCHED_MAX_ATTEMPTS = 10_000
Q4_TEST_ID = 401
MATCHED_DRAW_FAMILY = "matched_span_alpha1_both"
PATCH_BATCH_LIMIT = 512
POSITION_SETS = ("subject", "final", "both")
DIRECTIONS = ("singular_to_plural", "plural_to_singular")
EPS64 = float.fromhex("0x1.0000000000000p-52")


class Stage3Stop(RuntimeError):
    """Expected fail-closed stop; its status is never a scientific verdict."""

    def __init__(self, gate: str, message: str, *, status: str = "STOPPED") -> None:
        super().__init__(message)
        self.gate = gate
        self.status = status


class RuntimeCapStop(Stage3Stop):
    def __init__(self, message: str) -> None:
        super().__init__("max_wall_seconds", message, status="INCOMPLETE_RUNTIME_CAP")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _jsonable(value: Any) -> Any:
    """Convert tensors/scalars without importing a tensor package."""

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
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _jsonable(value.detach().cpu().tolist())
    # numpy/torch scalar objects expose item() but importing either here would
    # defeat the model-free split gate.
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported artifact value: {type(value)!r}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Stage3Stop("input_read", f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    encoded = "".join(json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    _atomic_text(path, encoded)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.expanduser().resolve().with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field)) for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path.expanduser().resolve())


def _resolve(value: str | Path, *, base: Path = HERE, reject_leaf_symlink: bool = False) -> Path:
    candidate = Path(value).expanduser()
    raw = candidate if candidate.is_absolute() else (base / candidate)
    if reject_leaf_symlink and raw.is_symlink():
        raise Stage3Stop("path_binding", f"runtime leaf path is a symlink and cannot be resolved: {raw}")
    return raw.resolve()


def _portable_artifact_name(path: Path, *, anchor: Path, label: str) -> str:
    """Return a same-directory basename suitable for a published manifest.

    Prepared artifacts are copied as a bundle.  Persisting an absolute source
    path (or a path containing ``..``) would make that bundle resolve against
    the machine that materialized it, so publication is deliberately limited
    to a basename next to the prepare manifest.
    """

    path = path.expanduser().resolve()
    anchor = anchor.expanduser().resolve()
    if path.parent != anchor.parent:
        raise Stage3Stop("path_binding", f"{label} must be beside the prepare manifest: {path}")
    return path.name


def _resolve_prepared_split_csv(prepare_path: Path, value: Any) -> Path:
    """Resolve the prepare manifest's split CSV binding without fallbacks."""

    if not isinstance(value, str) or not value.strip():
        raise Stage3Stop("prepare_cross_hash", "prepare manifest lacks a relative split CSV basename")
    candidate = Path(value)
    # Only a basename is accepted.  In particular, reject old absolute
    # manifests and traversal/subdirectory paths instead of trying a fallback.
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name in {"", ".", ".."} or candidate.as_posix() != candidate.name:
        raise Stage3Stop("prepare_cross_hash", "prepare manifest split CSV binding must be a relative basename")
    return _resolve(candidate, base=prepare_path.expanduser().resolve().parent)


def _validate_runtime_paths(*, immutable: Mapping[str, Path], runtime: Mapping[str, Path]) -> None:
    """Keep every writable runtime artifact distinct from inputs and each other."""

    immutable_resolved = {name: path.expanduser().resolve() for name, path in immutable.items()}
    runtime_resolved = {name: path.expanduser().resolve() for name, path in runtime.items()}
    seen: dict[Path, str] = {}
    for name, path in runtime_resolved.items():
        try:
            path.relative_to(SOURCE_WORKTREE)
        except ValueError:
            pass
        else:
            raise Stage3Stop("path_binding", f"runtime path {name!r} must be outside the immutable source worktree {SOURCE_WORKTREE}")
        if path in seen:
            raise Stage3Stop("path_binding", f"runtime paths {seen[path]!r} and {name!r} are identical")
        seen[path] = name
        if path in immutable_resolved.values():
            owner = next(key for key, value in immutable_resolved.items() if value == path)
            raise Stage3Stop("path_binding", f"runtime path {name!r} would overwrite immutable input {owner!r}")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise Stage3Stop("path_binding", f"existing runtime path {path} is not a regular file")


def _write_run_execution_tombstone(output_path: Path, checkpoint_path: Path, draw_path: Path, *, gate: str, detail: str, runtime: Mapping[str, Any] | None = None) -> None:
    """Replace any stale COMPLETE marker after an execution/provenance failure."""

    tombstone = {
        "schema": STAGE3_SCHEMA, "status": "EXECUTION_INCOMPLETE", "verdict": None,
        "runtime_tombstone": True, "reason": {"code": "EXECUTION_INCOMPLETE", "gate": gate, "detail": detail},
        "draw_csv_path": str(draw_path.expanduser().resolve()), "draw_csv_sha256": None,
        "runtime": dict(runtime or {}), "scientific_reuse_allowed": False, "resumable_for_adjudication": False,
    }
    try:
        _atomic_json(output_path, tombstone)
    finally:
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA, "phase": "run", "status": "EXECUTION_INCOMPLETE",
            "draw_csv_path": str(draw_path.expanduser().resolve()), "draw_csv_sha256": None,
            "runtime": dict(runtime or {}), "scientific_reuse_allowed": False, "resumable_for_adjudication": False,
            "runtime_tombstone": True, "reason": tombstone["reason"], "updated_at_epoch": time.time(),
        }
        _atomic_json(checkpoint_path, _with_self_hash(checkpoint))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Stage3Stop("missing_input", f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3Stop("invalid_input_json", f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Stage3Stop("invalid_input_schema", f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Stage3Stop("missing_input", f"{label} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Stage3Stop("invalid_input_schema", f"{label} line {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage3Stop("invalid_input_jsonl", f"cannot read {label} {path}: {exc}") from exc
    if not rows:
        raise Stage3Stop("empty_input", f"{label} has no records")
    return rows


def _git(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=HERE, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise Stage3Stop("git_unavailable", f"cannot inspect git provenance: {exc}") from exc
    if result.returncode:
        raise Stage3Stop("git_provenance", f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _source_tree_sha256() -> str:
    """Hash the tracked source tree without including runtime artifacts."""

    listing = _git(["ls-files", "-s"])
    return _sha256_bytes(listing.encode("utf-8"))


def _git_provenance(expected: str | None, require_clean: bool) -> dict[str, Any]:
    if require_clean is not True:
        raise Stage3Stop("clean_tree_required", "model-backed Stage-3 commands require --require-clean-tree")
    commit = _git(["rev-parse", "HEAD"])
    if expected and commit != expected:
        raise Stage3Stop("expected_git_commit", f"HEAD {commit} differs from expected {expected}")
    status = _git(["status", "--porcelain"])
    if require_clean and status:
        raise Stage3Stop("dirty_tree", "--require-clean-tree was requested but git status is not clean")
    return {
        "commit": commit,
        "expected_commit": expected,
        "require_clean_tree": require_clean,
        "status_porcelain": status,
        "source_worktree": str(SOURCE_WORKTREE),
        "source_tree_sha256": _source_tree_sha256(),
    }


def _assert_source_unchanged(start: Mapping[str, Any]) -> dict[str, Any]:
    final = _git_provenance(str(start.get("expected_commit") or ""), True)
    if dict(final) != dict(start):
        raise Stage3Stop("source_changed", "HEAD/source-tree hash/clean status changed during this invocation")
    return final


def _check_runtime(started: float, cap: float | None, where: str, *, prior_seconds: float = 0.0) -> None:
    if cap is not None and prior_seconds + time.perf_counter() - started >= cap:
        raise RuntimeCapStop(f"declared runtime cap {cap:.3f}s reached at {where}")


def _float32(value: Any) -> float:
    """Round a JSON number to the IEEE-754 float32 value used by the cache."""

    try:
        return float(struct.unpack("<f", struct.pack("<f", float(value)))[0])
    except (TypeError, ValueError, OverflowError, struct.error) as exc:
        raise Stage3Stop("cache_float32", f"cannot interpret cached value as float32: {value!r}") from exc


def _self_hash(value: Mapping[str, Any]) -> str:
    body = {str(key): item for key, item in value.items() if key != "self_sha256"}
    return _sha256_bytes(_json_bytes(body))


def _with_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["self_sha256"] = _self_hash(result)
    return result


def _new_invocation_id(*, protocol_hash: str, cache_hash: str, split_hash: str, prepare_hash: str, commit: str) -> str:
    return _sha256_bytes(_json_bytes({"protocol_sha256": protocol_hash, "gate_cache_sha256": cache_hash, "split_manifest_sha256": split_hash, "prepare_manifest_sha256": prepare_hash, "commit": commit, "nonce": time.time_ns()}))


def _seed_logical_fe(row: Mapping[str, Any]) -> int:
    execution = row.get("execution")
    records = execution.get("engine_records") if isinstance(execution, Mapping) else None
    if not isinstance(records, list):
        return 0
    total = 0
    for record in records:
        if isinstance(record, Mapping) and isinstance(record.get("batch"), (int, float)) and not isinstance(record.get("batch"), bool) and math.isfinite(float(record["batch"])):
            total += int(record["batch"])
    return total


def _rows_logical_fe(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(_seed_logical_fe(row) for row in rows)


def _preparation_accounting(*artifacts: Mapping[str, Any]) -> dict[str, Any]:
    values: list[float] = []
    for artifact in artifacts:
        candidates: list[Any] = [artifact.get("logical_fe"), artifact.get("logical_forward_equivalents")]
        runtime = artifact.get("runtime")
        if isinstance(runtime, Mapping):
            candidates.extend([runtime.get("logical_fe"), runtime.get("logical_forward_equivalents")])
        for value in candidates:
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                values.append(float(value))
                break
    original: int | float | None = sum(values) if values else None
    return {"reused_in_q4_invocation": True, "repeated_q4_fe": 0, "original_logical_fe": original, "status": "RECORDED" if original is not None else "NOT_REGISTERED"}


def _draw_binding_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed_row in rows:
        matched = seed_row.get("matched_draws")
        if not isinstance(matched, Mapping):
            continue
        accepted_by_attempt = {int(item["attempt"]): item for item in matched.get("accepted", []) if isinstance(item, Mapping)}
        results_by_draw = {int(item["draw_index"]): item for item in matched.get("results", []) if isinstance(item, Mapping)}
        for attempt in matched.get("attempts", []):
            if not isinstance(attempt, Mapping):
                raise Stage3Stop("draw_binding", f"seed {seed_row.get('seed')} contains a non-object matched attempt")
            accepted_row = accepted_by_attempt.get(int(attempt.get("attempt", -1))) if attempt.get("accepted") is True else None
            draw_index = int(accepted_row["draw_index"]) if accepted_row is not None else None
            result = results_by_draw.get(draw_index) if draw_index is not None else None
            effect = result.get("effect") if isinstance(result, Mapping) else None
            effect_hash = effect.get("result_hash") if isinstance(effect, Mapping) else None
            if attempt.get("accepted") is True and (accepted_row is None or not isinstance(effect_hash, str)):
                raise Stage3Stop("draw_binding", f"seed {seed_row.get('seed')} accepted attempt lacks its matched effect result")
            records.append({
                "invocation_id": attempt.get("invocation_id"),
                "seed_id": attempt.get("seed_id"),
                "draw_family": attempt.get("draw_family"),
                "draw_index": draw_index,
                "accepted_attempt_id": int(attempt["attempt"]) if accepted_row is not None else None,
                "draw_or_projector_hash": accepted_row.get("draw_or_projector_hash") if accepted_row is not None else None,
                "attempt": int(attempt["attempt"]),
                "attempt_sha256": attempt.get("attempt_sha256"),
                "latent_ids": attempt.get("latent_ids"),
                "rank": attempt.get("rank"),
                "tolerance": attempt.get("tolerance"),
                "accepted": bool(attempt.get("accepted")),
                "matched_effect_result_hash": effect_hash,
            })
    return sorted(records, key=lambda item: (int(item.get("seed_id", -1)), int(item.get("attempt", -1))))


SEED_RESULT_STATUSES = {"PASS", "COMPLETED_FAIL", "SCIENTIFIC_UNRESOLVED", "EXECUTION_INCOMPLETE"}


def _seed_result_hash(value: Mapping[str, Any]) -> str:
    body = {str(key): item for key, item in value.items() if key not in {"self_sha256", "seed_result_sha256"}}
    return _sha256_bytes(_json_bytes(body))


def _freeze_seed_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bind each per-seed row, rather than allowing a status-only checkpoint."""

    result = dict(value)
    result.pop("seed_result_sha256", None)
    result["seed_result_sha256"] = _seed_result_hash(result)
    return result


def _validate_seed_results(rows: Sequence[Mapping[str, Any]], expected_seeds: Sequence[int], *, context: str, invocation_id: str | None = None) -> list[dict[str, Any]]:
    expected = [int(seed) for seed in expected_seeds]
    if expected != sorted(set(expected)):
        raise Stage3Stop("seed_result_binding", f"{context} expected seed registry is not sorted/unique")
    if len(rows) != len(expected):
        raise Stage3Stop("seed_result_binding", f"{context} has {len(rows)} rows; expected exactly {len(expected)}")
    normalized = [dict(row) for row in rows]

    def finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    def hash_string(value: Any) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())

    def effect(value: Any, label: str) -> None:
        if not isinstance(value, Mapping):
            raise Stage3Stop("seed_result_binding", f"{context} {label} is not an effect object")
        for field in ("E", "pair_sign_consistency", "directed_sign_consistency"):
            if not finite(value.get(field)):
                raise Stage3Stop("seed_result_binding", f"{context} {label}.{field} is not finite")
        for field in ("directed_raw", "directed_sign_aligned", "pair_means"):
            values = value.get(field)
            if not isinstance(values, list) or not values or any(not finite(item) for item in values):
                raise Stage3Stop("seed_result_binding", f"{context} {label}.{field} is missing/non-finite")
        if not isinstance(value.get("pair_ids"), list) or not value["pair_ids"] or any(not isinstance(item, int) or isinstance(item, bool) for item in value["pair_ids"]):
            raise Stage3Stop("seed_result_binding", f"{context} {label}.pair_ids is missing/non-integral")
        if not isinstance(value.get("execution_cell_id"), str) or not value["execution_cell_id"] or not hash_string(value.get("tensor_hash")) or not hash_string(value.get("result_hash")):
            raise Stage3Stop("seed_result_binding", f"{context} {label} lacks execution/tensor/result hashes")
        body = dict(value)
        observed_result_hash = body.pop("result_hash")
        if observed_result_hash != _sha256_bytes(_json_bytes(body)):
            raise Stage3Stop("seed_result_binding", f"{context} {label}.result_hash does not bind the effect payload")

    def validate_effect_cell(value: Any, label: str, *, require_ratio: bool = True) -> None:
        if not isinstance(value, Mapping):
            raise Stage3Stop("seed_result_binding", f"{context} {label} is not a cell object")
        for key in ("full_delta", "target_span", "target_complement"):
            if key not in value:
                raise Stage3Stop("seed_result_binding", f"{context} {label} lacks {key}")
            effect(value[key], f"{label}.{key}")
        ratio = value.get("ratio_guard")
        if not isinstance(ratio, Mapping) or ratio.get("status") not in {"ESTIMABLE", "NON_ESTIMABLE_DENOMINATOR"}:
            raise Stage3Stop("seed_result_binding", f"{context} {label}.ratio_guard has no registered status")
        for key in ("D", "M", "tau"):
            if not finite(ratio.get(key)):
                raise Stage3Stop("seed_result_binding", f"{context} {label}.ratio_guard.{key} is not finite")
        if ratio["status"] == "ESTIMABLE":
            if not finite(ratio.get("R_span")) or not finite(ratio.get("R_comp")):
                raise Stage3Stop("seed_result_binding", f"{context} {label}.ratio_guard ratios are not finite")
        elif "R_span" in ratio or "R_comp" in ratio:
            raise Stage3Stop("seed_result_binding", f"{context} {label} reports a ratio under NON_ESTIMABLE_DENOMINATOR")
        fractions = value.get("geometric_fractions")
        if not isinstance(fractions, Mapping) or not finite(fractions.get("span")) or not finite(fractions.get("complement")):
            raise Stage3Stop("seed_result_binding", f"{context} {label} geometric fractions are incomplete")

    def validate_matchings(value: Any, pool_ids: Sequence[int], seed: int, invocation_id: str) -> None:
        if not isinstance(value, Mapping):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched_draws is not an object")
        rng = value.get("rng")
        accepted = value.get("accepted")
        attempts = value.get("attempts")
        results = value.get("results")
        if not isinstance(rng, Mapping) or rng.get("invocation_id") != invocation_id or int(rng.get("seed_id", -1)) != seed or rng.get("draw_family") != MATCHED_DRAW_FAMILY or int(rng.get("test_id", -1)) != Q4_TEST_ID or int(rng.get("accepted_count", -1)) != MATCHED_DRAW_COUNT or not isinstance(rng.get("generator"), str) or not isinstance(rng.get("seed_formula"), str):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched RNG binding is incomplete")
        if not isinstance(accepted, list) or len(accepted) != MATCHED_DRAW_COUNT or not isinstance(results, list) or len(results) != MATCHED_DRAW_COUNT or not isinstance(attempts, list) or len(attempts) < MATCHED_DRAW_COUNT:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched draws are not exactly 100 accepted/results with all attempts")
        allowed_pool = set(int(item) for item in pool_ids) - set(TARGET_LATENTS)
        accepted_by_index: dict[int, Mapping[str, Any]] = {}
        for item in accepted:
            if not isinstance(item, Mapping):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} accepted draw is not an object")
            index = int(item.get("draw_index", -1))
            ids = item.get("latent_ids")
            if index in accepted_by_index or item.get("invocation_id") != invocation_id or int(item.get("seed_id", -1)) != seed or item.get("draw_family") != MATCHED_DRAW_FAMILY or int(item.get("accepted_attempt_id", -1)) != int(item.get("attempt", -1)) or not hash_string(item.get("attempt_sha256")) or item.get("draw_or_projector_hash") != item.get("attempt_sha256") or not isinstance(ids, list) or len(ids) != MATCHED_SUBSET_SIZE or len(set(ids)) != MATCHED_SUBSET_SIZE or any(not isinstance(latent, int) or latent not in allowed_pool for latent in ids) or int(item.get("attempt", -1)) < 1 or int(item.get("rank", -1)) != MATCHED_SUBSET_SIZE or not finite(item.get("tolerance")):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} accepted matched draw is malformed")
            accepted_by_index[index] = item
        if sorted(accepted_by_index) != list(range(MATCHED_DRAW_COUNT)):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} accepted draw indices are not exactly 0..99")
        accepted_by_attempt = {int(item["attempt"]): item for item in accepted}
        if len(accepted_by_attempt) != MATCHED_DRAW_COUNT:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} accepted draws reuse an attempt number")
        result_by_index: dict[int, Mapping[str, Any]] = {}
        for item in results:
            if not isinstance(item, Mapping):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched result is not an object")
            index = int(item.get("draw_index", -1))
            ids = item.get("latent_ids")
            accepted_draw = accepted_by_index.get(index, {})
            if index in result_by_index or item.get("invocation_id") != invocation_id or int(item.get("seed_id", -1)) != seed or item.get("draw_family") != MATCHED_DRAW_FAMILY or not isinstance(ids, list) or ids != accepted_draw.get("latent_ids") or int(item.get("attempt", -1)) != int(accepted_draw.get("attempt", -1)) or int(item.get("accepted_attempt_id", -1)) != int(accepted_draw.get("attempt", -1)) or item.get("attempt_sha256") != accepted_draw.get("attempt_sha256") or not hash_string(item.get("effect_result_hash")) or not hash_string(item.get("projector_hash")) or not hash_string(item.get("projected_delta_hash")) or item.get("draw_or_projector_hash") != item.get("projected_delta_hash"):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched result does not bind the accepted draw")
            effect(item.get("effect"), f"seed {seed} matched result {index}.effect")
            if item.get("effect_result_hash") != item["effect"].get("result_hash"):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched result {index} effect hash is not bound")
            result_by_index[index] = item
        if sorted(result_by_index) != list(range(MATCHED_DRAW_COUNT)):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched result indices are not exactly 0..99")
        attempt_ids: set[int] = set()
        accepted_attempts = 0
        for item in attempts:
            if not isinstance(item, Mapping):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched attempt is not an object")
            attempt = int(item.get("attempt", -1))
            ids = item.get("latent_ids")
            accepted_draw = accepted_by_attempt.get(attempt)
            if attempt in attempt_ids or item.get("invocation_id") != invocation_id or int(item.get("seed_id", -1)) != seed or item.get("draw_family") != MATCHED_DRAW_FAMILY or not hash_string(item.get("attempt_sha256")) or attempt < 1 or not isinstance(ids, list) or len(ids) != MATCHED_SUBSET_SIZE or len(set(ids)) != MATCHED_SUBSET_SIZE or any(not isinstance(latent, int) or latent not in allowed_pool for latent in ids) or not finite(item.get("tolerance")) or not isinstance(item.get("accepted"), bool) or item.get("draw_index") != (int(accepted_draw["draw_index"]) if accepted_draw is not None else None) or item.get("accepted_attempt_id") != (attempt if accepted_draw is not None else None):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched attempt is malformed")
            attempt_ids.add(attempt)
            accepted_attempts += int(item["accepted"])
            body = dict(item)
            observed_attempt_hash = body.pop("attempt_sha256")
            if observed_attempt_hash != _sha256_bytes(_json_bytes(body)):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched attempt hash is not self-consistent")
        if accepted_attempts != MATCHED_DRAW_COUNT or int(rng.get("attempt_count", -1)) != len(attempts) or int(rng.get("rejected_count", -1)) != len(attempts) - MATCHED_DRAW_COUNT:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} matched attempt counts do not bind 100 accepted draws")
        attempts_by_number = {int(item["attempt"]): item for item in attempts}
        for item in accepted:
            attempt = attempts_by_number.get(int(item["attempt"]))
            if attempt is None or attempt.get("accepted") is not True or attempt.get("latent_ids") != item.get("latent_ids") or int(attempt.get("rank", -1)) != int(item.get("rank", -1)) or float(attempt.get("tolerance")) != float(item.get("tolerance")) or attempt.get("draw_index") != item.get("draw_index") or attempt.get("attempt_sha256") != item.get("attempt_sha256"):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} accepted draw does not exactly bind its accepted attempt record")

    def validate_complete_payload(row: Mapping[str, Any], seed: int) -> None:
        pool = row.get("candidate_pool")
        if not isinstance(pool, Mapping) or int(pool.get("budget", -1)) != MATCHED_POOL_SIZE or pool.get("rank_training_only") is not True:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} candidate pool budget/role is not frozen")
        frozen = pool.get("frozen_ranked_latent_ids")
        scores = pool.get("scores")
        if not isinstance(frozen, list) or len(frozen) != MATCHED_POOL_SIZE or len(set(frozen)) != MATCHED_POOL_SIZE or any(not isinstance(item, int) or isinstance(item, bool) for item in frozen) or any(item in TARGET_LATENTS for item in frozen) or not isinstance(scores, list) or len(scores) != MATCHED_POOL_SIZE:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} candidate pool is not exactly 128 unique ids/scores")
        if any(not isinstance(item, Mapping) or int(item.get("latent_id", -1)) != frozen[index] or not finite(item.get("score")) for index, item in enumerate(scores)):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} candidate scores do not bind frozen ranked ids")
        projector = row.get("projector")
        if not isinstance(projector, Mapping) or projector.get("arithmetic") != "float64" or projector.get("shape") != [TARGET_LATENT_COUNT, RESIDUAL_WIDTH] or projector.get("target_latent_ids") != list(TARGET_LATENTS) or int(projector.get("numerical_rank", -1)) != TARGET_LATENT_COUNT or not isinstance(projector.get("singular_values"), list) or len(projector["singular_values"]) != TARGET_LATENT_COUNT or any(not finite(item) for item in projector["singular_values"]) or not finite(projector.get("tolerance")) or not hash_string(projector.get("right_singular_rows_hash")):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} target projector payload is not frozen rank-12")
        cells = row.get("cells")
        if not isinstance(cells, Mapping) or any(kind not in cells for kind in POSITION_SETS):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} position cells are incomplete")
        for kind in POSITION_SETS:
            validate_effect_cell(cells[kind], f"seed {seed} cells.{kind}")
        for key in ("alpha_0_5/both", "alpha_1_0/both"):
            if key not in cells or not isinstance(cells[key], Mapping):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} {key} cell is missing")
        effect(cells["alpha_0_5/both"].get("effect"), f"seed {seed} cells.alpha_0_5/both.effect")
        alias = cells["alpha_1_0/both"]
        if alias.get("alias_of") != "both/full_delta/evaluation" or alias.get("rerun") is not False or alias.get("double_count") is not False or alias.get("execution_cell_id") != cells["both"]["full_delta"]["execution_cell_id"] or alias.get("tensor_hash") != cells["both"]["full_delta"]["tensor_hash"] or alias.get("result_hash") != cells["both"]["full_delta"]["result_hash"]:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} alpha=1 alias does not bind the full-delta cell")
        execution = row.get("execution")
        if not isinstance(execution, Mapping) or not isinstance(execution.get("train_pair_ids"), list) or len(execution["train_pair_ids"]) != TRAIN_PAIRS or len(set(execution["train_pair_ids"])) != TRAIN_PAIRS or not isinstance(execution.get("eval_pair_ids"), list) or not (40 <= len(execution["eval_pair_ids"] ) <= EVAL_CAP) or len(set(execution["eval_pair_ids"])) != len(execution["eval_pair_ids"]) or set(execution["train_pair_ids"]) & set(execution["eval_pair_ids"]) or not isinstance(execution.get("engine_records"), list) or not execution["engine_records"] or any(not isinstance(record, Mapping) or not isinstance(record.get("label"), str) or not isinstance(record.get("path"), str) or not finite(record.get("batch")) or not finite(record.get("seconds")) for record in execution["engine_records"]):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} execution roles/records are incomplete")
        validate_matchings(row.get("matched_draws"), frozen, seed, str(row["invocation_id"]))
        pca = row.get("pca_context")
        pca_cells = row.get("pca_cells")
        if not isinstance(pca, Mapping) or not isinstance(pca_cells, Mapping):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} PCA payload is missing")
        if pca.get("status") == "PCA_CONTEXT_NON_ESTIMABLE":
            if pca_cells:
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} non-estimable PCA context has cells")
        else:
            if int(pca.get("fit_rows", -1)) != 8192 or int(pca.get("width", -1)) != RESIDUAL_WIDTH or pca.get("dtype") != "float64" or pca.get("fallback_task_pool") is not False or not hash_string(pca.get("generic_pool_hash")) or not hash_string(pca.get("eigenvalues_hash")) or not finite(pca.get("orthogonality_max_abs")) or set(pca_cells) != {"PCA_span/subject", "PCA_span/final", "PCA_span/both"}:
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} PCA context/cell set is not frozen")
            for cell, payload in pca_cells.items():
                if not isinstance(payload, Mapping) or payload.get("role") != "descriptive_only" or not hash_string(payload.get("projector_hash")) or not hash_string(payload.get("projected_delta_hash")):
                    raise Stage3Stop("seed_result_binding", f"{context} seed {seed} PCA cell {cell} is incomplete")
                effect(payload.get("effect"), f"seed {seed} {cell}.effect")
        primary = cells["both"]["ratio_guard"]
        status = row["status"]
        if status in {"PASS", "COMPLETED_FAIL"}:
            edge = cells["both"].get("matched_edge_second_largest")
            exceeds = cells["both"].get("R_span_exceeds_matched_edge")
            matched_results = row["matched_draws"].get("results")
            matched_full_e = cells["both"].get("full_delta", {}).get("E")
            if not isinstance(matched_results, list) or not finite(matched_full_e):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} {status} lacks matched/full effects for edge recomputation")
            recomputed_edges = sorted(float(item["effect"]["E"]) / float(matched_full_e) for item in matched_results)
            expected_edge = recomputed_edges[-2] if len(recomputed_edges) >= 2 else float("nan")
            if primary.get("status") != "ESTIMABLE" or not finite(edge) or edge != expected_edge or not finite(primary.get("R_span")) or not isinstance(exceeds, bool) or exceeds != (float(primary.get("R_span")) > expected_edge) or exceeds != (status == "PASS"):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} {status} is not self-consistent with the matched edge")
        elif primary.get("status") == "NON_ESTIMABLE_DENOMINATOR" and row.get("reason") != "NON_ESTIMABLE_DENOMINATOR":
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} denominator unresolved reason is not registered")
    row_seeds: list[int] = []
    for index, row in enumerate(normalized):
        try:
            seed = int(row.get("seed", -1))
        except (TypeError, ValueError) as exc:
            raise Stage3Stop("seed_result_binding", f"{context} row {index} has a non-integer seed") from exc
        row_seeds.append(seed)
        row_invocation_id = row.get("invocation_id")
        if not isinstance(row_invocation_id, str) or len(row_invocation_id) != 64 or any(char not in "0123456789abcdef" for char in row_invocation_id.lower()) or (invocation_id is not None and row_invocation_id != invocation_id):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} lacks the exact invocation binding")
        if row.get("status") not in SEED_RESULT_STATUSES:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} has an unknown exact status")
        execution_status = row.get("execution_status")
        if execution_status not in {"EXECUTION_COMPLETE", "EXECUTION_INCOMPLETE"}:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} lacks an exact execution status")
        if (row.get("status") == "EXECUTION_INCOMPLETE") != (execution_status == "EXECUTION_INCOMPLETE"):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} scientific/execution status classes disagree")
        if row.get("seed_result_sha256") != _seed_result_hash(row):
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} lacks a valid seed-result self hash")
        try:
            retained = int(row.get("retained_pairs", -1))
            role_counts = row.get("role_counts")
            rank_count = int(role_counts.get("rank_training", -1)) if isinstance(role_counts, Mapping) else -1
            eval_count = int(role_counts.get("evaluation", -1)) if isinstance(role_counts, Mapping) else -1
        except (TypeError, ValueError) as exc:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} has malformed retained/role counts") from exc
        if retained < 0 or rank_count < 0 or eval_count < 0:
            raise Stage3Stop("seed_result_binding", f"{context} seed {seed} lacks complete retained/role counts")
        status = str(row["status"])
        if status in {"SCIENTIFIC_UNRESOLVED", "EXECUTION_INCOMPLETE"}:
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} unresolved/incomplete row lacks a structured reason")
        if status == "SCIENTIFIC_UNRESOLVED":
            for field in ("candidate_pool", "projector", "cells", "pca_context", "pca_cells", "matched_draws", "execution"):
                if field in row:
                    raise Stage3Stop("seed_result_binding", f"{context} seed {seed} scientific-unresolved row carries forbidden numeric payload {field}")
        if status in {"PASS", "COMPLETED_FAIL"}:
            for field in ("candidate_pool", "projector", "cells", "matched_draws", "execution", "pca_context"):
                if field not in row or not isinstance(row[field], Mapping):
                    raise Stage3Stop("seed_result_binding", f"{context} seed {seed} {status} row lacks frozen field {field}")
            cells = row.get("cells")
            if not isinstance(cells, Mapping) or not isinstance(cells.get("both"), Mapping):
                raise Stage3Stop("seed_result_binding", f"{context} seed {seed} {status} row lacks the primary both cell")
            validate_complete_payload(row, seed)
    if row_seeds != expected:
        raise Stage3Stop("seed_result_binding", f"{context} seed rows are not in exact registered order")
    if len(set(row_seeds)) != len(row_seeds):
        raise Stage3Stop("seed_result_binding", f"{context} seed rows contain duplicate seeds")
    return normalized


def _validate_run_checkpoint(value: Mapping[str, Any], *, protocol_hash: str, git: Mapping[str, Any], cache_hash: str, split_hash: str, prepare_hash: str, draw_csv_path: Path) -> tuple[list[dict[str, Any]], set[int], float]:
    if value.get("schema") != CHECKPOINT_SCHEMA or value.get("phase") != "run":
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint schema/phase is not frozen")
    if value.get("status") not in {"RUNNING", "INCOMPLETE_RUNTIME_CAP", "COMPLETE"}:
        raise Stage3Stop("checkpoint_binding", f"Stage-3 run checkpoint status {value.get('status')!r} is not resumable")
    if git.get("require_clean_tree") is not True or git.get("status_porcelain") != "" or value.get("protocol_sha256") != protocol_hash or value.get("git") != dict(git) or value.get("gate_cache_sha256") != cache_hash or value.get("split_manifest_sha256") != split_hash or value.get("prepare_manifest_sha256") != prepare_hash:
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint artifact/protocol/git binding differs")
    if value.get("draw_csv_path") != str(draw_csv_path.expanduser().resolve()):
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint draw CSV path differs")
    draw_hash = value.get("draw_csv_sha256")
    if draw_hash is not None:
        if not draw_csv_path.is_file() or _sha256_file(draw_csv_path) != draw_hash:
            raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint draw CSV hash does not match the bound file")
    if value.get("self_sha256") != _self_hash(value):
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint self hash is absent or invalid")
    if value.get("status") == "COMPLETE":
        observed_binding = value.get("checkpoint_binding_sha256")
        binding_body = {str(key): item for key, item in value.items() if key not in {"output_sha256", "checkpoint_binding_sha256", "self_sha256"}}
        if observed_binding != _sha256_bytes(_json_bytes(binding_body)):
            raise Stage3Stop("checkpoint_binding", "COMPLETE checkpoint binding hash is absent or does not cover its immutable body")
    completed_raw = value.get("completed_seeds")
    rows_raw = value.get("seed_rows")
    if not isinstance(completed_raw, list) or not isinstance(rows_raw, list):
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint completed_seeds/seed_rows are missing")
    try:
        completed = [int(seed) for seed in completed_raw]
    except (TypeError, ValueError) as exc:
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint contains a non-integer seed") from exc
    if completed != sorted(set(completed)) or not set(completed).issubset(set(STAGE3_SEEDS)):
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint seeds are not known, unique, and sorted")
    rows = [dict(item) for item in rows_raw if isinstance(item, Mapping)]
    if len(rows) != len(rows_raw) or len(rows) != len(completed):
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint rows must equal completed seed count")
    row_seeds = [int(row.get("seed", -1)) for row in rows]
    if row_seeds != completed or len(set(row_seeds)) != len(row_seeds):
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint row seeds do not equal completed_seeds")
    rows = _validate_seed_results(rows, completed, context="Stage-3 run checkpoint")
    try:
        prior = float(value.get("wall_clock_seconds", 0.0))
    except (TypeError, ValueError) as exc:
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint wall_clock_seconds is invalid") from exc
    if not math.isfinite(prior) or prior < 0:
        raise Stage3Stop("checkpoint_binding", "Stage-3 run checkpoint wall_clock_seconds is not finite/non-negative")
    return rows, set(completed), prior


def _protocol_hash(protocol: Mapping[str, Any]) -> str:
    return _sha256_bytes(_json_bytes(protocol))


def _require_equal(value: Any, expected: Any, gate: str, label: str) -> None:
    if value != expected:
        raise Stage3Stop(gate, f"{label}={value!r} differs from frozen {expected!r}")


def _validate_protocol(protocol: Mapping[str, Any], path: Path) -> None:
    _require_equal(protocol.get("schema"), PROTOCOL_SCHEMA, "protocol_schema", "protocol.schema")
    _require_equal(int(protocol.get("version", -1)), 1, "protocol_version", "protocol.version")
    design_freeze = protocol.get("design_freeze")
    if not isinstance(design_freeze, Mapping):
        raise Stage3Stop("protocol_design_freeze", "protocol.design_freeze is required")
    _require_equal(int(design_freeze.get("latest_amendment", -1)), 9, "protocol_design_freeze", "design_freeze.latest_amendment")
    _require_equal(tuple(int(item) for item in design_freeze.get("preserved_amendments", ())), tuple(range(1, 10)), "protocol_design_freeze", "design_freeze.preserved_amendments")
    model = protocol.get("model")
    if not isinstance(model, Mapping):
        raise Stage3Stop("protocol_model", "protocol.model must be an object")
    for key, expected in (("name", "gpt2-small"), ("mechanism_library", "TransformerLens"), ("activation_dtype", "float32"), ("residual_width", RESIDUAL_WIDTH)):
        _require_equal(model.get(key), expected, "protocol_model", f"protocol.model.{key}")
    expected_revisions = model.get("expected_local_snapshot_revisions")
    if not isinstance(expected_revisions, Mapping):
        raise Stage3Stop("protocol_model_revisions", "expected_local_snapshot_revisions is required")
    for key in ("gpt2", "sae"):
        revision = expected_revisions.get(key)
        if not isinstance(revision, str) or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()):
            raise Stage3Stop("protocol_model_revisions", f"protocol model revision {key!r} is not a 40-hex pin")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, Mapping):
        raise Stage3Stop("protocol_seeds", "protocol.seeds must be an object")
    _require_equal(tuple(int(seed) for seed in seeds.get("stage3_adjudication", ())), STAGE3_SEEDS, "protocol_seeds", "stage3 seeds")
    target = protocol.get("target_latents")
    if not isinstance(target, Mapping):
        raise Stage3Stop("protocol_target_latents", "protocol.target_latents is missing")
    _require_equal(tuple(int(item) for item in target.get("ids", ())), TARGET_LATENTS, "protocol_target_latents", "target ids")
    _require_equal(int(target.get("count", -1)), TARGET_LATENT_COUNT, "protocol_target_latents", "target count")
    if len(set(TARGET_LATENTS)) != TARGET_LATENT_COUNT:
        raise Stage3Stop("protocol_target_latents", "target ids are not unique")
    gate_a = protocol.get("gate_a")
    gate_a_conditions = gate_a.get("conditions") if isinstance(gate_a, Mapping) else None
    if not isinstance(gate_a_conditions, Mapping):
        raise Stage3Stop("protocol_gate_a", "protocol.gate_a.conditions is missing")
    for key, expected in (("both_members_signed_correct_fraction_at_least", 0.6), ("minimum_retained_pairs", 140), ("median_clean_d_gap_at_least", 1.0)):
        _require_equal(gate_a_conditions.get(key), expected, "protocol_gate_a", f"protocol.gate_a.conditions.{key}")
    stimuli = protocol.get("stimuli")
    if not isinstance(stimuli, Mapping):
        raise Stage3Stop("protocol_stimuli", "protocol.stimuli is missing")
    for key, expected in (("generator", "experiment_04_templates_and_gate_a"), ("family", "single_flip_minimal_pairs"), ("template", "The {ADJ} {SUBJ} {PREP} the {ATTRACTOR} ___"), ("requested_pairs", REQUESTED_PAIRS), ("attractor", "fixed_and_counterbalanced"), ("subject_token_rule", "last_subword_if_multi_token"), ("directed_edits_per_pair", 2), ("pair_cluster_rule", "both_directed_edits_stay_together")):
        _require_equal(stimuli.get(key), expected, "protocol_stimuli", f"protocol.stimuli.{key}")
    readout = protocol.get("readout")
    if not isinstance(readout, Mapping):
        raise Stage3Stop("protocol_readout", "protocol.readout is missing")
    for key, expected in (("name", "d"), ("definition", "logit(\" are\") - logit(\" is\") at the final position"), ("delta_definition", "sign-aligned source-minus-base change in d"), ("model_readout", "native_unembedding"), ("fitted_readout", False)):
        _require_equal(readout.get(key), expected, "protocol_readout", f"protocol.readout.{key}")
    q4 = protocol.get("q4")
    if not isinstance(q4, Mapping):
        raise Stage3Stop("protocol_q4", "protocol.q4 is missing")
    single_invocation = q4.get("single_invocation_runtime_provenance")
    if not isinstance(single_invocation, Mapping):
        raise Stage3Stop("protocol_single_invocation", "protocol.q4.single_invocation_runtime_provenance is required for Amendment 9")
    _require_equal(int(single_invocation.get("amendment", -1)), 9, "protocol_single_invocation", "q4.single_invocation_runtime_provenance.amendment")
    for key, expected in (
        ("all_eight_q4_runtime_seeds_one_fresh_invocation", True),
        ("cross_invocation_scientific_reuse_allowed", False),
        ("prior_checkpoint_role", "diagnostic_integrity_inspection_only"),
        ("restart_q4_runtime_from_zero_after_incomplete_attempt", True),
        ("preparation_inputs_may_be_reused_when_hash_and_receipt_match", True),
        ("accepted_draw_duplicates_or_missing_rows_forbidden", True),
        ("thresholds_and_estimands_unchanged", True),
    ):
        _require_equal(single_invocation.get(key), expected, "protocol_single_invocation", f"q4.single_invocation_runtime_provenance.{key}")
    _require_equal(single_invocation.get("preparation_inputs"), ["stage3_gate_a_cache", "stage3_split_manifest", "stage3_prepare_manifest", "independent_review_receipt", "independent_harness_receipt"], "protocol_single_invocation", "q4.single_invocation_runtime_provenance.preparation_inputs")
    clean_source = single_invocation.get("clean_source_requirement")
    if not isinstance(clean_source, Mapping):
        raise Stage3Stop("protocol_single_invocation", "q4.single_invocation_runtime_provenance.clean_source_requirement is required")
    for key, expected in (("require_clean_tree", True), ("start_and_finish_commit_source_hash_and_clean_status_match", True), ("runtime_outputs_outside_source_worktree", True), ("dirty_or_changed_source_status", "EXECUTION_INCOMPLETE")):
        _require_equal(clean_source.get(key), expected, "protocol_single_invocation", f"q4.single_invocation_runtime_provenance.clean_source_requirement.{key}")
    _require_equal(single_invocation.get("seed_execution_classes"), ["EXECUTION_COMPLETE", "EXECUTION_INCOMPLETE"], "protocol_single_invocation", "q4.single_invocation_runtime_provenance.seed_execution_classes")
    incomplete_seed = single_invocation.get("any_execution_incomplete_seed")
    if not isinstance(incomplete_seed, Mapping):
        raise Stage3Stop("protocol_single_invocation", "q4.single_invocation_runtime_provenance.any_execution_incomplete_seed is required")
    for key, expected in (("top_level_execution_status", "EXECUTION_INCOMPLETE"), ("top_level_scientific_verdict", None), ("resumable_for_adjudication", False)):
        _require_equal(incomplete_seed.get(key), expected, "protocol_single_invocation", f"q4.single_invocation_runtime_provenance.any_execution_incomplete_seed.{key}")
    unresolved = single_invocation.get("scientific_unresolved_output")
    if not isinstance(unresolved, Mapping):
        raise Stage3Stop("protocol_single_invocation", "q4.single_invocation_runtime_provenance.scientific_unresolved_output is required")
    for key, expected in (("execution_may_be_complete", True), ("dependent_value", None), ("dependent_axis_status", "SCIENTIFIC_UNRESOLVED"), ("reason_code", "frozen gate-specific reason"), ("optional_unvalidated_numeric_output_forbidden", True)):
        _require_equal(unresolved.get(key), expected, "protocol_single_invocation", f"q4.single_invocation_runtime_provenance.scientific_unresolved_output.{key}")
    _require_equal(single_invocation.get("accepted_draw_binding_key"), ["invocation_id", "seed_id", "draw_family", "draw_index", "accepted_attempt_id", "draw_or_projector_hash"], "protocol_single_invocation", "q4.single_invocation_runtime_provenance.accepted_draw_binding_key")
    _require_equal(single_invocation.get("accepted_draw_sets_exact_across"), ["accepted_draw_manifest", "accepted_attempt_records", "accepted_scientific_csv"], "protocol_single_invocation", "q4.single_invocation_runtime_provenance.accepted_draw_sets_exact_across")
    _require_equal(single_invocation.get("complete_requires"), [
        "all eight seeds have exact registered execution-cell coverage",
        "accepted draw set and multiplicity match manifest attempts and scientific CSV",
        "final manifest binds complete attempt ledger scientific CSV final checkpoint execution-cell registry immutable preparation hashes model-state fingerprint configuration source-tree identity and code revision",
    ], "protocol_single_invocation", "q4.single_invocation_runtime_provenance.complete_requires")
    _require_equal(single_invocation.get("final_manifest_binds"), [
        "accepted_draw_manifest",
        "complete_attempt_ledger",
        "accepted_scientific_csv",
        "final_checkpoint",
        "execution_cell_registry",
        "immutable_preparation_input_hashes",
        "model_state_fingerprint",
        "configuration",
        "source_tree_identity",
        "code_revision",
    ], "protocol_single_invocation", "q4.single_invocation_runtime_provenance.final_manifest_binds")
    _require_equal(single_invocation.get("compute_accounting_fields"), ["q4_invocation_id", "q4_attempt_logical_fe_executed", "q4_attempt_wall_time", "q4_attempt_status", "final_complete_q4_logical_fe", "final_complete_q4_wall_time", "prior_incomplete_q4_logical_fe", "cumulative_q4_logical_fe_all_attempts", "cumulative_q4_wall_time_all_attempts", "preparation_logical_fe_separately"], "protocol_single_invocation", "q4.single_invocation_runtime_provenance.compute_accounting_fields")
    for key, expected in (("adjudication_seeds", list(STAGE3_SEEDS)), ("layer", LAYER), ("hook", "hook_resid_pre"), ("position_sets", list(POSITION_SETS)), ("adjudicating_position_set", "both")):
        _require_equal(q4.get(key), expected, "protocol_q4", f"protocol.q4.{key}")
    if q4.get("independent_of_candidate_selection_and_stage2") is not True or q4.get("upstream_candidate_or_stage2_inputs") != []:
        raise Stage3Stop("protocol_q4_independence", "Q4 independence boundary is missing or changed")
    _require_equal(q4.get("delta"), "true activation delta x_src - x_base; no encoder and no reconstruction term", "protocol_q4", "protocol.q4.delta")
    interventions = q4.get("interventions")
    if not isinstance(interventions, Mapping):
        raise Stage3Stop("protocol_interventions", "Q4 interventions are missing")
    for key, expected in (("reference", "add delta"), ("span", "add P_V delta"), ("complement", "add (I - P_V) delta")):
        _require_equal(interventions.get(key), expected, "protocol_interventions", f"protocol.q4.interventions.{key}")
    alpha_cells = q4.get("alpha_cells")
    if not isinstance(alpha_cells, Mapping):
        raise Stage3Stop("protocol_alpha_cells", "Q4 alpha_cells are missing")
    for key, expected in (("scope", "both position set on held-out evaluation split"), ("alpha_0_5", "new cell"), ("alpha_1_0", "alias existing full-delta/both/evaluation cell"), ("alias_requires_same_execution_tensor_result_hashes", True), ("alpha_1_0_rerun", False), ("alpha_1_0_double_count", False), ("pca_alpha_1_logical_cell_count", 3), ("pca_alpha_1_never_expand_to_twelve_cells", True)):
        _require_equal(alpha_cells.get(key), expected, "protocol_alpha_cells", f"protocol.q4.alpha_cells.{key}")
    _require_equal(alpha_cells.get("pca_alpha_1_evaluation_cells"), ["PCA_span/subject", "PCA_span/final", "PCA_span/both"], "protocol_alpha_cells", "protocol.q4.alpha_cells.pca_alpha_1_evaluation_cells")
    _require_equal(alpha_cells.get("pca_alpha_1_primary_cell"), "PCA_span/both", "protocol_alpha_cells", "protocol.q4.alpha_cells.pca_alpha_1_primary_cell")
    _require_equal(alpha_cells.get("pca_alpha_1_fixed_controls"), ["PCA_span/subject", "PCA_span/final"], "protocol_alpha_cells", "protocol.q4.alpha_cells.pca_alpha_1_fixed_controls")
    projector = q4.get("projector")
    pool = q4.get("matched_pool")
    subsets = q4.get("matched_subsets")
    if not isinstance(projector, Mapping) or projector.get("decoder_matrix") != "target twelve decoder rows D in R^(12x768)" or projector.get("arithmetic") != "float64 thin SVD" or projector.get("rank_tolerance") != "max(D.shape) * eps_float64 * s_max" or projector.get("projection") != "P_V delta = V_r.T @ (V_r @ delta)" or projector.get("cast_to_model_dtype") != "only at injection" or projector.get("rank_block") != "if target numerical rank is not 12, block and require a dated amendment":
        raise Stage3Stop("protocol_projector", "Q4 float64 thin-SVD/rank block is not frozen")
    if not isinstance(pool, Mapping) or int(pool.get("candidate_count", -1)) != MATCHED_POOL_SIZE or pool.get("pre_filter") != "experiment-04 causal pre-filter" or pool.get("rank_and_freeze_reads_only") != "rank-training split" or pool.get("target_ids_excluded") is not True or pool.get("evaluation_cannot_change_pool") is not True:
        raise Stage3Stop("protocol_matched_pool", "Q4 matched pool is not frozen at 128 with target exclusion")
    if not isinstance(subsets, Mapping):
        raise Stage3Stop("protocol_matched_subsets", "Q4 matched_subsets is missing")
    for key, expected in (("subset_size", MATCHED_SUBSET_SIZE), ("draw_count", MATCHED_DRAW_COUNT), ("max_redraw_attempts", MATCHED_MAX_ATTEMPTS), ("rng_test_id", Q4_TEST_ID)):
        _require_equal(int(subsets.get(key, -1)), expected, "protocol_matched_subsets", f"q4.matched_subsets.{key}")
    for key, expected in (("scope", "both position set, held-out evaluation split, alpha=1.0 only; one cell per seed"), ("sampling", "uniform without replacement within each subset"), ("exclude_target_ids", True), ("full_rank_required", True), ("rank_deficient_subset_action", "reject and redraw; block seed if 100 full-rank subsets cannot be obtained"), ("insufficient_candidate_action", "block if fewer than 12 eligible candidate ids")):
        _require_equal(subsets.get(key), expected, "protocol_matched_subsets", f"protocol.q4.matched_subsets.{key}")
    stats = q4.get("statistics")
    if not isinstance(stats, Mapping):
        raise Stage3Stop("protocol_statistics", "Q4 statistics are missing")
    for key, expected in (("R_span", "E(delta_span) / E(delta)"), ("R_comp", "E(delta_comp) / E(delta)"), ("matched_edge", "second-largest of the 100 per-seed matched R_span values")):
        _require_equal(stats.get(key), expected, "protocol_statistics", f"protocol.q4.statistics.{key}")
    denominator_guard = stats.get("denominator_guard")
    if not isinstance(denominator_guard, Mapping):
        raise Stage3Stop("protocol_denominator_guard", "Q4 denominator guard is missing")
    for key, expected in (("D", "E(delta)"), ("M", "max(1, E|delta|, E|delta_span|, E|delta_comp|)"), ("tau", "sqrt(float64_eps) * M"), ("condition", "abs(D) <= tau"), ("status", "NON_ESTIMABLE_DENOMINATOR"), ("on_guard", "do not form either ratio and do not regularize or replace D"), ("evaluation_scope", "held-out evaluation split only")):
        _require_equal(denominator_guard.get(key), expected, "protocol_denominator_guard", f"protocol.q4.statistics.denominator_guard.{key}")
    _require_equal(q4.get("positive_rule"), "R_span exceeds the per-seed matched edge in >=6 of 8 seeds; a NON_ESTIMABLE_DENOMINATOR seed supplies no comparison and cannot count toward six", "protocol_statistics", "protocol.q4.positive_rule")
    _require_equal(q4.get("reporting", {}).get("non_estimable_reporting") if isinstance(q4.get("reporting"), Mapping) else None, "emit NON_ESTIMABLE_DENOMINATOR instead of a ratio", "protocol_statistics", "protocol.q4.reporting.non_estimable_reporting")
    data_roles = q4.get("data_roles")
    if not isinstance(data_roles, Mapping) or not isinstance(data_roles.get("role_table"), Mapping):
        raise Stage3Stop("protocol_data_roles", "Amendment-4 role table is missing")
    _require_equal(data_roles.get("split_generation"), "experiment-04 deterministic item-disjoint shuffle after Gate A", "protocol_data_roles", "protocol.q4.data_roles.split_generation")
    _require_equal(data_roles.get("pair_and_directions_stay_together"), True, "protocol_data_roles", "protocol.q4.data_roles.pair_and_directions_stay_together")
    gate_cache = data_roles.get("gate_a_cache")
    if not isinstance(gate_cache, Mapping):
        raise Stage3Stop("protocol_data_roles", "Gate-A cache contract is missing")
    for key, expected in (("selection_mode", "model-derived clean-logit/margin/provenance cache under separate authorization"), ("coverage", "every Stage-3 item and seed"), ("offline_materialization", "derive retained bits and immutable roles from cache only"), ("artifacts", ["stage3_gate_a_cache.jsonl", "split_manifest", "prepare_manifest"]), ("cross_hashes_required", True), ("hash_values_present_in_protocol", False), ("independent_pushed_review_before_q4", True)):
        _require_equal(gate_cache.get(key), expected, "protocol_data_roles", f"protocol.q4.data_roles.gate_a_cache.{key}")
    role_table = data_roles["role_table"]
    expected_roles = {
        "N_lt_40": {"rank_training": "all available", "evaluation": "none", "status": "BLOCKED_INSUFFICIENT_TRAIN"},
        "N_eq_40": {"rank_training": 40, "evaluation": 0, "status": "BLOCKED_INSUFFICIENT_EVAL"},
        "N_41_to_79": {"rank_training": 40, "evaluation": "remainder", "status": "BLOCKED_INSUFFICIENT_EVAL"},
        "N_80_to_190": {"rank_training": 40, "evaluation": "remainder", "status": "usable"},
        "N_gt_190": {"rank_training": 40, "evaluation": 150, "remaining": "unused_after_eval_cap", "status": "usable"},
    }
    _require_equal(role_table, expected_roles, "protocol_data_roles", "protocol.q4.data_roles.role_table")
    context = q4.get("context_arms")
    pca = context.get("pca") if isinstance(context, Mapping) else None
    if not isinstance(pca, Mapping):
        raise Stage3Stop("protocol_pca", "Q4 PCA context contract is missing")
    for key, expected in (("role", "descriptive only"), ("procedure", "experiment-04 generate_pool + fit_complete_bases"), ("fit_scope", "per Stage-3 seed"), ("generic_text_pool_tokens", 8192), ("basis_fit_dtype", "float64"), ("top12_projector", True), ("fallback_task_pool", False), ("primary_cell", "PCA_span/both"), ("logical_cell_count", 3)):
        _require_equal(pca.get(key), expected, "protocol_pca", f"protocol.q4.context_arms.pca.{key}")
    _require_equal(pca.get("preserve"), ["pool construction", "pool sampling", "layer-8 extraction", "centering", "fitting order", "serialization"], "protocol_pca", "protocol.q4.context_arms.pca.preserve")
    _require_equal(pca.get("implementation_deviation"), "experiment-04 code currently uses float32 eigh plus refinements; Stage3 freezes float64 and records this deviation", "protocol_pca", "protocol.q4.context_arms.pca.implementation_deviation")
    _require_equal(pca.get("fit_excludes"), ["task ids", "Gate-A bits", "role assignments", "task deltas", "evaluation items"], "protocol_pca", "protocol.q4.context_arms.pca.fit_excludes")
    _require_equal(pca.get("alpha_1_evaluation_cells"), ["PCA_span/subject", "PCA_span/final", "PCA_span/both"], "protocol_pca", "protocol.q4.context_arms.pca.alpha_1_evaluation_cells")
    _require_equal(pca.get("fixed_control_cells"), ["PCA_span/subject", "PCA_span/final"], "protocol_pca", "protocol.q4.context_arms.pca.fixed_control_cells")
    if not path.is_file():
        raise Stage3Stop("protocol_path", f"protocol path does not exist: {path}")


def _hf_cache_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _find_snapshot(repo_dir: str, revision: str) -> dict[str, Any]:
    repository = _hf_cache_root() / repo_dir
    root = repository / "snapshots" / revision
    if not root.is_dir():
        raise Stage3Stop("local_snapshot", f"required local snapshot is absent: {root}")
    ref_path = repository / "refs" / "main"
    if not ref_path.is_file():
        raise Stage3Stop("local_snapshot_ref", f"required Hugging Face refs/main is absent: {ref_path}")
    observed_ref = ref_path.read_text(encoding="utf-8").strip()
    if observed_ref != revision:
        raise Stage3Stop("local_snapshot_ref", f"{ref_path} points to {observed_ref!r}, expected {revision!r}")
    return {"expected_revision": revision, "observed_revision": root.name, "snapshot_path": str(root), "refs_main_path": str(ref_path), "refs_main_revision": observed_ref}


def _check_local_snapshots(protocol: Mapping[str, Any]) -> dict[str, Any]:
    revisions = protocol["model"]["expected_local_snapshot_revisions"]
    return {
        "gpt2": _find_snapshot("models--gpt2", str(revisions["gpt2"])),
        "sae": _find_snapshot("models--jbloom--GPT2-Small-SAEs-Reformatted", str(revisions["sae"])),
        "cache_root": str(_hf_cache_root()),
    }


def _offline_env() -> dict[str, Any]:
    observed = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MPLBACKEND")}
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if observed[key] != "1":
            raise Stage3Stop("offline_provenance", f"{key}=1 is required for a model-backed Stage-3 path")
    return observed


def _load_model_stack() -> tuple[Any, ...]:
    """Import the existing model primitives only after a model-backed gate."""

    if str(EXP04) not in sys.path:
        sys.path.insert(0, str(EXP04))
    from pilot import build_stimuli, clean_pass, directed_indices, logit_difference, load_model, require_one_token  # type: ignore
    from pilot import positions_for_kind, PatchEngine, gather_positions, feature_delta, decoded_delta  # type: ignore
    from pilot import set_determinism  # type: ignore
    from run_experiment import candidate_prefilter_full, generate_pool  # type: ignore
    return (build_stimuli, clean_pass, directed_indices, logit_difference, load_model, require_one_token, positions_for_kind, PatchEngine, gather_positions, feature_delta, decoded_delta, set_determinism, candidate_prefilter_full, generate_pool)


def _load_clean_stack() -> tuple[Any, ...]:
    """Load only the clean-logit primitives for the Gate-A cache phase."""

    if str(EXP04) not in sys.path:
        sys.path.insert(0, str(EXP04))
    from pilot import build_stimuli, load_model, require_one_token, set_determinism  # type: ignore
    return build_stimuli, load_model, require_one_token, set_determinism


def _clean_readouts(model: Any, tokens: Any, lengths: Any, is_id: int, are_id: int, *, chunk: int = 32) -> tuple[Any, Any, Any]:
    """Extract only the two final-position logits and d, never retain vocab logits."""

    torch = __import__("torch")
    is_values: list[Any] = []
    are_values: list[Any] = []
    for start in range(0, int(tokens.shape[0]), chunk):
        stop = min(start + chunk, int(tokens.shape[0]))
        with torch.no_grad():
            logits = model(tokens[start:stop], return_type="logits")
        rows = torch.arange(stop - start)
        final = lengths[start:stop] - 1
        is_values.append(logits[rows, final, is_id].detach().float().cpu())
        are_values.append(logits[rows, final, are_id].detach().float().cpu())
        del logits
    is_logits = torch.cat(is_values)
    are_logits = torch.cat(are_values)
    return is_logits, are_logits, are_logits - is_logits


def _cache_manifest(protocol: Mapping[str, Any], *, status: str, git: Mapping[str, Any], snapshots: Mapping[str, Any], env: Mapping[str, Any], seed_rows: Sequence[Mapping[str, Any]], protocol_path: Path, output_path: Path) -> dict[str, Any]:
    return {
        "record_type": "manifest",
        "schema": CACHE_SCHEMA,
        "status": status,
        "protocol_schema": PROTOCOL_SCHEMA,
        "protocol_sha256": _protocol_hash(protocol),
        "protocol_path": str(protocol_path),
        "output_path": str(output_path),
        "model": {"name": "gpt2-small", "device": "cpu", "dtype": "float32", "sae_loaded": False},
        "git": dict(git),
        "snapshots": dict(snapshots),
        "offline_env": dict(env),
        "stage3_seeds": list(STAGE3_SEEDS),
        "requested_pairs": REQUESTED_PAIRS,
        "gate_a": {"minimum_fraction": protocol["gate_a"]["conditions"]["both_members_signed_correct_fraction_at_least"], "minimum_retained_pairs": protocol["gate_a"]["conditions"]["minimum_retained_pairs"], "minimum_median_gap": protocol["gate_a"]["conditions"]["median_clean_d_gap_at_least"]},
        "coverage": {"completed_seed_count": len(seed_rows), "expected_seed_count": len(STAGE3_SEEDS), "rows_per_seed": REQUESTED_PAIRS},
        "input_provenance": {"stimulus_generator": "experiment_04.pilot.build_stimuli", "tokenization": "pinned local GPT-2 tokenizer", "readout_positions": "per-item lengths - 1", "cache_authorization": "separate offline model-backed Gate-A authorization"},
        "no_intervention": True,
        "no_sae": True,
    }


def _cache_pair_rows(seed: int, stimuli: Any, is_logits: Any, are_logits: Any, d: Any, token_ids: Mapping[str, int], protocol_sha256: str, provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_id in range(REQUESTED_PAIRS):
        singular = 2 * pair_id
        plural = singular + 1
        singular_d = float(d[singular])
        plural_d = float(d[plural])
        singular_is = float(is_logits[singular])
        singular_are = float(are_logits[singular])
        plural_is = float(is_logits[plural])
        plural_are = float(are_logits[plural])
        singular_pass = singular_d < 0.0
        plural_pass = plural_d > 0.0
        pair_retained = bool(singular_pass and plural_pass)
        record = stimuli.pair_records[pair_id]
        directions = [
            {
                "direction": DIRECTIONS[0], "base_item_id": singular, "source_item_id": plural, "sign_alignment": 1.0,
                "base_readout": singular_d, "source_readout": plural_d, "raw_readout_delta": plural_d - singular_d,
                "direction_pass": bool(singular_pass and plural_pass), "margin": min(-singular_d, plural_d),
            },
            {
                "direction": DIRECTIONS[1], "base_item_id": plural, "source_item_id": singular, "sign_alignment": -1.0,
                "base_readout": plural_d, "source_readout": singular_d, "raw_readout_delta": singular_d - plural_d,
                "direction_pass": bool(plural_pass and singular_pass), "margin": min(-singular_d, plural_d),
            },
        ]
        subject_position = int(stimuli.subject_positions[pair_id])
        def item_provenance(item_id: int, text_key: str) -> dict[str, Any]:
            length = int(stimuli.lengths[item_id])
            token_list = [int(item) for item in stimuli.tokens[item_id, :length].detach().cpu().tolist()]
            text = str(record.get(text_key, ""))
            return {"item_id": item_id, "text_sha256": _sha256_bytes(text.encode("utf-8")), "token_ids_sha256": _sha256_bytes(_json_bytes(token_list)), "subject_position": subject_position, "final_position": length - 1}
        rows.append({
            "record_type": "pair", "schema": CACHE_SCHEMA, "seed": int(seed), "pair_id": int(pair_id), "item_id": int(pair_id),
            "protocol_sha256": protocol_sha256, "token_ids": {"is": int(token_ids["is"]), "are": int(token_ids["are"])},
            "items": {
                "singular": {**item_provenance(singular, "singular_text"), "is_logit": singular_is, "are_logit": singular_are, "readout_d": singular_d, "signed_correct": bool(singular_pass)},
                "plural": {**item_provenance(plural, "plural_text"), "is_logit": plural_is, "are_logit": plural_are, "readout_d": plural_d, "signed_correct": bool(plural_pass)},
            },
            "directions": directions,
            "pair_retained": pair_retained,
            "provenance": dict(provenance),
        })
    return rows


def _load_cache(path: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    rows = _read_jsonl(path, "Stage-3 Gate-A cache")
    manifest = rows[0] if rows[0].get("record_type") == "manifest" else None
    if manifest is None or manifest.get("schema") != CACHE_SCHEMA:
        raise Stage3Stop("cache_manifest", "Gate-A cache must start with a complete manifest record")
    if manifest.get("status") != "COMPLETE":
        raise Stage3Stop("cache_status", f"Gate-A cache status is {manifest.get('status')!r}; materialization requires COMPLETE")
    if manifest.get("protocol_sha256") != _protocol_hash(protocol):
        raise Stage3Stop("cache_protocol_hash", "Gate-A cache protocol hash differs from the supplied frozen protocol")
    coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping) or int(coverage.get("completed_seed_count", -1)) != len(STAGE3_SEEDS) or int(coverage.get("expected_seed_count", -1)) != len(STAGE3_SEEDS) or int(coverage.get("rows_per_seed", -1)) != REQUESTED_PAIRS:
        raise Stage3Stop("cache_coverage", "Gate-A cache manifest coverage is not the complete frozen eight-seed/240-pair set")
    pairs = rows[1:]
    expected = {(seed, pair) for seed in STAGE3_SEEDS for pair in range(REQUESTED_PAIRS)}
    observed: set[tuple[int, int]] = set()
    for row in pairs:
        if row.get("record_type") != "pair" or row.get("schema") != CACHE_SCHEMA:
            raise Stage3Stop("cache_row_schema", "Gate-A cache contains a non-pair record")
        if row.get("protocol_sha256") != _protocol_hash(protocol):
            raise Stage3Stop("cache_protocol_hash", "Gate-A pair row protocol hash differs from the supplied frozen protocol")
        key = (int(row.get("seed", -1)), int(row.get("pair_id", -1)))
        if key in observed:
            raise Stage3Stop("cache_duplicate", f"duplicate Gate-A cache row {key}")
        observed.add(key)
        if key not in expected:
            raise Stage3Stop("cache_coverage", f"unexpected Gate-A cache row {key}")
        directions = row.get("directions")
        if not isinstance(directions, list) or [item.get("direction") for item in directions] != list(DIRECTIONS):
            raise Stage3Stop("cache_direction_order", f"pair {key} direction order is not frozen")
        token_pair = row.get("token_ids")
        if not isinstance(token_pair, Mapping) or set(token_pair) != {"is", "are"}:
            raise Stage3Stop("cache_token_ids", f"pair {key} lacks the frozen readout token ids")
        items = row.get("items")
        if not isinstance(items, Mapping):
            raise Stage3Stop("cache_item_schema", f"pair {key} lacks item provenance")
        for name, expected_id in (("singular", 2 * key[1]), ("plural", 2 * key[1] + 1)):
            item = items.get(name)
            if not isinstance(item, Mapping) or int(item.get("item_id", -1)) != expected_id:
                raise Stage3Stop("cache_item_schema", f"pair {key} {name} item id is not frozen")
            for field in ("text_sha256", "token_ids_sha256", "subject_position", "final_position", "is_logit", "are_logit", "readout_d"):
                if field not in item:
                    raise Stage3Stop("cache_item_schema", f"pair {key} {name} lacks {field}")
    if observed != expected:
        missing = sorted(expected - observed)[:8]
        raise Stage3Stop("cache_coverage", f"Gate-A cache is incomplete; missing examples={missing}")
    return manifest, pairs, _sha256_file(path)


def _recompute_gate_a(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    out: dict[int, dict[str, Any]] = {}
    for seed in STAGE3_SEEDS:
        seed_rows = sorted(by_seed.get(seed, []), key=lambda item: int(item["pair_id"]))
        if len(seed_rows) != REQUESTED_PAIRS:
            raise Stage3Stop("cache_coverage", f"seed {seed} has {len(seed_rows)} rows, expected {REQUESTED_PAIRS}")
        retained: list[int] = []
        gaps: list[float] = []
        for row in seed_rows:
            items = row.get("items")
            if not isinstance(items, Mapping) or not isinstance(items.get("singular"), Mapping) or not isinstance(items.get("plural"), Mapping):
                raise Stage3Stop("cache_item_schema", f"seed {seed} pair {row.get('pair_id')} lacks item logits")
            singular_is = _float32(items["singular"].get("is_logit"))
            singular_are = _float32(items["singular"].get("are_logit"))
            plural_is = _float32(items["plural"].get("is_logit"))
            plural_are = _float32(items["plural"].get("are_logit"))
            singular_expected = _float32(singular_are - singular_is)
            plural_expected = _float32(plural_are - plural_is)
            singular_d = _float32(items["singular"].get("readout_d"))
            plural_d = _float32(items["plural"].get("readout_d"))
            if singular_d != singular_expected or plural_d != plural_expected:
                raise Stage3Stop("cache_readout_recompute", f"seed {seed} pair {row.get('pair_id')} readout does not match float32 cached logits")
            if not (math.isfinite(singular_d) and math.isfinite(plural_d)):
                raise Stage3Stop("cache_nonfinite", f"seed {seed} pair {row.get('pair_id')} has non-finite readout")
            singular_pass = singular_d < 0.0
            plural_pass = plural_d > 0.0
            expected_retained = bool(singular_pass and plural_pass)
            if bool(row.get("pair_retained")) != expected_retained:
                raise Stage3Stop("cache_gate_recompute", f"seed {seed} pair {row.get('pair_id')} retained bit disagrees with cached logits")
            directions = row.get("directions")
            if not isinstance(directions, list) or len(directions) != len(DIRECTIONS):
                raise Stage3Stop("cache_direction_schema", f"seed {seed} pair {row.get('pair_id')} lacks two direction records")
            expected_margin = min(-singular_d, plural_d)
            for index, direction in enumerate(directions):
                if not isinstance(direction, Mapping) or direction.get("direction") != DIRECTIONS[index]:
                    raise Stage3Stop("cache_direction_schema", f"seed {seed} pair {row.get('pair_id')} direction order is not frozen")
                expected_base, expected_source, expected_sign = ((2 * int(row["pair_id"]), 2 * int(row["pair_id"]) + 1, 1.0) if index == 0 else (2 * int(row["pair_id"]) + 1, 2 * int(row["pair_id"]), -1.0))
                if int(direction.get("base_item_id", -1)) != expected_base or int(direction.get("source_item_id", -1)) != expected_source or float(direction.get("sign_alignment")) != expected_sign:
                    raise Stage3Stop("cache_direction_schema", f"seed {seed} pair {row.get('pair_id')} direction item/sign binding is not frozen")
                if bool(direction.get("direction_pass")) != expected_retained:
                    raise Stage3Stop("cache_gate_recompute", f"seed {seed} pair {row.get('pair_id')} direction bit disagrees with cached logits")
                if not math.isclose(float(direction.get("margin")), expected_margin, rel_tol=0.0, abs_tol=1e-7):
                    raise Stage3Stop("cache_gate_recompute", f"seed {seed} pair {row.get('pair_id')} direction margin disagrees with cached logits")
                expected_delta = plural_d - singular_d if index == 0 else singular_d - plural_d
                if not math.isclose(float(direction.get("raw_readout_delta")), expected_delta, rel_tol=0.0, abs_tol=1e-7):
                    raise Stage3Stop("cache_gate_recompute", f"seed {seed} pair {row.get('pair_id')} direction delta disagrees with cached logits")
            gaps.append(plural_d - singular_d)
            if expected_retained:
                retained.append(int(row["pair_id"]))
        sorted_gaps = sorted(gaps)
        # Match torch.Tensor.median exactly: for an even-length vector this is
        # the lower middle order statistic, not the average of the two middles.
        median = sorted_gaps[(len(sorted_gaps) - 1) // 2]
        fraction = len(retained) / float(REQUESTED_PAIRS)
        out[seed] = {
            "seed": seed, "generated_pairs": REQUESTED_PAIRS, "retained_pair_ids": retained,
            "retained_pairs": len(retained), "both_members_signed_correct_fraction": fraction,
            "median_d_gap_all_generated_pairs": median,
            "passed": bool(fraction >= 0.60 and len(retained) >= 140 and median >= 1.0),
        }
    return out


def _validate_stimuli_against_cache(stimuli: Any, seed: int, cache_rows: Sequence[Mapping[str, Any]], token_ids: Mapping[str, int]) -> None:
    """Rebuild the exact seed stimuli and bind every cache provenance field."""

    by_pair = {int(row.get("pair_id", -1)): row for row in cache_rows if int(row.get("seed", -1)) == seed}
    if set(by_pair) != set(range(REQUESTED_PAIRS)):
        raise Stage3Stop("cache_stimulus_coverage", f"seed {seed} cache rows do not cover all stimulus pairs")
    provenance = by_pair[0].get("provenance")
    if not isinstance(provenance, Mapping) or int(provenance.get("requested_pairs", -1)) != REQUESTED_PAIRS or int(provenance.get("seed", -1)) != seed:
        raise Stage3Stop("cache_stimulus_provenance", f"seed {seed} cache stimulus provenance is missing or mismatched")
    for pair_id in range(REQUESTED_PAIRS):
        row = by_pair[pair_id]
        items = row.get("items")
        if not isinstance(items, Mapping):
            raise Stage3Stop("cache_stimulus_provenance", f"seed {seed} pair {pair_id} lacks item provenance")
        record = stimuli.pair_records[pair_id]
        expected_subject = int(stimuli.subject_positions[pair_id])
        for name, item_id, text_key in (("singular", 2 * pair_id, "singular_text"), ("plural", 2 * pair_id + 1, "plural_text")):
            cached = items.get(name)
            if not isinstance(cached, Mapping):
                raise Stage3Stop("cache_stimulus_provenance", f"seed {seed} pair {pair_id} {name} provenance is missing")
            length = int(stimuli.lengths[item_id])
            tokens = [int(item) for item in stimuli.tokens[item_id, :length].detach().cpu().tolist()]
            text = str(record.get(text_key, ""))
            expected = {
                "item_id": item_id,
                "text_sha256": _sha256_bytes(text.encode("utf-8")),
                "token_ids_sha256": _sha256_bytes(_json_bytes(tokens)),
                "subject_position": expected_subject,
                "final_position": length - 1,
            }
            for field, value in expected.items():
                if cached.get(field) != value:
                    raise Stage3Stop("cache_stimulus_provenance", f"seed {seed} pair {pair_id} {name} {field} disagrees with rebuilt stimuli")
        token_pair = row.get("token_ids")
        if token_pair != {"is": int(token_ids["is"]), "are": int(token_ids["are"])}:
            raise Stage3Stop("cache_token_ids", f"seed {seed} pair {pair_id} readout token ids disagree with rebuilt tokenizer")


def _call_amendment4_split(pair_ids: Sequence[int], seed: int) -> dict[str, Any]:
    # The core module is pure statistics; importing it is allowed on the
    # model-free materialization path.  No local shuffle fallback is permitted.
    try:
        import exp05_core as core  # type: ignore
    except ImportError as exc:
        raise Stage3Stop("core_missing", "exp05_core.py is required for Amendment-4 split materialization") from exc
    function = getattr(core, "amendment4_split", None)
    if not callable(function):
        raise Stage3Stop("core_api", "exp05_core.amendment4_split is required")
    try:
        result = function(pair_ids, seed)
    except Exception as exc:
        raise Stage3Stop("core_split", f"exp05_core.amendment4_split failed: {exc}") from exc
    if hasattr(result, "as_dict"):
        result = result.as_dict()
    if not isinstance(result, Mapping):
        raise Stage3Stop("core_split_schema", "amendment4_split must return a mapping or SplitResult.as_dict()")
    return dict(result)


def _expected_role_status(n: int) -> str:
    if n < TRAIN_PAIRS:
        return "BLOCKED_INSUFFICIENT_TRAIN"
    if n < 80:
        return "BLOCKED_INSUFFICIENT_EVAL"
    return "READY"


def _validate_split_result(result: Mapping[str, Any], retained: Sequence[int], seed: int) -> dict[str, Any]:
    n = len(retained)
    expected_status = _expected_role_status(n)
    status = str(result.get("status", ""))
    if status != expected_status:
        raise Stage3Stop("split_contract", f"core split seed {seed} status {status!r} != frozen {expected_status!r}")
    assignments = result.get("assignments")
    if not isinstance(assignments, list):
        raise Stage3Stop("split_contract", f"core split seed {seed} lacks assignments")
    normalized: dict[int, str] = {}
    for row in assignments:
        if not isinstance(row, Mapping):
            raise Stage3Stop("split_contract", f"core split seed {seed} assignment is not an object")
        pair_id = int(row.get("pair_id", -1))
        role = str(row.get("role", ""))
        if pair_id in normalized or pair_id not in set(retained) or role not in {"rank_training", "evaluation", "unused_after_eval_cap"}:
            raise Stage3Stop("split_contract", f"invalid core assignment seed={seed} pair={pair_id} role={role!r}")
        normalized[pair_id] = role
    if set(normalized) != set(int(item) for item in retained):
        raise Stage3Stop("split_contract", f"core split seed {seed} does not assign every retained pair exactly once")
    expected_counts = {
        "rank_training": n if n < TRAIN_PAIRS else TRAIN_PAIRS,
        "evaluation": 0 if n < TRAIN_PAIRS else (n - TRAIN_PAIRS if n <= 190 else EVAL_CAP),
        "unused_after_eval_cap": 0 if n <= 190 else n - 190,
    }
    counts = {role: sum(value == role for value in normalized.values()) for role in expected_counts}
    if counts != expected_counts:
        raise Stage3Stop("split_contract", f"core split seed {seed} counts {counts} != expected {expected_counts}")
    result_copy = dict(result)
    result_copy["status"] = status
    result_copy["assignments"] = [{"pair_id": pair_id, "role": normalized[pair_id]} for pair_id in sorted(normalized)]
    result_copy["rank_training_pair_ids"] = sorted(pair_id for pair_id, role in normalized.items() if role == "rank_training")
    result_copy["evaluation_pair_ids"] = sorted(pair_id for pair_id, role in normalized.items() if role == "evaluation")
    result_copy["unused_after_eval_cap_pair_ids"] = sorted(pair_id for pair_id, role in normalized.items() if role == "unused_after_eval_cap")
    result_copy["retained_pair_count"] = n
    if "retained_pair_count" in result and int(result["retained_pair_count"]) != n:
        raise Stage3Stop("split_contract", f"core split seed {seed} retained_pair_count disagrees with input")
    for field in ("rank_training_pair_ids", "evaluation_pair_ids", "unused_after_eval_cap_pair_ids"):
        if field in result and sorted(int(item) for item in result[field]) != result_copy[field]:
            raise Stage3Stop("split_contract", f"core split seed {seed} {field} disagrees with assignments")
    if "block_reason" in result and status == "READY" and result.get("block_reason") is not None:
        raise Stage3Stop("split_contract", f"core split seed {seed} READY split has a block reason")
    return result_copy


def _materialize_splits(protocol_path: Path, cache_path: Path, split_path: Path, csv_path: Path, prepare_path: Path) -> dict[str, Any]:
    _validate_runtime_paths(immutable={"protocol": protocol_path, "gate_cache": cache_path}, runtime={"split_manifest": split_path, "split_csv": csv_path, "prepare_manifest": prepare_path})
    # The prepare manifest is the portable anchor for the split CSV.  Refuse
    # to publish a bundle whose CSV lives elsewhere; otherwise the manifest
    # could only be replayed on the original machine-specific path.
    split_manifest_name = _portable_artifact_name(split_path, anchor=prepare_path, label="split manifest")
    split_csv_name = _portable_artifact_name(csv_path, anchor=prepare_path, label="split CSV")
    protocol = _read_json(protocol_path, "protocol")
    _validate_protocol(protocol, protocol_path)
    manifest, rows, cache_hash = _load_cache(cache_path, protocol)
    gate = _recompute_gate_a(rows)
    seed_splits: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for seed in STAGE3_SEEDS:
        retained = gate[seed]["retained_pair_ids"]
        result = _validate_split_result(_call_amendment4_split(retained, seed), retained, seed)
        row = {"seed": seed, "cache_gate": gate[seed], "gate_a_passed": gate[seed]["passed"] is True, "split": result}
        seed_splits.append(row)
        roles = {int(item["pair_id"]): str(item["role"]) for item in result["assignments"]}
        for pair_id in sorted(set(retained)):
            csv_rows.append({"seed": seed, "pair_id": pair_id, "retained": True, "role": roles[pair_id], "rank_training": roles[pair_id] == "rank_training", "evaluation": roles[pair_id] == "evaluation", "unused_after_eval_cap": roles[pair_id] == "unused_after_eval_cap"})
    split_manifest = {
        "schema": SPLIT_SCHEMA, "status": "COMPLETE", "protocol_sha256": _protocol_hash(protocol), "cache_sha256": cache_hash,
        "cache_schema": manifest.get("schema"), "generated_by": "stage3.py materialize-splits", "stage3_seeds": list(STAGE3_SEEDS),
        "amendment4_split": "exp05_core.amendment4_split(pair_ids, seed)", "seeds": seed_splits,
    }
    _atomic_json(split_path, split_manifest)
    _atomic_csv(csv_path, csv_rows, ("seed", "pair_id", "retained", "role", "rank_training", "evaluation", "unused_after_eval_cap"))
    split_hash = _sha256_file(split_path)
    csv_hash = _sha256_file(csv_path)
    prepare = {
        "schema": PREPARE_SCHEMA, "status": "COMPLETE", "protocol_sha256": _protocol_hash(protocol), "gate_cache_sha256": cache_hash,
        # These are intentionally relative basenames.  At run time the CSV is
        # resolved only beside this prepare manifest (never beside stage3.py
        # and never through an arbitrary fallback path).
        "split_manifest_path": split_manifest_name, "split_csv_path": split_csv_name,
        "split_manifest_sha256": split_hash, "split_csv_sha256": csv_hash, "cross_hash_sha256": _sha256_bytes(_json_bytes({"gate_cache_sha256": cache_hash, "split_manifest_sha256": split_hash, "split_csv_sha256": csv_hash})),
        "independent_review_required": True, "q4_input_boundary": ["protocol", "stage3_gate_a_cache", "stage3_split_manifest", "stage3_prepare_manifest", "review_receipt"],
        "candidate_C_or_stage2_input": False,
        "seeds": [{"seed": seed, "status": row["split"]["status"], "gate_a_passed": row["gate_a_passed"], "retained_pairs": row["cache_gate"]["retained_pairs"], "rank_training_pairs": len(row["split"]["rank_training_pair_ids"]), "evaluation_pairs": len(row["split"]["evaluation_pair_ids"])} for seed, row in ((item["seed"], item) for item in seed_splits)],
    }
    _atomic_json(prepare_path, prepare)
    return {"status": "COMPLETE", "split_manifest": str(split_path), "split_csv": str(csv_path), "prepare_manifest": str(prepare_path), "cache_sha256": cache_hash, "split_manifest_sha256": _sha256_file(split_path), "split_csv_sha256": _sha256_file(csv_path)}


def _load_split_inputs(protocol: Mapping[str, Any], cache_path: Path, split_path: Path, prepare_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any], dict[str, Any], str]:
    cache_manifest, cache_rows, cache_hash = _load_cache(cache_path, protocol)
    gate = _recompute_gate_a(cache_rows)
    split = _read_json(split_path, "Stage-3 split manifest")
    prepare = _read_json(prepare_path, "Stage-3 prepare manifest")
    if split.get("schema") != SPLIT_SCHEMA or split.get("status") != "COMPLETE":
        raise Stage3Stop("split_manifest", "split manifest is not COMPLETE")
    if prepare.get("schema") != PREPARE_SCHEMA or prepare.get("status") != "COMPLETE":
        raise Stage3Stop("prepare_manifest", "prepare manifest is not COMPLETE")
    protocol_hash = _protocol_hash(protocol)
    if split.get("protocol_sha256") != protocol_hash or prepare.get("protocol_sha256") != protocol_hash:
        raise Stage3Stop("prepare_protocol_hash", "prepared Stage-3 artifacts do not match protocol")
    if split.get("cache_sha256") != cache_hash or prepare.get("gate_cache_sha256") != cache_hash:
        raise Stage3Stop("prepare_cache_hash", "prepared Stage-3 artifacts do not match Gate-A cache")
    split_manifest_value = prepare.get("split_manifest_path")
    if not isinstance(split_manifest_value, str):
        raise Stage3Stop("prepare_cross_hash", "prepare manifest lacks the relative split manifest basename")
    if split_manifest_value != _portable_artifact_name(split_path, anchor=prepare_path, label="split manifest"):
        raise Stage3Stop("prepare_cross_hash", "prepare manifest split manifest binding does not match the supplied manifest")
    split_csv_value = prepare.get("split_csv_path")
    split_csv_path = _resolve_prepared_split_csv(prepare_path, split_csv_value)
    if prepare.get("split_manifest_sha256") != _sha256_file(split_path) or prepare.get("split_csv_sha256") != _sha256_file(split_csv_path):
        raise Stage3Stop("prepare_cross_hash", "prepare manifest split hashes do not match prepared artifacts")
    cross_hash = _sha256_bytes(_json_bytes({"gate_cache_sha256": cache_hash, "split_manifest_sha256": _sha256_file(split_path), "split_csv_sha256": _sha256_file(split_csv_path)}))
    if prepare.get("cross_hash_sha256") != cross_hash:
        raise Stage3Stop("prepare_cross_hash", "prepare manifest cross hash does not recompute from its bound artifacts")
    expected_seed_set = set(STAGE3_SEEDS)
    split_seed_rows = split.get("seeds")
    if not isinstance(split_seed_rows, list) or {int(item.get("seed", -1)) for item in split_seed_rows if isinstance(item, Mapping)} != expected_seed_set or len(split_seed_rows) != len(STAGE3_SEEDS):
        raise Stage3Stop("split_seed_coverage", "split manifest must contain exactly one row for every registered seed")
    split_by_seed: dict[int, Mapping[str, Any]] = {}
    for item in split_seed_rows:
        if not isinstance(item, Mapping):
            raise Stage3Stop("split_seed_schema", "split manifest seed row is not an object")
        seed = int(item.get("seed", -1))
        if seed in split_by_seed:
            raise Stage3Stop("split_seed_coverage", f"duplicate split manifest seed {seed}")
        split_by_seed[seed] = item
        retained = gate[seed]["retained_pair_ids"]
        recomputed = _validate_split_result(_call_amendment4_split(retained, seed), retained, seed)
        observed_split = item.get("split")
        if not isinstance(observed_split, Mapping):
            raise Stage3Stop("split_contract", f"seed {seed} split payload is missing")
        if _json_bytes(_validate_split_result(observed_split, retained, seed)) != _json_bytes(recomputed):
            raise Stage3Stop("split_contract", f"seed {seed} split fields do not equal a fresh Amendment-4 recomputation")
        observed_gate = item.get("cache_gate")
        if not isinstance(observed_gate, Mapping) or _json_bytes(dict(observed_gate)) != _json_bytes(gate[seed]):
            raise Stage3Stop("split_gate_binding", f"seed {seed} split Gate-A facts do not equal cache recomputation")
        if bool(item.get("gate_a_passed")) is not bool(gate[seed]["passed"]):
            raise Stage3Stop("split_gate_binding", f"seed {seed} split Gate-A status is not bound to cache recomputation")
    prepare_seeds = prepare.get("seeds")
    if not isinstance(prepare_seeds, list) or {int(item.get("seed", -1)) for item in prepare_seeds if isinstance(item, Mapping)} != expected_seed_set or len(prepare_seeds) != len(STAGE3_SEEDS):
        raise Stage3Stop("prepare_seed_coverage", "prepare manifest must contain exactly one row for every registered seed")
    expected_prepare = {
        seed: {"seed": seed, "status": split_by_seed[seed]["split"]["status"], "gate_a_passed": bool(split_by_seed[seed]["gate_a_passed"]), "retained_pairs": gate[seed]["retained_pairs"], "rank_training_pairs": len(split_by_seed[seed]["split"]["rank_training_pair_ids"]), "evaluation_pairs": len(split_by_seed[seed]["split"]["evaluation_pair_ids"])}
        for seed in STAGE3_SEEDS
    }
    for item in prepare_seeds:
        if not isinstance(item, Mapping) or int(item.get("seed", -1)) not in expected_prepare or dict(item) != expected_prepare[int(item["seed"])]:
            raise Stage3Stop("prepare_seed_binding", "prepare manifest seed summary does not equal the split/cache facts")
    expected_csv: dict[tuple[int, int], dict[str, str]] = {}
    for seed in STAGE3_SEEDS:
        split_payload = split_by_seed[seed]["split"]
        roles = {int(item["pair_id"]): str(item["role"]) for item in split_payload["assignments"]}
        for pair_id in sorted(roles):
            role = roles[pair_id]
            expected_csv[(seed, pair_id)] = {"seed": str(seed), "pair_id": str(pair_id), "retained": "True", "role": role, "rank_training": str(role == "rank_training"), "evaluation": str(role == "evaluation"), "unused_after_eval_cap": str(role == "unused_after_eval_cap")}
    try:
        with split_csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["seed", "pair_id", "retained", "role", "rank_training", "evaluation", "unused_after_eval_cap"]:
                raise Stage3Stop("split_csv_schema", "split CSV columns differ from the frozen contract")
            observed_csv: dict[tuple[int, int], dict[str, str]] = {}
            for row in reader:
                key = (int(row.get("seed", -1)), int(row.get("pair_id", -1)))
                if key in observed_csv:
                    raise Stage3Stop("split_csv_coverage", f"duplicate split CSV row {key}")
                observed_csv[key] = {field: str(row.get(field, "")) for field in reader.fieldnames}
    except (OSError, ValueError) as exc:
        raise Stage3Stop("split_csv_schema", f"cannot read split CSV: {exc}") from exc
    if observed_csv != expected_csv:
        raise Stage3Stop("split_csv_binding", "split CSV rows are not exactly equal to the recomputed seed assignments")
    return cache_manifest, cache_rows, cache_hash, gate, split, _sha256_file(split_path)


def _review_receipt(path: Path, *, expected_commit: str, cache_hash: str, split_hash: str, prepare_hash: str) -> dict[str, Any]:
    receipt = _read_json(path, "independent prepared-artifact review receipt")
    if receipt.get("schema") != REVIEW_SCHEMA or receipt.get("status") != "ACCEPT":
        raise Stage3Stop("review_receipt", "review receipt must have frozen schema and status ACCEPT")
    if receipt.get("reviewed_commit") != expected_commit:
        raise Stage3Stop("review_receipt", "review receipt commit does not match expected HEAD")
    if not isinstance(receipt.get("reviewer"), str) or not receipt["reviewer"].strip():
        raise Stage3Stop("review_receipt", "review receipt must identify the independent reviewer")
    if not isinstance(receipt.get("model"), str) or not receipt["model"].strip():
        raise Stage3Stop("review_receipt", "review receipt must identify the reviewing model")
    if receipt.get("independent") is not True or not receipt.get("pushed_evidence"):
        raise Stage3Stop("review_receipt", "review receipt must carry independent=true and pushed evidence")
    if receipt.get("self_sha256") != _self_hash(receipt):
        raise Stage3Stop("review_receipt", "review receipt self hash is absent or invalid")
    for key, expected in (("gate_cache_sha256", cache_hash), ("split_manifest_sha256", split_hash), ("prepare_manifest_sha256", prepare_hash)):
        if receipt.get(key) != expected:
            raise Stage3Stop("review_receipt", f"review receipt {key} does not match prepared artifact")
    return receipt


def _harness_receipt(path: Path, *, expected_commit: str, protocol_hash: str) -> dict[str, Any]:
    receipt = _read_json(path, "independent Stage-3 harness receipt")
    if receipt.get("schema") != HARNESS_SCHEMA or receipt.get("status") != "ACCEPT":
        raise Stage3Stop("harness_receipt", "harness receipt must have frozen schema and status ACCEPT")
    if receipt.get("reviewed_commit") != expected_commit or receipt.get("protocol_sha256") != protocol_hash:
        raise Stage3Stop("harness_receipt", "harness receipt commit/protocol binding differs from this run")
    if not isinstance(receipt.get("reviewer"), str) or not receipt["reviewer"].strip() or not isinstance(receipt.get("model"), str) or not receipt["model"].strip():
        raise Stage3Stop("harness_receipt", "harness receipt must identify reviewer and model")
    if receipt.get("independent") is not True or not receipt.get("pushed_evidence"):
        raise Stage3Stop("harness_receipt", "harness receipt must carry independent=true and pushed evidence")
    if receipt.get("self_sha256") != _self_hash(receipt):
        raise Stage3Stop("harness_receipt", "harness receipt self hash is absent or invalid")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping):
        raise Stage3Stop("harness_receipt", "harness receipt checks are missing")
    expected_checks = ("causal_handle_floor", "zero", "start_at_layer8_equivalence", "prompt_swap")
    if set(checks) != set(expected_checks) or any(checks.get(key) != "PASS" for key in expected_checks):
        raise Stage3Stop("harness_receipt", "all four independent harness checks must be PASS")
    return receipt


def _tensor_hash(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    return _sha256_bytes(_json_bytes({"dtype": str(value.dtype), "shape": list(value.shape), "bytes_sha256": _sha256_bytes(value.numpy().tobytes())}))


def _target_projector(decoder_rows: Any) -> tuple[Any, dict[str, Any]]:
    torch = __import__("torch")
    matrix = decoder_rows.detach().cpu().double()
    if tuple(matrix.shape) != (TARGET_LATENT_COUNT, RESIDUAL_WIDTH):
        raise Stage3Stop("target_decoder_shape", f"target decoder rows have shape {tuple(matrix.shape)}, expected (12,768)")
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    s_max = float(singular[0]) if singular.numel() else 0.0
    tolerance = max(matrix.shape) * EPS64 * s_max
    rank = int((singular > tolerance).sum())
    if rank != TARGET_LATENT_COUNT:
        raise Stage3Stop("target_rank", f"target decoder numerical rank {rank} != 12")
    vr = vh[:rank].contiguous()
    return vr, {"arithmetic": "float64", "shape": list(matrix.shape), "target_latent_ids": list(TARGET_LATENTS), "singular_values": [float(item) for item in singular], "tolerance": tolerance, "numerical_rank": rank, "right_singular_rows_hash": _tensor_hash(vr)}


def _project(delta: Any, vr: Any) -> Any:
    # ``vr`` has [rank, d]; this is V_r.T @ (V_r @ delta), retaining float64.
    return (delta @ vr.T) @ vr


def _effect_summary(patched: Any, clean_d: Any, signs: Any, pair_ids: Sequence[int], directions: Sequence[str]) -> dict[str, Any]:
    raw = (patched - clean_d).detach().double().cpu()
    aligned = raw * signs.detach().double().cpu()
    pair = aligned.reshape(-1, 2)
    return {
        "pair_ids": [int(item) for item in pair_ids], "directions": list(directions), "directed_raw": [float(item) for item in raw],
        "directed_sign_aligned": [float(item) for item in aligned], "pair_means": [float(item) for item in pair.mean(dim=1)],
        "E": float(aligned.mean()), "pair_sign_consistency": float(((pair > 0).all(dim=1)).double().mean()),
        "directed_sign_consistency": float((aligned > 0).double().mean()),
    }


def _ratio_guard(full: Mapping[str, Any], span: Mapping[str, Any], comp: Mapping[str, Any]) -> dict[str, Any]:
    denominator = float(full["E"])
    def mean_abs(row: Mapping[str, Any]) -> float:
        values = [abs(float(value)) for value in row.get("directed_sign_aligned", [])]
        if not values:
            raise Stage3Stop("denominator_guard", "Q4 denominator guard received an empty effect vector")
        return sum(values) / len(values)

    scale = max(1.0, mean_abs(full), mean_abs(span), mean_abs(comp))
    tau = math.sqrt(EPS64) * scale
    result = {"D": denominator, "M": scale, "tau": tau}
    if abs(denominator) <= tau:
        result["status"] = "NON_ESTIMABLE_DENOMINATOR"
        return result
    result.update({"status": "ESTIMABLE", "R_span": float(span["E"]) / denominator, "R_comp": float(comp["E"]) / denominator})
    return result


def _freeze_target_excluded_pool(ranked: Sequence[tuple[int, float]], *, seed: int) -> list[tuple[int, float]]:
    """Freeze exactly 128 ranked non-target ids before any Q4 draw."""

    eligible: list[tuple[int, float]] = []
    seen: set[int] = set()
    for latent_id, score in ranked:
        latent_id = int(latent_id)
        if latent_id in seen:
            raise Stage3Stop("matched_pool", f"seed {seed} ranked candidate ids are not unique")
        seen.add(latent_id)
        if latent_id in TARGET_LATENTS:
            continue
        eligible.append((latent_id, float(score)))
    if len(eligible) < MATCHED_POOL_SIZE:
        raise Stage3Stop("matched_pool", f"seed {seed} has only {len(eligible)} target-excluded ranked candidates; need exactly {MATCHED_POOL_SIZE}")
    frozen = eligible[:MATCHED_POOL_SIZE]
    if len(frozen) != MATCHED_POOL_SIZE or any(latent_id in TARGET_LATENTS for latent_id, _ in frozen):
        raise Stage3Stop("matched_pool", f"seed {seed} could not freeze an exact target-excluded pool of {MATCHED_POOL_SIZE}")
    return frozen


def _matched_draws(pool_ids: Sequence[int], decoder: Any, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pool = [int(item) for item in pool_ids]
    if len(pool) != MATCHED_POOL_SIZE or len(set(pool)) != MATCHED_POOL_SIZE:
        raise Stage3Stop("matched_pool", f"seed {seed} matched pool must contain exactly {MATCHED_POOL_SIZE} unique ids before draws")
    if any(latent_id in TARGET_LATENTS for latent_id in pool):
        raise Stage3Stop("matched_pool", f"seed {seed} matched pool contains a target latent before draws")
    torch = __import__("torch")
    import numpy as np  # lazy: only the model-backed Q4 path needs it
    rng = np.random.default_rng(seed * 1000 + Q4_TEST_ID)
    accepted: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    eligible = sorted(pool)
    attempt = 0
    while len(accepted) < MATCHED_DRAW_COUNT and attempt < MATCHED_MAX_ATTEMPTS:
        attempt += 1
        chosen = sorted(int(item) for item in rng.choice(eligible, size=MATCHED_SUBSET_SIZE, replace=False).tolist())
        rows = decoder[chosen].detach().cpu().double()
        singular = torch.linalg.svdvals(rows)
        s_max = float(singular[0]) if singular.numel() else 0.0
        tol = max(rows.shape) * EPS64 * s_max
        rank = int((singular > tol).sum())
        accepted_flag = rank == MATCHED_SUBSET_SIZE
        record = {"attempt": attempt, "latent_ids": chosen, "rank": rank, "tolerance": tol, "accepted": accepted_flag}
        attempts.append(record)
        if accepted_flag:
            accepted.append({"draw_index": len(accepted), "attempt": attempt, "latent_ids": chosen, "rank": rank, "tolerance": tol})
    if len(accepted) != MATCHED_DRAW_COUNT:
        stop = Stage3Stop("matched_redraw_cap", f"seed {seed} obtained {len(accepted)}/{MATCHED_DRAW_COUNT} full-rank subsets in {attempt} attempts")
        stop.attempts = attempts  # type: ignore[attr-defined]
        stop.accepted = accepted  # type: ignore[attr-defined]
        stop.rng_meta = {"seed": seed, "test_id": Q4_TEST_ID, "generator": "numpy.random.default_rng", "seed_formula": "experiment_seed * 1000 + test_id", "attempt_count": attempt, "accepted_count": len(accepted), "rejected_count": len(attempts) - len(accepted)}  # type: ignore[attr-defined]
        raise stop
    return accepted, attempts, {"seed": seed, "test_id": Q4_TEST_ID, "generator": "numpy.random.default_rng", "seed_formula": "experiment_seed * 1000 + test_id", "attempt_count": attempt, "accepted_count": len(accepted), "rejected_count": len(attempts) - len(accepted)}


def _bind_matched_draws(accepted: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]], rng_meta: Mapping[str, Any], *, seed: int, invocation_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted_by_attempt = {int(item["attempt"]): item for item in accepted}
    bound_attempts: list[dict[str, Any]] = []
    for raw in attempts:
        item = dict(raw)
        attempt = int(item["attempt"])
        accepted_row = accepted_by_attempt.get(attempt)
        draw_index = int(accepted_row["draw_index"]) if accepted_row is not None else None
        item.update({"invocation_id": invocation_id, "seed": seed, "seed_id": seed, "family": MATCHED_DRAW_FAMILY, "draw_family": MATCHED_DRAW_FAMILY, "draw_index": draw_index, "accepted_attempt_id": attempt if accepted_row is not None else None})
        item["attempt_sha256"] = _sha256_bytes(_json_bytes(item))
        bound_attempts.append(item)
    attempt_hash_by_number = {int(item["attempt"]): str(item["attempt_sha256"]) for item in bound_attempts}
    bound_accepted: list[dict[str, Any]] = []
    for raw in accepted:
        item = dict(raw)
        item.update({"invocation_id": invocation_id, "seed": seed, "seed_id": seed, "family": MATCHED_DRAW_FAMILY, "draw_family": MATCHED_DRAW_FAMILY, "accepted_attempt_id": int(item["attempt"]), "attempt_sha256": attempt_hash_by_number[int(item["attempt"])]})
        item["draw_or_projector_hash"] = item["attempt_sha256"]
        bound_accepted.append(item)
    bound_rng = dict(rng_meta)
    bound_rng.update({"invocation_id": invocation_id, "seed": seed, "seed_id": seed, "family": MATCHED_DRAW_FAMILY, "draw_family": MATCHED_DRAW_FAMILY})
    return bound_accepted, bound_attempts, bound_rng


T_CRITICAL_DF7 = 2.365


def _cross_seed_numeric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize registered eight-seed numeric cells with frozen t(7)=2.365."""

    def summarize(slots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        finite_values = [float(slot["value"]) for slot in slots if isinstance(slot.get("value"), (int, float)) and not isinstance(slot.get("value"), bool) and math.isfinite(float(slot["value"]))]
        n = len(finite_values)
        if n == 0:
            mean = se = ci_low = ci_high = None
            reason: dict[str, Any] | None = {"code": "NO_FINITE_VALUES", "finite_count": 0, "required_finite_count_for_t7": len(STAGE3_SEEDS)}
        else:
            mean = sum(finite_values) / n
            if n >= 2:
                variance = sum((value - mean) ** 2 for value in finite_values) / (n - 1)
                se = math.sqrt(variance / n)
            else:
                se = None
            if n == len(STAGE3_SEEDS):
                ci_low = mean - T_CRITICAL_DF7 * se
                ci_high = mean + T_CRITICAL_DF7 * se
                t_critical = T_CRITICAL_DF7
                degrees_of_freedom = len(STAGE3_SEEDS) - 1
                reason = None
            else:
                ci_low = ci_high = None
                t_critical = degrees_of_freedom = None
                reason = {"code": "INCOMPLETE_FINITE_SEED_SET_FOR_T7_CI", "finite_count": n, "required_finite_count_for_t7": len(STAGE3_SEEDS)}
        return {"slots": [dict(slot) for slot in slots], "finite_count": n, "mean": mean, "standard_error": se, "t_critical": t_critical if n else None, "degrees_of_freedom": degrees_of_freedom if n else None, "ci95": {"low": ci_low, "high": ci_high}, "null_reason": reason}

    def slot_for(row: Mapping[str, Any], value: Any, reason: Mapping[str, Any] | None = None) -> dict[str, Any]:
        seed = int(row["seed"])
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return {"seed": seed, "value": float(value)}
        structured = dict(reason or {"code": "SEED_RESULT_VALUE_MISSING"})
        return {"seed": seed, "value": None, "reason": structured}

    def effect_slot(row: Mapping[str, Any], path: Sequence[str], *, missing_code: str) -> dict[str, Any]:
        current: Any = row
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                return slot_for(row, None, {"code": missing_code, "status": row.get("status"), "seed_reason": row.get("reason")})
            current = current[key]
        return slot_for(row, current, {"code": missing_code, "status": row.get("status"), "seed_reason": row.get("reason")})

    def ratio_slot(row: Mapping[str, Any], field: str) -> dict[str, Any]:
        cells = row.get("cells")
        both = cells.get("both") if isinstance(cells, Mapping) else None
        guard = both.get("ratio_guard") if isinstance(both, Mapping) else None
        if isinstance(guard, Mapping) and guard.get("status") == "NON_ESTIMABLE_DENOMINATOR":
            return slot_for(row, None, {"code": "NON_ESTIMABLE_DENOMINATOR", "status": row.get("status")})
        if isinstance(guard, Mapping) and guard.get("status") == "ESTIMABLE" and field in guard:
            return slot_for(row, guard.get(field), {"code": "NONFINITE_VALUE", "status": row.get("status")})
        return slot_for(row, None, {"code": "SEED_RESULT_VALUE_MISSING", "status": row.get("status"), "seed_reason": row.get("reason")})

    return {
        "t_7": {"degrees_of_freedom": len(STAGE3_SEEDS) - 1, "t_critical": T_CRITICAL_DF7},
        "R_span": summarize([ratio_slot(row, "R_span") for row in rows]),
        "R_comp": summarize([ratio_slot(row, "R_comp") for row in rows]),
        "alpha": {
            "alpha_0_5/both": summarize([effect_slot(row, ("cells", "alpha_0_5/both", "effect", "E"), missing_code="ALPHA_CELL_NONESTIMABLE") for row in rows]),
            "alpha_1_0/both": summarize([effect_slot(row, ("cells", "both", "full_delta", "E"), missing_code="ALPHA_CELL_NONESTIMABLE") for row in rows]),
        },
        "PCA": {cell: summarize([effect_slot(row, ("pca_cells", cell, "effect", "E"), missing_code="PCA_CONTEXT_NON_ESTIMABLE") for row in rows]) for cell in ("PCA_span/subject", "PCA_span/final", "PCA_span/both")},
    }


def _aggregate_q4(seed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        import exp05_core as core  # type: ignore
        aggregate = getattr(core, "aggregate_eight_seed_statuses")
    except (ImportError, AttributeError) as exc:
        raise Stage3Stop("core_aggregation", "exp05_core.aggregate_eight_seed_statuses is required") from exc
    try:
        result = aggregate(seed_rows)
    except Exception as exc:
        raise Stage3Stop("core_aggregation", f"core eight-seed aggregation failed: {exc}") from exc
    if hasattr(result, "as_dict"):
        result = result.as_dict()
    if not isinstance(result, Mapping):
        raise Stage3Stop("core_aggregation", "core eight-seed aggregation returned an invalid object")
    output = dict(result)
    output["status"] = output.get("verdict")
    return output


def _run_cache_gate_a(args: argparse.Namespace) -> int:
    protocol_path = _resolve(args.protocol)
    output_path = _resolve(args.output, reject_leaf_symlink=True)
    checkpoint_path = _resolve(args.checkpoint, reject_leaf_symlink=True)
    _validate_runtime_paths(immutable={"protocol": protocol_path}, runtime={"output": output_path, "checkpoint": checkpoint_path})
    protocol = _read_json(protocol_path, "protocol")
    _validate_protocol(protocol, protocol_path)
    env = _offline_env()
    git = _git_provenance(args.expected_git_commit, bool(args.require_clean_tree))
    snapshots = _check_local_snapshots(protocol)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    completed_seeds: set[int] = set()
    prior_wall_seconds = 0.0
    if checkpoint_path.is_file():
        checkpoint = _read_json(checkpoint_path, "Gate-A checkpoint")
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("phase") != "cache-gate-a" or checkpoint.get("status") not in {"RUNNING", "INCOMPLETE_RUNTIME_CAP"}:
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint schema/phase/status is not resumable")
        if checkpoint.get("protocol_sha256") != _protocol_hash(protocol) or checkpoint.get("git") != dict(git):
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint protocol/git binding differs")
        if checkpoint.get("self_sha256") != _self_hash(checkpoint):
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint self hash is absent or invalid")
        completed_values = checkpoint.get("completed_seeds")
        pair_values = checkpoint.get("pair_rows")
        if not isinstance(completed_values, list) or not isinstance(pair_values, list):
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint completed_seeds/pair_rows are missing")
        completed_list = [int(item) for item in completed_values]
        if completed_list != sorted(set(completed_list)) or not set(completed_list).issubset(set(STAGE3_SEEDS)):
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint seeds are not known, unique, and sorted")
        rows = [dict(item) for item in pair_values if isinstance(item, Mapping)]
        if len(rows) != len(pair_values) or len(rows) != REQUESTED_PAIRS * len(completed_list):
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint pair rows do not equal completed seed coverage")
        keys = [(int(row.get("seed", -1)), int(row.get("pair_id", -1))) for row in rows]
        if len(set(keys)) != len(keys) or {seed for seed, _ in keys} != set(completed_list) or {pair for _, pair in keys} != set(range(REQUESTED_PAIRS)):
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint pair rows are not exact and unique")
        completed_seeds = set(completed_list)
        try:
            prior_wall_seconds = float(checkpoint.get("wall_clock_seconds", 0.0))
        except (TypeError, ValueError) as exc:
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint wall_clock_seconds is invalid") from exc
        if not math.isfinite(prior_wall_seconds) or prior_wall_seconds < 0:
            raise Stage3Stop("checkpoint_binding", "Gate-A checkpoint wall_clock_seconds is not finite/non-negative")
    build_stimuli, load_model, require_one_token, set_determinism = _load_clean_stack()
    model = load_model()
    token_ids = {"is": require_one_token(model.tokenizer, " is"), "are": require_one_token(model.tokenizer, " are")}
    model_config = {name: int(getattr(model.cfg, name, -1)) for name in ("n_layers", "n_heads", "d_model", "d_vocab")}
    expected_model_config = {"n_layers": 12, "n_heads": 12, "d_model": 768, "d_vocab": 50_257}
    if model_config != expected_model_config:
        raise Stage3Stop("model_architecture", f"pinned GPT-2-small architecture mismatch: {model_config}")
    tokenizer_meta = {"class": type(model.tokenizer).__name__, "vocab_size": int(getattr(model.tokenizer, "vocab_size", -1)), "eos_token_id": int(getattr(model.tokenizer, "eos_token_id", -1)), "readout_token_ids": token_ids}
    for seed in STAGE3_SEEDS:
        if seed in completed_seeds:
            continue
        try:
            _check_runtime(started, args.max_wall_seconds, f"before seed {seed}", prior_seconds=prior_wall_seconds)
        except RuntimeCapStop:
            checkpoint = {"schema": CHECKPOINT_SCHEMA, "phase": "cache-gate-a", "status": "INCOMPLETE_RUNTIME_CAP", "protocol_sha256": _protocol_hash(protocol), "git": git, "completed_seeds": sorted(completed_seeds), "pair_rows": rows, "wall_clock_seconds": prior_wall_seconds + time.perf_counter() - started, "updated_at_epoch": time.time()}
            _atomic_json(checkpoint_path, _with_self_hash(checkpoint))
            manifest = _cache_manifest(protocol, status="INCOMPLETE_RUNTIME_CAP", git=git, snapshots=snapshots, env=env, seed_rows=[{"seed": seed} for seed in sorted(completed_seeds)], protocol_path=protocol_path, output_path=output_path)
            manifest["model_config"] = model_config
            manifest["tokenizer"] = tokenizer_meta
            _atomic_jsonl(output_path, [manifest])
            return 2
        set_determinism(seed)
        stimuli = build_stimuli(model.tokenizer, REQUESTED_PAIRS, seed)
        is_logits, are_logits, d = _clean_readouts(model, stimuli.tokens, stimuli.lengths, token_ids["is"], token_ids["are"])
        provenance = {"seed": seed, "generator": "experiment_04.pilot.build_stimuli", "requested_pairs": REQUESTED_PAIRS, "attempted": int(stimuli.attempted), "rejected": int(stimuli.rejected), "readout": "logit(are)-logit(is) at lengths-1", "intervention": False}
        rows.extend(_cache_pair_rows(seed, stimuli, is_logits, are_logits, d, token_ids, _protocol_hash(protocol), provenance))
        completed_seeds.add(seed)
        checkpoint = {"schema": CHECKPOINT_SCHEMA, "phase": "cache-gate-a", "status": "RUNNING", "protocol_sha256": _protocol_hash(protocol), "git": git, "completed_seeds": sorted(completed_seeds), "pair_rows": rows, "wall_clock_seconds": prior_wall_seconds + time.perf_counter() - started, "updated_at_epoch": time.time()}
        _atomic_json(checkpoint_path, _with_self_hash(checkpoint))
    if len(completed_seeds) != len(STAGE3_SEEDS):
        manifest = _cache_manifest(protocol, status="INCOMPLETE_RUNTIME_CAP", git=git, snapshots=snapshots, env=env, seed_rows=[], protocol_path=protocol_path, output_path=output_path)
        manifest["model_config"] = model_config
        manifest["tokenizer"] = tokenizer_meta
        _atomic_jsonl(output_path, [manifest])
        return 2
    final_git = _assert_source_unchanged(git)
    manifest = _cache_manifest(protocol, status="COMPLETE", git=git, snapshots=snapshots, env=env, seed_rows=[{"seed": seed} for seed in sorted(completed_seeds)], protocol_path=protocol_path, output_path=output_path)
    manifest["model_config"] = model_config
    manifest["tokenizer"] = tokenizer_meta
    manifest["git_start"] = git
    manifest["git_final"] = final_git
    manifest["source_provenance"] = {"start": git, "final": final_git}
    # Validate the newly produced rows before publishing the complete marker.
    pair_rows = sorted(rows, key=lambda item: (int(item["seed"]), int(item["pair_id"])))
    gate_rows = _recompute_gate_a(pair_rows)
    if set(gate_rows) != set(STAGE3_SEEDS):
        raise Stage3Stop("cache_coverage", "newly produced Gate-A cache does not cover every Stage-3 seed")
    _atomic_jsonl(output_path, [manifest, *pair_rows])
    _atomic_json(checkpoint_path, _with_self_hash({"schema": CHECKPOINT_SCHEMA, "phase": "cache-gate-a", "status": "COMPLETE", "protocol_sha256": _protocol_hash(protocol), "git": git, "git_start": git, "git_final": final_git, "source_provenance": {"start": git, "final": final_git}, "completed_seeds": sorted(completed_seeds), "output_sha256": _sha256_file(output_path), "wall_clock_seconds": prior_wall_seconds + time.perf_counter() - started}))
    return 0


def _load_residual_cache(model: Any, tokens: Any) -> Any:
    torch = __import__("torch")
    parts: list[Any] = []
    for start in range(0, int(tokens.shape[0]), 32):
        stop = min(start + 32, int(tokens.shape[0]))
        with torch.no_grad():
            result, cache = model.run_with_cache(tokens[start:stop], names_filter=lambda name: name == "blocks.8.hook_resid_pre", return_type=None)
        if result is not None:
            raise Stage3Stop("residual_cache", "Q4 residual cache unexpectedly returned logits")
        parts.append(cache["blocks.8.hook_resid_pre"].detach().float().cpu())
        del cache
    return torch.cat(parts)


def _encode_chunked(sae: Any, values: Any, *, chunk: int = 32) -> Any:
    torch = __import__("torch")
    out: list[Any] = []
    flat = values.reshape(-1, values.shape[-1])
    for start in range(0, int(flat.shape[0]), chunk):
        out.append(sae.encode(flat[start : start + chunk]).detach().float().cpu())
    return torch.cat(out).reshape(*values.shape[:-1], SAE_WIDTH)


def _causal_coordinate_scores(model: Any, engine: Any, stimuli: Any, residuals: Any, base: Any, source: Any, signs: Any, positions: Any, clean_d: Any, decoder: Any, code_delta: Any, candidates: Any, logit_difference: Any, *, label: str) -> Any:
    torch = __import__("torch")
    scores: list[Any] = []
    per_call = max(1, PATCH_BATCH_LIMIT // int(base.numel()))
    for start in range(0, int(candidates.numel()), per_call):
        current = candidates[start : start + per_call]
        coefficients = code_delta[:, :, current].permute(2, 0, 1)
        deltas = torch.einsum("cbm,cd->cbmd", coefficients, decoder[current]).reshape(current.numel() * base.numel(), positions.shape[1], RESIDUAL_WIDTH)
        repeated_base = base.repeat(current.numel())
        repeated_signs = signs.repeat(current.numel())
        repeated_positions = positions.repeat(current.numel(), 1)
        patched = engine.run(layer=LAYER, base_tokens=stimuli.tokens[repeated_base], base_residual=residuals[repeated_base], positions=repeated_positions, deltas=deltas, label=label)
        d_patched = logit_difference(patched, stimuli.lengths[repeated_base], int(model.tokenizer(" is", add_special_tokens=False)["input_ids"][0]), int(model.tokenizer(" are", add_special_tokens=False)["input_ids"][0]))
        effect = (d_patched - clean_d[repeated_base]).reshape(current.numel(), base.numel()) * repeated_signs.reshape(current.numel(), base.numel())
        scores.append(effect.double().mean(dim=1).cpu())
    return torch.cat(scores)


def _fit_pca_context(model: Any, generate_pool: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    torch = __import__("torch")
    generic_pool, generation_meta = generate_pool(model, seed)
    values = generic_pool.detach().cpu().double()
    if tuple(values.shape) != (8192, RESIDUAL_WIDTH):
        raise Stage3Stop("pca_generic_pool", f"generic context pool shape {tuple(values.shape)} is not the frozen (8192,768)")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    top = vectors[:, order[:TARGET_LATENT_COUNT]].T.contiguous()
    orthogonality_error = float((top @ top.T - torch.eye(TARGET_LATENT_COUNT, dtype=torch.float64)).abs().max())
    ordered_eigenvalues = eigenvalues[order]
    return top, {
        "fit_rows": int(values.shape[0]), "width": RESIDUAL_WIDTH, "dtype": "float64", "float64_mirror": {"pool_cast": "float64 before centering/covariance/eigh", "eigen_order": "descending", "projector_cast": "float64", "cast_to_model_dtype": "only at injection"},
        "generic_pool_hash": _tensor_hash(values), "mean_hash": _tensor_hash(mean), "covariance_hash": _tensor_hash(covariance), "eigenvalues_hash": _tensor_hash(ordered_eigenvalues), "eigenvalues_descending": [float(item) for item in ordered_eigenvalues[:TARGET_LATENT_COUNT]], "orthogonality_max_abs": orthogonality_error, "projector_rank": TARGET_LATENT_COUNT, "top12_projector_hash": _tensor_hash(top), "generation": generation_meta, "fit_excludes": ["task ids", "Gate-A bits", "role assignments", "task deltas", "evaluation items"], "fallback_task_pool": False, "alpha_1_evaluation_cells": ["PCA_span/subject", "PCA_span/final", "PCA_span/both"], "primary_cell": "PCA_span/both", "fixed_control_cells": ["PCA_span/subject", "PCA_span/final"], "logical_cell_count": 3,
    }


def _run_one_seed(model: Any, sae: Any, stack: tuple[Any, ...], seed: int, invocation_id: str, gate: Mapping[str, Any], split_row: Mapping[str, Any], target_vr: Any, projector_meta: Mapping[str, Any], started: float, cap: float | None, *, prior_seconds: float = 0.0) -> dict[str, Any]:
    torch = __import__("torch")
    seed_started = time.perf_counter()
    build_stimuli, _, directed_indices, logit_difference, _, require_one_token, positions_for_kind, PatchEngine, gather_positions, feature_delta, decoded_delta, set_determinism, candidate_prefilter_full, generate_pool = stack
    _check_runtime(started, cap, f"before Q4 seed {seed}", prior_seconds=prior_seconds)
    set_determinism(seed)
    stimuli = build_stimuli(model.tokenizer, REQUESTED_PAIRS, seed)
    # The Gate-A cache, not a live recomputation, fixes roles and retained ids.
    retained = [int(item) for item in gate["retained_pair_ids"]]
    split = split_row["split"]
    train_ids = [int(item) for item in split["rank_training_pair_ids"]]
    eval_ids = [int(item) for item in split["evaluation_pair_ids"]]
    if not eval_ids:
        return {"seed": seed, "invocation_id": invocation_id, "execution_status": "EXECUTION_COMPLETE", "status": "SCIENTIFIC_UNRESOLVED", "reason": "BLOCKED_INSUFFICIENT_EVAL", "retained_pairs": len(retained), "role_counts": {"rank_training": len(train_ids), "evaluation": 0}, "elapsed_seconds": time.perf_counter() - seed_started}
    if len(train_ids) != TRAIN_PAIRS:
        return {"seed": seed, "invocation_id": invocation_id, "execution_status": "EXECUTION_COMPLETE", "status": "SCIENTIFIC_UNRESOLVED", "reason": "BLOCKED_INSUFFICIENT_TRAIN", "retained_pairs": len(retained), "role_counts": {"rank_training": len(train_ids), "evaluation": len(eval_ids)}, "elapsed_seconds": time.perf_counter() - seed_started}
    token_ids = {"is": require_one_token(model.tokenizer, " is"), "are": require_one_token(model.tokenizer, " are")}
    _validate_stimuli_against_cache(stimuli, seed, gate.get("cache_rows", ()), token_ids)
    # Use cached readouts for all clean baselines.  Residual caching requests no
    # logits and cannot alter the prepared Gate-A roles.
    clean_d_values = []
    # The caller stores these values from the immutable cache in the gate row.
    for pair_id in range(REQUESTED_PAIRS):
        pair = gate["readout_by_pair"][str(pair_id)]
        clean_d_values.extend((float(pair["singular"]), float(pair["plural"])))
    clean_d = torch.tensor(clean_d_values, dtype=torch.float32)
    residuals = _load_residual_cache(model, stimuli.tokens)
    rank_base, rank_source, rank_signs = directed_indices(REQUESTED_PAIRS, train_ids)
    eval_base, eval_source, eval_signs = directed_indices(REQUESTED_PAIRS, eval_ids)
    rank_positions = positions_for_kind(stimuli, rank_base, "both")
    engine = PatchEngine(model, start_at_layer8=True)
    rank_x_base = gather_positions(residuals[rank_base], rank_positions)
    rank_x_source = gather_positions(residuals[rank_source], rank_positions)
    rank_delta = rank_x_source - rank_x_base
    rank_code_source = _encode_chunked(sae, rank_x_source)
    rank_code_base = _encode_chunked(sae, rank_x_base)
    rank_code_delta = rank_code_source - rank_code_base
    try:
        # Exclude the twelve adjudicating target ids before the frozen 128
        # candidate pre-filter.  Zeroing only source/base activity preserves
        # the experiment-04 active-union/proxy threshold and makes its exact
        # budget an exact target-excluded pool budget.
        prefilter_source = rank_code_source.clone()
        prefilter_base = rank_code_base.clone()
        prefilter_source[..., list(TARGET_LATENTS)] = 0
        prefilter_base[..., list(TARGET_LATENTS)] = 0
        candidates, prefilter = candidate_prefilter_full(code_delta=rank_code_delta, source_code=prefilter_source, base_code=prefilter_base, signs=rank_signs, decoder=sae.W_dec, budget=MATCHED_POOL_SIZE, active_mode="positive")
        scores = _causal_coordinate_scores(model, engine, stimuli, residuals, rank_base, rank_source, rank_signs, rank_positions, clean_d, sae.W_dec, rank_code_delta, candidates, logit_difference, label=f"stage3_rank_train_128_seed_{seed}")
    except Exception as exc:
        if getattr(exc, "gate", None) == "ranking_candidate_coverage":
            raise Stage3Stop("ranking_candidate_coverage", str(exc)) from exc
        raise
    _check_runtime(started, cap, f"after rank-training pool seed {seed}", prior_seconds=prior_seconds)
    ranked = sorted((int(item), float(score)) for item, score in zip(candidates.tolist(), scores.tolist()))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    try:
        frozen_ranked = _freeze_target_excluded_pool(ranked, seed=seed)
    except Stage3Stop as exc:
        if exc.gate == "matched_pool":
            return {"seed": seed, "invocation_id": invocation_id, "execution_status": "EXECUTION_COMPLETE", "status": "SCIENTIFIC_UNRESOLVED", "reason": "matched_pool_size", "detail": str(exc), "retained_pairs": len(retained), "role_counts": {"rank_training": len(train_ids), "evaluation": len(eval_ids)}, "elapsed_seconds": time.perf_counter() - seed_started}
        raise
    frozen_pool = [item[0] for item in frozen_ranked]
    try:
        pca_vr, pca_meta = _fit_pca_context(model, generate_pool, seed)
    except Exception as exc:
        # PCA is a descriptive comparator only.  A failed generic-text fit
        # must not erase an otherwise estimable target-span Q4 seed.
        pca_vr = None
        pca_meta = {"status": "PCA_CONTEXT_NON_ESTIMABLE", "error_type": type(exc).__name__, "error": str(exc), "role": "descriptive_only", "fit_rows": None, "generic_pool_hash": None, "eigenvalues_hash": None, "orthogonality_max_abs": None, "float64_mirror": {"required": True, "cast_to_model_dtype": "only at injection"}, "fallback_task_pool": False, "alpha_1_evaluation_cells": ["PCA_span/subject", "PCA_span/final", "PCA_span/both"], "primary_cell": "PCA_span/both", "fixed_control_cells": ["PCA_span/subject", "PCA_span/final"], "logical_cell_count": 3}
    eval_positions_by_kind = {kind: positions_for_kind(stimuli, eval_base, kind) for kind in POSITION_SETS}
    eval_delta_by_kind = {kind: gather_positions(residuals[eval_source], eval_positions_by_kind[kind]) - gather_positions(residuals[eval_base], eval_positions_by_kind[kind]) for kind in POSITION_SETS}
    directions = [DIRECTIONS[index % 2] for index in range(len(eval_ids) * 2)]
    cells: dict[str, Any] = {}
    for kind in POSITION_SETS:
        positions = eval_positions_by_kind[kind]
        delta = eval_delta_by_kind[kind]
        full_patched = engine.run(layer=LAYER, base_tokens=stimuli.tokens[eval_base], base_residual=residuals[eval_base], positions=positions, deltas=delta, label=f"stage3_full_delta_{kind}_seed_{seed}")
        delta64 = delta.detach().double()
        span_delta64 = _project(delta64, target_vr)
        comp_delta64 = delta64 - span_delta64
        span_patched = engine.run(layer=LAYER, base_tokens=stimuli.tokens[eval_base], base_residual=residuals[eval_base], positions=positions, deltas=span_delta64.float(), label=f"stage3_target_span_{kind}_seed_{seed}")
        comp_patched = engine.run(layer=LAYER, base_tokens=stimuli.tokens[eval_base], base_residual=residuals[eval_base], positions=positions, deltas=comp_delta64.float(), label=f"stage3_target_complement_{kind}_seed_{seed}")
        full = _effect_summary(logit_difference(full_patched, stimuli.lengths[eval_base], token_ids["is"], token_ids["are"]), clean_d[eval_base], eval_signs, eval_ids, directions)
        span = _effect_summary(logit_difference(span_patched, stimuli.lengths[eval_base], token_ids["is"], token_ids["are"]), clean_d[eval_base], eval_signs, eval_ids, directions)
        comp = _effect_summary(logit_difference(comp_patched, stimuli.lengths[eval_base], token_ids["is"], token_ids["are"]), clean_d[eval_base], eval_signs, eval_ids, directions)
        for effect, label, tensor in ((span, "target_span", span_delta64), (comp, "target_complement", comp_delta64)):
            effect["execution_cell_id"] = f"{label}/{kind}/evaluation/seed_{seed}"
            effect["tensor_hash"] = _tensor_hash(tensor)
            effect["result_hash"] = _sha256_bytes(_json_bytes(effect))
        full["execution_cell_id"] = f"full_delta/{kind}/evaluation/seed_{seed}"
        full["tensor_hash"] = _tensor_hash(delta64)
        full["result_hash"] = _sha256_bytes(_json_bytes(full))
        cells[kind] = {"full_delta": full, "target_span": span, "target_complement": comp, "ratio_guard": _ratio_guard(full, span, comp), "geometric_fractions": {"span": float((span_delta64.norm(dim=-1) ** 2).sum() / max(float((delta64.norm(dim=-1) ** 2).sum()), 1e-300)), "complement": float((comp_delta64.norm(dim=-1) ** 2).sum() / max(float((delta64.norm(dim=-1) ** 2).sum()), 1e-300))}}
        _check_runtime(started, cap, f"after target projector position set {kind} seed {seed}", prior_seconds=prior_seconds)
    alpha05_delta = eval_delta_by_kind["both"] * 0.5
    alpha05_patched = engine.run(layer=LAYER, base_tokens=stimuli.tokens[eval_base], base_residual=residuals[eval_base], positions=eval_positions_by_kind["both"], deltas=alpha05_delta, label=f"stage3_alpha_0_5_full_delta_both_seed_{seed}")
    alpha05 = _effect_summary(logit_difference(alpha05_patched, stimuli.lengths[eval_base], token_ids["is"], token_ids["are"]), clean_d[eval_base], eval_signs, eval_ids, directions)
    alpha05["execution_cell_id"] = f"alpha_0_5/full_delta/both/evaluation/seed_{seed}"
    alpha05["tensor_hash"] = _tensor_hash(alpha05_delta)
    alpha05["result_hash"] = _sha256_bytes(_json_bytes(alpha05))
    cells["alpha_0_5/both"] = {"effect": alpha05, "execution_cell_id": f"alpha_0_5/full_delta/both/evaluation/seed_{seed}", "tensor_hash": _tensor_hash(alpha05_delta), "result_hash": _sha256_bytes(_json_bytes(alpha05))}
    cells["alpha_1_0/both"] = {"alias_of": "both/full_delta/evaluation", "execution_cell_id": cells["both"]["full_delta"]["execution_cell_id"], "tensor_hash": cells["both"]["full_delta"]["tensor_hash"], "result_hash": cells["both"]["full_delta"]["result_hash"], "rerun": False, "double_count": False}
    try:
        matched, attempts, rng_meta = _matched_draws(frozen_pool, sae.W_dec, seed)
        matched, attempts, rng_meta = _bind_matched_draws(matched, attempts, rng_meta, seed=seed, invocation_id=invocation_id)
    except Stage3Stop as exc:
        if exc.gate in {"matched_pool", "matched_redraw_cap"}:
            return {"seed": seed, "invocation_id": invocation_id, "execution_status": "EXECUTION_COMPLETE", "status": "SCIENTIFIC_UNRESOLVED", "reason": exc.gate, "detail": str(exc), "retained_pairs": len(retained), "role_counts": {"rank_training": len(train_ids), "evaluation": len(eval_ids)}, "elapsed_seconds": time.perf_counter() - seed_started}
        raise
    matched_results: list[dict[str, Any]] = []
    for draw in matched:
        matched_vr, _ = _target_projector(sae.W_dec[draw["latent_ids"]].detach().float().cpu())
        projected64 = _project(eval_delta_by_kind["both"].detach().double(), matched_vr)
        patched = engine.run(layer=LAYER, base_tokens=stimuli.tokens[eval_base], base_residual=residuals[eval_base], positions=eval_positions_by_kind["both"], deltas=projected64.float(), label=f"stage3_matched_span_{draw['draw_index']}_seed_{seed}")
        effect = _effect_summary(logit_difference(patched, stimuli.lengths[eval_base], token_ids["is"], token_ids["are"]), clean_d[eval_base], eval_signs, eval_ids, directions)
        effect["execution_cell_id"] = f"matched_span/{draw['draw_index']}/both/evaluation/seed_{seed}"
        effect["tensor_hash"] = _tensor_hash(projected64)
        effect["result_hash"] = _sha256_bytes(_json_bytes(effect))
        projected_delta_hash = _tensor_hash(projected64)
        matched_results.append({"invocation_id": invocation_id, "seed": seed, "seed_id": seed, "family": MATCHED_DRAW_FAMILY, "draw_family": MATCHED_DRAW_FAMILY, "draw_index": draw["draw_index"], "attempt": draw["attempt"], "accepted_attempt_id": draw["attempt"], "attempt_sha256": draw["attempt_sha256"], "latent_ids": draw["latent_ids"], "effect": effect, "effect_result_hash": effect["result_hash"], "projector_hash": _tensor_hash(matched_vr), "projected_delta_hash": projected_delta_hash, "draw_or_projector_hash": projected_delta_hash})
        _check_runtime(started, cap, f"after matched draw {draw['draw_index']} seed {seed}", prior_seconds=prior_seconds)
    matched_values = sorted(float(item["effect"]["E"]) / float(cells["both"]["full_delta"]["E"]) for item in matched_results) if cells["both"]["ratio_guard"]["status"] == "ESTIMABLE" else []
    primary = cells["both"]["ratio_guard"]
    if primary["status"] == "NON_ESTIMABLE_DENOMINATOR":
        q4_status = "SCIENTIFIC_UNRESOLVED"
    else:
        edge = matched_values[-2] if len(matched_values) >= 2 else float("nan")
        q4_status = "PASS" if float(primary["R_span"]) > edge else "COMPLETED_FAIL"
        cells["both"]["matched_edge_second_largest"] = edge
        cells["both"]["R_span_exceeds_matched_edge"] = q4_status == "PASS"
    pca_cells: dict[str, Any] = {}
    if pca_vr is not None:
        for kind in POSITION_SETS:
            pca_delta64 = _project(eval_delta_by_kind[kind].detach().double(), pca_vr)
            patched = engine.run(layer=LAYER, base_tokens=stimuli.tokens[eval_base], base_residual=residuals[eval_base], positions=eval_positions_by_kind[kind], deltas=pca_delta64.float(), label=f"stage3_pca_span_{kind}_seed_{seed}")
            effect = _effect_summary(logit_difference(patched, stimuli.lengths[eval_base], token_ids["is"], token_ids["are"]), clean_d[eval_base], eval_signs, eval_ids, directions)
            effect["execution_cell_id"] = f"PCA_span/{kind}/evaluation/seed_{seed}"
            effect["tensor_hash"] = _tensor_hash(pca_delta64)
            effect["result_hash"] = _sha256_bytes(_json_bytes(effect))
            pca_cells[f"PCA_span/{kind}"] = {"effect": effect, "projector_hash": _tensor_hash(pca_vr), "projected_delta_hash": _tensor_hash(pca_delta64), "role": "descriptive_only"}
    if q4_status == "SCIENTIFIC_UNRESOLVED":
        return {"seed": seed, "invocation_id": invocation_id, "execution_status": "EXECUTION_COMPLETE", "status": q4_status, "reason": "NON_ESTIMABLE_DENOMINATOR", "retained_pairs": len(retained), "role_counts": {"rank_training": len(train_ids), "evaluation": len(eval_ids)}, "elapsed_seconds": time.perf_counter() - seed_started}
    return {"seed": seed, "invocation_id": invocation_id, "execution_status": "EXECUTION_COMPLETE", "status": q4_status, "reason": None, "retained_pairs": len(retained), "role_counts": {"rank_training": len(train_ids), "evaluation": len(eval_ids)}, "candidate_pool": {"budget": MATCHED_POOL_SIZE, "rank_training_only": True, "prefilter": prefilter, "frozen_ranked_latent_ids": frozen_pool, "scores": [{"latent_id": item[0], "score": item[1]} for item in frozen_ranked]}, "projector": projector_meta, "cells": cells, "pca_context": pca_meta, "pca_cells": pca_cells, "matched_draws": {"rng": rng_meta, "accepted": matched, "attempts": attempts, "results": matched_results}, "execution": {"engine_records": list(engine.records), "eval_pair_ids": eval_ids, "train_pair_ids": train_ids, "clean_d_source": "Gate-A cache", "candidate_or_stage2_inputs": []}, "elapsed_seconds": time.perf_counter() - seed_started}


def _run_q4_impl(args: argparse.Namespace) -> int:
    protocol_path = _resolve(args.protocol)
    gate_cache_path = _resolve(args.gate_cache)
    split_manifest_path = _resolve(args.split_manifest)
    prepare_path = _resolve(args.prepare_manifest)
    review_path = _resolve(args.review_receipt)
    harness_path = _resolve(args.harness_receipt)
    output_path = _resolve(args.output, reject_leaf_symlink=True)
    draw_path = _resolve(args.draw_csv, reject_leaf_symlink=True)
    checkpoint_path = _resolve(args.checkpoint, reject_leaf_symlink=True)
    _validate_runtime_paths(immutable={"protocol": protocol_path, "gate_cache": gate_cache_path, "split_manifest": split_manifest_path, "prepare_manifest": prepare_path, "review_receipt": review_path, "harness_receipt": harness_path}, runtime={"output": output_path, "draw_csv": draw_path, "checkpoint": checkpoint_path})
    protocol = _read_json(protocol_path, "protocol")
    _validate_protocol(protocol, protocol_path)
    env = _offline_env()
    git = _git_provenance(args.expected_git_commit, bool(args.require_clean_tree))
    snapshots = _check_local_snapshots(protocol)
    cache_manifest, cache_rows, cache_hash, gate, split, split_hash = _load_split_inputs(protocol, gate_cache_path, split_manifest_path, prepare_path)
    prepare_manifest = _read_json(prepare_path, "Stage-3 prepare manifest")
    prepare_hash = _sha256_file(prepare_path)
    preparation_accounting = _preparation_accounting(cache_manifest, split, prepare_manifest)
    _review_receipt(review_path, expected_commit=git["commit"], cache_hash=cache_hash, split_hash=split_hash, prepare_hash=prepare_hash)
    harness = _harness_receipt(harness_path, expected_commit=git["commit"], protocol_hash=_protocol_hash(protocol))
    # Attach cache-derived readouts to each seed only in memory.  They are not
    # recomputed and are not written as a second Gate-A artifact.
    readout_by_seed: dict[int, dict[str, dict[str, float]]] = {seed: {} for seed in STAGE3_SEEDS}
    cache_rows_by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in STAGE3_SEEDS}
    for row in cache_rows:
        items = row["items"]
        readout_by_seed[int(row["seed"])][str(int(row["pair_id"]))] = {"singular": float(items["singular"]["readout_d"]), "plural": float(items["plural"]["readout_d"])}
        cache_rows_by_seed[int(row["seed"])].append(dict(row))
    split_rows = {int(row["seed"]): row for row in split["seeds"]}
    # Attach only immutable cache facts required by the seed runner; no Stage-2
    # or candidate artifact is accepted, read, hashed, or recorded anywhere.
    seed_inputs = []
    for seed in STAGE3_SEEDS:
        row = dict(gate[seed])
        row["readout_by_pair"] = readout_by_seed[seed]
        row["cache_rows"] = cache_rows_by_seed[seed]
        row["split_row"] = split_rows[seed]
        seed_inputs.append(row)
    started = time.perf_counter()
    invocation_id = _new_invocation_id(protocol_hash=_protocol_hash(protocol), cache_hash=cache_hash, split_hash=split_hash, prepare_hash=prepare_hash, commit=git["commit"])
    completed_rows: list[dict[str, Any]] = []
    completed_seeds: set[int] = set()
    prior_incomplete_q4_logical_fe = 0
    prior_incomplete_q4_wall_time = 0.0
    prior_checkpoint_diagnostic: dict[str, Any] = {"status": "ABSENT", "rows_discarded": 0, "scientific_reuse_allowed": False, "resumable_for_adjudication": False}
    if checkpoint_path.is_file():
        checkpoint = _read_json(checkpoint_path, "Stage-3 checkpoint")
        checkpoint_status = str(checkpoint.get("status", ""))
        if checkpoint_status == "EXECUTION_INCOMPLETE":
            if checkpoint.get("self_sha256") != _self_hash(checkpoint) or checkpoint.get("runtime_tombstone") is not True:
                raise Stage3Stop("checkpoint_binding", "prior EXECUTION_INCOMPLETE checkpoint tombstone is not self-consistent")
            prior_runtime = checkpoint.get("runtime")
            if isinstance(prior_runtime, Mapping):
                prior_fe_value = prior_runtime.get("cumulative_q4_logical_fe_all_attempts", prior_runtime.get("q4_attempt_logical_fe_executed"))
                prior_wall_value = prior_runtime.get("cumulative_q4_wall_time_all_attempts", prior_runtime.get("q4_attempt_wall_time"))
                if isinstance(prior_fe_value, (int, float)) and not isinstance(prior_fe_value, bool) and math.isfinite(float(prior_fe_value)) and float(prior_fe_value) >= 0:
                    prior_incomplete_q4_logical_fe = int(prior_fe_value)
                if isinstance(prior_wall_value, (int, float)) and not isinstance(prior_wall_value, bool) and math.isfinite(float(prior_wall_value)) and float(prior_wall_value) >= 0:
                    prior_incomplete_q4_wall_time = float(prior_wall_value)
            prior_checkpoint_diagnostic = {"status": checkpoint_status, "rows_discarded": 0, "prior_logical_fe": prior_incomplete_q4_logical_fe, "prior_wall_time": prior_incomplete_q4_wall_time, "scientific_reuse_allowed": False, "resumable_for_adjudication": False}
        else:
            prior_rows, _, prior_wall = _validate_run_checkpoint(checkpoint, protocol_hash=_protocol_hash(protocol), git=git, cache_hash=cache_hash, split_hash=split_hash, prepare_hash=prepare_hash, draw_csv_path=draw_path)
            if checkpoint_status != "COMPLETE":
                prior_runtime = checkpoint.get("attempt_runtime")
                prior_fe_value = prior_runtime.get("cumulative_q4_logical_fe_all_attempts") if isinstance(prior_runtime, Mapping) else None
                prior_wall_value = prior_runtime.get("cumulative_q4_wall_time_all_attempts") if isinstance(prior_runtime, Mapping) else None
                prior_incomplete_q4_logical_fe = int(prior_fe_value) if isinstance(prior_fe_value, (int, float)) and not isinstance(prior_fe_value, bool) and math.isfinite(float(prior_fe_value)) and float(prior_fe_value) >= 0 else _rows_logical_fe(prior_rows)
                prior_incomplete_q4_wall_time = float(prior_wall_value) if isinstance(prior_wall_value, (int, float)) and not isinstance(prior_wall_value, bool) and math.isfinite(float(prior_wall_value)) and float(prior_wall_value) >= 0 else float(prior_wall)
            prior_checkpoint_diagnostic = {"status": checkpoint_status, "rows_discarded": len(prior_rows), "prior_logical_fe": prior_incomplete_q4_logical_fe, "prior_wall_time": prior_incomplete_q4_wall_time, "scientific_reuse_allowed": False, "resumable_for_adjudication": False}
        # Amendment 9: prior seed rows/cells are diagnostic-only and never
        # enter this invocation's fresh adjudicating dataset.
        completed_rows, completed_seeds = [], set()
    _atomic_json(checkpoint_path, _with_self_hash({
        "schema": CHECKPOINT_SCHEMA, "phase": "run", "status": "RUNNING", "protocol_sha256": _protocol_hash(protocol), "git": git, "git_start": git, "invocation_id": invocation_id,
        "gate_cache_sha256": cache_hash, "split_manifest_sha256": split_hash, "prepare_manifest_sha256": prepare_hash,
        "draw_csv_path": str(draw_path), "draw_csv_sha256": None, "completed_seeds": [], "seed_rows": [],
        "prior_checkpoint_diagnostic": prior_checkpoint_diagnostic, "scientific_reuse_allowed": False, "resumable_for_adjudication": False,
        "attempt_runtime": {"q4_invocation_id": invocation_id, "q4_attempt_logical_fe_executed": 0, "q4_attempt_wall_time": 0.0, "q4_attempt_status": "RUNNING", "final_complete_q4_logical_fe": 0, "final_complete_q4_wall_time": 0.0, "prior_incomplete_q4_logical_fe": prior_incomplete_q4_logical_fe, "cumulative_q4_logical_fe_all_attempts": prior_incomplete_q4_logical_fe, "prior_incomplete_q4_wall_time": prior_incomplete_q4_wall_time, "cumulative_q4_wall_time_all_attempts": prior_incomplete_q4_wall_time, "preparation_logical_fe_separately": dict(preparation_accounting)},
        "wall_clock_seconds": 0.0,
        "updated_at_epoch": time.time(),
    }))
    stack = _load_model_stack()
    load_model = stack[4]
    from pilot import load_direct_res_jb  # type: ignore
    model = load_model()
    model_config = {name: int(getattr(model.cfg, name, -1)) for name in ("n_layers", "n_heads", "d_model", "d_vocab")}
    if model_config != {"n_layers": 12, "n_heads": 12, "d_model": 768, "d_vocab": 50_257}:
        raise Stage3Stop("model_architecture", f"pinned GPT-2-small architecture mismatch: {model_config}")
    sae = load_direct_res_jb(LAYER)
    if tuple(sae.W_dec.shape) != (SAE_WIDTH, RESIDUAL_WIDTH):
        raise Stage3Stop("sae_shape", f"SAE decoder shape {tuple(sae.W_dec.shape)} != (24576,768)")
    target_rows = sae.W_dec[list(TARGET_LATENTS)].detach().float().cpu()
    target_vr, projector_meta = _target_projector(target_rows)

    def write_checkpoint(status: str) -> None:
        attempt_wall = time.perf_counter() - started
        attempt_fe = _rows_logical_fe(completed_rows)
        payload = {
            "schema": CHECKPOINT_SCHEMA, "phase": "run", "status": status, "protocol_sha256": _protocol_hash(protocol), "git": git, "git_start": git, "invocation_id": invocation_id,
            "gate_cache_sha256": cache_hash, "split_manifest_sha256": split_hash, "prepare_manifest_sha256": prepare_hash,
            "completed_seeds": sorted(completed_seeds), "seed_rows": sorted(completed_rows, key=lambda item: int(item["seed"])),
            "draw_csv_path": str(draw_path), "draw_csv_sha256": None,
            "prior_checkpoint_diagnostic": prior_checkpoint_diagnostic, "scientific_reuse_allowed": False, "resumable_for_adjudication": False,
            "attempt_runtime": {"q4_invocation_id": invocation_id, "q4_attempt_logical_fe_executed": attempt_fe, "q4_attempt_wall_time": attempt_wall, "q4_attempt_status": status, "final_complete_q4_logical_fe": 0, "final_complete_q4_wall_time": 0.0, "prior_incomplete_q4_logical_fe": prior_incomplete_q4_logical_fe, "cumulative_q4_logical_fe_all_attempts": prior_incomplete_q4_logical_fe + attempt_fe, "prior_incomplete_q4_wall_time": prior_incomplete_q4_wall_time, "cumulative_q4_wall_time_all_attempts": prior_incomplete_q4_wall_time + attempt_wall, "preparation_logical_fe_separately": dict(preparation_accounting)},
            "wall_clock_seconds": attempt_wall, "updated_at_epoch": time.time(),
        }
        _atomic_json(checkpoint_path, _with_self_hash(payload))

    for item in seed_inputs:
        seed = int(item["seed"])
        if seed in completed_seeds:
            continue
        split_status = str(item["split_row"].get("split", {}).get("status", ""))
        if split_status != "READY" or item.get("passed") is not True:
            row = {"seed": seed, "invocation_id": invocation_id, "execution_status": "EXECUTION_COMPLETE", "status": "SCIENTIFIC_UNRESOLVED", "reason": "GATE_A_FAIL" if item.get("passed") is not True else split_status, "retained_pairs": item["retained_pairs"], "role_counts": {"rank_training": len(item["split_row"]["split"]["rank_training_pair_ids"]), "evaluation": len(item["split_row"]["split"]["evaluation_pair_ids"])}, "elapsed_seconds": 0.0}
            completed_rows.append(_freeze_seed_result(row))
            completed_seeds.add(seed)
            completed_rows = _validate_seed_results(sorted(completed_rows, key=lambda item: int(item["seed"])), sorted(completed_seeds), context="new Stage-3 seed results", invocation_id=invocation_id)
            write_checkpoint("RUNNING")
            continue
        try:
            row = _run_one_seed(model, sae, stack, seed, invocation_id, item, item["split_row"], target_vr, projector_meta, started, args.max_wall_seconds, prior_seconds=0.0)
        except RuntimeCapStop:
            attempt_wall = time.perf_counter() - started
            attempt_fe = _rows_logical_fe(completed_rows)
            _write_run_execution_tombstone(output_path, checkpoint_path, draw_path, gate="max_wall_seconds", detail="declared runtime cap reached before completing the next seed", runtime={"q4_invocation_id": invocation_id, "q4_attempt_logical_fe_executed": attempt_fe, "q4_attempt_wall_time": attempt_wall, "q4_attempt_status": "EXECUTION_INCOMPLETE", "final_complete_q4_logical_fe": 0, "final_complete_q4_wall_time": 0.0, "prior_incomplete_q4_logical_fe": prior_incomplete_q4_logical_fe, "cumulative_q4_logical_fe_all_attempts": prior_incomplete_q4_logical_fe + attempt_fe, "prior_incomplete_q4_wall_time": prior_incomplete_q4_wall_time, "cumulative_q4_wall_time_all_attempts": prior_incomplete_q4_wall_time + attempt_wall, "preparation_logical_fe_separately": dict(preparation_accounting)})
            return 2
        except Stage3Stop as exc:
            if exc.gate == "ranking_candidate_coverage":
                row = {"seed": seed, "invocation_id": invocation_id, "execution_status": "EXECUTION_COMPLETE", "status": "SCIENTIFIC_UNRESOLVED", "reason": "matched_pool", "detail": str(exc), "retained_pairs": item["retained_pairs"], "role_counts": {"rank_training": len(item["split_row"]["split"]["rank_training_pair_ids"]), "evaluation": len(item["split_row"]["split"]["evaluation_pair_ids"])}, "elapsed_seconds": None}
            else:
                raise
        completed_rows.append(_freeze_seed_result(row))
        completed_seeds.add(seed)
        completed_rows = _validate_seed_results(sorted(completed_rows, key=lambda item: int(item["seed"])), sorted(completed_seeds), context="new Stage-3 seed results", invocation_id=invocation_id)
        write_checkpoint("RUNNING")
    if len(completed_seeds) != len(STAGE3_SEEDS):
        raise Stage3Stop("execution_incomplete", "Stage-3 ended without exact eight registered seed results")
    completed_rows = _validate_seed_results(sorted(completed_rows, key=lambda item: int(item["seed"])), list(STAGE3_SEEDS), context="final exact-eight seed results", invocation_id=invocation_id)
    final_git = _assert_source_unchanged(git)
    summary = _aggregate_q4(completed_rows)
    if any(row.get("status") == "EXECUTION_INCOMPLETE" or row.get("execution_status") == "EXECUTION_INCOMPLETE" for row in completed_rows):
        attempt_wall = time.perf_counter() - started
        attempt_fe = _rows_logical_fe(completed_rows)
        _write_run_execution_tombstone(output_path, checkpoint_path, draw_path, gate="seed_execution_incomplete", detail="at least one registered seed is EXECUTION_INCOMPLETE; no scientific verdict is publishable", runtime={"q4_invocation_id": invocation_id, "q4_attempt_logical_fe_executed": attempt_fe, "q4_attempt_wall_time": attempt_wall, "q4_attempt_status": "EXECUTION_INCOMPLETE", "final_complete_q4_logical_fe": 0, "final_complete_q4_wall_time": 0.0, "prior_incomplete_q4_logical_fe": prior_incomplete_q4_logical_fe, "cumulative_q4_logical_fe_all_attempts": prior_incomplete_q4_logical_fe + attempt_fe, "prior_incomplete_q4_wall_time": prior_incomplete_q4_wall_time, "cumulative_q4_wall_time_all_attempts": prior_incomplete_q4_wall_time + attempt_wall, "preparation_logical_fe_separately": dict(preparation_accounting)})
        return 2
    final_complete_q4_logical_fe = _rows_logical_fe(completed_rows)
    final_complete_q4_wall_time = time.perf_counter() - started
    runtime = {"q4_invocation_id": invocation_id, "q4_attempt_logical_fe_executed": final_complete_q4_logical_fe, "q4_attempt_wall_time": final_complete_q4_wall_time, "q4_attempt_status": "COMPLETE", "final_complete_q4_logical_fe": final_complete_q4_logical_fe, "final_complete_q4_wall_time": final_complete_q4_wall_time, "prior_incomplete_q4_logical_fe": prior_incomplete_q4_logical_fe, "cumulative_q4_logical_fe_all_attempts": prior_incomplete_q4_logical_fe + final_complete_q4_logical_fe, "prior_incomplete_q4_wall_time": prior_incomplete_q4_wall_time, "cumulative_q4_wall_time_all_attempts": prior_incomplete_q4_wall_time + final_complete_q4_wall_time, "preparation_logical_fe_separately": dict(preparation_accounting)}
    draw_rows = _draw_binding_records(completed_rows)
    draw_binding_set_sha256 = _sha256_bytes(_json_bytes(draw_rows))
    accepted_draw_rows = [dict(row) for row in draw_rows if row.get("accepted") is True]
    accepted_draw_binding_set_sha256 = _sha256_bytes(_json_bytes(accepted_draw_rows))
    _atomic_csv(draw_path, draw_rows, ("invocation_id", "seed_id", "draw_family", "draw_index", "accepted_attempt_id", "draw_or_projector_hash", "attempt", "attempt_sha256", "latent_ids", "rank", "tolerance", "accepted", "matched_effect_result_hash"))
    draw_hash = _sha256_file(draw_path)
    preparation = {"gate_cache_sha256": cache_hash, "split_manifest_sha256": split_hash, "prepare_manifest_sha256": prepare_hash, "review_receipt_sha256": _sha256_file(review_path), "harness_receipt_sha256": _sha256_file(harness_path), "logical_fe_separately": dict(preparation_accounting)}
    accepted_draw_csv_binding_set_sha256 = _sha256_bytes(_json_bytes(accepted_draw_rows))
    immutable_preparation_input_hashes = {
        "gate_cache_sha256": cache_hash,
        "split_manifest_sha256": split_hash,
        "prepare_manifest_sha256": prepare_hash,
        "review_receipt_sha256": preparation["review_receipt_sha256"],
        "harness_receipt_sha256": preparation["harness_receipt_sha256"],
    }

    def collect_execution_cell_ids(value: Any, output: list[str]) -> None:
        if isinstance(value, Mapping):
            cell_id = value.get("execution_cell_id")
            if isinstance(cell_id, str) and cell_id:
                output.append(cell_id)
            for child in value.values():
                collect_execution_cell_ids(child, output)
        elif isinstance(value, list):
            for child in value:
                collect_execution_cell_ids(child, output)

    execution_cell_rows: list[dict[str, Any]] = []
    for row in sorted(completed_rows, key=lambda item: int(item["seed"])):
        cell_ids: list[str] = []
        collect_execution_cell_ids(row, cell_ids)
        execution_cell_rows.append({"seed": int(row["seed"]), "execution_status": row.get("execution_status"), "status": row.get("status"), "execution_cell_ids": sorted(set(cell_ids)), "cell_count": len(set(cell_ids))})
    execution_cell_registry = {
        "expected_seed_ids": list(STAGE3_SEEDS),
        "registered_seed_ids": [int(row["seed"]) for row in execution_cell_rows],
        "coverage_exact": [int(row["seed"]) for row in execution_cell_rows] == list(STAGE3_SEEDS),
        "rows": execution_cell_rows,
    }
    model_state_fingerprint = {
        "snapshots": snapshots,
        "model_config": model_config,
        "sae": {"layer": LAYER, "decoder_shape": [SAE_WIDTH, RESIDUAL_WIDTH], "target_latent_ids": list(TARGET_LATENTS)},
    }
    configuration = {
        "protocol_sha256": _protocol_hash(protocol),
        "stage3_schema": STAGE3_SCHEMA,
        "seeds": list(STAGE3_SEEDS),
        "layer": LAYER,
        "position_sets": list(POSITION_SETS),
        "matched_draw_family": MATCHED_DRAW_FAMILY,
        "matched_draw_count": MATCHED_DRAW_COUNT,
    }
    source_tree_identity = {
        "source_worktree": git.get("source_worktree"),
        "start_source_tree_sha256": git.get("source_tree_sha256"),
        "final_source_tree_sha256": final_git.get("source_tree_sha256"),
        "clean_status_match": git.get("status_porcelain") == final_git.get("status_porcelain") == "",
    }
    code_revision = {
        "expected_commit": git.get("expected_commit"),
        "start_commit": git.get("commit"),
        "final_commit": final_git.get("commit"),
    }
    final_manifest_bind_keys = [
        "accepted_draw_manifest",
        "complete_attempt_ledger",
        "accepted_scientific_csv",
        "final_checkpoint",
        "execution_cell_registry",
        "immutable_preparation_input_hashes",
        "model_state_fingerprint",
        "configuration",
        "source_tree_identity",
        "code_revision",
    ]
    accepted_scientific_csv = {
        "path": str(draw_path),
        "full_csv_sha256": draw_hash,
        "accepted_binding_set_sha256": accepted_draw_csv_binding_set_sha256,
        "accepted_row_count": len(accepted_draw_rows),
        "binding_key": ["invocation_id", "seed_id", "draw_family", "draw_index", "accepted_attempt_id", "draw_or_projector_hash"],
    }
    checkpoint_core = {"schema": CHECKPOINT_SCHEMA, "phase": "run", "status": "COMPLETE", "protocol_sha256": _protocol_hash(protocol), "git": git, "git_start": git, "git_final": final_git, "source_provenance": {"start": git, "final": final_git}, "invocation_id": invocation_id, "gate_cache_sha256": cache_hash, "split_manifest_sha256": split_hash, "prepare_manifest_sha256": prepare_hash, "preparation": preparation, "draw_csv_path": str(draw_path), "draw_csv_sha256": draw_hash, "draw_binding_set_sha256": draw_binding_set_sha256, "accepted_draw_binding_set_sha256": accepted_draw_binding_set_sha256, "completed_seeds": sorted(completed_seeds), "seed_rows": sorted(completed_rows, key=lambda item: int(item["seed"])), "summary": summary, "attempt_runtime": runtime, "wall_clock_seconds": final_complete_q4_wall_time, "scientific_reuse_allowed": False, "resumable_for_adjudication": False}
    checkpoint_binding_sha256 = _sha256_bytes(_json_bytes(checkpoint_core))
    final_checkpoint = {"path": str(checkpoint_path), "schema": CHECKPOINT_SCHEMA, "phase": "run", "status": "COMPLETE", "binding_sha256": checkpoint_binding_sha256}
    final_manifest_bindings = {
        "accepted_draw_manifest": {"sha256": accepted_draw_binding_set_sha256, "row_count": len(accepted_draw_rows)},
        "complete_attempt_ledger": {"sha256": draw_binding_set_sha256, "row_count": len(draw_rows)},
        "accepted_scientific_csv": accepted_scientific_csv,
        "final_checkpoint": final_checkpoint,
        "execution_cell_registry": execution_cell_registry,
        "immutable_preparation_input_hashes": immutable_preparation_input_hashes,
        "model_state_fingerprint": model_state_fingerprint,
        "configuration": configuration,
        "source_tree_identity": source_tree_identity,
        "code_revision": code_revision,
    }
    complete_requires = {
        "all_eight_seeds_have_exact_registered_execution_cell_coverage": execution_cell_registry,
        "accepted_draw_set_and_multiplicity_match_manifest_attempts_and_scientific_csv": {
            "accepted_manifest_sha256": accepted_draw_binding_set_sha256,
            "attempt_ledger_sha256": draw_binding_set_sha256,
            "accepted_scientific_csv_sha256": accepted_draw_csv_binding_set_sha256,
            "accepted_row_count": len(accepted_draw_rows),
        },
        "final_manifest_binds": final_manifest_bind_keys,
    }
    if execution_cell_registry["registered_seed_ids"] != list(STAGE3_SEEDS) or execution_cell_registry["coverage_exact"] is not True:
        raise Stage3Stop("execution_cell_registry", "COMPLETE publication requires an exact registered eight-seed execution-cell registry")
    if list(final_manifest_bindings) != final_manifest_bind_keys:
        raise Stage3Stop("manifest_binding", "COMPLETE publication manifest binding keys differ from the frozen Amendment-9 set")
    manifest = {"schema": STAGE3_SCHEMA, "status": "COMPLETE", "protocol_sha256": _protocol_hash(protocol), "git": git, "git_start": git, "git_final": final_git, "source_provenance": {"start": git, "final": final_git}, "invocation_id": invocation_id, "offline_env": env, "snapshots": snapshots, "model_config": model_config, "sae": model_state_fingerprint["sae"], "model_state_fingerprint": model_state_fingerprint, "configuration": configuration, "source_tree_identity": source_tree_identity, "code_revision": code_revision, "preparation": preparation, "immutable_preparation_input_hashes": immutable_preparation_input_hashes, "gate_cache_sha256": cache_hash, "split_manifest_sha256": split_hash, "prepare_manifest_sha256": prepare_hash, "review_receipt_sha256": preparation["review_receipt_sha256"], "harness_receipt_sha256": preparation["harness_receipt_sha256"], "draw_csv_path": str(draw_path), "accepted_scientific_csv": accepted_scientific_csv, "final_checkpoint": final_checkpoint, "execution_cell_registry": execution_cell_registry, "final_manifest_binds": final_manifest_bind_keys, "final_manifest_bindings": final_manifest_bindings, "complete_requires": complete_requires, "seed_results": sorted(completed_rows, key=lambda item: int(item["seed"])), "summary": summary, "draw_csv_sha256": draw_hash, "draw_binding_set_sha256": draw_binding_set_sha256, "accepted_draw_binding_set_sha256": accepted_draw_binding_set_sha256, "draw_bindings": draw_rows, "accepted_draw_bindings": accepted_draw_rows, "cross_seed_summaries": _cross_seed_numeric_summary(completed_rows), "runtime": runtime, "wall_clock_seconds": final_complete_q4_wall_time, "q4_input_boundary": ["protocol", "stage3_gate_a_cache", "stage3_split_manifest", "stage3_prepare_manifest", "review_receipt", "harness_receipt"], "candidate_C_or_stage2_input": False, "scientific_reuse_allowed": False, "resumable_for_adjudication": False}
    complete_checkpoint = dict(checkpoint_core)
    complete_checkpoint.update({"checkpoint_binding_sha256": checkpoint_binding_sha256, "output_sha256": None})
    try:
        _atomic_json(output_path, manifest)
        complete_checkpoint["output_sha256"] = _sha256_file(output_path)
        _atomic_json(checkpoint_path, _with_self_hash(complete_checkpoint))
    except Exception as exc:
        try:
            _write_run_execution_tombstone(output_path, checkpoint_path, draw_path, gate="complete_publish", detail=str(exc))
        except Exception:
            pass
        raise Stage3Stop("execution_incomplete", f"COMPLETE publication transaction failed: {exc}") from exc
    return 0


def _run_q4(args: argparse.Namespace) -> int:
    """Run Q4 and tombstone every unexpected stop or exception."""

    output_path = _resolve(args.output, reject_leaf_symlink=True)
    draw_path = _resolve(args.draw_csv, reject_leaf_symlink=True)
    checkpoint_path = _resolve(args.checkpoint, reject_leaf_symlink=True)
    try:
        return _run_q4_impl(args)
    except Exception as exc:
        try:
            _validate_runtime_paths(
                immutable={
                    "protocol": _resolve(args.protocol), "gate_cache": _resolve(args.gate_cache),
                    "split_manifest": _resolve(args.split_manifest), "prepare_manifest": _resolve(args.prepare_manifest),
                    "review_receipt": _resolve(args.review_receipt), "harness_receipt": _resolve(args.harness_receipt),
                },
                runtime={"output": output_path, "draw_csv": draw_path, "checkpoint": checkpoint_path},
            )
            _write_run_execution_tombstone(output_path, checkpoint_path, draw_path, gate=getattr(exc, "gate", "unexpected_exception"), detail=str(exc))
        except Exception:
            pass
        if isinstance(exc, Stage3Stop):
            raise
        raise Stage3Stop("execution_incomplete", f"unexpected Stage-3 exception: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 05 Stage 3/Q4 fail-closed runner")
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache-gate-a", help="offline clean-logit Gate-A cache only")
    cache.add_argument("--protocol", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--checkpoint", required=True)
    cache.add_argument("--expected-git-commit", required=True)
    cache.add_argument("--require-clean-tree", action="store_true", required=True)
    cache.add_argument("--max-wall-seconds", type=float, required=True)
    split = sub.add_parser("materialize-splits", help="model-free Gate-A cache to Amendment-4 roles")
    split.add_argument("--protocol", required=True)
    split.add_argument("--gate-cache", required=True)
    split.add_argument("--split-manifest", required=True)
    split.add_argument("--split-csv", required=True)
    split.add_argument("--prepare-manifest", required=True)
    run = sub.add_parser("run", help="independent Q4 adjudication")
    run.add_argument("--protocol", required=True)
    run.add_argument("--gate-cache", required=True)
    run.add_argument("--split-manifest", required=True)
    run.add_argument("--prepare-manifest", required=True)
    run.add_argument("--review-receipt", required=True)
    run.add_argument("--harness-receipt", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--draw-csv", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--expected-git-commit", required=True)
    run.add_argument("--require-clean-tree", action="store_true", required=True)
    run.add_argument("--max-wall-seconds", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "cache-gate-a":
            return _run_cache_gate_a(args)
        if args.command == "materialize-splits":
            result = _materialize_splits(_resolve(args.protocol), _resolve(args.gate_cache), _resolve(args.split_manifest, reject_leaf_symlink=True), _resolve(args.split_csv, reject_leaf_symlink=True), _resolve(args.prepare_manifest, reject_leaf_symlink=True))
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "run":
            return _run_q4(args)
    except Stage3Stop as exc:
        print(f"STAGE3 STOP [{exc.gate}] {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover - execution is separately authorised
    raise SystemExit(main())
