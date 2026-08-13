"""Fresh-seed head -> SAE-span -> readout bridge rescue.

This runner is deliberately a small follow-up to the frozen Stage-3 result,
not a second candidate-selection protocol.  It uses fresh seeds (20260814--21)
to build the same number-agreement stimuli, applies Gate-A, and deterministically
keeps the first at most 150 retained pairs.  The target decoder rows and the
100 matched rank-12 sets are copied by ordinal from the supplied Q4 result;
there is no re-selection on the fresh data.

The causal timing is fixed by the Advisor decision ``L7_ONLY_RESID_PRE8``:
first patch L7H4's Q3 frozen-pattern ``hook_z`` on the base prompt, then cache
``blocks.8.hook_resid_pre`` for the true and source-A arms.  All rescue edits
are additive edits to the source-A residual at ``resid_pre8``.  Models and
SAEs are imported and loaded only inside ``run``; importing this module is
model-free and safe for offline contract tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
EXP04 = HERE.parent / "04_causal_feature_interchange"
SOURCE_WORKTREE = HERE.parents[1]

SCHEMA = "exp05-number-agreement-bridge-rescue-v1"
Q4_SCHEMA_PREFIX = "exp05-number-agreement-stage3-v1"
FRESH_SEEDS = tuple(range(20_260_814, 20_260_822))
Q4_SEED_COUNT = 8
REQUESTED_PAIRS = 240
MAX_EVAL_PAIRS = 150
GATE_A_MIN_FRACTION = 0.60
GATE_A_MIN_RETAINED = 140
GATE_A_MIN_MEDIAN_GAP = 1.0
L7 = 7
L8 = 8
HEAD = 4  # H4, zero indexed
READER_HEAD = 5  # L8H5, zero indexed
TARGET_COUNT = 12
MATCHED_COUNT = 100
T_CRITICAL_DF7 = 2.365
RESIDUAL_WIDTH = 768


class BridgeStop(RuntimeError):
    """Expected fail-closed stop; never carries a scientific verdict."""

    def __init__(self, gate: str, message: str, *, status: str = "STOPPED") -> None:
        super().__init__(message)
        self.gate = gate
        self.status = status


class RuntimeCapStop(BridgeStop):
    def __init__(self, message: str) -> None:
        super().__init__("max_wall_seconds", message, status="STOPPED")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite artifact value: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _jsonable(value.detach().cpu().tolist())
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
        raise BridgeStop("input_read", f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resolve_path_alias(value: str | Path, *, base: Path = HERE) -> Path:
    """Resolve a portable path without silently changing a supplied absolute path.

    ``@exp05/…`` points beside this file and ``@repo/…`` points at the source
    worktree.  Relative paths use the explicit caller-provided base.  Unknown
    ``@`` prefixes are rejected instead of falling through to a surprising
    machine-specific path.
    """

    raw = str(value).strip()
    if not raw:
        raise BridgeStop("path_alias", "empty path is not allowed")
    if raw.startswith("@exp05/"):
        return (HERE / raw[len("@exp05/") :]).expanduser().resolve()
    if raw.startswith("@repo/"):
        return (SOURCE_WORKTREE / raw[len("@repo/") :]).expanduser().resolve()
    if raw.startswith("@"):
        raise BridgeStop("path_alias", f"unknown path alias: {raw.split('/', 1)[0]}")
    candidate = Path(raw).expanduser()
    return (candidate if candidate.is_absolute() else (base / candidate)).resolve()


def _git(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=SOURCE_WORKTREE, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise BridgeStop("git_provenance", f"cannot inspect git: {exc}") from exc
    if result.returncode:
        raise BridgeStop("git_provenance", result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _source_tree_sha256() -> str:
    return _sha256_bytes(_git(["ls-files", "-s"]).encode("utf-8"))


def _git_provenance(expected: str | None, require_clean: bool) -> dict[str, Any]:
    if not require_clean:
        raise BridgeStop("clean_tree_required", "--require-clean-tree is required for a model-backed bridge run")
    commit = _git(["rev-parse", "HEAD"])
    if expected and commit != expected:
        raise BridgeStop("expected_git_commit", f"HEAD {commit} differs from expected {expected}")
    status = _git(["status", "--porcelain"])
    if status:
        raise BridgeStop("dirty_tree", "working tree is not clean")
    return {
        "commit": commit,
        "expected_commit": expected,
        "require_clean_tree": True,
        "status_porcelain": status,
        "source_worktree": str(SOURCE_WORKTREE),
        "source_tree_sha256": _source_tree_sha256(),
    }


def _assert_source_unchanged(start: Mapping[str, Any]) -> dict[str, Any]:
    final = _git_provenance(str(start.get("expected_commit") or ""), True)
    if dict(final) != dict(start):
        raise BridgeStop("source_changed", "HEAD/source tree/clean status changed during the run")
    return final


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BridgeStop("missing_input", f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeStop("invalid_input_json", f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BridgeStop("invalid_input_schema", f"{label} must be a JSON object")
    return value


def _check_runtime(started: float, cap: float | None, where: str) -> None:
    if cap is not None and time.perf_counter() - started >= cap:
        raise RuntimeCapStop(f"declared runtime cap {cap:.3f}s reached at {where}")


def _finite_list(values: Iterable[Any], *, label: str) -> list[float]:
    out = [float(value) for value in values]
    if not out or not all(math.isfinite(value) for value in out):
        raise BridgeStop("nonfinite_statistic", f"{label} is empty or non-finite")
    return out


def mean_ratio(signs: Sequence[float], arm_d: Sequence[float], baseline_d: Sequence[float], full_d: Sequence[float]) -> float:
    """Compute the prescribed ratio of directed signed means, fail closed."""

    if not (len(signs) == len(arm_d) == len(baseline_d) == len(full_d)) or not signs:
        raise BridgeStop("ratio_shape", "ratio inputs must have the same non-empty length")
    numerator = math.fsum(float(s) * (float(a) - float(b)) for s, a, b in zip(signs, arm_d, baseline_d)) / len(signs)
    denominator = math.fsum(float(s) * (float(f) - float(b)) for s, f, b in zip(signs, full_d, baseline_d)) / len(signs)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        raise BridgeStop("denominator_guard", "rescue ratio denominator is non-finite or too close to zero")
    return float(numerator / denominator)


def _mean_signed_delta(signs: Sequence[float], arm_d: Sequence[float], baseline_d: Sequence[float]) -> float:
    if not (len(signs) == len(arm_d) == len(baseline_d)) or not signs:
        raise BridgeStop("effect_shape", "effect inputs must have the same non-empty length")
    value = math.fsum(float(s) * (float(a) - float(b)) for s, a, b in zip(signs, arm_d, baseline_d)) / len(signs)
    if not math.isfinite(value):
        raise BridgeStop("nonfinite_effect", "signed rescue effect is non-finite")
    return float(value)


def _t7_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = _finite_list(values, label="cross-seed values")
    mean = math.fsum(finite) / len(finite)
    if len(finite) != 8:
        return {"finite_count": len(finite), "mean": mean, "ci95": {"low": None, "high": None}, "t_critical": None, "degrees_of_freedom": None, "status": "SCIENTIFIC_UNRESOLVED"}
    variance = math.fsum((value - mean) ** 2 for value in finite) / 7.0
    se = math.sqrt(variance / 8.0)
    half = T_CRITICAL_DF7 * se
    return {"finite_count": 8, "mean": mean, "standard_error": se, "t_critical": T_CRITICAL_DF7, "degrees_of_freedom": 7, "ci95": {"low": mean - half, "high": mean + half}, "status": "ESTIMABLE"}


def _flatten_double(value: Any) -> Any:
    # Keep tensor import lazy while accepting both tensors and numpy-like arrays.
    if hasattr(value, "detach"):
        return value.detach().cpu().double().reshape(-1)
    torch = __import__("torch")
    return torch.as_tensor(value, dtype=torch.float64).reshape(-1)


def decoder_row_projector(rows: Any) -> tuple[Any, dict[str, Any]]:
    """Return a float64 orthonormal row basis and metadata for rank-12 rows."""

    torch = __import__("torch")
    matrix = rows.detach().cpu().double() if hasattr(rows, "detach") else torch.as_tensor(rows, dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape[0] != TARGET_COUNT or matrix.shape[1] != RESIDUAL_WIDTH:
        raise BridgeStop("projector_shape", f"decoder rows must have shape (12,768), got {tuple(matrix.shape)}")
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    s_max = float(singular[0]) if singular.numel() else 0.0
    tolerance = max(matrix.shape) * float.fromhex("0x1.0000000000000p-52") * s_max
    rank = int((singular > tolerance).sum())
    if rank != TARGET_COUNT:
        raise BridgeStop("projector_rank", f"decoder rows have numerical rank {rank}, expected {TARGET_COUNT}")
    basis = vh[:TARGET_COUNT].contiguous()
    return basis, {
        "arithmetic": "float64",
        "shape": list(matrix.shape),
        "rank": rank,
        "singular_values": [float(item) for item in singular],
        "tolerance": tolerance,
    }


def project_float64(delta: Any, basis: Any) -> Any:
    """Project only in float64; callers cast to model dtype at injection."""

    torch = __import__("torch")
    values = delta.detach().double() if hasattr(delta, "detach") else torch.as_tensor(delta, dtype=torch.float64)
    rows = basis.detach().double() if hasattr(basis, "detach") else torch.as_tensor(basis, dtype=torch.float64)
    if rows.ndim != 2 or rows.shape[1] != values.shape[-1]:
        raise BridgeStop("projector_shape", "projector width does not match delta width")
    return (values @ rows.T) @ rows


def reader_projection(z_arm: Any, z_a: Any, z_full: Any) -> dict[str, Any]:
    """Describe the L8H5 response in the full rescue direction."""

    torch = __import__("torch")
    arm = _flatten_double(z_arm)
    baseline = _flatten_double(z_a)
    full = _flatten_double(z_full)
    if not (arm.numel() == baseline.numel() == full.numel()):
        raise BridgeStop("reader_shape", "L8 reader response arrays have different sizes")
    arm_delta = arm - baseline
    full_delta = full - baseline
    denom = float(torch.dot(full_delta, full_delta))
    arm_norm = float(torch.linalg.vector_norm(arm_delta))
    full_norm = float(torch.linalg.vector_norm(full_delta))
    if denom <= 1e-24 or full_norm <= 1e-12:
        return {"status": "NON_ESTIMABLE_FULL_READER_DELTA", "coefficient": None, "cosine": None, "arm_delta_norm": arm_norm, "full_delta_norm": full_norm}
    dot = float(torch.dot(arm_delta, full_delta))
    cosine = dot / max(arm_norm * full_norm, 1e-300) if arm_norm > 0.0 else 0.0
    return {"status": "ESTIMABLE", "coefficient": dot / denom, "cosine": cosine, "arm_delta_norm": arm_norm, "full_delta_norm": full_norm}


def parse_q4_frozen_sets(q4: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate ordinal target/matched sets without fresh reselection."""

    if not str(q4.get("schema", "")).startswith(Q4_SCHEMA_PREFIX) or q4.get("status") != "COMPLETE":
        raise BridgeStop("q4_results", "Q4 result must be COMPLETE with the frozen Stage-3 schema")
    rows = q4.get("seed_results")
    if not isinstance(rows, list) or len(rows) != Q4_SEED_COUNT:
        raise BridgeStop("q4_results", "Q4 result must contain exactly eight seed results")
    by_seed: dict[int, Mapping[str, Any]] = {}
    target_ids: tuple[int, ...] | None = None
    frozen: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise BridgeStop("q4_results", "Q4 seed result is not an object")
        seed = int(row.get("seed", -1))
        if seed in by_seed:
            raise BridgeStop("q4_results", f"duplicate Q4 seed {seed}")
        by_seed[seed] = row
        projector = row.get("projector")
        if not isinstance(projector, Mapping):
            raise BridgeStop("q4_results", f"Q4 seed {seed} lacks frozen target projector")
        current_target = tuple(int(item) for item in projector.get("target_latent_ids", ()))
        if len(current_target) != TARGET_COUNT or len(set(current_target)) != TARGET_COUNT:
            raise BridgeStop("q4_results", f"Q4 seed {seed} target ids are not exactly twelve unique ids")
        if target_ids is None:
            target_ids = current_target
        elif current_target != target_ids:
            raise BridgeStop("q4_results", "Q4 target decoder rows differ across ordinal seeds")
        matched = row.get("matched_draws")
        accepted = matched.get("accepted") if isinstance(matched, Mapping) else None
        if not isinstance(accepted, list) or len(accepted) != MATCHED_COUNT:
            raise BridgeStop("q4_results", f"Q4 seed {seed} lacks exactly 100 accepted matched draws")
        draws: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for draw in accepted:
            if not isinstance(draw, Mapping):
                raise BridgeStop("q4_results", f"Q4 seed {seed} contains a malformed matched draw")
            index = int(draw.get("draw_index", -1))
            ids = tuple(int(item) for item in draw.get("latent_ids", ()))
            if index in seen_indices or len(ids) != TARGET_COUNT or len(set(ids)) != TARGET_COUNT or set(ids).intersection(target_ids):
                raise BridgeStop("q4_results", f"Q4 seed {seed} matched draw {index} violates frozen id binding")
            seen_indices.add(index)
            draws.append({"draw_index": index, "latent_ids": list(ids), "source_q4_seed": seed})
        if seen_indices != set(range(MATCHED_COUNT)):
            raise BridgeStop("q4_results", f"Q4 seed {seed} matched draw indices are not exactly 0..99")
        frozen.append({"source_q4_seed": seed, "target_latent_ids": list(target_ids), "matched": sorted(draws, key=lambda item: item["draw_index"])})
    if target_ids is None:
        raise BridgeStop("q4_results", "Q4 target ids were not found")
    source_seeds = sorted(by_seed)
    if len(source_seeds) != Q4_SEED_COUNT or len(set(source_seeds)) != Q4_SEED_COUNT:
        raise BridgeStop("q4_results", "Q4 source seeds must be exactly eight unique ids")
    by_source_seed = {int(item["source_q4_seed"]): item for item in frozen}
    if set(by_source_seed) != set(source_seeds):
        raise BridgeStop("q4_results", "Q4 ordinal seed binding is incomplete")
    # Raw JSON row order is not a scientific input.  Ordinal reuse is defined
    # by sorted source seed id, so a reordered Q4 artifact cannot silently map
    # a fresh seed to a different matched-set family.
    ordinal_sets = [by_source_seed[seed] for seed in source_seeds]
    return {"q4_seeds": source_seeds, "target_latent_ids": list(target_ids), "ordinal_sets": ordinal_sets}


def _validate_runtime_paths(output_path: Path, immutable: Sequence[Path]) -> None:
    resolved = [path.expanduser().resolve() for path in immutable]
    output = output_path.expanduser().resolve()
    if output in resolved:
        raise BridgeStop("runtime_path_alias", "output aliases an immutable input")
    try:
        output.relative_to(SOURCE_WORKTREE)
    except ValueError:
        return
    raise BridgeStop("runtime_path_alias", "runtime output must be outside the immutable source worktree")


def _load_stack() -> dict[str, Any]:
    """Import model primitives only on the model-backed execution path."""

    sys.dont_write_bytecode = True
    # ``bridge_rescue`` is also imported by tests and review tooling, where
    # Python does not automatically put this script's directory on sys.path.
    # Prefer the Exp05 implementations of ``calibrate``/``stage1`` while
    # retaining Exp04's ``pilot`` as the shared model/SAE loader.
    for module_dir in (EXP04, HERE):
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
    from calibrate import make_source_a  # type: ignore
    from pilot import build_stimuli, directed_indices, gather_positions, load_direct_res_jb, load_model, logit_difference, positions_for_kind, require_one_token, set_determinism  # type: ignore
    from stage1 import _patch_hook, clean_readout_microbatched  # type: ignore
    return {
        "make_source_a": make_source_a,
        "build_stimuli": build_stimuli,
        "directed_indices": directed_indices,
        "gather_positions": gather_positions,
        "load_direct_res_jb": load_direct_res_jb,
        "load_model": load_model,
        "logit_difference": logit_difference,
        "positions_for_kind": positions_for_kind,
        "require_one_token": require_one_token,
        "set_determinism": set_determinism,
        "patch_hook": _patch_hook,
        "clean_readout_microbatched": clean_readout_microbatched,
    }


def _cache_hooks(model: Any, tokens: Any, names: set[str], *, batch_size: int) -> dict[str, Any]:
    torch = __import__("torch")
    parts: dict[str, list[Any]] = {name: [] for name in names}
    for start in range(0, int(tokens.shape[0]), batch_size):
        stop = min(start + batch_size, int(tokens.shape[0]))
        with torch.no_grad():
            result, cache = model.run_with_cache(tokens[start:stop], names_filter=lambda name: name in names, return_type=None)
        if result is not None:
            raise BridgeStop("cache_return_type", "activation cache unexpectedly returned logits")
        for name in names:
            parts[name].append(cache[name].detach().float().cpu().clone())
        del cache
    return {name: torch.cat(values) for name, values in parts.items()}


def _run_l7_arm(model: Any, tokens: Any, final_positions: Any, replacement: Any, *, patch_hook: Any, logit_difference: Any, lengths: Any, is_id: int, are_id: int, batch_size: int, label: str, started: float, cap: float | None) -> dict[str, Any]:
    torch = __import__("torch")
    d_values: list[Any] = []
    residual_parts: list[Any] = []
    reader_parts: list[Any] = []
    for start in range(0, int(tokens.shape[0]), batch_size):
        _check_runtime(started, cap, f"before {label} batch {start}")
        stop = min(start + batch_size, int(tokens.shape[0]))
        local_positions = final_positions[start:stop]
        local_replacement = replacement[start:stop]
        captured: dict[str, Any] = {}

        def z_patch(activation: Any, hook: Any) -> Any:
            del hook
            return patch_hook(activation, base_positions=local_positions, replacement=local_replacement, head=HEAD, expected_heads=int(model.cfg.n_heads), expected_d_head=int(model.cfg.d_head))

        def capture_resid(activation: Any, hook: Any) -> Any:
            del hook
            captured["resid"] = activation.detach().float().cpu().clone()
            return activation

        def capture_reader(activation: Any, hook: Any) -> Any:
            del hook
            rows = torch.arange(activation.shape[0], device=activation.device)
            captured["reader"] = activation[rows, local_positions.to(activation.device), READER_HEAD, :].detach().float().cpu().clone()
            return activation

        hooks = [
            (f"blocks.{L7}.attn.hook_z", z_patch),
            (f"blocks.{L8}.hook_resid_pre", capture_resid),
            (f"blocks.{L8}.attn.hook_z", capture_reader),
        ]
        with torch.no_grad():
            logits = model.run_with_hooks(tokens[start:stop], fwd_hooks=hooks, return_type="logits")
        d_values.append(logit_difference(logits, lengths[start:stop], is_id, are_id).detach().float().cpu())
        residual_parts.append(captured["resid"])
        reader_parts.append(captured["reader"])
        del logits
        _check_runtime(started, cap, f"after {label} batch {start}")
    return {"d": torch.cat(d_values), "resid": torch.cat(residual_parts), "reader": torch.cat(reader_parts), "label": label}


def _run_rescue_natural(model: Any, a_resid: Any, positions: Any, delta: Any, final_positions: Any, *, logit_difference: Any, lengths: Any, is_id: int, are_id: int, batch_size: int, label: str, started: float, cap: float | None) -> dict[str, Any]:
    torch = __import__("torch")
    d_values: list[Any] = []
    reader_parts: list[Any] = []
    rows_all = torch.arange(a_resid.shape[0])
    edited = a_resid.clone()
    for slot in range(int(positions.shape[1])):
        edited[rows_all, positions[:, slot]] += delta[:, slot].float()
    for start in range(0, int(edited.shape[0]), batch_size):
        _check_runtime(started, cap, f"before {label} batch {start}")
        stop = min(start + batch_size, int(edited.shape[0]))
        local_final = final_positions[start:stop]
        captured: dict[str, Any] = {}

        def capture_reader(activation: Any, hook: Any) -> Any:
            del hook
            rows = torch.arange(activation.shape[0], device=activation.device)
            captured["reader"] = activation[rows, local_final.to(activation.device), READER_HEAD, :].detach().float().cpu().clone()
            return activation

        with torch.no_grad():
            logits = model.run_with_hooks(edited[start:stop], fwd_hooks=[(f"blocks.{L8}.attn.hook_z", capture_reader)], start_at_layer=L8, return_type="logits")
        d_values.append(logit_difference(logits, lengths[start:stop], is_id, are_id).detach().float().cpu())
        reader_parts.append(captured["reader"])
        del logits
        _check_runtime(started, cap, f"after {label} batch {start}")
    return {"d": torch.cat(d_values), "reader": torch.cat(reader_parts), "label": label}


def _run_rescue_clamped(model: Any, tokens: Any, l7_positions: Any, l7_replacement: Any, positions: Any, delta: Any, final_positions: Any, clamp_reader: Any, *, patch_hook: Any, logit_difference: Any, lengths: Any, is_id: int, are_id: int, batch_size: int, label: str, started: float, cap: float | None) -> dict[str, Any]:
    torch = __import__("torch")
    d_values: list[Any] = []
    rows_all = torch.arange(tokens.shape[0])
    for start in range(0, int(tokens.shape[0]), batch_size):
        _check_runtime(started, cap, f"before {label} batch {start}")
        stop = min(start + batch_size, int(tokens.shape[0]))
        local_l7_positions = l7_positions[start:stop]
        local_l7_replacement = l7_replacement[start:stop]
        local_positions = positions[start:stop]
        local_delta = delta[start:stop].float()
        local_final = final_positions[start:stop]
        local_clamp = clamp_reader[start:stop]

        def z_patch(activation: Any, hook: Any) -> Any:
            del hook
            return patch_hook(activation, base_positions=local_l7_positions, replacement=local_l7_replacement, head=HEAD, expected_heads=int(model.cfg.n_heads), expected_d_head=int(model.cfg.d_head))

        def resid_patch(activation: Any, hook: Any) -> Any:
            del hook
            rows = torch.arange(activation.shape[0], device=activation.device)
            for slot in range(int(local_positions.shape[1])):
                activation[rows, local_positions[:, slot].to(activation.device)] += local_delta[:, slot].to(activation.device, dtype=activation.dtype)
            return activation

        def reader_clamp(activation: Any, hook: Any) -> Any:
            del hook
            rows = torch.arange(activation.shape[0], device=activation.device)
            positions_device = local_final.to(activation.device)
            # TransformerLens hook_z is the complete per-head output after the
            # attention pattern has aggregated values. This overwrites L8H5 at
            # the final query position; it is not a value-only intervention.
            activation[rows, positions_device, READER_HEAD, :] = local_clamp.to(device=activation.device, dtype=activation.dtype)
            return activation

        hooks = [
            (f"blocks.{L7}.attn.hook_z", z_patch),
            (f"blocks.{L8}.hook_resid_pre", resid_patch),
            (f"blocks.{L8}.attn.hook_z", reader_clamp),
        ]
        with torch.no_grad():
            logits = model.run_with_hooks(tokens[start:stop], fwd_hooks=hooks, return_type="logits")
        d_values.append(logit_difference(logits, lengths[start:stop], is_id, are_id).detach().float().cpu())
        del logits
        _check_runtime(started, cap, f"after {label} batch {start}")
    return {"d": torch.cat(d_values), "label": label}


def _gate_a(clean_d: Any) -> dict[str, Any]:
    values = [float(item) for item in clean_d.detach().cpu().tolist()]
    if len(values) != 2 * REQUESTED_PAIRS:
        raise BridgeStop("gate_a_shape", "fresh clean readout does not cover 240 pairs")
    retained: list[int] = []
    gaps: list[float] = []
    for pair_id in range(REQUESTED_PAIRS):
        singular, plural = values[2 * pair_id], values[2 * pair_id + 1]
        gaps.append(plural - singular)
        if singular < 0.0 and plural > 0.0:
            retained.append(pair_id)
    ordered = sorted(gaps)
    median = ordered[(len(ordered) - 1) // 2]
    fraction = len(retained) / float(REQUESTED_PAIRS)
    passed = bool(fraction >= GATE_A_MIN_FRACTION and len(retained) >= GATE_A_MIN_RETAINED and median >= GATE_A_MIN_MEDIAN_GAP)
    return {"generated_pairs": REQUESTED_PAIRS, "retained_pair_ids": retained, "retained_pairs": len(retained), "fraction": fraction, "median_gap": median, "passed": passed, "thresholds": {"fraction_at_least": GATE_A_MIN_FRACTION, "retained_at_least": GATE_A_MIN_RETAINED, "median_gap_at_least": GATE_A_MIN_MEDIAN_GAP}}


def _arm_summary(values: Any, signs: Any, baseline: Any, full: Any | None = None) -> dict[str, Any]:
    d = [float(item) for item in values.detach().cpu().tolist()]
    s = [float(item) for item in signs.detach().cpu().tolist()]
    a = [float(item) for item in baseline.detach().cpu().tolist()]
    payload: dict[str, Any] = {"mean_d": math.fsum(d) / len(d), "signed_delta_mean_vs_source_A": _mean_signed_delta(s, d, a)}
    if full is not None:
        f = [float(item) for item in full.detach().cpu().tolist()]
        payload["R"] = mean_ratio(s, d, a, f)
    return payload


def _run_seed(model: Any, sae: Any, stack: Mapping[str, Any], *, seed: int, ordinal: int, frozen: Mapping[str, Any], started: float, cap: float | None, batch_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = __import__("torch")
    build_stimuli = stack["build_stimuli"]
    make_source_a = stack["make_source_a"]
    directed_indices = stack["directed_indices"]
    positions_for_kind = stack["positions_for_kind"]
    require_one_token = stack["require_one_token"]
    clean_readout = stack["clean_readout_microbatched"]
    set_determinism = stack["set_determinism"]
    patch_hook = stack["patch_hook"]
    logit_difference = stack["logit_difference"]
    set_determinism(seed)
    base = build_stimuli(model.tokenizer, REQUESTED_PAIRS, seed)
    is_id = require_one_token(model.tokenizer, " is")
    are_id = require_one_token(model.tokenizer, " are")
    clean_d = clean_readout(model, base.tokens, base.lengths, is_id, are_id)
    gate = _gate_a(clean_d)
    if not gate["passed"]:
        return {"seed": seed, "source_q4_seed": int(frozen["source_q4_seed"]), "status": "SCIENTIFIC_UNRESOLVED", "reason": "GATE_A_FAIL", "gate_a": gate}, []
    eval_pair_ids = gate["retained_pair_ids"][:MAX_EVAL_PAIRS]
    if not eval_pair_ids:
        return {"seed": seed, "source_q4_seed": int(frozen["source_q4_seed"]), "status": "SCIENTIFIC_UNRESOLVED", "reason": "NO_RETAINED_PAIRS", "gate_a": gate}, []
    base_indices, source_indices, signs = directed_indices(REQUESTED_PAIRS, eval_pair_ids)
    base_tokens = base.tokens[base_indices]
    true_tokens = base.tokens[source_indices]
    source_a = make_source_a(model.tokenizer, base, seed)
    source_a_tokens = source_a.tokens[base_indices]
    base_lengths = base.lengths[base_indices]
    if not torch.equal(base_lengths, base.lengths[source_indices]) or not torch.equal(base_lengths, source_a.lengths[base_indices]):
        raise BridgeStop("stimulus_lengths", f"seed {seed} true/source-A sequence lengths differ")
    base_final = positions_for_kind(base, base_indices, "final").squeeze(1)
    base_subject = positions_for_kind(base, base_indices, "subject").squeeze(1)
    source_subject = positions_for_kind(base, source_indices, "subject").squeeze(1)
    source_a_subject = positions_for_kind(source_a, base_indices, "subject").squeeze(1)
    positions = positions_for_kind(base, base_indices, "both")
    names_base = {f"blocks.{L7}.attn.hook_z", f"blocks.{L7}.attn.hook_v", f"blocks.{L7}.attn.hook_pattern"}
    base_cache = _cache_hooks(model, base_tokens, names_base, batch_size=batch_size)
    true_cache = _cache_hooks(model, true_tokens, {f"blocks.{L7}.attn.hook_v"}, batch_size=batch_size)
    a_cache = _cache_hooks(model, source_a_tokens, {f"blocks.{L7}.attn.hook_v"}, batch_size=batch_size)
    rows = torch.arange(base_tokens.shape[0])
    base_z = base_cache[f"blocks.{L7}.attn.hook_z"][rows, base_final, HEAD, :]
    pattern = base_cache[f"blocks.{L7}.attn.hook_pattern"][rows, HEAD, base_final, base_subject]
    base_v = base_cache[f"blocks.{L7}.attn.hook_v"][rows, base_subject, HEAD, :]
    true_v = true_cache[f"blocks.{L7}.attn.hook_v"][rows, source_subject, HEAD, :]
    a_v = a_cache[f"blocks.{L7}.attn.hook_v"][rows, source_a_subject, HEAD, :]
    z_true = base_z + pattern.unsqueeze(-1) * (true_v - base_v)
    z_a = base_z + pattern.unsqueeze(-1) * (a_v - base_v)
    a_arm = _run_l7_arm(model, base_tokens, base_final, z_a, patch_hook=patch_hook, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_source_A_L7H4_seed_{seed}", started=started, cap=cap)
    true_arm = _run_l7_arm(model, base_tokens, base_final, z_true, patch_hook=patch_hook, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_true_L7H4_seed_{seed}", started=started, cap=cap)
    a_resid, true_resid = a_arm["resid"], true_arm["resid"]
    delta = (true_resid.detach().double() - a_resid.detach().double())
    selected_a = a_resid[rows[:, None], positions]
    selected_true = true_resid[rows[:, None], positions]
    selected_delta = delta[rows[:, None], positions]
    selected_identity_error = float((selected_a.double() + selected_delta).sub(selected_true.double()).abs().max())
    non_final_mask = torch.ones(delta.shape[:2], dtype=torch.bool)
    non_final_mask[rows, base_final] = False
    non_final_delta_error = float(delta.abs()[non_final_mask].max()) if bool(non_final_mask.any()) else 0.0
    identity_diagnostics = {
        "selected_positions_max_abs": selected_identity_error,
        "non_final_positions_max_abs": non_final_delta_error,
        "non_final_tolerance": 1e-6,
        "status": "PASS" if non_final_delta_error <= 1e-6 else "FAIL",
    }
    if non_final_delta_error > 1e-6:
        raise BridgeStop("timing_identity", f"L7H4 final-only intervention changed non-final resid_pre8 positions by {non_final_delta_error:.6g}")
    target_rows = sae.W_dec[list(frozen["target_latent_ids"])].detach().float().cpu()
    target_basis, target_meta = decoder_row_projector(target_rows)
    target_delta = project_float64(selected_delta, target_basis)
    complement_delta = selected_delta - target_delta
    natural: dict[str, Any] = {}
    full = _run_rescue_natural(model, a_resid, positions, selected_delta, base_final, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_full_seed_{seed}", started=started, cap=cap)
    target = _run_rescue_natural(model, a_resid, positions, target_delta, base_final, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_target_seed_{seed}", started=started, cap=cap)
    complement = _run_rescue_natural(model, a_resid, positions, complement_delta, base_final, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_complement_seed_{seed}", started=started, cap=cap)
    baseline_d = a_arm["d"]
    full_d = full["d"]
    natural["source_A"] = _arm_summary(baseline_d, signs, baseline_d)
    natural["full"] = {**_arm_summary(full_d, signs, baseline_d, full_d), "reader": reader_projection(full["reader"], a_arm["reader"], full["reader"])}
    full_logit_error = float((full_d - true_arm["d"]).abs().max())
    identity_diagnostics["full_vs_true_final_logit_max_abs"] = full_logit_error
    identity_diagnostics["full_vs_true_final_logit_tolerance"] = 1e-5
    if full_logit_error > 1e-5:
        raise BridgeStop("timing_identity", f"full rescue does not reproduce direct true L7H4 final logits: max_abs={full_logit_error:.6g}")
    natural["target"] = {**_arm_summary(target["d"], signs, baseline_d, full_d), "reader": reader_projection(target["reader"], a_arm["reader"], full["reader"])}
    natural["complement"] = {**_arm_summary(complement["d"], signs, baseline_d, full_d), "reader": reader_projection(complement["reader"], a_arm["reader"], full["reader"])}
    matched_rows: list[dict[str, Any]] = []
    for draw in frozen["matched"]:
        _check_runtime(started, cap, f"before matched draw {draw['draw_index']} seed {seed}")
        matched_rows_tensor, matched_meta = decoder_row_projector(sae.W_dec[list(draw["latent_ids"])].detach().float().cpu())
        matched_delta = project_float64(selected_delta, matched_rows_tensor)
        matched = _run_rescue_natural(model, a_resid, positions, matched_delta, base_final, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_matched_{draw['draw_index']}_seed_{seed}", started=started, cap=cap)
        matched_rows.append({"seed": seed, "source_q4_seed": int(draw["source_q4_seed"]), "draw_index": int(draw["draw_index"]), "latent_ids": list(draw["latent_ids"]), "R_matched": mean_ratio([float(item) for item in signs.tolist()], [float(item) for item in matched["d"].tolist()], [float(item) for item in baseline_d.tolist()], [float(item) for item in full_d.tolist()]), "signed_delta_mean_vs_source_A": _mean_signed_delta([float(item) for item in signs.tolist()], [float(item) for item in matched["d"].tolist()], [float(item) for item in baseline_d.tolist()]), "reader": reader_projection(matched["reader"], a_arm["reader"], full["reader"]), "projector": {"rank": matched_meta["rank"], "arithmetic": matched_meta["arithmetic"]}})
    matched_values = [float(item["R_matched"]) for item in matched_rows]
    matched_sorted = sorted(matched_values)
    # Clamp-to-source-A baseline is intentionally limited to full and target.
    zero = torch.zeros_like(selected_delta)
    clamp_a = _run_rescue_clamped(model, base_tokens, base_final, z_a, positions, zero, base_final, a_arm["reader"], patch_hook=patch_hook, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_clamp_source_A_seed_{seed}", started=started, cap=cap)
    clamp_full = _run_rescue_clamped(model, base_tokens, base_final, z_a, positions, selected_delta, base_final, a_arm["reader"], patch_hook=patch_hook, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_clamp_full_seed_{seed}", started=started, cap=cap)
    clamp_target = _run_rescue_clamped(model, base_tokens, base_final, z_a, positions, target_delta, base_final, a_arm["reader"], patch_hook=patch_hook, logit_difference=logit_difference, lengths=base_lengths, is_id=is_id, are_id=are_id, batch_size=batch_size, label=f"bridge_clamp_target_seed_{seed}", started=started, cap=cap)
    clamp_a_d, clamp_full_d, clamp_target_d = clamp_a["d"], clamp_full["d"], clamp_target["d"]
    clamped = {
        "source_A": _arm_summary(clamp_a_d, signs, clamp_a_d),
        "full": _arm_summary(clamp_full_d, signs, clamp_a_d, clamp_full_d),
        "target": _arm_summary(clamp_target_d, signs, clamp_a_d, clamp_full_d),
        "reader_clamp": {
            "hook": f"blocks.{L8}.attn.hook_z",
            "head": f"L{L8}H{READER_HEAD}",
            "query_position": "final",
            "replacement": "natural source-A L7H4-arm hook_z output",
            "semantics": "complete per-head z after attention-pattern-weighted value aggregation",
            "value_only": False,
        },
    }
    def empirical_quantile(values: Sequence[float], q: float) -> float:
        if not values:
            raise BridgeStop("matched_summary", "matched draw values are empty")
        ordered = sorted(float(item) for item in values)
        position = (len(ordered) - 1) * float(q)
        lower = int(math.floor(position))
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    seed_result = {
        "seed": seed,
        "source_q4_seed": int(frozen["source_q4_seed"]),
        "status": "COMPLETE",
        "gate_a": gate,
        "evaluation_pair_ids": [int(item) for item in eval_pair_ids],
        "evaluation_item_count": int(len(eval_pair_ids) * 2),
        "timing_decision": "L7_ONLY_RESID_PRE8",
        "l7_head": {"layer": L7, "head": HEAD},
        "reader_head": {"layer": L8, "head": READER_HEAD},
        "target_latent_ids": list(frozen["target_latent_ids"]),
        "target_projector": target_meta,
        "identity_diagnostics": identity_diagnostics,
        "full_identity_max_abs": selected_identity_error,
        "full_vs_true_final_logit_max_abs": full_logit_error,
        "natural": natural,
        "matched_summary": {"count": len(matched_values), "mean": math.fsum(matched_values) / len(matched_values), "median": empirical_quantile(matched_values, 0.50), "p05": empirical_quantile(matched_values, 0.05), "p95": empirical_quantile(matched_values, 0.95), "second_largest": matched_sorted[-2], "max": matched_sorted[-1]},
        "clamped": clamped,
    }
    return seed_result, matched_rows


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    output_path = resolve_path_alias(args.output, base=HERE)
    q4_path = resolve_path_alias(args.q4_results, base=HERE)
    protocol_path = resolve_path_alias(args.protocol, base=HERE) if getattr(args, "protocol", None) else None
    _validate_runtime_paths(output_path, [q4_path, *( [protocol_path] if protocol_path else [])])
    started = time.perf_counter()
    git: dict[str, Any] | None = None
    try:
        git = _git_provenance(args.expected_git_commit, bool(args.require_clean_tree))
        q4 = _read_json(q4_path, "Q4 results")
        frozen = parse_q4_frozen_sets(q4)
        q4_hash = _sha256_file(q4_path)
        protocol_hash = None
        if protocol_path is not None:
            protocol_payload = _read_json(protocol_path, "protocol")
            # Stage-3 records the canonical JSON hash, not the byte hash of
            # the pretty-printed protocol file.
            protocol_hash = _sha256_bytes(_json_bytes(protocol_payload))
            if q4.get("protocol_sha256") != protocol_hash:
                raise BridgeStop("protocol_hash", "supplied protocol does not match Q4 result")
        _atomic_json(output_path, {"schema": SCHEMA, "status": "RUNNING", "verdict": None, "scientific_verdict_emitted": False, "inputs": {"q4_results_sha256": q4_hash, "protocol_sha256": protocol_hash, "fresh_seeds": list(FRESH_SEEDS)}, "git": git})
        if len(frozen["ordinal_sets"]) != len(FRESH_SEEDS):
            raise BridgeStop("q4_ordinal_binding", "Q4 ordinal frozen sets do not cover the eight fresh seeds")
        offline = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        if any(value != "1" for value in offline.values()):
            raise BridgeStop("offline_provenance", "HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        stack = _load_stack()
        model = stack["load_model"]()
        sae = stack["load_direct_res_jb"](L8)
        if tuple(sae.W_dec.shape) != (24_576, RESIDUAL_WIDTH):
            raise BridgeStop("sae_shape", f"unexpected layer-8 SAE decoder shape {tuple(sae.W_dec.shape)}")
        batch_size = max(1, int(getattr(args, "batch_size", 32)))
        seed_results: list[dict[str, Any]] = []
        matched_rows: list[dict[str, Any]] = []
        for ordinal, (seed, frozen_row) in enumerate(zip(FRESH_SEEDS, frozen["ordinal_sets"])):
            _check_runtime(started, args.max_wall_seconds, f"before fresh seed {seed}")
            result, draws = _run_seed(model, sae, stack, seed=seed, ordinal=ordinal, frozen=frozen_row, started=started, cap=args.max_wall_seconds, batch_size=batch_size)
            seed_results.append(result)
            matched_rows.extend(draws)
        if any(row.get("status") != "COMPLETE" for row in seed_results):
            raise BridgeStop("seed_gate", "at least one fresh seed failed Gate-A; no scientific verdict is emitted")
        target_values = [float(row["natural"]["target"]["R"]) for row in seed_results]
        comp_values = [float(row["natural"]["complement"]["R"]) for row in seed_results]
        matched_means = [float(row["matched_summary"]["mean"]) for row in seed_results]
        clamp_target_values = [float(row["clamped"]["target"]["R"]) for row in seed_results]
        aggregate = {"R_target": _t7_summary(target_values), "R_complement": _t7_summary(comp_values), "R_matched_mean": _t7_summary(matched_means), "R_target_clamped": _t7_summary(clamp_target_values)}
        final_git = _assert_source_unchanged(git)
        output = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "verdict": None,
            "scientific_verdict_emitted": False,
            "inputs": {"q4_results_path": str(q4_path), "q4_results_sha256": q4_hash, "protocol_path": str(protocol_path) if protocol_path else None, "protocol_sha256": protocol_hash, "q4_seed_ordinals": [int(item["source_q4_seed"]) for item in frozen["ordinal_sets"]], "fresh_seeds": list(FRESH_SEEDS), "target_latent_ids": list(frozen["target_latent_ids"]), "q4_frozen_draw_count_per_seed": MATCHED_COUNT},
            "design": {"timing_decision": "L7_ONLY_RESID_PRE8", "l7_intervention": "Q3 frozen-pattern z_star for L7H4 true-source and source-A arms", "baseline": "source-A L7H4 arm at blocks.8.hook_resid_pre", "estimand": "mean(sign*(d_arm-d_A))/mean(sign*(d_full-d_A))", "fresh_selection": "Gate-A then sorted retained pair ids capped at 150; no fresh latent re-selection", "natural_arms": ["full", "target", "complement", "100 matched"], "clamped_arms": ["full", "target"], "reader_clamp": "complete L8H5 hook_z@final output; not value-only", "zero_ablation": False},
            "offline_env": offline,
            "aggregate": aggregate,
            "seed_results": seed_results,
            "matched_rows": matched_rows,
            "claim_boundary": {"supports": ["a fresh-seed causal rescue estimand for the tested L7H4 -> resid_pre8 intervention", "comparison of the fixed target span with fixed matched rank-12 spans", "a dependence control on the complete L8H5 hook_z output at the final query position"], "does_not_support": ["activation reconstruction", "individual-latent causality", "necessity", "complete circuit", "partial or full mediation", "an identified L7H4-to-span-to-reader path", "an all-position-clamp versus parallel-route distinction", "generalization beyond the tested GPT-2-small prompts and intervention"], "reader_interpretation": "natural versus clamped is an exploratory dependence control on complete L8H5 hook_z@final output; it is not a partial-mediation estimand"},
            "runtime": {"wall_clock_seconds": time.perf_counter() - started, "batch_size": batch_size, "fresh_seed_count": len(seed_results), "matched_row_count": len(matched_rows)},
            "git": git,
            "git_final": final_git,
        }
        _atomic_json(output_path, output)
        return output, 0
    except Exception as exc:
        detail = {"schema": SCHEMA, "status": "STOPPED", "verdict": None, "scientific_verdict_emitted": False, "reason": {"gate": getattr(exc, "gate", "unexpected_exception"), "type": type(exc).__name__, "detail": str(exc)}, "runtime": {"wall_clock_seconds": time.perf_counter() - started}}
        try:
            _atomic_json(output_path, detail)
        except Exception:
            pass
        return detail, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fresh-seed L7H4 -> SAE-span -> readout rescue")
    parser.add_argument("--q4-results", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--require-clean-tree", action="store_true", required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, code = run(args)
    return int(code)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
