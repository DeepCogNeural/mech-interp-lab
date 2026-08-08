"""Experiment 05 same-invocation fresh true/source-A selection sweeps.

This runner reuses the shipped Stage-1 patching primitives but measures both
registered 144-head sweeps afresh on one in-memory model and one immutable
clean-base cache.  The shipped Stage-1 JSON is descriptive cross-check evidence
only; it is never used for Gate A, ranking, ties, fallback, or candidate input.

The output is an input to the frozen Stage-2 candidate constructor.  It never
constructs candidate ``C`` and it never adjudicates Q1--Q4.  In particular, a
partial run is represented as ``INCOMPLETE_RUNTIME_CAP`` (or ``STOPPED``) and
cannot be mistaken for a complete selection artifact.

The executable is deliberately offline-first.  It validates every operative
protocol field before loading GPT-2, requires the expected commit, fingerprints
model/config/tokenizer/cache state across both sweeps, and records raw hashes of
its inputs.  Checkpoints are atomic and accepted only after their canonical
self-hash, metadata, rows, and ``L0H0, ... L11H11`` order validate exactly; a
new invocation validates but does not reuse prior rows.

Run from this directory (after the experiment commit is clean):

.. code-block:: console

   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg \
     ../../.venv/bin/python selection_source_a.py \
     --protocol protocol_v1.json --calibration calibration_results.json \
     --stage1 stage1_results.json --source A --all-heads \
     --expected-head-count 144 --expected-git-commit <commit> \
     --require-clean-tree --output selection_source_a.json \
     --pair-output selection_source_a_pairs.json \
     --checkpoint selection_source_a.checkpoint.json

No model, experiment, or test is run on import; work starts only through
``main``/``run``.
"""

from __future__ import annotations

import argparse
import hashlib
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

# Importing the Stage-1 module is the only supported implementation path.  It
# imports Experiment 04's reusable definitions and disables bytecode writes,
# exactly as the shipped Stage-1 program does.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from calibrate import gate_a, make_source_a  # noqa: E402
from pilot import (  # noqa: E402
    CleanPass,
    GateStop,
    build_stimuli,
    directed_indices,
    load_model,
    positions_for_kind,
    require_one_token,
    set_determinism,
)
from stage1 import (  # noqa: E402
    AttentionPatchRunner,
    HEADS_PER_LAYER,
    HOOK_Z,
    LAYER_COUNT,
    PATCH_BATCH_SIZE,
    _source_values,
    assert_hook_z_layout,
    cached_stage1_clean_pass,
    clean_readout_microbatched,
)
from exp05_core import (  # noqa: E402
    MIN_RETAINED_PAIRS,
    PAIR_DIRECTIONS,
    finite_float,
    group_pair_records,
    linear_percentile,
    paired_effect_array,
    pair_sign_consistency,
    validate_protocol,
)


PROTOCOL_SCHEMA = "exp05-number-agreement-protocol"
STAGE1_SCHEMA = "exp05-number-agreement-stage1-v1; frozen Stage 1 only; CPU float32; signed source-to-base head replacements"
CALIBRATION_SCHEMA_PREFIX = "exp05-number-agreement-calibration-v1"
SWEEP_SCHEMA = "exp05.stage_sweep.v1"
SELECTION_SCHEMA = "exp05.selection.v1"
CHECKPOINT_SCHEMA = "exp05.selection_fresh_pair_sweeps.checkpoint.v2"
PAIR_OUTPUT_SCHEMA = "exp05.selection_fresh_pair_sweeps.pairs.v2"
FRESH_SWEEP_SNAPSHOT_STATUS = "FRESH_SAME_INVOCATION_MODEL_STATE_FINGERPRINT_MATCHED"
SNAPSHOT_PROVENANCE_STATUS = "READY"
TRUE_SOURCE_ROW_LABEL = "true_single_flip_fresh_same_invocation"
SOURCE_A_ROW_LABEL = "source_A_same_number_different_noun_fresh_same_invocation"
SEED = 20_260_801
REQUESTED_PAIRS = 240
EXPECTED_HEADS = 144
EXPECTED_DIRECTED_EDITS = 472
DIRECTIONS = ("singular_to_plural", "plural_to_singular")


class SelectionStop(RuntimeError):
    """A fail-closed stop which is safe to publish as a non-candidate artifact."""

    def __init__(self, gate: str, message: str, *, status: str = "STOPPED"):
        super().__init__(message)
        self.gate = gate
        self.status = status


class RuntimeCapStop(SelectionStop):
    """The declared wall-clock budget was reached before all heads completed."""

    def __init__(self, message: str):
        super().__init__("max_wall_seconds", message, status="INCOMPLETE_RUNTIME_CAP")


def _json_bytes(value: Any) -> bytes:
    """Canonical JSON used for input/checkpoint hashes.

    ``exp05_core.canonical_json_bytes`` is intentionally not imported at module
    import time.  The core module is allowed to supply the exact same helper at
    integration, but the selection runner remains inspectable before that file
    exists and does not silently change any experiment math.
    """

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha256_bytes(_json_bytes({key: item for key, item in value.items() if key != field}))


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    payload = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "bytes_sha256": _sha256_bytes(value.numpy().tobytes()),
    }
    return _sha256_bytes(_json_bytes(payload))


def _normalize_json_value(value: Any, *, label: str) -> Any:
    """Normalize config/tokenizer metadata without lossy repr fallbacks."""

    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SelectionStop("normalized_metadata", f"{label} contains a non-finite float.")
        return value
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise SelectionStop("normalized_metadata", f"{label} contains a non-string mapping key {key!r}.")
            normalized[key] = _normalize_json_value(value[key], label=f"{label}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    raise SelectionStop("normalized_metadata", f"{label} contains unsupported value type {type(value)!r}.")


def _normalized_model_config(model: Any) -> dict[str, Any]:
    cfg = model.cfg
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        raw = cfg.to_dict()
    elif hasattr(cfg, "__dict__"):
        raw = vars(cfg)
    else:
        raise SelectionStop("model_config", "Model config exposes neither to_dict() nor __dict__.")
    normalized = _normalize_json_value(raw, label="model.cfg")
    if not isinstance(normalized, dict):
        raise SelectionStop("model_config", "Normalized model config is not an object.")
    return {
        "schema": "exp05.normalized_model_config.v1",
        "sha256": _sha256_bytes(_json_bytes(normalized)),
        "field_count": len(normalized),
        "normalization": "recursive finite JSON; mapping keys lexicographic; no repr fallback",
    }


def _tokenizer_asset_hashes(tokenizer: Any) -> dict[str, Any]:
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:
        raise SelectionStop("tokenizer_assets", f"Tokenizer vocabulary is unavailable: {exc}") from exc
    if not isinstance(vocab, Mapping) or not vocab:
        raise SelectionStop("tokenizer_assets", "Tokenizer get_vocab() must return a non-empty mapping.")
    normalized_vocab: dict[str, int] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str) or type(token_id) is not int:
            raise SelectionStop("tokenizer_assets", "Tokenizer vocabulary must map strings to plain integer ids.")
        normalized_vocab[token] = token_id
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise SelectionStop("tokenizer_assets", "Fast-tokenizer backend serialization is unavailable.")
    try:
        backend_serialized = backend.to_str()
    except Exception as exc:
        raise SelectionStop("tokenizer_assets", f"Could not serialize tokenizer backend: {exc}") from exc
    if not isinstance(backend_serialized, str) or not backend_serialized:
        raise SelectionStop("tokenizer_assets", "Tokenizer backend serialization is empty.")
    special_tokens = _normalize_json_value(
        getattr(tokenizer, "special_tokens_map", {}), label="tokenizer.special_tokens_map"
    )
    local_files: list[dict[str, str]] = []
    file_names = getattr(tokenizer, "vocab_files_names", {})
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if isinstance(file_names, Mapping):
        for asset_key in sorted(file_names):
            candidate = getattr(tokenizer, str(asset_key), None)
            if candidate is None and isinstance(init_kwargs, Mapping):
                candidate = init_kwargs.get(asset_key)
            if isinstance(candidate, str) and Path(candidate).is_file():
                local_files.append(
                    {"asset": str(asset_key), "path": str(Path(candidate).resolve()), "sha256": _sha256_file(Path(candidate))}
                )
    material = {
        "schema": "exp05.tokenizer_assets.v1",
        "tokenizer_class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": len(normalized_vocab),
        "vocab_sha256": _sha256_bytes(_json_bytes(normalized_vocab)),
        "backend_tokenizer_sha256": _sha256_bytes(backend_serialized.encode("utf-8")),
        "special_tokens_map_sha256": _sha256_bytes(_json_bytes(special_tokens)),
        "local_files": local_files,
    }
    material["aggregate_sha256"] = _self_hash(material, "aggregate_sha256")
    return material


def _model_state_fingerprint(model: Any) -> dict[str, Any]:
    """Hash uncast contiguous state_dict bytes in lexicographic key order."""

    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise SelectionStop("model_state_fingerprint", "model.state_dict() is empty or not a mapping.")
    digest = hashlib.sha256()
    digest.update(b"exp05.model_state_fingerprint.v1\0")
    entries: list[dict[str, Any]] = []
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise SelectionStop("model_state_fingerprint", f"Invalid state_dict entry {key!r}.")
        value = tensor.detach().cpu().contiguous()
        raw_bytes = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        metadata = {"key": key, "dtype": str(value.dtype), "shape": list(value.shape)}
        encoded_metadata = _json_bytes(metadata)
        digest.update(len(encoded_metadata).to_bytes(8, "big"))
        digest.update(encoded_metadata)
        digest.update(len(raw_bytes).to_bytes(8, "big"))
        digest.update(raw_bytes)
        entries.append(
            {
                **metadata,
                "byte_length": len(raw_bytes),
                "bytes_sha256": _sha256_bytes(raw_bytes),
            }
        )
    return {
        "schema": "exp05.model_state_fingerprint.v1",
        "sha256": digest.hexdigest(),
        "key_count": len(entries),
        "entries": entries,
        "scheme": "lexicographic state_dict keys; key/dtype/shape JSON plus uncast contiguous tensor bytes; uint64 length framing",
        "encoding_detail": "canonical JSON metadata; unsigned uint64 big-endian metadata and raw-byte lengths",
    }


def _cache_fingerprint(
    *,
    base_tokens: torch.Tensor,
    base_lengths: torch.Tensor,
    base_final: torch.Tensor,
    clean_base_d: torch.Tensor,
    signs: torch.Tensor,
    true_z: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    material = {
        "base_tokens": _tensor_hash(base_tokens),
        "base_lengths": _tensor_hash(base_lengths),
        "base_final_positions": _tensor_hash(base_final),
        "clean_base_d": _tensor_hash(clean_base_d),
        "sign_alignment": _tensor_hash(signs),
        "true_source_z_by_layer": {str(layer): _tensor_hash(true_z[layer]) for layer in range(LAYER_COUNT)},
    }
    return {
        "schema": "exp05.immutable_clean_base_cache.v1",
        "sha256": _sha256_bytes(_json_bytes(material)),
        "tensor_hashes": material,
    }


def _environment_provenance() -> dict[str, Any]:
    material = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "device": "cpu",
    }
    return {**material, "sha256": _sha256_bytes(_json_bytes(material))}


def _require_fingerprint_match(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    label: str,
    hash_field: str = "sha256",
) -> None:
    before_hash = before.get(hash_field)
    after_hash = after.get(hash_field)
    if not _is_sha256_hex(before_hash) or before_hash != after_hash:
        raise SelectionStop(
            "snapshot_provenance_changed",
            f"{label} changed within the fresh true/source-A selection invocation.",
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value cannot enter selection artifact: {value}")
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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionStop("missing_input", f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionStop("invalid_input_json", f"Could not read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectionStop("invalid_input_schema", f"{label} must be a JSON object: {path}")
    return value


def _resolve_path(value: str | Path, *, base: Path = HERE) -> Path:
    path = Path(value).expanduser()
    absolute = path if path.is_absolute() else base / path
    return absolute.resolve(strict=False)


def _resolve_runtime_path(value: str | Path, *, base: Path = HERE) -> Path:
    path = Path(value).expanduser()
    absolute = path if path.is_absolute() else base / path
    if absolute.is_symlink():
        raise SelectionStop("runtime_paths", f"Runtime artifact leaf must not be a symlink: {absolute}")
    return absolute.resolve(strict=False)


def _git(args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=HERE, check=False, capture_output=True, text=True
        )
    except OSError as exc:
        raise SelectionStop("git_unavailable", f"Cannot inspect git provenance: {exc}") from exc
    if result.returncode != 0:
        raise SelectionStop("git_provenance", f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.rstrip("\n")


def _require_distinct_run_paths(
    *,
    inputs: Sequence[Path],
    output: Path,
    pair_output: Path,
    checkpoint: Path | None,
) -> tuple[Path, ...]:
    run_paths = tuple(path for path in (output, pair_output, checkpoint) if path is not None)
    if len(set(run_paths)) != len(run_paths):
        raise SelectionStop("runtime_paths", "output, pair-output, and checkpoint must be distinct paths.")
    input_set = set(inputs)
    overlap = [path for path in run_paths if path in input_set]
    if overlap:
        raise SelectionStop("runtime_paths", f"Runtime artifacts cannot overwrite frozen inputs: {overlap}")
    for path in run_paths:
        if path.exists() and not path.is_file():
            raise SelectionStop("runtime_paths", f"Existing runtime artifact is not a regular file: {path}")
    return run_paths


def _require_clean_tree_except(run_paths: Sequence[Path]) -> dict[str, Any]:
    """Require a clean code/input tree while permitting exact resumable outputs.

    The permitted paths are outputs of this executable, never scientific inputs.
    Their pre-run hashes are recorded, and any checkpoint is separately validated
    against its canonical self-hash and full metadata before a row is reused.
    """

    repository_root = Path(_git(["rev-parse", "--show-toplevel"])).resolve()
    allowed_relative: dict[str, Path] = {}
    for path in run_paths:
        try:
            relative = path.resolve().relative_to(repository_root).as_posix()
        except ValueError:
            continue
        allowed_relative[relative] = path
    raw_status = _git(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    allowed_observed: list[dict[str, Any]] = []
    unexpected: list[str] = []
    for record in raw_status.split("\0"):
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            unexpected.append(record)
            continue
        status = record[:2]
        relative = record[3:]
        if status not in {"??", " M"} or relative not in allowed_relative:
            unexpected.append(record)
            continue
        artifact = allowed_relative[relative]
        allowed_observed.append(
            {
                "path": str(artifact),
                "repository_relative_path": relative,
                "git_status": status,
                "preexisting_sha256": _sha256_file(artifact) if artifact.is_file() else None,
            }
        )
    if unexpected:
        raise SelectionStop(
            "dirty_tree",
            "--require-clean-tree permits only the exact output/pair-output/checkpoint paths; "
            f"unexpected git status entries: {unexpected[:8]!r}",
        )
    return {
        "dirty": False,
        "git_status": "clean",
        "clean_tree_scope": "repository excluding exact declared runtime artifacts",
        "allowed_runtime_artifacts": allowed_observed,
    }


def _canonical_heads(head_count: int) -> list[dict[str, int]]:
    if head_count != EXPECTED_HEADS:
        raise SelectionStop("head_count", f"Frozen GPT-2-small sweep requires {EXPECTED_HEADS} heads, got {head_count}.")
    return [{"layer": layer, "head": head} for layer in range(LAYER_COUNT) for head in range(HEADS_PER_LAYER)]


def _head_key(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def _lookup_nested(value: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present key from a small provenance spelling set."""

    for key in keys:
        if key in value:
            return value[key]
    return None


def _validate_protocol(protocol: Mapping[str, Any], protocol_path: Path, head_count: int) -> None:
    schema = _lookup_nested(protocol, "schema", "protocol_schema", "id", "protocol")
    if schema != PROTOCOL_SCHEMA or int(protocol.get("version", -1)) != 1:
        raise SelectionStop(
            "protocol_schema",
            f"Expected protocol schema/version {PROTOCOL_SCHEMA!r}/1, got {schema!r}/{protocol.get('version')!r} from {protocol_path}.",
        )
    status = protocol.get("status")
    if status not in {"designed_not_executed", "FROZEN", "COMPLETE"}:
        raise SelectionStop("protocol_status", f"Unexpected protocol status {status!r}; fail closed.")
    head_universe = protocol.get("head_universe", {})
    if not isinstance(head_universe, Mapping):
        raise SelectionStop("protocol_head_universe", "Protocol head_universe must be an object.")
    declared_heads = _lookup_nested(protocol, "head_count", "expected_head_count", "n_heads")
    if declared_heads is None:
        declared_heads = head_universe.get("total_heads")
    if declared_heads is not None and int(declared_heads) != head_count:
        raise SelectionStop("protocol_head_count", f"Protocol declares {declared_heads} heads, expected {head_count}.")
    if int(head_universe.get("layer_count", -1)) != LAYER_COUNT or int(head_universe.get("heads_per_layer", -1)) != HEADS_PER_LAYER:
        raise SelectionStop("protocol_head_universe", "Protocol layer/head dimensions differ from GPT-2-small's 12x12 universe.")
    model = protocol.get("model", {})
    if not isinstance(model, Mapping):
        raise SelectionStop("protocol_model", "Protocol model provenance must be an object.")
    for key, want in {
        "name": "gpt2-small",
        "mechanism_library": "TransformerLens",
        "activation_dtype": "float32",
        "residual_width": 768,
    }.items():
        if model.get(key) != want:
            raise SelectionStop("protocol_model", f"Protocol model.{key}={model.get(key)!r} != frozen {want!r}.")
    seeds = protocol.get("seeds", {})
    if not isinstance(seeds, Mapping):
        raise SelectionStop("protocol_seeds", "Protocol seeds must be an object.")
    declared_seed = _lookup_nested(protocol, "stage1_seed", "seed")
    if declared_seed is None:
        declared_seed = seeds.get("stage1_and_source_a_selection")
    if declared_seed is not None and int(declared_seed) != SEED:
        raise SelectionStop("protocol_seed", f"Protocol declares seed {declared_seed}, expected frozen seed {SEED}.")
    stage1 = protocol.get("stage1", {})
    if not isinstance(stage1, Mapping):
        raise SelectionStop("protocol_stage1", "Protocol stage1 section must be an object.")
    source_a = stage1.get("source_a_selection_supplement", {})
    if not isinstance(source_a, Mapping) or source_a.get("head_count") != EXPECTED_HEADS or source_a.get("same_retained_base_pairs") is not True:
        raise SelectionStop("protocol_source_a", "Protocol does not register the full same-retained-pairs source-A supplement.")
    declared_directions = protocol.get("directions")
    if declared_directions is not None and tuple(declared_directions) != DIRECTIONS:
        raise SelectionStop("protocol_directions", f"Protocol directions are not the frozen order {DIRECTIONS}.")
    declared_heads_map = protocol.get("canonical_heads")
    if declared_heads_map is not None and declared_heads_map != _canonical_heads(head_count):
        raise SelectionStop("protocol_head_order", "Protocol canonical head order differs from L0H0...L11H11.")
    try:
        validate_protocol(protocol)
    except Exception as exc:
        raise SelectionStop("protocol_frozen_fields", f"Operative frozen protocol validation failed: {exc}") from exc


def _hf_hub_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _local_snapshot_revision(repo_dir: str, expected_revision: str, *, label: str) -> dict[str, Any]:
    repository = (_hf_hub_root() / repo_dir).resolve(strict=False)
    raw_ref_path = repository / "refs" / "main"
    if raw_ref_path.is_symlink():
        raise SelectionStop("model_revision", f"Pinned local {label} refs/main must not be a symlink: {raw_ref_path}")
    ref_path = raw_ref_path.resolve(strict=False)
    snapshot_path = (repository / "snapshots" / expected_revision).resolve(strict=False)
    if not ref_path.is_file():
        raise SelectionStop("model_revision", f"Pinned local {label} ref is missing: {ref_path}")
    if not snapshot_path.is_dir():
        raise SelectionStop("model_revision", f"Pinned local {label} snapshot is missing: {snapshot_path}")
    try:
        ref_bytes = ref_path.read_bytes()
        observed_revision = ref_bytes.decode("utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise SelectionStop("model_revision", f"Could not read pinned local {label} refs/main: {exc}") from exc
    if observed_revision != expected_revision:
        raise SelectionStop(
            "model_revision",
            f"Local {label} refs/main is {observed_revision!r}, expected frozen {expected_revision!r}.",
        )
    return {
        "expected_revision": expected_revision,
        "observed_revision": observed_revision,
        "refs_main_path": str(ref_path),
        "refs_main_sha256": _sha256_bytes(ref_bytes),
        "snapshot_path": str(snapshot_path),
        "snapshot_present": True,
        "revision_check": "exact local refs/main and snapshots/<revision> match",
    }


def _validate_model_provenance(model: Any, protocol: Mapping[str, Any]) -> dict[str, Any]:
    # Selection does not use SAE activations, but A7 still requires the pinned
    # local GPT-2 and SAE revisions to be fingerprinted in the same invocation.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise SelectionStop("offline_provenance", f"{key}=1 is required; refusing a network-dependent run.")
    cfg = model.cfg
    expected = {"n_layers": 12, "n_heads": 12, "d_model": 768, "d_vocab": 50_257}
    observed = {name: int(getattr(cfg, name, -1)) for name in expected}
    for name, want in expected.items():
        if observed[name] != want:
            raise SelectionStop("model_architecture", f"GPT-2-small {name}={want} required, got {observed[name]}.")
    model_name = str(getattr(cfg, "model_name", ""))
    if model_name and model_name.lower() not in {"gpt2-small", "gpt2"}:
        raise SelectionStop("model_revision", f"Unexpected TransformerLens model name {model_name!r}.")
    revisions = protocol.get("model", {}).get("expected_local_snapshot_revisions", {})
    if (
        not isinstance(revisions, Mapping)
        or not isinstance(revisions.get("gpt2"), str)
        or not isinstance(revisions.get("sae"), str)
    ):
        raise SelectionStop("model_revision", "Protocol must pin both GPT-2 and SAE local revisions.")
    local_snapshots = {
        "gpt2": _local_snapshot_revision("models--gpt2", str(revisions["gpt2"]), label="GPT-2"),
        "sae": _local_snapshot_revision(
            "models--jbloom--GPT2-Small-SAEs-Reformatted",
            str(revisions["sae"]),
            label="res-jb SAE",
        ),
    }
    revisions_fingerprint = _sha256_bytes(_json_bytes(local_snapshots))
    gpt2_snapshot = local_snapshots["gpt2"]
    return {
        "model_name": model_name or "gpt2-small (architecture-pinned)",
        "architecture": observed,
        "device": "cpu",
        "dtype": "float32",
        "offline": True,
        "snapshot_revision_expected": gpt2_snapshot["expected_revision"],
        "snapshot_revision_observed": gpt2_snapshot["observed_revision"],
        "snapshot_ref_path": gpt2_snapshot["refs_main_path"],
        "snapshot_revision_check": "exact local refs/main match",
        "local_snapshot_revisions": local_snapshots,
        "local_snapshot_revisions_sha256": revisions_fingerprint,
        "local_model_revision": local_snapshots["gpt2"]["observed_revision"],
        "local_sae_revision": local_snapshots["sae"]["observed_revision"],
        "sae_loaded": False,
        "sae_revision_fingerprint_present": True,
        "sae_requirement": "weights not loaded for selection; pinned local revision/ref fingerprint required",
    }


def _validate_calibration(calibration: Mapping[str, Any], path: Path) -> None:
    schema = str(calibration.get("schema", ""))
    if not schema.startswith(CALIBRATION_SCHEMA_PREFIX):
        raise SelectionStop("calibration_schema", f"Unexpected calibration schema {schema!r} in {path}.")
    constants = calibration.get("theta_spec")
    if constants is not None and not isinstance(constants, Mapping):
        raise SelectionStop("calibration_constants", "Calibration theta_spec must be an object when present.")


def _validate_stage1_schema(stage1: Mapping[str, Any], path: Path, head_count: int) -> None:
    if stage1.get("schema") != STAGE1_SCHEMA:
        raise SelectionStop("stage1_schema", f"Stage-1 schema mismatch in {path}: {stage1.get('schema')!r}.")
    if stage1.get("status") != "completed_stage1_only":
        raise SelectionStop("stage1_status", f"Stage-1 is not a completed frozen artifact: {stage1.get('status')!r}.")
    config = stage1.get("configuration", {})
    if config.get("seed") != SEED or config.get("requested_pairs") != REQUESTED_PAIRS:
        raise SelectionStop("stage1_configuration", "Stage-1 seed/requested-pairs do not match the frozen source-A slice.")
    if config.get("patch_microbatch_size") != PATCH_BATCH_SIZE:
        raise SelectionStop("stage1_microbatch", "Stage-1 microbatch size differs from the shipped AttentionPatchRunner.")
    if config.get("z_sweep") != "all 144 heads, hook_z at final position only":
        raise SelectionStop("stage1_sweep", "Stage-1 z sweep declaration is not the required all-head final-position sweep.")
    gate = stage1.get("gate_A") or {}
    if not gate.get("passed"):
        raise SelectionStop("stage1_gate_A", "Stage-1 Gate A was not passed; selection is not admissible.")
    if int(gate.get("retained_pairs", -1)) * 2 != EXPECTED_DIRECTED_EDITS:
        raise SelectionStop("stage1_retention_count", "Stage-1 retained-pair count is not the frozen 472 directed edits.")
    position = stage1.get("position_indexing") or {}
    for key, want in {
        "final_position_formula": "per-sequence lengths - 1",
        "final_positions_hard_coded": False,
        "source_and_base_final_positions_equal_all_directed_edits": True,
        "source_and_base_subject_positions_equal_all_directed_edits": True,
        "directed_edit_count": EXPECTED_DIRECTED_EDITS,
    }.items():
        if position.get(key) != want:
            raise SelectionStop("stage1_positions", f"Stage-1 position convention {key}={position.get(key)!r} != {want!r}.")
    heads = stage1.get("heads")
    if not isinstance(heads, list) or len(heads) != head_count:
        raise SelectionStop("stage1_head_records", f"Stage-1 must contain exactly {head_count} per-head records.")


def _validate_directed_edits(
    stage1: Mapping[str, Any],
    base_indices: torch.Tensor,
    source_indices: torch.Tensor,
    signs: torch.Tensor,
    retained_pairs: Sequence[int],
) -> None:
    edits = stage1.get("directed_edits")
    if not isinstance(edits, Mapping):
        raise SelectionStop("stage1_directed_edits", "Stage-1 directed_edits is missing.")
    expected = {
        "pair_indices": [int(pair) for pair in retained_pairs for _ in (0, 1)],
        "base_item_indices": [int(value) for value in base_indices.tolist()],
        "source_item_indices": [int(value) for value in source_indices.tolist()],
        "sign_alignment": [float(value) for value in signs.tolist()],
        "direction": list(DIRECTIONS) * len(retained_pairs),
    }
    for key, want in expected.items():
        observed = edits.get(key)
        if key == "sign_alignment":
            if (
                not isinstance(observed, list)
                or len(observed) != len(want)
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in observed)
                or any(not math.isfinite(float(value)) for value in observed)
                or any(float(a) != float(b) for a, b in zip(observed, want))
            ):
                raise SelectionStop("retained_index_contract", f"Stage-1 {key} differs from the freshly rebuilt Gate-A indices.")
        elif key in {"pair_indices", "base_item_indices", "source_item_indices"}:
            if (
                not isinstance(observed, list)
                or any(type(value) is not int for value in observed)
                or observed != want
            ):
                raise SelectionStop("retained_index_contract", f"Stage-1 {key} differs from the freshly rebuilt Gate-A indices.")
        elif not isinstance(observed, list) or any(not isinstance(value, str) for value in observed) or observed != want:
            raise SelectionStop("retained_index_contract", f"Stage-1 {key} differs from the freshly rebuilt Gate-A indices.")


def _stage1_true_sweep(stage1: Mapping[str, Any], stage1_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert shipped true-source rows to the scalar core sweep contract.

    This is a cross-check only.  The source-A measurements below are always
    produced by fresh forward passes and never read Stage-1 effect values.
    """

    edits = stage1["directed_edits"]
    pair_ids = [int(value) for value in edits["pair_indices"]]
    directions = list(edits["direction"])
    expected_heads = _canonical_heads(EXPECTED_HEADS)
    rows = sorted(stage1["heads"], key=lambda row: (int(row["layer"]), int(row["head"])))
    if [(int(row.get("layer", -1)), int(row.get("head", -1))) for row in rows] != [
        (item["layer"], item["head"]) for item in expected_heads
    ]:
        raise SelectionStop("stage1_head_order", "Stage-1 per-head records are not in the canonical 144-head set.")
    ranking = stage1.get("ranking", {}).get("signed_descending")
    if not isinstance(ranking, list) or len(ranking) != EXPECTED_HEADS:
        raise SelectionStop("stage1_ranking", "Stage-1 signed ranking is not exhaustive over 144 heads.")
    declared_top = [(int(row["layer"]), int(row["head"])) for row in ranking]
    true_heads: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    for row in rows:
        values = row.get("delta_d_sign_aligned")
        if not isinstance(values, list) or len(values) != len(pair_ids):
            raise SelectionStop("stage1_per_edit_schema", f"Stage-1 row L{row['layer']}H{row['head']} lacks 472 true per-edit values.")
        effects = [float(value) for value in values]
        pair_records = [
            {"pair_id": pair_id, "direction": direction, "effect": effect}
            for pair_id, direction, effect in zip(pair_ids, directions, effects)
        ]
        grouped = group_pair_records(pair_records, directions=PAIR_DIRECTIONS, effect_key="effect")
        pair_values = paired_effect_array(grouped)
        mean = float(pair_values.mean())
        if not math.isclose(mean, float(row["E_delta_d"]), rel_tol=1e-6, abs_tol=1e-7):
            raise SelectionStop("stage1_effect_recompute", f"Stage-1 L{row['layer']}H{row['head']} mean disagrees with E_delta_d.")
        true_row = {
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "E_delta_d": mean,
            "abs_E_delta_d": abs(mean),
            "minimal_pair_both_directions_positive_fraction": pair_sign_consistency(pair_values),
            "pair_records": pair_records,
            "source_record": "stage1_results.json true_single_flip; cross-check only",
        }
        true_row["row_sha256"] = _self_hash(true_row, "row_sha256")
        true_heads.append(true_row)
        detailed.append({"layer": int(row["layer"]), "head": int(row["head"]), "effects": effects})
    canonical_rank = sorted(
        true_heads,
        key=lambda row: (-float(row["E_delta_d"]), int(row["layer"]), int(row["head"])),
    )
    canonical_rank_ids = [(int(row["layer"]), int(row["head"])) for row in canonical_rank]
    if declared_top != canonical_rank_ids:
        raise SelectionStop(
            "stage1_top_ranking",
            "Stage-1 declared signed ranking does not reproduce under the canonical float64 pair reduction.",
        )
    true_sweep = {
        "schema": SWEEP_SCHEMA,
        "status": "COMPLETE",
        "dirty": False,
        "seed": SEED,
        "source": "true",
        "head_count": EXPECTED_HEADS,
        "directions": list(DIRECTIONS),
        "heads": true_heads,
        "signed_rank_order": [
            {"layer": layer, "head": head, "flat_id": layer * HEADS_PER_LAYER + head + 1}
            for layer, head in canonical_rank_ids
        ],
        "stage1_sha256": _sha256_file(stage1_path),
        "model_snapshot_status": "UNVERIFIED_FROM_SHIPPED_STAGE1_ARTIFACT",
    }
    return true_sweep, detailed


def _read_shipped_stage1_crosscheck(stage1_path: Path, head_count: int) -> tuple[dict[str, Any], str | None]:
    """Read shipped Stage 1 as descriptive evidence; never as selection input."""

    base = {
        "role": "descriptive_non_blocking_only",
        "used_for_gate": False,
        "used_for_selection": False,
        "used_for_ties": False,
        "used_as_fallback": False,
        "logical_forward_equivalents": 0,
        "path": str(stage1_path),
        "stage1_sha256": None,
        "snapshot_status": "UNVERIFIED_FROM_SHIPPED_STAGE1_ARTIFACT",
    }
    if not stage1_path.is_file():
        return (
            {
                **base,
                "status": "UNAVAILABLE_NONBLOCKING",
                "unavailable_reason": "MISSING",
                "error": "shipped Stage-1 file is missing",
            },
            None,
        )
    try:
        stage1_hash = _sha256_file(stage1_path)
    except Exception as exc:
        return (
            {
                **base,
                "status": "UNAVAILABLE_NONBLOCKING",
                "unavailable_reason": "READ_OR_HASH_ERROR",
                "stage1_sha256": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            None,
        )
    try:
        stage1 = _read_json(stage1_path, "shipped Stage-1 descriptive crosscheck")
        _validate_stage1_schema(stage1, stage1_path, head_count)
        shipped_sweep, _ = _stage1_true_sweep(stage1, stage1_path)
        top10 = list(shipped_sweep["signed_rank_order"][:10])
        return (
            {
                **base,
                "status": "AVAILABLE",
                "stage1_sha256": stage1_hash,
                "top10_signed_canonical_float64": top10,
                "snapshot_status": "UNVERIFIED_FROM_SHIPPED_STAGE1_ARTIFACT",
            },
            stage1_hash,
        )
    except Exception as exc:
        return (
            {
                **base,
                "status": "UNAVAILABLE_NONBLOCKING",
                "unavailable_reason": "INVALID_ARTIFACT",
                "stage1_sha256": stage1_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            stage1_hash,
        )


def _compare_shipped_and_fresh_top10(
    shipped: Mapping[str, Any], fresh_rank_order: Sequence[Mapping[str, int]]
) -> dict[str, Any]:
    fresh_top10 = [dict(row) for row in fresh_rank_order[:10]]
    shipped_top10 = shipped.get("top10_signed_canonical_float64")
    fresh_membership = sorted(fresh_top10, key=lambda row: (int(row["layer"]), int(row["head"])))
    comparison = {
        **dict(shipped),
        "fresh_top10_signed_canonical_float64": fresh_top10,
        "fresh_top10_order": fresh_top10,
        "fresh_top10_membership": fresh_membership,
        "divergence_is_blocking": False,
        "selection_source": "fresh true sweep only",
    }
    if not isinstance(shipped_top10, list):
        comparison.update(
            {
                "comparison_status": "UNAVAILABLE_NONBLOCKING",
                "membership_exact": None,
                "order_exact": None,
                "overlap_count": None,
                "overlap_heads_in_fresh_order": None,
                "shipped_stage1_top10_membership": None,
                "shipped_stage1_top10_order": None,
                "membership_overlap": None,
                "order_overlap": None,
            }
        )
        return comparison
    shipped_keys = [(row.get("layer"), row.get("head")) for row in shipped_top10 if isinstance(row, Mapping)]
    fresh_keys = [(row["layer"], row["head"]) for row in fresh_top10]
    shipped_set = set(shipped_keys)
    fresh_set = set(fresh_keys)
    order_exact = shipped_keys == fresh_keys
    membership_exact = shipped_set == fresh_set and len(shipped_keys) == len(fresh_keys)
    overlap = [row for row, key in zip(fresh_top10, fresh_keys) if key in shipped_set]
    order_overlap = [
        {"rank": rank, "head": dict(fresh_row)}
        for rank, (fresh_row, fresh_key, shipped_key) in enumerate(
            zip(fresh_top10, fresh_keys, shipped_keys), start=1
        )
        if fresh_key == shipped_key
    ]
    comparison.update(
        {
            "comparison_status": "MATCH" if order_exact else "DIVERGED_NONBLOCKING",
            "membership_exact": membership_exact,
            "order_exact": order_exact,
            "overlap_count": len(overlap),
            "overlap_heads_in_fresh_order": overlap,
            "shipped_stage1_top10_membership": sorted(
                [dict(row) for row in shipped_top10 if isinstance(row, Mapping)],
                key=lambda row: (int(row["layer"]), int(row["head"])),
            ),
            "shipped_stage1_top10_order": [dict(row) for row in shipped_top10 if isinstance(row, Mapping)],
            "membership_overlap": overlap,
            "order_overlap": order_overlap,
        }
    )
    return comparison


def _build_gate_and_indices(model: Any) -> tuple[Any, list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = {text: require_one_token(model.tokenizer, text) for text in (" is", " are")}
    base = build_stimuli(model.tokenizer, REQUESTED_PAIRS, SEED)
    clean_d = clean_readout_microbatched(model, base.tokens, base.lengths, token_ids[" is"], token_ids[" are"])
    compact_width = max(token_ids.values()) + 1
    compact_logits = torch.zeros((base.tokens.shape[0], base.tokens.shape[1], compact_width), dtype=torch.float32)
    rows = torch.arange(base.tokens.shape[0])
    finals = base.lengths - 1
    compact_logits[rows, finals, token_ids[" are"]] = clean_d
    gate_clean = CleanPass(logits=compact_logits, residuals={})
    gate, retained_pairs, _ = gate_a(base, gate_clean, token_ids[" is"], token_ids[" are"])
    if not gate.get("passed"):
        raise SelectionStop("Gate_A_base", f"Fresh base Gate A failed: {gate}")
    if len(retained_pairs) < MIN_RETAINED_PAIRS:
        raise SelectionStop(
            "retained_pairs",
            f"Fresh Gate-A retained only {len(retained_pairs)} pairs; protocol requires at least {MIN_RETAINED_PAIRS}.",
        )
    if any(type(pair_id) is not int or not 0 <= pair_id < REQUESTED_PAIRS for pair_id in retained_pairs):
        raise SelectionStop("retained_pairs", "Fresh Gate-A returned a non-canonical pair id.")
    base_indices, source_indices, signs = directed_indices(REQUESTED_PAIRS, retained_pairs)
    return base, retained_pairs, base_indices, source_indices, signs, clean_d


def _pair_records(
    effects: Sequence[float],
    *,
    pair_ids: Sequence[int],
    directions: Sequence[str],
) -> list[dict[str, Any]]:
    if len(effects) != len(pair_ids) or len(effects) != len(directions):
        raise SelectionStop("pair_record_length", "Per-edit effects and canonical direction arrays differ in length.")
    return [
        {"pair_id": int(pair_id), "direction": str(direction), "effect": float(effect)}
        for pair_id, direction, effect in zip(pair_ids, directions, effects)
    ]


def _fresh_head_row(
    *,
    layer: int,
    head: int,
    source: str,
    patched_d: torch.Tensor,
    clean_base_d: torch.Tensor,
    signs: torch.Tensor,
    pair_ids: Sequence[int],
    directions: Sequence[str],
    base_item_indices: Sequence[int],
) -> dict[str, Any]:
    raw = (patched_d - clean_base_d).detach().float().cpu()
    aligned = (raw * signs).detach().float().cpu()
    effects = [float(value) for value in aligned.tolist()]
    pair_records = _pair_records(effects, pair_ids=pair_ids, directions=directions)
    grouped = group_pair_records(pair_records, directions=PAIR_DIRECTIONS, effect_key="effect")
    pair_values = paired_effect_array(grouped)
    canonical_mean = float(pair_values.mean())
    directed_consistency = sum(effect > 0.0 for effect in effects) / len(effects)
    pair_consistency = pair_sign_consistency(pair_values)
    pair_summary = []
    for offset, pair_id in enumerate(pair_ids[::2]):
        pair_effects = effects[2 * offset : 2 * offset + 2]
        pair_summary.append(
            {
                "pair_id": int(pair_id),
                "directions": list(directions[2 * offset : 2 * offset + 2]),
                "effects": pair_effects,
                "mean_effect": (pair_effects[0] + pair_effects[1]) / 2.0,
                "both_positive": bool(pair_effects[0] > 0.0 and pair_effects[1] > 0.0),
            }
        )
    material = {
        "layer": int(layer),
        "head": int(head),
        "hook": HOOK_Z,
        "write_position": "final",
        "source": source,
        "E_delta_d": canonical_mean,
        "abs_E_delta_d": abs(canonical_mean),
        "directed_sign_consistency_positive_fraction": directed_consistency,
        "minimal_pair_both_directions_positive_fraction": pair_consistency,
        "pair_records": pair_records,
        "pair_summary": pair_summary,
        "delta_d_raw": [float(value) for value in raw.tolist()],
        "delta_d_sign_aligned": effects,
        "d_patched": [float(value) for value in patched_d.detach().float().cpu().tolist()],
        "base_item_indices": [int(value) for value in base_item_indices],
    }
    material["row_sha256"] = _self_hash(material, "row_sha256")
    return material


def _checkpoint_metadata(
    *,
    commit: str,
    protocol_hash: str,
    calibration_hash: str,
    stage1_hash: str | None,
    input_hash: str,
    canonical_heads: Sequence[Mapping[str, int]],
    invocation_id: str,
    model_state_sha256: str,
    normalized_config_sha256: str,
    tokenizer_assets_sha256: str,
    clean_base_cache_sha256: str,
    local_snapshot_revisions_sha256: str,
    environment_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "commit": commit,
        "protocol_sha256": protocol_hash,
        "calibration_sha256": calibration_hash,
        "stage1_sha256": stage1_hash,
        "input_sha256": input_hash,
        "source": "fresh_true_and_source_A",
        "canonical_head_order": [dict(item) for item in canonical_heads],
        "invocation_id": invocation_id,
        "model_state_sha256_before_sweeps": model_state_sha256,
        "normalized_config_sha256": normalized_config_sha256,
        "tokenizer_assets_sha256": tokenizer_assets_sha256,
        "clean_base_cache_sha256": clean_base_cache_sha256,
        "local_snapshot_revisions_sha256": local_snapshot_revisions_sha256,
        "environment_sha256": environment_sha256,
        "resume_policy": "validate_prior_checkpoint_then_restart_both_sweeps_in_new_invocation",
    }


def _finite_list(value: Any, *, label: str, expected_length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise SelectionStop("checkpoint_schema", f"{label} must contain exactly {expected_length} finite values.")
    try:
        return [finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)]
    except Exception as exc:
        raise SelectionStop("checkpoint_schema", f"{label} contains a non-finite or non-numeric value: {exc}") from exc


def _validate_checkpoint_row(
    *,
    key: str,
    row: Any,
    layer: int,
    head: int,
    source: str,
    pair_ids: Sequence[int],
    directions: Sequence[str],
    base_item_indices: Sequence[int],
    signs: Sequence[float],
    clean_base_d: Sequence[float],
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} is not an object.")
    material = dict(row)
    declared_hash = material.get("row_sha256")
    if not _is_sha256_hex(declared_hash) or declared_hash != _self_hash(material, "row_sha256"):
        raise SelectionStop("checkpoint_hash", f"Checkpoint row {key} has a missing or mismatched canonical row hash.")
    if (
        type(material.get("layer")) is not int
        or type(material.get("head")) is not int
        or material.get("layer") != layer
        or material.get("head") != head
    ):
        raise SelectionStop("checkpoint_schema", f"Checkpoint key {key} disagrees with its row layer/head.")
    expected_fixed = {
        "hook": HOOK_Z,
        "write_position": "final",
        "source": source,
    }
    for name, expected in expected_fixed.items():
        if material.get(name) != expected:
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} field {name} differs from {expected!r}.")
    records = material.get("pair_records")
    if not isinstance(records, list) or len(records) != len(pair_ids):
        raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair_records is incomplete.")
    record_effects: list[float] = []
    for index, (record, expected_pair, expected_direction) in enumerate(zip(records, pair_ids, directions)):
        if not isinstance(record, Mapping):
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair record {index} is not an object.")
        if type(record.get("pair_id")) is not int or record.get("pair_id") != expected_pair:
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair id/order differs at directed edit {index}.")
        if record.get("direction") != expected_direction:
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} direction/order differs at directed edit {index}.")
        try:
            record_effects.append(finite_float(record.get("effect"), f"{key}.pair_records[{index}].effect"))
        except Exception as exc:
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} has an invalid effect: {exc}") from exc
    grouped = group_pair_records(records, directions=PAIR_DIRECTIONS, effect_key="effect")
    grouped_ids = tuple(record.pair_id for record in grouped)
    if grouped_ids != tuple(pair_ids[::2]):
        raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} does not preserve the frozen retained pair ids.")
    pair_values = paired_effect_array(grouped)
    canonical_mean = float(pair_values.mean())
    aligned_values = _finite_list(material.get("delta_d_sign_aligned"), label=f"{key}.delta_d_sign_aligned", expected_length=len(pair_ids))
    if aligned_values != record_effects:
        raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair_records and aligned values disagree.")
    raw_values = _finite_list(material.get("delta_d_raw"), label=f"{key}.delta_d_raw", expected_length=len(pair_ids))
    patched_values = _finite_list(material.get("d_patched"), label=f"{key}.d_patched", expected_length=len(pair_ids))
    observed_base_indices = material.get("base_item_indices")
    if (
        not isinstance(observed_base_indices, list)
        or any(type(value) is not int for value in observed_base_indices)
        or observed_base_indices != list(base_item_indices)
    ):
        raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} base item indices differ from the rebuilt input.")
    for index, (raw_value, aligned_value, sign, patched_value, clean_value) in enumerate(
        zip(raw_values, aligned_values, signs, patched_values, clean_base_d)
    ):
        if not math.isclose(raw_value * sign, aligned_value, rel_tol=0.0, abs_tol=1e-7):
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} sign alignment disagrees at directed edit {index}.")
        if not math.isclose(patched_value - clean_value, raw_value, rel_tol=0.0, abs_tol=1e-6):
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} raw delta disagrees with patched-clean at edit {index}.")
    expected_derived = {
        "E_delta_d": canonical_mean,
        "abs_E_delta_d": abs(canonical_mean),
        "directed_sign_consistency_positive_fraction": sum(value > 0.0 for value in aligned_values) / len(aligned_values),
        "minimal_pair_both_directions_positive_fraction": pair_sign_consistency(pair_values),
    }
    for name, expected in expected_derived.items():
        try:
            observed = finite_float(material.get(name), f"{key}.{name}")
        except Exception as exc:
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} has invalid {name}: {exc}") from exc
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=0.0):
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} derived field {name} does not recompute.")
    summaries = material.get("pair_summary")
    if not isinstance(summaries, list) or len(summaries) != len(grouped_ids):
        raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair_summary is incomplete.")
    for index, (summary, pair_id) in enumerate(zip(summaries, grouped_ids)):
        expected_effects = aligned_values[2 * index : 2 * index + 2]
        if (
            not isinstance(summary, Mapping)
            or type(summary.get("pair_id")) is not int
            or summary.get("pair_id") != pair_id
        ):
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair_summary id differs at pair {index}.")
        if summary.get("directions") != list(PAIR_DIRECTIONS) or summary.get("effects") != expected_effects:
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair_summary values differ at pair {index}.")
        expected_pair_mean = (expected_effects[0] + expected_effects[1]) / 2.0
        if not math.isclose(finite_float(summary.get("mean_effect"), "pair mean"), expected_pair_mean, rel_tol=0.0, abs_tol=0.0):
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair mean differs at pair {index}.")
        expected_positive = expected_effects[0] > 0.0 and expected_effects[1] > 0.0
        if type(summary.get("both_positive")) is not bool or summary.get("both_positive") is not expected_positive:
            raise SelectionStop("checkpoint_schema", f"Checkpoint row {key} pair sign marker differs at pair {index}.")
    return material


def _load_checkpoint(
    path: Path,
    metadata: Mapping[str, Any],
    *,
    canonical_heads: Sequence[Mapping[str, int]],
    pair_ids: Sequence[int],
    directions: Sequence[str],
    base_item_indices: Sequence[int],
    signs: Sequence[float],
    clean_base_d: Sequence[float],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, bool]], dict[str, Any]]:
    checkpoint = _read_json(path, "checkpoint")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SelectionStop("checkpoint_schema", f"Checkpoint must declare schema {CHECKPOINT_SCHEMA!r}.")
    declared_hash = checkpoint.get("checkpoint_sha256")
    if not _is_sha256_hex(declared_hash) or declared_hash != _self_hash(checkpoint, "checkpoint_sha256"):
        raise SelectionStop("checkpoint_hash", "Checkpoint canonical self-hash is missing or mismatched.")
    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping) or _json_bytes(checkpoint_metadata) != _json_bytes(dict(metadata)):
        raise SelectionStop("checkpoint_mismatch", "Checkpoint metadata does not match commit/protocol/inputs/head order.")
    status = checkpoint.get("status")
    if status not in {"RUNNING", "INCOMPLETE_RUNTIME_CAP", "COMPLETE"}:
        raise SelectionStop("checkpoint_schema", f"Checkpoint status {status!r} is not resumable.")
    completed_root = checkpoint.get("completed_heads")
    rows_root = checkpoint.get("head_rows")
    if (
        not isinstance(completed_root, Mapping)
        or set(completed_root) != {"true", "source_a"}
        or not isinstance(rows_root, Mapping)
        or set(rows_root) != {"true", "source_a"}
    ):
        raise SelectionStop("checkpoint_schema", "Checkpoint must contain exact true/source_a completion and row maps.")
    expected_heads = {_head_key(int(item["layer"]), int(item["head"])): (int(item["layer"]), int(item["head"])) for item in canonical_heads}
    validated_root: dict[str, dict[str, Any]] = {}
    completed_maps: dict[str, dict[str, bool]] = {}
    source_labels = {"true": TRUE_SOURCE_ROW_LABEL, "source_a": SOURCE_A_ROW_LABEL}
    for sweep_name in ("true", "source_a"):
        completed = completed_root[sweep_name]
        rows = rows_root[sweep_name]
        if (
            not isinstance(completed, list)
            or any(not isinstance(value, str) for value in completed)
            or len(completed) != len(set(completed))
            or not isinstance(rows, Mapping)
        ):
            raise SelectionStop("checkpoint_schema", f"Checkpoint {sweep_name} completion/row schema is invalid.")
        row_map = dict(rows)
        completed_map = {value: True for value in completed}
        if set(row_map) != set(completed_map):
            raise SelectionStop("checkpoint_schema", f"Checkpoint {sweep_name} completed-head and row keys differ.")
        unknown = set(row_map) - set(expected_heads)
        if unknown:
            raise SelectionStop("checkpoint_schema", f"Checkpoint {sweep_name} contains non-canonical heads: {sorted(unknown)}")
        expected_completed_order = [key for key in expected_heads if key in completed_map]
        if completed != expected_completed_order:
            raise SelectionStop("checkpoint_schema", f"Checkpoint {sweep_name} heads are not in canonical order.")
        if status == "COMPLETE" and len(row_map) != EXPECTED_HEADS:
            raise SelectionStop("checkpoint_schema", "A COMPLETE checkpoint must contain 144 true and 144 source-A heads.")
        validated: dict[str, Any] = {}
        for key, row in row_map.items():
            layer, head = expected_heads[key]
            validated[key] = _validate_checkpoint_row(
                key=f"{sweep_name}:{key}",
                row=row,
                layer=layer,
                head=head,
                source=source_labels[sweep_name],
                pair_ids=pair_ids,
                directions=directions,
                base_item_indices=base_item_indices,
                signs=signs,
                clean_base_d=clean_base_d,
            )
        validated_root[sweep_name] = validated
        completed_maps[sweep_name] = completed_map
    return validated_root, completed_maps, checkpoint


def _write_checkpoint(
    path: Path,
    metadata: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    completed: Mapping[str, Iterable[str]],
    *,
    status: str,
) -> None:
    if status not in {"RUNNING", "INCOMPLETE_RUNTIME_CAP", "COMPLETE"}:
        raise SelectionStop("checkpoint_schema", f"Refusing to write unsupported checkpoint status {status!r}.")
    if set(rows) != {"true", "source_a"} or set(completed) != {"true", "source_a"}:
        raise SelectionStop("checkpoint_schema", "Refusing to write checkpoint without exact true/source_a maps.")
    canonical_keys = [_head_key(layer, head) for layer in range(LAYER_COUNT) for head in range(HEADS_PER_LAYER)]
    ordered_completed_root: dict[str, list[str]] = {}
    ordered_rows_root: dict[str, dict[str, Any]] = {}
    for sweep_name in ("true", "source_a"):
        completed_values = list(completed[sweep_name])
        completed_set = set(completed_values)
        if (
            any(not isinstance(value, str) for value in completed_values)
            or len(completed_values) != len(completed_set)
            or completed_set != set(rows[sweep_name])
            or not completed_set.issubset(canonical_keys)
        ):
            raise SelectionStop("checkpoint_schema", f"Refusing malformed {sweep_name} checkpoint rows.")
        ordered_completed = [key for key in canonical_keys if key in completed_set]
        if status == "COMPLETE" and len(ordered_completed) != EXPECTED_HEADS:
            raise SelectionStop("checkpoint_schema", "A COMPLETE checkpoint write requires both 144-head sweeps.")
        ordered_completed_root[sweep_name] = ordered_completed
        ordered_rows_root[sweep_name] = {key: rows[sweep_name][key] for key in ordered_completed}
    material = {
        "schema": CHECKPOINT_SCHEMA,
        "metadata": dict(metadata),
        "status": status,
        "completed_heads": ordered_completed_root,
        "head_rows": ordered_rows_root,
    }
    material["checkpoint_sha256"] = _self_hash(material, "checkpoint_sha256")
    _atomic_write_json(path, material)


def _base_manifest(
    *,
    args: argparse.Namespace,
    commit: str,
    protocol_path: Path,
    calibration_path: Path,
    stage1_path: Path,
    protocol_hash: str,
    calibration_hash: str,
    stage1_hash: str | None,
    canonical_heads: Sequence[Mapping[str, int]],
    shipped_crosscheck: Mapping[str, Any],
    invocation_id: str,
    clean_provenance: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "schema": SELECTION_SCHEMA,
        "status": "RUNNING",
        "snapshot_provenance_status": "PENDING_FRESH_SWEEPS",
        "dirty": False,
        "artifact_kind": "selection_fresh_true_and_source_A",
        "candidate_C": None,
        "source": "fresh_true_and_source_A",
        "source_definition": "fresh true single-flip and same-number different-noun source A in one invocation",
        "seed": SEED,
        "requested_pairs": REQUESTED_PAIRS,
        "head_count": EXPECTED_HEADS,
        "directions": list(DIRECTIONS),
        "canonical_head_order": [dict(item) for item in canonical_heads],
        "true_sweep": {"schema": SWEEP_SCHEMA, "status": "RUNNING", "dirty": False, "seed": SEED, "source": "true", "head_count": EXPECTED_HEADS, "directions": list(DIRECTIONS), "heads": []},
        "source_a_sweep": {"schema": SWEEP_SCHEMA, "status": "RUNNING", "dirty": False, "seed": SEED, "source": "source_a", "head_count": EXPECTED_HEADS, "directions": list(DIRECTIONS), "heads": []},
        "shipped_stage1_crosscheck": dict(shipped_crosscheck),
        "provenance": {
            "commit": commit,
            "invocation_id": invocation_id,
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "calibration": str(calibration_path),
            "calibration_sha256": calibration_hash,
            "stage1": str(stage1_path),
            "stage1_sha256": stage1_hash,
            "require_clean_tree": True,
            "dirty": False,
            "git_status": "clean",
            "clean_tree_scope": clean_provenance.get("clean_tree_scope"),
            "allowed_runtime_artifacts": list(clean_provenance.get("allowed_runtime_artifacts", [])),
            "snapshot_provenance_status": "PENDING_FRESH_SWEEPS",
            "stage1_model_snapshot": {
                "status": "UNVERIFIED_FROM_SHIPPED_STAGE1_ARTIFACT",
                "role": "descriptive_non_blocking_only",
                "reason": "shipped Stage 1 is never used for gate, selection, ties, or fallback",
            },
        },
        "runtime": {
            "started_unix": time.time(),
            "max_wall_seconds": args.max_wall_seconds,
            "microbatch_size": PATCH_BATCH_SIZE,
            "device": "cpu",
            "dtype": "float32",
            "wall_clock_seconds": 0.0,
            "completed_head_count": 0,
            "completed_heads_by_sweep": {"true": 0, "source_a": 0},
            "resume_policy": "validate_prior_checkpoint_then_restart_both_sweeps_in_new_invocation",
        },
        "selection": {
            "source_a_abs_mean_linear_p99": None,
            "source_a_edge_definition": "linear 99th percentile across all 144 absolute per-head E(delta_d)",
            "candidate_pool": "computed later by freeze_candidate.py; this artifact does not construct C",
        },
        "failed_gate": None,
        "error": None,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def _check_runtime_cap(started: float, max_wall_seconds: float | None, *, where: str) -> None:
    if max_wall_seconds is not None and (time.perf_counter() - started) >= max_wall_seconds:
        raise RuntimeCapStop(f"Declared runtime cap {max_wall_seconds:.3f}s reached at {where}.")


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Execute fresh true/source-A sweeps and return ``(manifest, exit_code)``."""

    started = time.perf_counter()
    started_unix_ns = time.time_ns()
    protocol_path = _resolve_path(args.protocol)
    calibration_path = _resolve_path(args.calibration)
    stage1_path = _resolve_path(args.stage1)
    output_path = _resolve_path(args.output)
    pair_output_path = _resolve_path(args.pair_output)
    checkpoint_path = _resolve_path(args.checkpoint) if args.checkpoint else None
    run_paths: tuple[Path, ...] = tuple()
    runtime_paths_validated = False
    manifest: dict[str, Any] | None = None
    checkpoint_metadata: dict[str, Any] | None = None
    checkpoint_rows: dict[str, dict[str, Any]] = {"true": {}, "source_a": {}}
    checkpoint_completed: dict[str, dict[str, bool]] = {"true": {}, "source_a": {}}
    try:
        output_path = _resolve_runtime_path(args.output)
        pair_output_path = _resolve_runtime_path(args.pair_output)
        checkpoint_path = _resolve_runtime_path(args.checkpoint) if args.checkpoint else None
        run_paths = _require_distinct_run_paths(
            inputs=(protocol_path, calibration_path, stage1_path),
            output=output_path,
            pair_output=pair_output_path,
            checkpoint=checkpoint_path,
        )
        runtime_paths_validated = True
        if args.source != "A":
            raise SelectionStop("source", "This executable accepts source A only.")
        if not args.all_heads:
            raise SelectionStop("all_heads_required", "--all-heads is required; partial head sweeps cannot select candidates.")
        if int(args.expected_head_count) != EXPECTED_HEADS:
            raise SelectionStop("head_count", f"--expected-head-count must be exactly {EXPECTED_HEADS}.")
        if checkpoint_path is None:
            raise SelectionStop("checkpoint_required", "A COMPLETE selection run requires an atomic checkpoint path.")
        protocol = _read_json(protocol_path, "protocol")
        calibration = _read_json(calibration_path, "calibration")
        _validate_protocol(protocol, protocol_path, int(args.expected_head_count))
        _validate_calibration(calibration, calibration_path)
        protocol_hash = _sha256_file(protocol_path)
        calibration_hash = _sha256_file(calibration_path)
        shipped_crosscheck, stage1_hash = _read_shipped_stage1_crosscheck(
            stage1_path, int(args.expected_head_count)
        )
        commit = _git(["rev-parse", "HEAD"])
        expected_commit = args.expected_git_commit or protocol.get("git_commit")
        if not expected_commit:
            raise SelectionStop("expected_git_commit_required", "Pass --expected-git-commit; provenance must be explicit.")
        if str(expected_commit) != commit:
            raise SelectionStop("git_revision", f"Expected commit {expected_commit}, running {commit}.")
        if not args.require_clean_tree:
            raise SelectionStop(
                "clean_tree_required",
                "A selection artifact can become COMPLETE only with --require-clean-tree.",
            )
        clean_provenance = _require_clean_tree_except(run_paths)
        canonical_heads = _canonical_heads(int(args.expected_head_count))
        invocation_id = _sha256_bytes(
            _json_bytes({"commit": commit, "pid": os.getpid(), "started_unix_ns": started_unix_ns})
        )
        manifest = _base_manifest(
            args=args,
            commit=commit,
            protocol_path=protocol_path,
            calibration_path=calibration_path,
            stage1_path=stage1_path,
            protocol_hash=protocol_hash,
            calibration_hash=calibration_hash,
            stage1_hash=stage1_hash,
            canonical_heads=canonical_heads,
            shipped_crosscheck=shipped_crosscheck,
            invocation_id=invocation_id,
            clean_provenance=clean_provenance,
            started=started,
        )
        # Invalidate any prior COMPLETE selection before model work begins.  The
        # final COMPLETE publication occurs only after pair/checkpoint artifacts
        # are atomically complete and bound by raw hashes below.
        _atomic_write_json(output_path, manifest)
        _check_runtime_cap(started, args.max_wall_seconds, where="pre-model validation")
        set_determinism(SEED)
        model = load_model()
        model_provenance = _validate_model_provenance(model, protocol)
        manifest["provenance"]["model"] = model_provenance
        local_snapshot_revisions_sha256 = model_provenance["local_snapshot_revisions_sha256"]
        manifest["provenance"]["local_model_revision"] = model_provenance["local_model_revision"]
        manifest["provenance"]["local_sae_revision"] = model_provenance["local_sae_revision"]
        manifest["provenance"]["activation_dtype"] = "float32"
        manifest["provenance"]["local_snapshot_revision_checks"] = {
            "before_sweeps_sha256": local_snapshot_revisions_sha256,
            "after_true_sweep_sha256": None,
            "after_source_a_sweep_sha256": None,
            "all_exact_match": False,
        }
        environment = _environment_provenance()
        config_before = _normalized_model_config(model)
        tokenizer_before = _tokenizer_asset_hashes(model.tokenizer)
        model_state_before = _model_state_fingerprint(model)
        manifest["provenance"]["environment"] = environment
        manifest["provenance"]["runtime_environment_fingerprint"] = environment["sha256"]
        manifest["provenance"]["normalized_model_config"] = config_before
        manifest["provenance"]["tokenizer_assets"] = tokenizer_before
        manifest["provenance"]["normalized_model_config_checks"] = {
            "before_sweeps_sha256": config_before["sha256"],
            "after_true_sweep_sha256": None,
            "after_source_a_sweep_sha256": None,
            "all_exact_match": False,
        }
        manifest["provenance"]["tokenizer_asset_checks"] = {
            "before_sweeps_sha256": tokenizer_before["aggregate_sha256"],
            "after_true_sweep_sha256": None,
            "after_source_a_sweep_sha256": None,
            "all_exact_match": False,
        }
        manifest["provenance"]["model_state_fingerprints"] = {
            "scheme": model_state_before["scheme"],
            "before_sweeps": model_state_before,
            "after_true_sweep": None,
            "after_source_a_sweep": None,
            "all_exact_match": False,
        }
        base, retained_pairs, base_indices, source_indices, signs, clean_d = _build_gate_and_indices(model)
        source_a = make_source_a(model.tokenizer, base, SEED)
        if len(source_a.pair_records) != REQUESTED_PAIRS or source_a.tokens.shape != base.tokens.shape:
            raise SelectionStop("source_a_shape", "Source A did not preserve the frozen 240-pair token layout.")
        source_a_indices = base_indices.clone()
        base_final = positions_for_kind(base, base_indices, "final").squeeze(1)
        true_source_final = positions_for_kind(base, source_indices, "final").squeeze(1)
        source_a_final = positions_for_kind(source_a, source_a_indices, "final").squeeze(1)
        if not torch.equal(base_final, true_source_final):
            raise SelectionStop("true_source_positions", "True-source and base final positions differ within a retained pair.")
        if not torch.equal(base_final, source_a_final):
            raise SelectionStop("source_a_positions", "Source-A and base final positions differ under the frozen per-sequence rule.")
        if not torch.equal(base.lengths[base_indices], source_a.lengths[source_a_indices]):
            raise SelectionStop("source_a_lengths", "Source-A/base lengths differ for a retained directed edit.")
        base_tokens = base.tokens[base_indices]
        source_a_tokens = source_a.tokens[source_a_indices]
        base_lengths = base.lengths[base_indices]
        base_clean, base_z, base_attn_out = cached_stage1_clean_pass(model, base_tokens)
        source_clean, source_z, source_attn_out = cached_stage1_clean_pass(model, source_a_tokens)
        layout = assert_hook_z_layout(model, base_z, base_attn_out)
        source_layout = assert_hook_z_layout(model, source_z, source_attn_out)
        if layout["hook_z_shapes_by_layer"] != source_layout["hook_z_shapes_by_layer"]:
            raise SelectionStop("source_a_z_layout", "Source-A hook_z layout differs from base layout.")
        clean_base_d = clean_d[base_indices].detach().float().cpu().clone()
        del base_clean, source_clean, clean_d
        local_indices = torch.arange(base_indices.numel())
        true_source_local_indices = local_indices ^ 1
        if not torch.equal(source_indices, base_indices[true_source_local_indices]):
            raise SelectionStop(
                "true_source_pair_order",
                "Fresh directed indices do not preserve the canonical within-pair opposite-source order.",
            )
        if not torch.equal(local_indices // PATCH_BATCH_SIZE, true_source_local_indices // PATCH_BATCH_SIZE):
            raise SelectionStop("true_source_microbatch_alignment", "A retained true-source pair crosses a patch microbatch.")
        true_source_local_final = base_final[true_source_local_indices]
        pair_ids = [int(pair_id) for pair_id in retained_pairs for _ in DIRECTIONS]
        directions = list(DIRECTIONS) * len(retained_pairs)
        clean_base_cache_before = _cache_fingerprint(
            base_tokens=base_tokens,
            base_lengths=base_lengths,
            base_final=base_final,
            clean_base_d=clean_base_d,
            signs=signs,
            true_z=base_z,
        )
        manifest["provenance"]["immutable_clean_base_cache"] = {
            "before_sweeps": clean_base_cache_before,
            "after_true_sweep_sha256": None,
            "after_source_a_sweep_sha256": None,
            "all_exact_match": False,
            "shared_by": ["fresh_true_sweep", "fresh_source_a_sweep"],
        }
        true_source_cache_sha256 = _sha256_bytes(
            _json_bytes({str(layer): _tensor_hash(base_z[layer]) for layer in range(LAYER_COUNT)})
        )
        source_a_cache_sha256 = _sha256_bytes(
            _json_bytes({str(layer): _tensor_hash(source_z[layer]) for layer in range(LAYER_COUNT)})
        )
        input_descriptor = {
            "seed": SEED,
            "base_tokens": _tensor_hash(base_tokens),
            "source_a_tokens": _tensor_hash(source_a_tokens),
            "base_lengths": _tensor_hash(base_lengths),
            "base_final_positions": _tensor_hash(base_final),
            "true_source_final_positions": _tensor_hash(true_source_final),
            "source_a_final_positions": _tensor_hash(source_a_final),
            "clean_base_d": _tensor_hash(clean_base_d),
            "clean_base_cache_sha256": clean_base_cache_before["sha256"],
            "true_source_cache_sha256": true_source_cache_sha256,
            "source_a_cache_sha256": source_a_cache_sha256,
            "retained_pair_ids": [int(pair) for pair in retained_pairs],
            "base_item_indices": [int(value) for value in base_indices.tolist()],
            "true_source_item_indices": [int(value) for value in source_indices.tolist()],
            "source_a_item_indices": [int(value) for value in source_a_indices.tolist()],
            "sign_alignment": [float(value) for value in signs.tolist()],
            "directions": directions,
            "model_state_sha256_before_sweeps": model_state_before["sha256"],
            "normalized_config_sha256": config_before["sha256"],
            "tokenizer_assets_sha256": tokenizer_before["aggregate_sha256"],
            "local_snapshot_revisions_sha256": local_snapshot_revisions_sha256,
            "environment_sha256": environment["sha256"],
        }
        input_hash = _sha256_bytes(_json_bytes(input_descriptor))
        manifest["provenance"]["input_sha256"] = input_hash
        manifest["input_sha256"] = input_hash
        metadata = _checkpoint_metadata(
            commit=commit,
            protocol_hash=protocol_hash,
            calibration_hash=calibration_hash,
            stage1_hash=stage1_hash,
            input_hash=input_hash,
            canonical_heads=canonical_heads,
            invocation_id=invocation_id,
            model_state_sha256=model_state_before["sha256"],
            normalized_config_sha256=config_before["sha256"],
            tokenizer_assets_sha256=tokenizer_before["aggregate_sha256"],
            clean_base_cache_sha256=clean_base_cache_before["sha256"],
            local_snapshot_revisions_sha256=local_snapshot_revisions_sha256,
            environment_sha256=environment["sha256"],
        )
        checkpoint_metadata = metadata
        head_rows: dict[str, dict[str, Any]] = {"true": {}, "source_a": {}}
        completed: dict[str, dict[str, bool]] = {"true": {}, "source_a": {}}
        if checkpoint_path and checkpoint_path.exists():
            prior_raw = _read_json(checkpoint_path, "prior checkpoint")
            prior_metadata = prior_raw.get("metadata")
            prior_invocation = prior_metadata.get("invocation_id") if isinstance(prior_metadata, Mapping) else None
            if not _is_sha256_hex(prior_invocation):
                raise SelectionStop("checkpoint_schema", "Prior checkpoint lacks a canonical invocation id.")
            expected_prior_metadata = dict(metadata)
            expected_prior_metadata["invocation_id"] = prior_invocation
            prior_rows, prior_completed, prior_checkpoint = _load_checkpoint(
                checkpoint_path,
                expected_prior_metadata,
                canonical_heads=canonical_heads,
                pair_ids=pair_ids,
                directions=directions,
                base_item_indices=[int(value) for value in base_indices.tolist()],
                signs=[float(value) for value in signs.tolist()],
                clean_base_d=[float(value) for value in clean_base_d.tolist()],
            )
            manifest["provenance"]["prior_checkpoint"] = {
                "status": prior_checkpoint["status"],
                "checkpoint_sha256": prior_checkpoint["checkpoint_sha256"],
                "invocation_id": prior_invocation,
                "validated_head_counts": {
                    "true": len(prior_completed["true"]),
                    "source_a": len(prior_completed["source_a"]),
                },
                "rows_loaded_for_validation_only": len(prior_rows["true"]) + len(prior_rows["source_a"]),
                "rows_reused_for_selection": 0,
                "policy": "discard and restart both sweeps to preserve same-invocation provenance",
            }
        checkpoint_rows = head_rows
        checkpoint_completed = completed
        runner = AttentionPatchRunner(model)
        is_id = require_one_token(model.tokenizer, " is")
        are_id = require_one_token(model.tokenizer, " are")

        for item in canonical_heads:
            _check_runtime_cap(started, args.max_wall_seconds, where=f"before fresh true {_head_key(item['layer'], item['head'])}")
            key = _head_key(item["layer"], item["head"])
            layer, head = int(item["layer"]), int(item["head"])
            replacement = _source_values(
                base_z[layer], true_source_local_indices, true_source_local_final, head
            )
            patched_d = runner.run_one(
                hook_kind=HOOK_Z,
                layer=layer,
                head=head,
                base_tokens=base_tokens,
                base_positions=base_final,
                replacement=replacement,
                label="selection_fresh_true_z_final",
                lengths=base_lengths,
                is_id=is_id,
                are_id=are_id,
            )
            head_rows["true"][key] = _fresh_head_row(
                layer=layer,
                head=head,
                source=TRUE_SOURCE_ROW_LABEL,
                patched_d=patched_d,
                clean_base_d=clean_base_d,
                signs=signs,
                pair_ids=pair_ids,
                directions=directions,
                base_item_indices=base_indices.tolist(),
            )
            completed["true"][key] = True
            if checkpoint_path:
                _write_checkpoint(checkpoint_path, metadata, head_rows, completed, status="RUNNING")
            _check_runtime_cap(started, args.max_wall_seconds, where=f"after fresh true {key}")
        if len(completed["true"]) != EXPECTED_HEADS:
            raise SelectionStop("head_completion", f"Only {len(completed['true'])}/{EXPECTED_HEADS} fresh true heads completed.")
        true_rows = [head_rows["true"][_head_key(item["layer"], item["head"])] for item in canonical_heads]
        true_rank_rows = sorted(
            true_rows,
            key=lambda row: (-float(row["E_delta_d"]), int(row["layer"]), int(row["head"])),
        )
        true_rank_order = [
            {"layer": int(row["layer"]), "head": int(row["head"]), "flat_id": int(row["layer"]) * HEADS_PER_LAYER + int(row["head"]) + 1}
            for row in true_rank_rows
        ]
        model_state_after_true = _model_state_fingerprint(model)
        config_after_true = _normalized_model_config(model)
        tokenizer_after_true = _tokenizer_asset_hashes(model.tokenizer)
        model_provenance_after_true = _validate_model_provenance(model, protocol)
        clean_base_cache_after_true = _cache_fingerprint(
            base_tokens=base_tokens,
            base_lengths=base_lengths,
            base_final=base_final,
            clean_base_d=clean_base_d,
            signs=signs,
            true_z=base_z,
        )
        _require_fingerprint_match(model_state_before, model_state_after_true, label="model state after fresh true sweep")
        _require_fingerprint_match(config_before, config_after_true, label="normalized model config after fresh true sweep")
        _require_fingerprint_match(
            tokenizer_before,
            tokenizer_after_true,
            label="tokenizer assets after fresh true sweep",
            hash_field="aggregate_sha256",
        )
        _require_fingerprint_match(clean_base_cache_before, clean_base_cache_after_true, label="clean-base cache after fresh true sweep")
        _require_fingerprint_match(
            model_provenance,
            model_provenance_after_true,
            label="local GPT-2/SAE revisions after fresh true sweep",
            hash_field="local_snapshot_revisions_sha256",
        )
        # Keep the complete canonical fingerprint registry at all three A7
        # checkpoints.  Stage 2 independently validates the schema, hashing
        # scheme, lexicographic state entries, and final digest rather than
        # trusting a digest-only producer summary.
        manifest["provenance"]["model_state_fingerprints"]["after_true_sweep"] = {
            **model_state_after_true,
            "exact_match_before": True,
        }
        manifest["provenance"]["normalized_model_config_checks"]["after_true_sweep_sha256"] = config_after_true["sha256"]
        manifest["provenance"]["tokenizer_asset_checks"]["after_true_sweep_sha256"] = tokenizer_after_true["aggregate_sha256"]
        manifest["provenance"]["immutable_clean_base_cache"]["after_true_sweep_sha256"] = clean_base_cache_after_true["sha256"]
        manifest["provenance"]["local_snapshot_revision_checks"]["after_true_sweep_sha256"] = model_provenance_after_true["local_snapshot_revisions_sha256"]
        manifest["true_sweep"] = {
            "schema": SWEEP_SCHEMA,
            "status": "COMPLETE",
            "dirty": False,
            "seed": SEED,
            "source": "true",
            "head_count": EXPECTED_HEADS,
            "directions": list(DIRECTIONS),
            "heads": true_rows,
            "signed_rank_order": true_rank_order,
            "measurement_origin": "fresh_same_invocation",
            "invocation_id": invocation_id,
            "model_snapshot_status": FRESH_SWEEP_SNAPSHOT_STATUS,
            "model_state_sha256": model_state_before["sha256"],
            "normalized_config_sha256": config_before["sha256"],
            "tokenizer_assets_sha256": tokenizer_before["aggregate_sha256"],
            "clean_base_cache_sha256": clean_base_cache_before["sha256"],
            "local_snapshot_revisions_sha256": local_snapshot_revisions_sha256,
            "source_cache_sha256": true_source_cache_sha256,
            "activation_dtype": "float32",
            "row_hash_scheme": "sha256(canonical JSON row excluding row_sha256)",
        }
        manifest["shipped_stage1_crosscheck"] = _compare_shipped_and_fresh_top10(
            shipped_crosscheck, true_rank_order
        )

        for item in canonical_heads:
            _check_runtime_cap(started, args.max_wall_seconds, where=f"before fresh source A {_head_key(item['layer'], item['head'])}")
            key = _head_key(item["layer"], item["head"])
            layer, head = int(item["layer"]), int(item["head"])
            replacement = _source_values(source_z[layer], local_indices, source_a_final, head)
            patched_d = runner.run_one(
                hook_kind=HOOK_Z,
                layer=layer,
                head=head,
                base_tokens=base_tokens,
                base_positions=base_final,
                replacement=replacement,
                label="selection_fresh_source_A_z_final",
                lengths=base_lengths,
                is_id=is_id,
                are_id=are_id,
            )
            head_rows["source_a"][key] = _fresh_head_row(
                layer=layer,
                head=head,
                source=SOURCE_A_ROW_LABEL,
                patched_d=patched_d,
                clean_base_d=clean_base_d,
                signs=signs,
                pair_ids=pair_ids,
                directions=directions,
                base_item_indices=base_indices.tolist(),
            )
            completed["source_a"][key] = True
            if checkpoint_path:
                _write_checkpoint(checkpoint_path, metadata, head_rows, completed, status="RUNNING")
            _check_runtime_cap(started, args.max_wall_seconds, where=f"after fresh source A {key}")
        if len(completed["source_a"]) != EXPECTED_HEADS:
            raise SelectionStop("head_completion", f"Only {len(completed['source_a'])}/{EXPECTED_HEADS} fresh source-A heads completed.")
        source_a_rows = [head_rows["source_a"][_head_key(item["layer"], item["head"])] for item in canonical_heads]
        edge_values = [abs(float(row["E_delta_d"])) for row in source_a_rows]
        edge = linear_percentile(edge_values, q=99.0)
        manifest["selection"].update(
            {
                "source_a_abs_mean_linear_p99": edge,
                "source_a_abs_mean_values": edge_values,
                "source_a_head_order": [_head_key(item["layer"], item["head"]) for item in canonical_heads],
            }
        )
        manifest["source_a_sweep"] = {
            "schema": SWEEP_SCHEMA,
            "status": "COMPLETE",
            "dirty": False,
            "seed": SEED,
            "source": "source_a",
            "head_count": EXPECTED_HEADS,
            "directions": list(DIRECTIONS),
            "heads": source_a_rows,
            "source_a_abs_mean_linear_p99": edge,
            "measurement_origin": "fresh_same_invocation",
            "invocation_id": invocation_id,
            "model_snapshot_status": FRESH_SWEEP_SNAPSHOT_STATUS,
            "model_state_sha256": model_state_before["sha256"],
            "normalized_config_sha256": config_before["sha256"],
            "tokenizer_assets_sha256": tokenizer_before["aggregate_sha256"],
            "clean_base_cache_sha256": clean_base_cache_before["sha256"],
            "local_snapshot_revisions_sha256": local_snapshot_revisions_sha256,
            "source_cache_sha256": source_a_cache_sha256,
            "activation_dtype": "float32",
            "row_hash_scheme": "sha256(canonical JSON row excluding row_sha256)",
        }
        model_state_after_source_a = _model_state_fingerprint(model)
        config_after_source_a = _normalized_model_config(model)
        tokenizer_after_source_a = _tokenizer_asset_hashes(model.tokenizer)
        model_provenance_after_source_a = _validate_model_provenance(model, protocol)
        clean_base_cache_after_source_a = _cache_fingerprint(
            base_tokens=base_tokens,
            base_lengths=base_lengths,
            base_final=base_final,
            clean_base_d=clean_base_d,
            signs=signs,
            true_z=base_z,
        )
        _require_fingerprint_match(model_state_before, model_state_after_source_a, label="model state after fresh source-A sweep")
        _require_fingerprint_match(config_before, config_after_source_a, label="normalized model config after fresh source-A sweep")
        _require_fingerprint_match(
            tokenizer_before,
            tokenizer_after_source_a,
            label="tokenizer assets after fresh source-A sweep",
            hash_field="aggregate_sha256",
        )
        _require_fingerprint_match(
            clean_base_cache_before,
            clean_base_cache_after_source_a,
            label="clean-base cache after fresh source-A sweep",
        )
        _require_fingerprint_match(
            model_provenance,
            model_provenance_after_source_a,
            label="local GPT-2/SAE revisions after fresh source-A sweep",
            hash_field="local_snapshot_revisions_sha256",
        )
        manifest["provenance"]["model_state_fingerprints"]["after_source_a_sweep"] = {
            **model_state_after_source_a,
            "exact_match_before": True,
        }
        manifest["provenance"]["model_state_fingerprints"]["all_exact_match"] = True
        manifest["provenance"]["normalized_model_config_checks"]["after_source_a_sweep_sha256"] = config_after_source_a["sha256"]
        manifest["provenance"]["normalized_model_config_checks"]["all_exact_match"] = True
        manifest["provenance"]["tokenizer_asset_checks"]["after_source_a_sweep_sha256"] = tokenizer_after_source_a["aggregate_sha256"]
        manifest["provenance"]["tokenizer_asset_checks"]["all_exact_match"] = True
        manifest["provenance"]["immutable_clean_base_cache"]["after_source_a_sweep_sha256"] = clean_base_cache_after_source_a["sha256"]
        manifest["provenance"]["immutable_clean_base_cache"]["all_exact_match"] = True
        manifest["provenance"]["local_snapshot_revision_checks"]["after_source_a_sweep_sha256"] = model_provenance_after_source_a["local_snapshot_revisions_sha256"]
        manifest["provenance"]["local_snapshot_revision_checks"]["all_exact_match"] = True
        manifest["provenance"]["snapshot_provenance_status"] = SNAPSHOT_PROVENANCE_STATUS
        manifest["snapshot_provenance_status"] = SNAPSHOT_PROVENANCE_STATUS
        for sweep_name, source_label in (("true", TRUE_SOURCE_ROW_LABEL), ("source_a", SOURCE_A_ROW_LABEL)):
            for item in canonical_heads:
                key = _head_key(item["layer"], item["head"])
                _validate_checkpoint_row(
                    key=f"final:{sweep_name}:{key}",
                    row=head_rows[sweep_name][key],
                    layer=int(item["layer"]),
                    head=int(item["head"]),
                    source=source_label,
                    pair_ids=pair_ids,
                    directions=directions,
                    base_item_indices=[int(value) for value in base_indices.tolist()],
                    signs=[float(value) for value in signs.tolist()],
                    clean_base_d=[float(value) for value in clean_base_d.tolist()],
                )
        manifest["retained_pairs"] = [int(pair) for pair in retained_pairs]
        manifest["retained_directed_edits"] = {
            "count": int(base_indices.numel()),
            "pair_ids": pair_ids,
            "directions": directions,
            "base_item_indices": [int(value) for value in base_indices.tolist()],
            "true_source_item_indices": [int(value) for value in source_indices.tolist()],
            "source_a_item_indices": [int(value) for value in source_a_indices.tolist()],
            "sign_alignment": [float(value) for value in signs.tolist()],
            "base_final_positions": [int(value) for value in base_final.tolist()],
            "true_source_final_positions": [int(value) for value in true_source_final.tolist()],
            "source_a_final_positions": [int(value) for value in source_a_final.tolist()],
        }
        manifest["correctness"] = {
            "snapshot_provenance_status": SNAPSHOT_PROVENANCE_STATUS,
            "fresh_true_and_source_a_same_invocation": True,
            "same_in_memory_model": True,
            "model_state_before_after_both_sweeps_exact": True,
            "normalized_config_unchanged": True,
            "tokenizer_assets_unchanged": True,
            "pinned_gpt2_and_sae_revisions_unchanged": True,
            "sae_loaded": False,
            "sae_revision_fingerprint_present": True,
            "immutable_clean_base_cache_shared_and_unchanged": True,
            "shipped_stage1_crosscheck_nonblocking": manifest["shipped_stage1_crosscheck"],
            "hook_z_layout": layout,
            "source_a_layout": source_layout,
            "source_a_same_number_pair_layout": True,
            "fresh_base_gate_A": True,
            "canonical_float64_pair_reduction": True,
        }
        pair_payload = {
            "schema": PAIR_OUTPUT_SCHEMA,
            "status": "COMPLETE",
            "input_sha256": input_hash,
            "commit": commit,
            "invocation_id": invocation_id,
            "seed": SEED,
            "snapshot_provenance_status": SNAPSHOT_PROVENANCE_STATUS,
            "model_state_sha256": model_state_before["sha256"],
            "normalized_config_sha256": config_before["sha256"],
            "tokenizer_assets_sha256": tokenizer_before["aggregate_sha256"],
            "clean_base_cache_sha256": clean_base_cache_before["sha256"],
            "local_snapshot_revisions_sha256": local_snapshot_revisions_sha256,
            "environment_sha256": environment["sha256"],
            "retained_pairs": [int(pair) for pair in retained_pairs],
            "pair_records": [
                {
                    "pair_id": int(pair_id),
                    "base_pair": base.pair_records[int(pair_id)],
                    "true_source_pair": base.pair_records[int(pair_id)],
                    "source_a_pair": source_a.pair_records[int(pair_id)],
                }
                for pair_id in retained_pairs
            ],
            "true_heads": true_rows,
            "source_a_heads": source_a_rows,
        }
        _atomic_write_json(pair_output_path, pair_payload)
        manifest["provenance"]["pair_output"] = str(pair_output_path)
        manifest["provenance"]["pair_output_sha256"] = _sha256_file(pair_output_path)
        manifest["runtime"]["completed_head_count"] = 2 * EXPECTED_HEADS
        manifest["runtime"]["completed_heads_by_sweep"] = {"true": EXPECTED_HEADS, "source_a": EXPECTED_HEADS}
        manifest["runtime"]["runner_records"] = runner.records
        manifest["runtime"]["logical_patch_count"] = 2 * EXPECTED_HEADS
        manifest["runtime"]["logical_forward_equivalents"] = 291
        manifest["runtime"]["logical_forward_equivalents_definition"] = (
            "1 clean + 1 true cache + 1 source-A cache + 144 fresh true heads + 144 fresh source-A heads"
        )
        manifest["runtime"]["logical_forward_equivalents_breakdown"] = {
            "clean": 1,
            "fresh_true_source_cache": 1,
            "fresh_source_A_cache": 1,
            "fresh_true_source_144_heads": 144,
            "fresh_source_A_144_heads": 144,
            "historical_stage1_crosscheck": 0,
            "total": 291,
        }
        manifest["runtime"]["logical_edited_sequence_count"] = 2 * EXPECTED_HEADS * int(base_indices.numel())
        manifest["runtime"]["logical_forward_calls"] = sum(int(record["forward_calls"]) for record in runner.records)
        manifest["runtime"]["logical_forward_evaluations"] = sum(
            int(record["batch"]) * int(record["forward_calls"]) for record in runner.records
        )
        manifest["runtime"]["wall_clock_seconds"] = time.perf_counter() - started
        manifest["wall_clock_seconds"] = manifest["runtime"]["wall_clock_seconds"]
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, metadata, head_rows, completed, status="COMPLETE")
            manifest["provenance"]["checkpoint"] = str(checkpoint_path)
            manifest["provenance"]["checkpoint_sha256"] = _sha256_file(checkpoint_path)
        completion_commit = _git(["rev-parse", "HEAD"])
        if completion_commit != commit:
            raise SelectionStop(
                "git_revision_changed",
                f"Repository HEAD changed during the selection run: started {commit}, ended {completion_commit}.",
            )
        completion_clean = _require_clean_tree_except(run_paths)
        completion_clean["commit"] = completion_commit
        manifest["provenance"]["completion_clean_tree_check"] = completion_clean
        manifest["status"] = "COMPLETE"
        _atomic_write_json(output_path, manifest)
        return manifest, 0
    except RuntimeCapStop as exc:
        if manifest is None:
            manifest = {
                "schema": SELECTION_SCHEMA,
                "status": exc.status,
                "artifact_kind": "selection_fresh_true_and_source_A",
                "candidate_C": None,
                "source": "fresh_true_and_source_A",
                "failed_gate": exc.gate,
                "error": str(exc),
            }
        else:
            manifest["status"] = exc.status
            manifest["failed_gate"] = exc.gate
            manifest["error"] = str(exc)
            manifest["candidate_C"] = None
            manifest["runtime"]["wall_clock_seconds"] = time.perf_counter() - started
            completed_counts = {
                "true": len(checkpoint_completed["true"]),
                "source_a": len(checkpoint_completed["source_a"]),
            }
            manifest["runtime"]["completed_head_count"] = sum(completed_counts.values())
            manifest["runtime"]["completed_heads_by_sweep"] = completed_counts
            for sweep_name, manifest_key in (("true", "true_sweep"), ("source_a", "source_a_sweep")):
                ordered_partial = [
                    checkpoint_rows[sweep_name][_head_key(layer, head)]
                    for layer in range(LAYER_COUNT)
                    for head in range(HEADS_PER_LAYER)
                    if _head_key(layer, head) in checkpoint_rows[sweep_name]
                ]
                manifest[manifest_key]["heads"] = ordered_partial
                if len(ordered_partial) != EXPECTED_HEADS:
                    manifest[manifest_key]["status"] = exc.status
                manifest[manifest_key]["measurement_origin"] = "fresh_same_invocation_partial_not_selectable"
            manifest["wall_clock_seconds"] = manifest["runtime"]["wall_clock_seconds"]
        if checkpoint_path and checkpoint_metadata is not None:
            _write_checkpoint(
                checkpoint_path,
                checkpoint_metadata,
                checkpoint_rows,
                checkpoint_completed,
                status=exc.status,
            )
            if isinstance(manifest.get("provenance"), Mapping):
                manifest["provenance"]["checkpoint"] = str(checkpoint_path)
                manifest["provenance"]["checkpoint_sha256"] = _sha256_file(checkpoint_path)
        if runtime_paths_validated:
            _atomic_write_json(output_path, manifest)
        return manifest, 2
    except (SelectionStop, GateStop) as exc:
        if manifest is None:
            manifest = {
                "schema": SELECTION_SCHEMA,
                "status": "STOPPED",
                "artifact_kind": "selection_fresh_true_and_source_A",
                "candidate_C": None,
                "source": "fresh_true_and_source_A",
                "failed_gate": getattr(exc, "gate", "environment"),
                "error": str(exc),
            }
        else:
            manifest["status"] = "STOPPED"
            manifest["failed_gate"] = getattr(exc, "gate", "environment")
            manifest["error"] = str(exc)
            manifest["candidate_C"] = None
            manifest["runtime"]["wall_clock_seconds"] = time.perf_counter() - started
            manifest["wall_clock_seconds"] = manifest["runtime"]["wall_clock_seconds"]
        if runtime_paths_validated:
            _atomic_write_json(output_path, manifest)
        return manifest, 1
    except Exception as exc:  # pragma: no cover - fail closed at the executable boundary
        if manifest is None:
            manifest = {
                "schema": SELECTION_SCHEMA,
                "status": "STOPPED",
                "artifact_kind": "selection_fresh_true_and_source_A",
                "candidate_C": None,
                "source": "fresh_true_and_source_A",
                "failed_gate": "implementation_or_environment",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        else:
            manifest["status"] = "STOPPED"
            manifest["failed_gate"] = "implementation_or_environment"
            manifest["error_type"] = type(exc).__name__
            manifest["error"] = str(exc)
            manifest["candidate_C"] = None
            manifest["runtime"]["wall_clock_seconds"] = time.perf_counter() - started
            manifest["wall_clock_seconds"] = manifest["runtime"]["wall_clock_seconds"]
        if runtime_paths_validated:
            _atomic_write_json(output_path, manifest)
        return manifest, 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Experiment 05's same-invocation fresh true/source-A 144-head sweeps.")
    parser.add_argument("--protocol", default=str(HERE / "protocol_v1.json"))
    parser.add_argument("--calibration", default=str(HERE / "calibration_results.json"))
    parser.add_argument("--stage1", default=str(HERE / "stage1_results.json"))
    parser.add_argument("--source", choices=("A",), required=True)
    parser.add_argument("--all-heads", action="store_true")
    parser.add_argument("--expected-head-count", type=int, default=EXPECTED_HEADS)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--require-clean-tree", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float)
    parser.add_argument("--output", default=str(HERE / "selection_source_a.json"))
    parser.add_argument("--pair-output", default=str(HERE / "selection_source_a_pairs.json"))
    parser.add_argument("--checkpoint", default=str(HERE / "selection_source_a.checkpoint.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_wall_seconds is not None and args.max_wall_seconds <= 0:
        raise SystemExit("--max-wall-seconds must be positive when supplied")
    _, exit_code = run(args)
    return int(exit_code)


if __name__ == "__main__":  # pragma: no cover - execution is intentionally user-authorized
    raise SystemExit(main())
