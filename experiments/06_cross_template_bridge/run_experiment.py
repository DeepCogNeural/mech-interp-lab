"""Fixed-object bridge on a calibration-exposed, mechanism-held-out template family.

This module is model-free on import. Model, SAE, and tensor-backed Experiment 05
helpers are loaded only inside :func:`run`. The scientific objects are inherited
without reselection from the hash-bound Experiment 05 Q4 result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
EXP05 = HERE.parent / "05_number_agreement_circuit"
SOURCE_WORKTREE = HERE.parents[1]

sys.dont_write_bytecode = True
if str(EXP05) not in sys.path:
    sys.path.insert(0, str(EXP05))

import bridge_rescue as bridge  # noqa: E402


SCHEMA = "exp06-cross-template-bridge-result-v1"
PROTOCOL_SCHEMA = "exp06-cross-template-bridge-protocol-v1"
FRESH_SEEDS = tuple(range(20_260_822, 20_260_830))
REQUESTED_PAIRS = 240
MAX_EVAL_PAIRS = 150
MIN_VALID_SEEDS = 8
MECHANISM_EFFECT_FLOOR = 0.05
L7 = 7
HEAD = 4
L8 = 8
TARGET_COUNT = 12
MATCHED_COUNT = 100
RESIDUAL_WIDTH = 768
NON_FINAL_TOLERANCE = 1e-6
FULL_LOGIT_TOLERANCE = 1e-5
Q4_RAW_SHA256 = "81a917da187a103a1e76d79ce86f672347d01110cd50c4a93a241302237671ac"
T_CRITICAL_DF7 = 2.365
ASSET_CONTRACT: dict[str, Any] = {
    "gpt2": {
        "repo_id": "gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "files": {
            "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
            "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
            "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
            "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
            "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
            "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
        },
    },
    "sae": {
        "repo_id": "jbloom/GPT2-Small-SAEs-Reformatted",
        "revision": "57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9",
        "files": {
            "blocks.8.hook_resid_pre/sae_weights.safetensors": "d46a41aa7a9e0475c135e41dde0cb3d8510528ce3140568a18a0e29d3a624d8b",
            "blocks.8.hook_resid_pre/cfg.json": "f8c690c02ea3e1a29713748dd3922404fe59310cafaab6e376b4526b5f69c12c",
        },
    },
    "runtime_versions": {
        "torch": "2.13.0",
        "transformer-lens": "3.5.1",
        "transformers": "5.14.1",
        "huggingface_hub": "1.23.0",
        "safetensors": "0.8.0",
    },
    "state_fingerprint": {
        "model_scheme": "lexicographic state_dict keys; key/dtype/shape JSON plus uncast contiguous tensor bytes; uint64 length framing",
        "sae_decoder_scheme": "dtype/shape JSON plus uncast contiguous tensor bytes; uint64 length framing",
        "require_before_after_equal": True,
    },
}
SOURCE_A_LEMMA_ORDER = (
    ("cat", "cats"),
    ("dog", "dogs"),
    ("child", "children"),
    ("student", "students"),
    ("teacher", "teachers"),
    ("artist", "artists"),
    ("pilot", "pilots"),
    ("singer", "singers"),
    ("doctor", "doctors"),
    ("farmer", "farmers"),
    ("driver", "drivers"),
    ("actor", "actors"),
    ("chef", "chefs"),
    ("writer", "writers"),
    ("friend", "friends"),
    ("neighbor", "neighbors"),
    ("visitor", "visitors"),
    ("player", "players"),
    ("dancer", "dancers"),
    ("reader", "readers"),
)
FROZEN_PROTOCOL: dict[str, Any] = {
    "schema": PROTOCOL_SCHEMA,
    "status": "FROZEN_NO_RUN",
    "date": "2026-08-13",
    "result_observed": False,
    "contract_id": "exp06-v1-ai-advisor-reviewed-raw-effect-8of8-20260813",
    "advisor_review": {
        "status": "INTERACTIVE_AI_REVIEW_COMPLETE",
        "decision": "SECTION_3_PREREG_FINAL",
        "comparator_decision": "SECTION_3_COMPARATOR: APPROVE_CLARIFICATION",
        "scope": "result-blind protocol and implementation-boundary review by an interactive AI advisor; not external expert validation; final-SHA receipt pending",
        "receipt": None,
    },
    "scientific_object": {
        "model": "gpt2-small",
        "sae": "res-jb layer-8 direct decoder",
        "l7_head": {"layer_zero_indexed": 7, "head_zero_indexed": 4},
        "capture_hook": "blocks.8.hook_resid_pre",
        "intervention_timing": "L7H4 frozen base-pattern subject-value replacement at final query",
        "edited_resid_pre8_positions": ["matrix_subject", "final"],
        "reader_clamp": None,
    },
    "asset_contract": ASSET_CONTRACT,
    "frozen_q4_input": {
        "raw_sha256": Q4_RAW_SHA256,
        "target_decoder_row_count": TARGET_COUNT,
        "target_must_be_constant_across_q4_seeds": True,
        "matched_draws_per_ordinal_seed": MATCHED_COUNT,
        "matched_rank": TARGET_COUNT,
        "matched_target_overlap_allowed": False,
        "binding": "sort Q4 source seeds; zip to sorted Experiment 06 seeds",
        "fresh_reselection_allowed": False,
    },
    "registered_seeds": list(FRESH_SEEDS),
    "stimuli": {
        "family": "source_C_relative_clause_with_adverb",
        "template": "The {SUBJ} that the {ATTRACTOR} often {RELVERB}",
        "public_label": "mechanism-held-out evaluation on a calibration-exposed template family",
        "fully_unseen_claim_allowed": False,
        "generated_pairs_per_seed": REQUESTED_PAIRS,
        "max_evaluated_pairs_per_seed": MAX_EVAL_PAIRS,
        "retained_pair_order": "ascending pair id",
        "directions_per_pair": 2,
        "source_true": "opposite number, same matrix-subject lemma",
        "source_A": "same number as base; fixed cyclic successor of the matrix-subject lemma; same template",
        "source_A_lemma_order": [list(pair) for pair in SOURCE_A_LEMMA_ORDER],
        "source_A_mapping": "cyclic successor with reader mapping to cat; no fixed points; identical for both directions and all seeds",
        "held_constant": ["attractor", "relative_verb", "adverb", "all_non_subject_tokens", "template_positions"],
    },
    "gate_a": {
        "both_members_signed_correct_fraction_at_least": 0.6,
        "retained_pairs_at_least": 140,
        "median_plural_minus_singular_gap_at_least": 1.0,
    },
    "arms": [
        "source_A",
        "true",
        "full_delta",
        "target_projection",
        "complement_projection",
        "100_frozen_matched_projections",
    ],
    "estimands": {
        "readout": "logit(' are') - logit(' is')",
        "direction_sign": "+1 singular-base to plural-true; -1 plural-base to singular-true",
        "D_s": "mean(sign * (d_true - d_A))",
        "T_s": "mean(sign * (d_target - d_A))",
        "M_sj": "mean(sign * (d_matched_j - d_A))",
        "E_s": "second-largest_j(M_sj), the raw-effect edge over the 100 frozen Q4 matched latent sets",
        "A_s": "T_s - E_s",
        "verdict_uses_ratios": False,
        "descriptive_only": [
            "target_over_direct_ratio",
            "matched_over_direct_ratios",
            "complement",
            "target_plus_complement_signed_effect_sum",
            "target_plus_complement_minus_full_signed_effect",
            "target_plus_complement_over_full_ratio",
        ],
    },
    "coverage": {
        "required_gate_a_valid_registered_seeds": 8,
        "outcome_based_seed_selection_allowed": False,
        "scientific_non_estimable_trigger": "any registered seed fails Gate A population sufficiency",
        "execution_guards": [
            "q4_hash_schema_and_status",
            "target_and_matched_binding",
            "projector_rank",
            "token_and_position_identity",
            "finite_statistics",
            "non_final_resid_pre8_delta_tolerance",
            "full_rescue_final_logit_identity",
            "clean_source_provenance",
        ],
    },
    "statistics": {
        "unit": "registered_seed",
        "interval": "two-sided 95% Student-t",
        "degrees_of_freedom": 7,
        "analysis_population": "all eight registered seeds; a Gate-A population failure makes the result non-estimable, while missing execution or guard failure stops without a scientific verdict",
    },
    "decision_rule": {
        "status": "FROZEN",
        "non_estimable": "any of the eight registered seeds fails Gate A population sufficiency",
        "stopped_without_scientific_verdict": "any missing execution, runtime cap, unexpected exception, binding, artifact-rank, token/position, non-final-delta, full-rescue-identity, or nonfinite-statistic guard failure",
        "mechanism_transfer": {
            "all_required": [
                "lower_95pct_t_bound(mean(D_s)) > 0",
                "count_registered_seeds(D_s >= 0.05) >= 6",
            ],
            "absolute_floor": MECHANISM_EFFECT_FLOOR,
            "floor_source": "chosen result-blind at less than half the previous original-template minimum direct handle 0.104946173032125",
        },
        "mechanism_negative": "all eight seeds pass Gate A and mechanism_transfer fails; span not adjudicated",
        "positive": {
            "requires_mechanism_transfer": True,
            "all_required": [
                "lower_95pct_t_bound(mean(T_s)) > 0",
                "lower_95pct_t_bound(mean(A_s)) > 0",
                "count_registered_seeds(T_s > 0 and A_s > 0) >= 6",
            ],
        },
        "span_negative": "mechanism_transfer passes and any positive-span condition fails",
    },
    "artifact_contract": {
        "model_output_path_must_be_outside_source_tree": True,
        "model_output_path_must_not_exist_before_run": True,
        "clean_tree_and_exact_commit_required": True,
        "offline_model_and_sae_loading_required": True,
        "compact_result_includes_all_seed_rows": True,
        "adjudicable_eight_seed_run_requires_exactly_800_matched_rows": True,
        "non_estimable_run_may_have_fewer_matched_rows": True,
        "activation_payloads_allowed": False,
        "directional_verdict_on_incomplete_eight_seed_evidence": False,
    },
    "claim_boundary": {
        "strongest_positive": "fixed-object reappearance under a second prompt/control construction in a mechanism-held-out evaluation on one calibration-exposed relative-clause family",
        "control_construction_disclosure": "the relative-clause family and source-A construction both differ from the Experiment 05 exploratory bridge; this is not a one-factor template-only contrast",
        "comparator_disclosure": "reuses Q4 fixed latent sets and second-largest tail-order statistic, not Q4 normalized R estimand",
        "does_not_support": [
            "fully unseen template transfer",
            "independent external validation",
            "natural or monosemantic latent semantics",
            "individual-latent causality",
            "necessity or sufficiency",
            "mediation",
            "complete circuit",
            "generalisation across models or tasks",
            "isolating template-family change from source-A-control construction",
        ],
    },
}

BridgeStop = bridge.BridgeStop


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _json_object_from_bytes(data: bytes, label: str) -> dict[str, Any]:
    """Parse the same bytes that were hashed, rejecting NaN and infinities."""

    try:
        value = json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BridgeStop("json_input", f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BridgeStop("json_input", f"{label} must be a JSON object")
    return value


def _read_bound_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BridgeStop("json_input", f"cannot read {label}: {exc}") from exc
    return _json_object_from_bytes(data, label), hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_exact_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _stage_frozen_assets() -> tuple[tempfile.TemporaryDirectory[str], dict[str, Path], dict[str, Any]]:
    """Bind exact cached bytes, then stage those bytes for the model loaders.

    The loaders never reopen the mutable Hugging Face cache paths. They consume
    private copies written from the exact buffers whose hashes are recorded.
    """

    from huggingface_hub import hf_hub_download

    versions: dict[str, str] = {}
    for distribution, expected in ASSET_CONTRACT["runtime_versions"].items():
        try:
            observed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise BridgeStop("asset_runtime", f"required distribution {distribution!r} is unavailable") from exc
        if observed != expected:
            raise BridgeStop(
                "asset_runtime",
                f"{distribution} version {observed!r} differs from frozen {expected!r}",
            )
        versions[distribution] = observed

    holder: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="exp06-frozen-assets-")
    root = Path(holder.name)
    paths: dict[str, Path] = {}
    observed_files: dict[str, dict[str, str]] = {}
    try:
        for group in ("gpt2", "sae"):
            contract = ASSET_CONTRACT[group]
            group_hashes: dict[str, str] = {}
            for filename, expected_sha in contract["files"].items():
                try:
                    cache_path = Path(
                        hf_hub_download(
                            repo_id=contract["repo_id"],
                            filename=filename,
                            revision=contract["revision"],
                            local_files_only=True,
                        )
                    )
                    data = cache_path.read_bytes()
                except (OSError, RuntimeError, ValueError) as exc:
                    raise BridgeStop("asset_bytes", f"cannot resolve frozen {group} asset {filename}: {exc}") from exc
                observed_sha = hashlib.sha256(data).hexdigest()
                if observed_sha != expected_sha:
                    raise BridgeStop(
                        "asset_bytes",
                        f"{group} asset {filename} SHA-256 {observed_sha} differs from frozen {expected_sha}",
                    )
                staged = root / group / filename
                _write_exact_bytes(staged, data)
                paths[f"{group}:{filename}"] = staged
                group_hashes[filename] = observed_sha
            observed_files[group] = group_hashes
    except Exception:
        holder.cleanup()
        raise
    receipt = {
        "contract_sha256": _canonical_sha256(ASSET_CONTRACT),
        "repositories": {
            group: {
                "repo_id": ASSET_CONTRACT[group]["repo_id"],
                "revision": ASSET_CONTRACT[group]["revision"],
                "files": observed_files[group],
            }
            for group in ("gpt2", "sae")
        },
        "runtime_versions": versions,
        "loader_boundary": (
            "loaders consumed private copies written from the exact hash-validated cache byte buffers; "
            "TransformerLens config and weights were constructed from the staged local directory "
            "without a model-name lookup"
        ),
    }
    return holder, paths, receipt


def _build_hooked_transformer_from_staged(
    gpt2_dir: Path,
    hf_model: Any,
    tokenizer: Any,
    *,
    torch_module: Any,
    hooked_transformer_cls: Any,
    loading_module: Any,
) -> Any:
    """Convert a staged HF model without consulting a mutable named cache."""

    staged_dir = gpt2_dir.resolve()
    if not staged_dir.is_dir():
        raise BridgeStop("asset_loader", f"staged GPT-2 directory is missing: {staged_dir}")
    hf_config = hf_model.config.to_dict()
    cfg = loading_module.get_pretrained_model_config(
        str(staged_dir),
        hf_cfg=hf_config,
        fold_ln=True,
        device="cpu",
        n_devices=1,
        dtype=torch_module.float32,
        local_files_only=True,
    )
    state_dict = loading_module.get_pretrained_state_dict(
        str(staged_dir),
        cfg,
        hf_model=hf_model,
        dtype=torch_module.float32,
        local_files_only=True,
    )
    model = hooked_transformer_cls(cfg, tokenizer, move_to_device=False)
    model.load_and_process_state_dict(
        state_dict,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    )
    model.move_model_modules_to_device()
    return model


def _load_frozen_model_and_sae(paths: Mapping[str, Path]) -> tuple[Any, Any]:
    """Load the exact staged GPT-2 snapshot and SAE safetensor without network access."""

    import torch
    from safetensors.torch import load_file
    from transformer_lens import HookedTransformer
    from transformer_lens import loading_from_pretrained as loading
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpt2_dir = paths["gpt2:config.json"].parent
    tokenizer = AutoTokenizer.from_pretrained(
        str(gpt2_dir),
        local_files_only=True,
        use_fast=True,
        add_bos_token=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(gpt2_dir),
        local_files_only=True,
        dtype=torch.float32,
    )
    model = _build_hooked_transformer_from_staged(
        gpt2_dir,
        hf_model,
        tokenizer,
        torch_module=torch,
        hooked_transformer_cls=HookedTransformer,
        loading_module=loading,
    )
    model.eval()
    del hf_model

    state = load_file(str(paths["sae:blocks.8.hook_resid_pre/sae_weights.safetensors"]), device="cpu")
    if "W_dec" not in state:
        raise BridgeStop("sae_shape", f"frozen SAE lacks W_dec; found {sorted(state)}")

    class _FrozenSae:
        pass

    sae = _FrozenSae()
    sae.W_dec = state["W_dec"].detach().float().cpu()
    return model, sae


def _model_state_fingerprint(model: Any) -> dict[str, Any]:
    """Hash uncast state_dict bytes with the frozen length-framed scheme."""

    torch = __import__("torch")
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise BridgeStop("model_state_fingerprint", "model.state_dict() is empty or invalid")
    digest = hashlib.sha256()
    digest.update(b"exp06.model_state_fingerprint.v1\0")
    key_count = 0
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise BridgeStop("model_state_fingerprint", f"invalid model state entry {key!r}")
        value = tensor.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        metadata = json.dumps(
            {"key": key, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        key_count += 1
    return {
        "schema": "exp06-model-state-fingerprint-v1",
        "sha256": digest.hexdigest(),
        "key_count": key_count,
        "scheme": ASSET_CONTRACT["state_fingerprint"]["model_scheme"],
    }


def _tensor_fingerprint(tensor: Any) -> dict[str, Any]:
    torch = __import__("torch")
    if not isinstance(tensor, torch.Tensor):
        raise BridgeStop("sae_state_fingerprint", "SAE decoder is not a tensor")
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    metadata = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"exp06.sae_decoder_fingerprint.v1\0")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return {
        "schema": "exp06-sae-decoder-fingerprint-v1",
        "sha256": digest.hexdigest(),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "scheme": ASSET_CONTRACT["state_fingerprint"]["sae_decoder_scheme"],
    }


def resolve_path(value: str | Path, *, base: Path = HERE) -> Path:
    """Resolve explicit portable aliases without falling back silently."""

    raw = str(value).strip()
    if not raw:
        raise BridgeStop("path_alias", "empty path is not allowed")
    if raw.startswith("@exp06/"):
        return (HERE / raw[len("@exp06/") :]).expanduser().resolve()
    if raw.startswith("@exp05/"):
        return (EXP05 / raw[len("@exp05/") :]).expanduser().resolve()
    if raw.startswith("@repo/"):
        return (SOURCE_WORKTREE / raw[len("@repo/") :]).expanduser().resolve()
    if raw.startswith("@"):
        raise BridgeStop("path_alias", f"unknown path alias: {raw.split('/', 1)[0]}")
    candidate = Path(raw).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _require_outside_source_tree(output_path: Path) -> None:
    try:
        output_path.resolve().relative_to(SOURCE_WORKTREE.resolve())
    except ValueError:
        return
    raise BridgeStop("runtime_path_alias", "model output must be outside the immutable source tree")


def _reserve_output(output_path: Path) -> None:
    """Atomically claim one never-before-used run path before any mutable work."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write('{"schema":"exp06-cross-template-bridge-result-v1","status":"RESERVED","scientific_verdict_emitted":false}\n')
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BridgeStop("output_exists", f"refusing to overwrite existing run artifact: {output_path}") from exc
    except OSError as exc:
        raise BridgeStop("output_reservation", f"cannot reserve run artifact {output_path}: {exc}") from exc


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject any byte-level JSON meaning that drifts from the approved contract."""

    actual = dict(protocol)
    if actual != FROZEN_PROTOCOL:
        changed = sorted(
            key
            for key in set(actual).union(FROZEN_PROTOCOL)
            if actual.get(key) != FROZEN_PROTOCOL.get(key)
        )
        raise BridgeStop("protocol_contract", f"canonical protocol drifted in top-level sections: {changed}")


def _load_stack() -> dict[str, Any]:
    """Load tensor-backed dependencies only for an authorized model run."""

    stack = dict(bridge._load_stack())
    from calibrate import NOUNS, build_stimuli_from_rows, make_source_c_relative_clause  # type: ignore

    stack.update(
        {
            "nouns": NOUNS,
            "build_stimuli_from_rows": build_stimuli_from_rows,
            "make_relative_base": make_source_c_relative_clause,
        }
    )
    return stack


def make_relative_source_a(
    tokenizer: Any,
    base: Any,
    *,
    nouns: Sequence[tuple[str, str]],
    build_stimuli_from_rows: Any,
) -> Any:
    """Apply one frozen no-fixed-point lemma map in both directions and all seeds."""

    vetted = tuple((str(singular), str(plural)) for singular, plural in nouns)
    if vetted != SOURCE_A_LEMMA_ORDER:
        raise BridgeStop("source_A_mapping", "runtime noun order differs from the frozen source-A map")
    mapping = {
        pair: SOURCE_A_LEMMA_ORDER[(index + 1) % len(SOURCE_A_LEMMA_ORDER)]
        for index, pair in enumerate(SOURCE_A_LEMMA_ORDER)
    }
    if any(source == target for source, target in mapping.items()):
        raise BridgeStop("source_A_mapping", "frozen source-A lemma map contains a fixed point")
    rows: list[dict[str, Any]] = []
    for record in base.pair_records:
        if record.get("template") != "The {SUBJ} that the {ATTRACTOR} often {RELVERB}":
            raise BridgeStop("stimulus_template", "base row is not the registered relative-clause template")
        original = (str(record["subject_singular"]), str(record["subject_plural"]))
        if original not in mapping:
            raise BridgeStop("source_A_mapping", f"base lemma {original!r} is absent from the frozen map")
        subject_singular, subject_plural = mapping[original]
        attractor = str(record["attractor"])
        relative_verb = str(record["relative_verb"])
        rows.append(
            {
                "source_kind": "exp06_same_template_source_A",
                "template": record["template"],
                "subject_singular": subject_singular,
                "subject_plural": subject_plural,
                "attractor": attractor,
                "relative_verb": relative_verb,
                "subject_token_index": int(record["subject_token_index"]),
                "singular_text": f"The {subject_singular} that the {attractor} often {relative_verb}",
                "plural_text": f"The {subject_plural} that the {attractor} often {relative_verb}",
            }
        )
    return build_stimuli_from_rows(tokenizer, family="exp06_relative_clause_source_A", rows=rows)


def _assert_only_subject_changed(base_tokens: Any, source_tokens: Any, subject_positions: Any, label: str) -> None:
    torch = __import__("torch")
    if tuple(base_tokens.shape) != tuple(source_tokens.shape):
        raise BridgeStop("stimulus_lengths", f"{label} token arrays have different shapes")
    rows = torch.arange(base_tokens.shape[0])
    subject_positions = subject_positions.to(dtype=torch.long)
    mask = torch.ones_like(base_tokens, dtype=torch.bool)
    mask[rows, subject_positions] = False
    if not torch.equal(base_tokens[mask], source_tokens[mask]):
        raise BridgeStop("stimulus_identity", f"{label} changed a non-subject token")
    if bool(torch.any(base_tokens[rows, subject_positions] == source_tokens[rows, subject_positions])):
        raise BridgeStop("stimulus_identity", f"{label} failed to change a subject token")


def _as_float_list(value: Any) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def _signed_effect(signs: Any, arm: Any, baseline: Any) -> float:
    return bridge._mean_signed_delta(_as_float_list(signs), _as_float_list(arm), _as_float_list(baseline))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def _matched_edge_summary(effects: Sequence[float]) -> tuple[float, float]:
    """Return the frozen raw-effect edge and descriptive maximum.

    This pure helper gives the runner's second-largest choice its own
    model-free contract. The public packager deliberately reimplements the
    order statistic rather than importing this helper.
    """

    finite = [float(value) for value in effects]
    if len(finite) != MATCHED_COUNT:
        raise BridgeStop("matched_summary", f"expected exactly {MATCHED_COUNT} matched effects")
    if not all(math.isfinite(value) for value in finite):
        raise BridgeStop("nonfinite_statistic", "matched effects contain a non-finite value")
    ordered = sorted(finite)
    return ordered[-2], ordered[-1]


def _t_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if not finite or not all(math.isfinite(value) for value in finite):
        raise BridgeStop("nonfinite_statistic", "cross-seed values are empty or non-finite")
    n = len(finite)
    if n != len(FRESH_SEEDS):
        raise BridgeStop("technical_coverage", f"registered intervals require all eight seeds, got n={n}")
    degrees = 7
    critical = T_CRITICAL_DF7
    mean = math.fsum(finite) / n
    variance = math.fsum((value - mean) ** 2 for value in finite) / degrees
    standard_error = math.sqrt(variance / n)
    half_width = critical * standard_error
    return {
        "n": n,
        "mean": mean,
        "standard_error": standard_error,
        "t_critical": critical,
        "degrees_of_freedom": degrees,
        "ci95": {"low": mean - half_width, "high": mean + half_width},
    }


def adjudicate(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen staged rule without outcome-conditioned seed selection."""

    observed_seeds = [int(row["seed"]) for row in seed_results]
    valid = [row for row in seed_results if row.get("status") == "ADJUDICABLE"]
    gate_a_invalid = [int(row["seed"]) for row in seed_results if row.get("status") == "GATE_A_POPULATION_INVALID"]
    missing = [seed for seed in FRESH_SEEDS if seed not in set(observed_seeds)]
    complete_binding = len(observed_seeds) == len(FRESH_SEEDS) and sorted(observed_seeds) == list(FRESH_SEEDS)
    known_statuses = {"ADJUDICABLE", "GATE_A_POPULATION_INVALID"}
    if not complete_binding or any(row.get("status") not in known_statuses for row in seed_results):
        raise BridgeStop("seed_execution", "adjudication requires exactly one completed row for every registered seed")
    coverage = {
        "registered_seed_count": len(FRESH_SEEDS),
        "observed_seed_count": len(observed_seeds),
        "gate_a_valid_seed_count": len(valid),
        "gate_a_population_invalid_seeds": gate_a_invalid,
        "missing_registered_seeds": missing,
        "minimum_required": MIN_VALID_SEEDS,
        "passed": len(valid) == len(FRESH_SEEDS),
    }
    if not coverage["passed"]:
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
    summaries = {
        "D_s": _t_summary(direct),
        "T_s": _t_summary(target),
        "A_s": _t_summary(advantage),
    }
    mechanism_floor_count = sum(value >= MECHANISM_EFFECT_FLOOR for value in direct)
    mechanism_conditions = {
        "direct_lower_bound_above_zero": summaries["D_s"]["ci95"]["low"] > 0.0,
        "registered_seed_floor_count": mechanism_floor_count,
        "registered_seed_floor_count_at_least_six": mechanism_floor_count >= 6,
        "absolute_floor": MECHANISM_EFFECT_FLOOR,
    }
    mechanism_passed = bool(all((mechanism_conditions["direct_lower_bound_above_zero"], mechanism_conditions["registered_seed_floor_count_at_least_six"])))
    if not mechanism_passed:
        return {
            "verdict": "MECHANISM_NEGATIVE",
            "coverage": coverage,
            "mechanism_transfer": {"passed": False, "conditions": mechanism_conditions},
            "span_transfer": {"adjudicated": False, "reason": "mechanism transfer gate failed"},
            "statistics": summaries,
        }

    joint_positive_count = sum(
        target_value > 0.0 and advantage_value > 0.0
        for target_value, advantage_value in zip(target, advantage)
    )
    span_conditions = {
        "target_lower_bound_above_zero": summaries["T_s"]["ci95"]["low"] > 0.0,
        "advantage_lower_bound_above_zero": summaries["A_s"]["ci95"]["low"] > 0.0,
        "joint_target_and_advantage_positive_registered_seed_count": joint_positive_count,
        "joint_target_and_advantage_positive_registered_seed_count_at_least_six": joint_positive_count >= 6,
    }
    span_passed = bool(
        all(
            (
                span_conditions["target_lower_bound_above_zero"],
                span_conditions["advantage_lower_bound_above_zero"],
                span_conditions["joint_target_and_advantage_positive_registered_seed_count_at_least_six"],
            )
        )
    )
    return {
        "verdict": "POSITIVE" if span_passed else "SPAN_NEGATIVE",
        "coverage": coverage,
        "mechanism_transfer": {"passed": True, "conditions": mechanism_conditions},
        "span_transfer": {"adjudicated": True, "passed": span_passed, "conditions": span_conditions},
        "statistics": summaries,
    }


def _run_seed(
    model: Any,
    sae: Any,
    stack: Mapping[str, Any],
    *,
    seed: int,
    frozen: Mapping[str, Any],
    started: float,
    cap: float | None,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch = __import__("torch")
    stack["set_determinism"](seed)
    base = stack["make_relative_base"](model.tokenizer, REQUESTED_PAIRS, seed, with_adverb=True)
    is_id = stack["require_one_token"](model.tokenizer, " is")
    are_id = stack["require_one_token"](model.tokenizer, " are")
    clean_d = stack["clean_readout_microbatched"](model, base.tokens, base.lengths, is_id, are_id)
    gate = bridge._gate_a(clean_d)
    if not gate["passed"]:
        return (
            {
                "seed": seed,
                "source_q4_seed": int(frozen["source_q4_seed"]),
                "status": "GATE_A_POPULATION_INVALID",
                "reason": "GATE_A_FAIL",
                "gate_a": gate,
            },
            [],
        )

    eval_pair_ids = [int(item) for item in gate["retained_pair_ids"][:MAX_EVAL_PAIRS]]
    base_indices, source_indices, signs = stack["directed_indices"](REQUESTED_PAIRS, eval_pair_ids)
    base_tokens = base.tokens[base_indices]
    true_tokens = base.tokens[source_indices]
    source_a = make_relative_source_a(
        model.tokenizer,
        base,
        nouns=stack["nouns"],
        build_stimuli_from_rows=stack["build_stimuli_from_rows"],
    )
    source_a_tokens = source_a.tokens[base_indices]
    base_lengths = base.lengths[base_indices]
    if not torch.equal(base_lengths, base.lengths[source_indices]) or not torch.equal(base_lengths, source_a.lengths[base_indices]):
        raise BridgeStop("stimulus_lengths", f"seed {seed} base/true/source-A lengths differ")

    base_final = stack["positions_for_kind"](base, base_indices, "final").squeeze(1)
    base_subject = stack["positions_for_kind"](base, base_indices, "subject").squeeze(1)
    true_subject = stack["positions_for_kind"](base, source_indices, "subject").squeeze(1)
    source_a_subject = stack["positions_for_kind"](source_a, base_indices, "subject").squeeze(1)
    if not torch.equal(base_subject, true_subject) or not torch.equal(base_subject, source_a_subject):
        raise BridgeStop("stimulus_positions", f"seed {seed} matrix-subject positions differ")
    _assert_only_subject_changed(base_tokens, true_tokens, base_subject, "true source")
    _assert_only_subject_changed(base_tokens, source_a_tokens, base_subject, "source A")

    positions = stack["positions_for_kind"](base, base_indices, "both")
    names_base = {
        f"blocks.{L7}.attn.hook_z",
        f"blocks.{L7}.attn.hook_v",
        f"blocks.{L7}.attn.hook_pattern",
    }
    base_cache = bridge._cache_hooks(model, base_tokens, names_base, batch_size=batch_size)
    true_cache = bridge._cache_hooks(model, true_tokens, {f"blocks.{L7}.attn.hook_v"}, batch_size=batch_size)
    source_a_cache = bridge._cache_hooks(model, source_a_tokens, {f"blocks.{L7}.attn.hook_v"}, batch_size=batch_size)
    rows = torch.arange(base_tokens.shape[0])
    base_z = base_cache[f"blocks.{L7}.attn.hook_z"][rows, base_final, HEAD, :]
    pattern = base_cache[f"blocks.{L7}.attn.hook_pattern"][rows, HEAD, base_final, base_subject]
    base_v = base_cache[f"blocks.{L7}.attn.hook_v"][rows, base_subject, HEAD, :]
    true_v = true_cache[f"blocks.{L7}.attn.hook_v"][rows, true_subject, HEAD, :]
    source_a_v = source_a_cache[f"blocks.{L7}.attn.hook_v"][rows, source_a_subject, HEAD, :]
    z_true = base_z + pattern.unsqueeze(-1) * (true_v - base_v)
    z_a = base_z + pattern.unsqueeze(-1) * (source_a_v - base_v)

    arm_args = {
        "patch_hook": stack["patch_hook"],
        "logit_difference": stack["logit_difference"],
        "lengths": base_lengths,
        "is_id": is_id,
        "are_id": are_id,
        "batch_size": batch_size,
        "started": started,
        "cap": cap,
    }
    a_arm = bridge._run_l7_arm(model, base_tokens, base_final, z_a, label=f"exp06_source_A_seed_{seed}", **arm_args)
    true_arm = bridge._run_l7_arm(model, base_tokens, base_final, z_true, label=f"exp06_true_seed_{seed}", **arm_args)
    a_resid = a_arm["resid"]
    true_resid = true_arm["resid"]
    delta = true_resid.detach().double() - a_resid.detach().double()
    selected_delta = delta[rows[:, None], positions]
    selected_a = a_resid[rows[:, None], positions]
    selected_true = true_resid[rows[:, None], positions]
    selected_identity_error = float((selected_a.double() + selected_delta - selected_true.double()).abs().max())
    if not math.isfinite(selected_identity_error):
        raise BridgeStop("timing_identity", f"seed {seed} selected-position identity is non-finite")
    non_final_mask = torch.ones(delta.shape[:2], dtype=torch.bool)
    non_final_mask[rows, base_final] = False
    non_final_error = float(delta.abs()[non_final_mask].max()) if bool(non_final_mask.any()) else 0.0
    if not math.isfinite(non_final_error):
        raise BridgeStop("timing_identity", f"seed {seed} non-final resid_pre8 identity is non-finite")
    if non_final_error > NON_FINAL_TOLERANCE:
        raise BridgeStop("timing_identity", f"seed {seed} non-final resid_pre8 delta is {non_final_error:.6g}")

    target_rows = sae.W_dec[list(frozen["target_latent_ids"])].detach().float().cpu()
    target_basis, target_meta = bridge.decoder_row_projector(target_rows)
    target_delta = bridge.project_float64(selected_delta, target_basis)
    complement_delta = selected_delta - target_delta
    rescue_args = {
        "logit_difference": stack["logit_difference"],
        "lengths": base_lengths,
        "is_id": is_id,
        "are_id": are_id,
        "batch_size": batch_size,
        "started": started,
        "cap": cap,
    }
    full = bridge._run_rescue_natural(model, a_resid, positions, selected_delta, base_final, label=f"exp06_full_seed_{seed}", **rescue_args)
    target = bridge._run_rescue_natural(model, a_resid, positions, target_delta, base_final, label=f"exp06_target_seed_{seed}", **rescue_args)
    complement = bridge._run_rescue_natural(model, a_resid, positions, complement_delta, base_final, label=f"exp06_complement_seed_{seed}", **rescue_args)
    full_logit_error = float((full["d"] - true_arm["d"]).abs().max())
    if not math.isfinite(full_logit_error):
        raise BridgeStop("timing_identity", f"seed {seed} full-rescue logit identity is non-finite")
    if full_logit_error > FULL_LOGIT_TOLERANCE:
        raise BridgeStop("timing_identity", f"seed {seed} full rescue logit error is {full_logit_error:.6g}")

    direct_effect = _signed_effect(signs, true_arm["d"], a_arm["d"])
    full_effect = _signed_effect(signs, full["d"], a_arm["d"])
    target_effect = _signed_effect(signs, target["d"], a_arm["d"])
    complement_effect = _signed_effect(signs, complement["d"], a_arm["d"])
    matched_rows: list[dict[str, Any]] = []
    matched_effects: list[float] = []
    for draw in frozen["matched"]:
        bridge._check_runtime(started, cap, f"before Exp06 matched draw {draw['draw_index']} seed {seed}")
        matched_decoder_rows = sae.W_dec[list(draw["latent_ids"])].detach().float().cpu()
        matched_basis, matched_meta = bridge.decoder_row_projector(matched_decoder_rows)
        matched_delta = bridge.project_float64(selected_delta, matched_basis)
        matched = bridge._run_rescue_natural(
            model,
            a_resid,
            positions,
            matched_delta,
            base_final,
            label=f"exp06_matched_{draw['draw_index']}_seed_{seed}",
            **rescue_args,
        )
        effect = _signed_effect(signs, matched["d"], a_arm["d"])
        matched_effects.append(effect)
        matched_rows.append(
            {
                "seed": seed,
                "source_q4_seed": int(draw["source_q4_seed"]),
                "draw_index": int(draw["draw_index"]),
                "latent_ids": [int(item) for item in draw["latent_ids"]],
                "M_sj": effect,
                "descriptive_ratio": _safe_ratio(effect, direct_effect),
                "projector": {"rank": int(matched_meta["rank"]), "arithmetic": matched_meta["arithmetic"]},
            }
        )
    matched_second_largest, matched_max = _matched_edge_summary(matched_effects)
    target_advantage = target_effect - matched_second_largest
    target_plus_complement = target_effect + complement_effect
    seed_result = {
        "seed": seed,
        "source_q4_seed": int(frozen["source_q4_seed"]),
        "status": "ADJUDICABLE",
        "gate_a": gate,
        "evaluation_pair_ids": eval_pair_ids,
        "evaluation_item_count": 2 * len(eval_pair_ids),
        "target_latent_ids": [int(item) for item in frozen["target_latent_ids"]],
        "target_projector": target_meta,
        "identity_diagnostics": {
            "selected_positions_max_abs": selected_identity_error,
            "non_final_positions_max_abs": non_final_error,
            "non_final_tolerance": NON_FINAL_TOLERANCE,
            "full_vs_true_final_logit_max_abs": full_logit_error,
            "full_vs_true_final_logit_tolerance": FULL_LOGIT_TOLERANCE,
        },
        "estimands": {
            "D_s": direct_effect,
            "full_signed_effect": full_effect,
            "T_s": target_effect,
            "E_s_second_largest_matched": matched_second_largest,
            "M_s_max_descriptive": matched_max,
            "A_s": target_advantage,
            "complement_signed_effect": complement_effect,
            "target_plus_complement_signed_effect_sum": target_plus_complement,
            "target_plus_complement_minus_full_signed_effect": target_plus_complement - full_effect,
            "target_plus_complement_over_full_ratio_descriptive": _safe_ratio(target_plus_complement, full_effect),
            "target_over_direct_ratio_descriptive": _safe_ratio(target_effect, direct_effect),
            "complement_over_direct_ratio_descriptive": _safe_ratio(complement_effect, direct_effect),
        },
    }
    return seed_result, matched_rows


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    output_path = resolve_path(args.output)
    q4_path = resolve_path(args.q4_results)
    protocol_path = resolve_path(args.protocol)
    canonical_protocol_path = (HERE / "protocol_v1.json").resolve()
    if protocol_path != canonical_protocol_path:
        raise BridgeStop("protocol_path", f"protocol must be the tracked canonical file {canonical_protocol_path}")
    _require_outside_source_tree(output_path)
    if output_path in (q4_path, protocol_path):
        raise BridgeStop("runtime_path_alias", "output aliases an immutable input")
    _reserve_output(output_path)
    started = time.perf_counter()
    git: dict[str, Any] | None = None
    known_inputs: dict[str, Any] = {
        "protocol_path": str(protocol_path),
        "q4_results_path": str(q4_path),
        "registered_seeds": list(FRESH_SEEDS),
        "asset_contract_sha256": _canonical_sha256(ASSET_CONTRACT),
    }
    offline: dict[str, str | None] | None = None
    asset_holder: tempfile.TemporaryDirectory[str] | None = None
    asset_provenance: dict[str, Any] | None = None
    seed_results: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    try:
        git = bridge._git_provenance(args.expected_git_commit, bool(args.require_clean_tree))
        protocol, protocol_file_sha = _read_bound_json(protocol_path, "Experiment 06 protocol")
        validate_protocol(protocol)
        protocol_sha = _canonical_sha256(protocol)
        q4, q4_sha = _read_bound_json(q4_path, "Q4 raw result")
        known_inputs.update(
            {
                "protocol_sha256": protocol_sha,
                "protocol_file_sha256": protocol_file_sha,
                "q4_raw_sha256": q4_sha,
            }
        )
        if q4_sha != Q4_RAW_SHA256:
            raise BridgeStop("q4_hash", f"Q4 raw SHA-256 {q4_sha} differs from the frozen input")
        frozen = bridge.parse_q4_frozen_sets(q4)
        if len(frozen["ordinal_sets"]) != len(FRESH_SEEDS):
            raise BridgeStop("q4_ordinal_binding", "Q4 ordinal sets do not cover all registered seeds")
        bridge._atomic_json(
            output_path,
            {
                "schema": SCHEMA,
                "status": "RUNNING",
                "scientific_verdict_emitted": False,
                "inputs": {"protocol_sha256": protocol_sha, "q4_raw_sha256": q4_sha},
                "git": git,
            },
        )
        offline = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        if any(value != "1" for value in offline.values()):
            raise BridgeStop("offline_provenance", "HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
        stack = _load_stack()
        asset_holder, asset_paths, asset_receipt = _stage_frozen_assets()
        model, sae = _load_frozen_model_and_sae(asset_paths)
        if tuple(sae.W_dec.shape) != (24_576, RESIDUAL_WIDTH):
            raise BridgeStop("sae_shape", f"unexpected layer-8 decoder shape {tuple(sae.W_dec.shape)}")
        model_state_before = _model_state_fingerprint(model)
        sae_decoder_before = _tensor_fingerprint(sae.W_dec)
        asset_provenance = {
            **asset_receipt,
            "model_state": {"before": model_state_before},
            "sae_decoder": {"before": sae_decoder_before},
        }
        batch_size = max(1, int(args.batch_size))
        for seed, frozen_row in zip(FRESH_SEEDS, frozen["ordinal_sets"]):
            bridge._check_runtime(started, args.max_wall_seconds, f"before Experiment 06 seed {seed}")
            seed_result, seed_matched = _run_seed(
                model,
                sae,
                stack,
                seed=seed,
                frozen=frozen_row,
                started=started,
                cap=args.max_wall_seconds,
                batch_size=batch_size,
            )
            seed_results.append(seed_result)
            matched_rows.extend(seed_matched)
        adjudication = adjudicate(seed_results)
        if adjudication["verdict"] != "NON_ESTIMABLE" and len(matched_rows) != len(FRESH_SEEDS) * MATCHED_COUNT:
            raise BridgeStop("matched_rows", "an adjudicable run must contain exactly 800 matched rows")
        model_state_after = _model_state_fingerprint(model)
        sae_decoder_after = _tensor_fingerprint(sae.W_dec)
        if model_state_after != model_state_before:
            raise BridgeStop("model_state_fingerprint", "model state changed during Experiment 06")
        if sae_decoder_after != sae_decoder_before:
            raise BridgeStop("sae_state_fingerprint", "SAE decoder changed during Experiment 06")
        asset_provenance["model_state"].update({"after": model_state_after, "exact_match": True})
        asset_provenance["sae_decoder"].update({"after": sae_decoder_after, "exact_match": True})
        final_git = bridge._assert_source_unchanged(git)
        output = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "verdict": adjudication["verdict"],
            "scientific_verdict_emitted": True,
            "inputs": {
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_sha,
                "protocol_file_sha256": protocol_file_sha,
                "q4_results_path": str(q4_path),
                "q4_raw_sha256": q4_sha,
                "asset_contract_sha256": _canonical_sha256(ASSET_CONTRACT),
                "registered_seeds": list(FRESH_SEEDS),
                "q4_seed_ordinals": [int(row["source_q4_seed"]) for row in frozen["ordinal_sets"]],
                "target_latent_ids": [int(item) for item in frozen["target_latent_ids"]],
            },
            "design": {
                "template_family": "source_C_relative_clause_with_adverb",
                "public_label": "mechanism-held-out evaluation on a calibration-exposed template family",
                "l7_intervention": "frozen base-pattern L7H4 subject-value replacement at final query",
                "capture_hook": "blocks.8.hook_resid_pre",
                "source_A": "same template and number, different matrix-subject lemma",
                "source_A_mapping": "fixed cyclic next lemma in the registered 20-lemma order; no fixed points",
                "fresh_selection": "Gate A then first at most 150 retained pair ids; no object reselection",
                "reader_clamp": None,
            },
            "adjudication": adjudication,
            "seed_results": seed_results,
            "matched_rows": matched_rows,
            "claim_boundary": protocol["claim_boundary"],
            "offline_env": offline,
            "asset_provenance": asset_provenance,
            "runtime": {
                "wall_clock_seconds": time.perf_counter() - started,
                "batch_size": batch_size,
                "adjudicable_seed_count": adjudication["coverage"]["gate_a_valid_seed_count"],
                "matched_row_count": len(matched_rows),
            },
            "git": git,
            "git_final": final_git,
        }
        bridge._atomic_json(output_path, output)
        if asset_holder is not None:
            asset_holder.cleanup()
        return output, 0
    except Exception as exc:
        detail = {
            "schema": SCHEMA,
            "status": "STOPPED",
            "verdict": None,
            "scientific_verdict_emitted": False,
            "reason": {
                "gate": getattr(exc, "gate", "unexpected_exception"),
                "type": type(exc).__name__,
                "detail": str(exc),
            },
            "inputs": known_inputs,
            "git": git,
            "offline_env": offline,
            "asset_provenance": asset_provenance,
            "progress": {
                "completed_seed_results": seed_results,
                "completed_seed_count": len(seed_results),
                "completed_matched_row_count": len(matched_rows),
            },
            "runtime": {"wall_clock_seconds": time.perf_counter() - started},
        }
        try:
            bridge._atomic_json(output_path, detail)
        except Exception:
            pass
        if asset_holder is not None:
            try:
                asset_holder.cleanup()
            except Exception:
                pass
        return detail, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-object L7H4/SAE bridge on a relative-clause template")
    parser.add_argument("--q4-results", required=True)
    parser.add_argument("--protocol", required=True)
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
