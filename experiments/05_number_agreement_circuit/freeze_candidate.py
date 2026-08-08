"""Freeze Experiment 05's selection-only candidate pool.

This CLI is model-free.  The upstream selection runner must have produced one
``exp05.selection.v1`` JSON object with ``status=COMPLETE`` and two nested,
scalar ``exp05.stage_sweep.v1`` artifacts (``true_sweep`` and
``source_a_sweep``).  Each sweep must itself say ``status=COMPLETE`` and contain
all 144 canonical heads, each with ``pair_records`` entries of the form
``{pair_id, direction, effect}``.  The shipped ``stage1_results.json`` is not a
selection artifact: it has parallel arrays and no source-A head sweep, so this
program rejects it instead of fabricating the missing evidence.

The protocol is the committed ``protocol_v1.json`` schema
``exp05-number-agreement-protocol``.  Its nested frozen constants are validated
by :func:`exp05_core.construct_candidate`; no primary threshold can be supplied
on this command line.  The output is written atomically and carries both the raw
input hashes and a self-hash over all immutable fields.

Example (after a COMPLETE selection-only sweep exists)::

    python freeze_candidate.py --protocol protocol_v1.json \
      --selection selection_source_a.json --require-status COMPLETE \
      --output candidate.json
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from exp05_core import (
    ArtifactError,
    CandidateResult,
    CoreError,
    DirtyArtifactError,
    FRESH_SWEEP_SNAPSHOT_STATUS,
    HashMismatchError,
    HEAD_COUNT,
    HeadSchemaError,
    IncompleteArtifactError,
    MIN_RETAINED_PAIRS,
    PAIR_DIRECTIONS,
    SELECTION_SCHEMA,
    STAGE_SWEEP_SCHEMA,
    construct_candidate,
    finite_float,
    group_pair_records,
    linear_percentile,
    paired_effect_array,
    pair_sign_consistency,
    sha256_file,
    sha256_bytes,
    sha256_json,
)


CHECKPOINT_SCHEMA = "exp05.selection_fresh_pair_sweeps.checkpoint.v2"
PAIR_OUTPUT_SCHEMA = "exp05.selection_fresh_pair_sweeps.pairs.v2"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} JSON root must be an object")
    return value


def _declared_protocol_hash(selection: Mapping[str, Any]) -> str:
    provenance = selection.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("protocol_sha256") is not None:
        value = provenance.get("protocol_sha256")
    else:
        value = selection.get("protocol_sha256")
    if not _is_sha256_hex(value):
        raise HashMismatchError("selection artifact must declare provenance.protocol_sha256")
    return value


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_clean_and_complete(selection: Mapping[str, Any], require_status: str) -> None:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ArtifactError(f"selection must declare schema {SELECTION_SCHEMA!r}")
    if selection.get("status") != require_status:
        raise IncompleteArtifactError(
            f"selection status must be {require_status!r}; got {selection.get('status')!r}"
        )
    if selection.get("dirty") is not False:
        raise DirtyArtifactError("selection artifact must explicitly declare dirty=false")
    provenance = selection.get("provenance")
    if not isinstance(provenance, Mapping):
        raise DirtyArtifactError("selection provenance must be an object")
    if provenance.get("dirty") is not False:
        raise DirtyArtifactError("selection provenance must explicitly declare dirty=false")
    if provenance.get("git_status") != "clean":
        raise DirtyArtifactError(f"selection provenance git_status is not clean: {provenance.get('git_status')!r}")
    if provenance.get("require_clean_tree") is not True:
        raise DirtyArtifactError("selection provenance must declare require_clean_tree=true")
    if provenance.get("clean_tree_scope") != "repository excluding exact declared runtime artifacts":
        raise DirtyArtifactError("selection provenance clean-tree scope is missing or changed")
    completion_clean = provenance.get("completion_clean_tree_check")
    if (
        not isinstance(completion_clean, Mapping)
        or completion_clean.get("dirty") is not False
        or completion_clean.get("git_status") != "clean"
        or completion_clean.get("clean_tree_scope") != "repository excluding exact declared runtime artifacts"
        or completion_clean.get("commit") != provenance.get("commit")
    ):
        raise DirtyArtifactError("selection lacks an exact clean-tree check at COMPLETE finalization")
    if not _is_git_commit(provenance.get("commit")):
        raise DirtyArtifactError("selection provenance must contain one exact lowercase 40-hex git commit")
    if selection.get("snapshot_provenance_status") != "READY" or provenance.get("snapshot_provenance_status") != "READY":
        raise ArtifactError("selection snapshot provenance is not READY")
    invocation_id = provenance.get("invocation_id")
    if not _is_sha256_hex(invocation_id):
        raise ArtifactError("selection provenance lacks a canonical invocation id")
    fingerprints = provenance.get("model_state_fingerprints")
    before = fingerprints.get("before_sweeps") if isinstance(fingerprints, Mapping) else None
    after_true = fingerprints.get("after_true_sweep") if isinstance(fingerprints, Mapping) else None
    after_a = fingerprints.get("after_source_a_sweep") if isinstance(fingerprints, Mapping) else None
    if (
        not isinstance(before, Mapping)
        or not isinstance(after_true, Mapping)
        or not isinstance(after_a, Mapping)
        or fingerprints.get("all_exact_match") is not True
        or not _is_sha256_hex(before.get("sha256"))
        or after_true.get("sha256") != before.get("sha256")
        or after_a.get("sha256") != before.get("sha256")
        or after_true.get("exact_match_before") is not True
        or after_a.get("exact_match_before") is not True
    ):
        raise ArtifactError("selection model state was not fingerprinted identically before/after both fresh sweeps")
    entries = before.get("entries")
    if (
        before.get("schema") != "exp05.model_state_fingerprint.v1"
        or before.get("scheme") != "lexicographic state_dict keys; key/dtype/shape JSON plus uncast contiguous tensor bytes; uint64 length framing"
        or before.get("encoding_detail") != "canonical JSON metadata; unsigned uint64 big-endian metadata and raw-byte lengths"
        or not isinstance(entries, list)
        or before.get("key_count") != len(entries)
        or not entries
    ):
        raise ArtifactError("selection model-state fingerprint contract is incomplete")
    entry_keys: list[str] = []
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("key"), str)
            or not isinstance(entry.get("dtype"), str)
            or not isinstance(entry.get("shape"), list)
            or any(type(size) is not int or size < 0 for size in entry.get("shape", []))
            or type(entry.get("byte_length")) is not int
            or entry.get("byte_length") < 0
            or not _is_sha256_hex(entry.get("bytes_sha256"))
        ):
            raise ArtifactError("selection model-state fingerprint contains a malformed entry")
        entry_keys.append(entry["key"])
    if entry_keys != sorted(entry_keys) or len(entry_keys) != len(set(entry_keys)):
        raise ArtifactError("selection model-state fingerprint keys are not unique lexicographic state_dict order")
    config = provenance.get("normalized_model_config")
    tokenizer = provenance.get("tokenizer_assets")
    environment = provenance.get("environment")
    clean_cache = provenance.get("immutable_clean_base_cache")
    if not isinstance(config, Mapping) or not _is_sha256_hex(config.get("sha256")):
        raise ArtifactError("selection normalized model-config hash is missing")
    if not isinstance(tokenizer, Mapping) or not _is_sha256_hex(tokenizer.get("aggregate_sha256")):
        raise ArtifactError("selection tokenizer asset aggregate hash is missing")
    config_checks = provenance.get("normalized_model_config_checks")
    tokenizer_checks = provenance.get("tokenizer_asset_checks")
    if (
        not isinstance(config_checks, Mapping)
        or config_checks.get("before_sweeps_sha256") != config.get("sha256")
        or config_checks.get("after_true_sweep_sha256") != config.get("sha256")
        or config_checks.get("after_source_a_sweep_sha256") != config.get("sha256")
        or config_checks.get("all_exact_match") is not True
    ):
        raise ArtifactError("selection normalized model config was not stable across both fresh sweeps")
    if (
        not isinstance(tokenizer_checks, Mapping)
        or tokenizer_checks.get("before_sweeps_sha256") != tokenizer.get("aggregate_sha256")
        or tokenizer_checks.get("after_true_sweep_sha256") != tokenizer.get("aggregate_sha256")
        or tokenizer_checks.get("after_source_a_sweep_sha256") != tokenizer.get("aggregate_sha256")
        or tokenizer_checks.get("all_exact_match") is not True
    ):
        raise ArtifactError("selection tokenizer assets were not stable across both fresh sweeps")
    if not isinstance(environment, Mapping) or not _is_sha256_hex(environment.get("sha256")):
        raise ArtifactError("selection environment hash is missing")
    cache_before = clean_cache.get("before_sweeps") if isinstance(clean_cache, Mapping) else None
    if (
        not isinstance(cache_before, Mapping)
        or not _is_sha256_hex(cache_before.get("sha256"))
        or clean_cache.get("after_true_sweep_sha256") != cache_before.get("sha256")
        or clean_cache.get("after_source_a_sweep_sha256") != cache_before.get("sha256")
        or clean_cache.get("all_exact_match") is not True
    ):
        raise ArtifactError("selection immutable clean-base cache was not identical across both fresh sweeps")
    runtime = selection.get("runtime")
    expected_fe_breakdown = {
        "clean": 1,
        "fresh_true_source_cache": 1,
        "fresh_source_A_cache": 1,
        "fresh_true_source_144_heads": 144,
        "fresh_source_A_144_heads": 144,
        "historical_stage1_crosscheck": 0,
        "total": 291,
    }
    observed_fe_breakdown = runtime.get("logical_forward_equivalents_breakdown") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(runtime, Mapping)
        or type(runtime.get("logical_forward_equivalents")) is not int
        or runtime.get("logical_forward_equivalents") != 291
        or not isinstance(observed_fe_breakdown, Mapping)
        or set(observed_fe_breakdown) != set(expected_fe_breakdown)
        or any(
            type(observed_fe_breakdown.get(name)) is not int
            or observed_fe_breakdown.get(name) != expected
            for name, expected in expected_fe_breakdown.items()
        )
        or runtime.get("logical_forward_equivalents_definition")
        != "1 clean + 1 true cache + 1 source-A cache + 144 fresh true heads + 144 fresh source-A heads"
    ):
        raise ArtifactError("selection logical forward-equivalent accounting is not the exact A7 291 breakdown")
    shipped = selection.get("shipped_stage1_crosscheck")
    shipped_status = shipped.get("status") if isinstance(shipped, Mapping) else None
    shipped_hash = shipped.get("stage1_sha256") if isinstance(shipped, Mapping) else None
    fresh_order = shipped.get("fresh_top10_order") if isinstance(shipped, Mapping) else None
    fresh_membership = shipped.get("fresh_top10_membership") if isinstance(shipped, Mapping) else None
    historical_snapshot = provenance.get("stage1_model_snapshot")
    if (
        not isinstance(shipped, Mapping)
        or shipped.get("role") != "descriptive_non_blocking_only"
        or shipped.get("used_for_selection") is not False
        or shipped.get("used_for_gate") is not False
        or shipped.get("used_for_ties") is not False
        or shipped.get("used_as_fallback") is not False
        or shipped.get("divergence_is_blocking") is not False
        or type(shipped.get("logical_forward_equivalents")) is not int
        or shipped.get("logical_forward_equivalents") != 0
        or shipped.get("snapshot_status") != "UNVERIFIED_FROM_SHIPPED_STAGE1_ARTIFACT"
        or shipped_status not in {"AVAILABLE", "UNAVAILABLE_NONBLOCKING"}
        or not isinstance(shipped.get("path"), str)
        or not Path(shipped["path"]).is_absolute()
        or (shipped_hash is not None and not _is_sha256_hex(shipped_hash))
        or (shipped_status == "AVAILABLE" and not _is_sha256_hex(shipped_hash))
        or not isinstance(fresh_order, list)
        or len(fresh_order) != 10
        or not isinstance(fresh_membership, list)
        or len(fresh_membership) != 10
        or not isinstance(historical_snapshot, Mapping)
        or historical_snapshot.get("status") != "UNVERIFIED_FROM_SHIPPED_STAGE1_ARTIFACT"
        or historical_snapshot.get("role") != "descriptive_non_blocking_only"
    ):
        raise ArtifactError("shipped Stage-1 crosscheck is not explicitly nonblocking and selection-independent")
    if shipped_status == "AVAILABLE":
        if (
            shipped.get("comparison_status") not in {"MATCH", "DIVERGED_NONBLOCKING"}
            or type(shipped.get("membership_exact")) is not bool
            or type(shipped.get("order_exact")) is not bool
            or type(shipped.get("overlap_count")) is not int
            or not 0 <= shipped.get("overlap_count") <= 10
            or not isinstance(shipped.get("shipped_stage1_top10_order"), list)
            or len(shipped["shipped_stage1_top10_order"]) != 10
            or not isinstance(shipped.get("shipped_stage1_top10_membership"), list)
            or len(shipped["shipped_stage1_top10_membership"]) != 10
            or not isinstance(shipped.get("membership_overlap"), list)
            or not isinstance(shipped.get("order_overlap"), list)
        ):
            raise ArtifactError("available shipped Stage-1 crosscheck lacks its recorded top-10 comparison structure")
    elif (
        shipped.get("comparison_status") != "UNAVAILABLE_NONBLOCKING"
        or any(
            shipped.get(name) is not None
            for name in (
                "membership_exact",
                "order_exact",
                "overlap_count",
                "shipped_stage1_top10_order",
                "shipped_stage1_top10_membership",
                "membership_overlap",
                "order_overlap",
            )
        )
    ):
        raise ArtifactError("unavailable shipped Stage-1 crosscheck must not fabricate historical top-10 data")
    if provenance.get("stage1_sha256") != shipped_hash or provenance.get("stage1") != shipped.get("path"):
        raise HashMismatchError("selection provenance does not bind the recorded nonblocking Stage-1 crosscheck")
    if selection.get("manual_override") is True or selection.get("candidate_manual_override") is True:
        raise ArtifactError("manual candidate override is forbidden")
    if selection.get("candidate_C") not in (None, []):
        raise ArtifactError("selection runner must not pre-populate candidate_C; freeze computes it")
    if selection.get("head_count") != 144:
        raise HeadSchemaError("selection artifact must declare head_count=144")
    nested_order = selection.get("canonical_head_order")
    if not isinstance(nested_order, list) or len(nested_order) != HEAD_COUNT:
        raise HeadSchemaError("selection canonical_head_order must contain all 144 heads")
    expected = [(layer, head) for layer in range(12) for head in range(12)]
    observed = [(row.get("layer"), row.get("head")) for row in nested_order if isinstance(row, Mapping)]
    if (
        len(observed) != HEAD_COUNT
        or any(type(layer) is not int or type(head) is not int for layer, head in observed)
        or observed != expected
    ):
        raise HeadSchemaError("selection canonical_head_order must be L0H0 ... L11H11")


def _extract_sweeps(selection: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    true_sweep = selection.get("true_sweep")
    source_a_sweep = selection.get("source_a_sweep")
    if not isinstance(true_sweep, Mapping) or not isinstance(source_a_sweep, Mapping):
        raise ArtifactError(
            "selection must contain separate true_sweep and source_a_sweep objects; "
            "do not pass stage1_results.json or a fabricated source-A placeholder"
        )
    for name, sweep in (("true_sweep", true_sweep), ("source_a_sweep", source_a_sweep)):
        if sweep.get("schema") != STAGE_SWEEP_SCHEMA:
            raise ArtifactError(f"{name} must declare schema {STAGE_SWEEP_SCHEMA!r}")
        if sweep.get("status") != "COMPLETE":
            raise IncompleteArtifactError(f"{name} status must be COMPLETE; got {sweep.get('status')!r}")
        if sweep.get("dirty") is not False:
            raise DirtyArtifactError(f"{name} must explicitly declare dirty=false")
    return true_sweep, source_a_sweep


def _require_close(observed: Any, expected: float, label: str, *, tolerance: float = 0.0) -> None:
    try:
        value = finite_float(observed, label)
    except Exception as exc:
        raise ArtifactError(f"{label} is not a finite scalar: {exc}") from exc
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ArtifactError(f"{label}={value!r} does not recompute to {expected!r}")


def _canonical_sweep_summary(sweep: Mapping[str, Any], label: str) -> dict[str, Any]:
    heads = sweep.get("heads")
    if not isinstance(heads, list) or len(heads) != HEAD_COUNT:
        raise HeadSchemaError(f"{label} must contain exactly {HEAD_COUNT} head rows")
    ordered = sorted(
        heads,
        key=lambda row: (int(row.get("layer", -1)), int(row.get("head", -1))) if isinstance(row, Mapping) else (-1, -1),
    )
    expected_heads = [(layer, head) for layer in range(12) for head in range(12)]
    observed_heads = [
        (row.get("layer"), row.get("head")) if isinstance(row, Mapping) else (None, None)
        for row in ordered
    ]
    if observed_heads != expected_heads:
        raise HeadSchemaError(f"{label} head rows are not the canonical L0H0 ... L11H11 universe")
    frozen_pair_ids: tuple[Any, ...] | None = None
    means: list[float] = []
    expected_source = (
        "true_single_flip_fresh_same_invocation"
        if label == "true_sweep"
        else "source_A_same_number_different_noun_fresh_same_invocation"
    )
    for row in ordered:
        declared_hash = row.get("row_sha256")
        material = {key: value for key, value in row.items() if key != "row_sha256"}
        if not _is_sha256_hex(declared_hash) or declared_hash != sha256_json(material):
            raise HashMismatchError(f"{label} L{row['layer']}H{row['head']} row_sha256 is missing or mismatched")
        if row.get("hook") != "hook_z" or row.get("write_position") != "final" or row.get("source") != expected_source:
            raise ArtifactError(f"{label} L{row['layer']}H{row['head']} is not a fresh final-position hook_z row")
        records = row.get("pair_records")
        if not isinstance(records, list):
            raise ArtifactError(f"{label} L{row['layer']}H{row['head']} pair_records is missing")
        grouped = group_pair_records(records, directions=PAIR_DIRECTIONS, effect_key="effect")
        pair_ids = tuple(record.pair_id for record in grouped)
        if frozen_pair_ids is None:
            frozen_pair_ids = pair_ids
        elif pair_ids != frozen_pair_ids:
            raise ArtifactError(f"{label} head rows do not share one retained pair-id population")
        pair_values = paired_effect_array(grouped)
        mean = float(pair_values.mean())
        _require_close(row.get("E_delta_d"), mean, f"{label} L{row['layer']}H{row['head']} E_delta_d")
        _require_close(row.get("abs_E_delta_d"), abs(mean), f"{label} L{row['layer']}H{row['head']} abs_E_delta_d")
        _require_close(
            row.get("minimal_pair_both_directions_positive_fraction"),
            pair_sign_consistency(pair_values),
            f"{label} L{row['layer']}H{row['head']} pair sign consistency",
        )
        means.append(mean)
    if frozen_pair_ids is None or len(frozen_pair_ids) < MIN_RETAINED_PAIRS:
        observed_count = 0 if frozen_pair_ids is None else len(frozen_pair_ids)
        raise ArtifactError(f"{label} has {observed_count} retained pairs; at least {MIN_RETAINED_PAIRS} are required")
    return {
        "pair_ids": frozen_pair_ids,
        "means": means,
        "rank_order": [
            {"layer": int(row["layer"]), "head": int(row["head"]), "flat_id": int(row["layer"]) * 12 + int(row["head"]) + 1}
            for _, row in sorted(
                zip(means, ordered),
                key=lambda item: (-item[0], int(item[1]["layer"]), int(item[1]["head"])),
            )
        ],
    }


def _validate_selection_derivatives(
    selection: Mapping[str, Any],
    true_sweep: Mapping[str, Any],
    source_a_sweep: Mapping[str, Any],
    result: CandidateResult,
) -> dict[str, Any]:
    true_summary = _canonical_sweep_summary(true_sweep, "true_sweep")
    source_a_summary = _canonical_sweep_summary(source_a_sweep, "source_a_sweep")
    if true_summary["pair_ids"] != source_a_summary["pair_ids"]:
        raise ArtifactError("true/source-A sweeps do not share the same retained pair ids")
    retained = selection.get("retained_pairs")
    if not isinstance(retained, list) or any(type(pair_id) is not int for pair_id in retained):
        raise ArtifactError("selection.retained_pairs must be a list of plain integer ids")
    if tuple(retained) != true_summary["pair_ids"]:
        raise ArtifactError("selection.retained_pairs differs from the nested sweep pair population")
    directed = selection.get("retained_directed_edits")
    if not isinstance(directed, Mapping):
        raise ArtifactError("selection.retained_directed_edits is missing")
    expected_pair_ids = [pair_id for pair_id in retained for _ in PAIR_DIRECTIONS]
    expected_directions = list(PAIR_DIRECTIONS) * len(retained)
    if directed.get("count") != len(expected_pair_ids):
        raise ArtifactError("selection retained directed-edit count does not match retained pairs")
    if directed.get("pair_ids") != expected_pair_ids or directed.get("directions") != expected_directions:
        raise ArtifactError("selection retained directed-edit pair ids/directions differ from the nested sweeps")
    declared_rank = true_sweep.get("signed_rank_order")
    if declared_rank != true_summary["rank_order"]:
        raise ArtifactError("true_sweep signed_rank_order does not reproduce under the canonical float64 reduction")
    selection_block = selection.get("selection")
    if not isinstance(selection_block, Mapping):
        raise ArtifactError("selection derived-statistics block is missing")
    source_a_abs_means = [abs(float(value)) for value in source_a_summary["means"]]
    expected_edge = linear_percentile(source_a_abs_means, 99.0)
    if selection_block.get("source_a_abs_mean_values") != source_a_abs_means:
        raise ArtifactError("selection source-A absolute means do not match canonical row reductions")
    expected_head_order = [f"L{layer}H{head}" for layer in range(12) for head in range(12)]
    if selection_block.get("source_a_head_order") != expected_head_order:
        raise ArtifactError("selection source-A head order is not canonical")
    _require_close(selection_block.get("source_a_abs_mean_linear_p99"), expected_edge, "selection source-A P99")
    _require_close(source_a_sweep.get("source_a_abs_mean_linear_p99"), expected_edge, "source_a_sweep P99")
    if not result.evidence:
        raise ArtifactError("candidate constructor returned no top-10 evidence rows")
    _require_close(result.evidence[0].source_a_noise_edge, expected_edge, "candidate source-A P99")
    evidence_heads = [entry.head.as_dict() for entry in result.evidence]
    if evidence_heads != true_summary["rank_order"][:10]:
        raise ArtifactError("candidate evidence does not use the canonical Stage-1 top-10 rank order")
    return {
        "retained_pair_count": len(retained),
        "retained_pair_ids": list(retained),
        "source_a_abs_mean_linear_p99": expected_edge,
        "true_signed_rank_order": list(true_summary["rank_order"]),
    }


def _validate_linked_runtime_artifacts(selection: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, str]:
    pair_hash = provenance.get("pair_output_sha256")
    pair_path_value = provenance.get("pair_output")
    checkpoint_hash = provenance.get("checkpoint_sha256")
    checkpoint_path_value = provenance.get("checkpoint")
    if (
        not _is_sha256_hex(pair_hash)
        or not isinstance(pair_path_value, str)
        or not Path(pair_path_value).is_absolute()
    ):
        raise HashMismatchError("selection pair-output path/hash is missing")
    if (
        not _is_sha256_hex(checkpoint_hash)
        or not isinstance(checkpoint_path_value, str)
        or not Path(checkpoint_path_value).is_absolute()
    ):
        raise HashMismatchError("selection checkpoint path/hash is missing")
    pair_path_raw = Path(pair_path_value).expanduser()
    checkpoint_path_raw = Path(checkpoint_path_value).expanduser()
    if pair_path_raw.is_symlink() or checkpoint_path_raw.is_symlink():
        raise ArtifactError("selection linked runtime artifacts must not be symlinks")
    pair_path = pair_path_raw.resolve(strict=False)
    checkpoint_path = checkpoint_path_raw.resolve(strict=False)
    if sha256_file(pair_path) != pair_hash:
        raise HashMismatchError("selection pair-output file differs from its declared raw hash")
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise HashMismatchError("selection checkpoint file differs from its declared raw hash")
    pair_payload = _read_json(pair_path, "selection pair-output")
    if pair_payload.get("schema") != PAIR_OUTPUT_SCHEMA or pair_payload.get("status") != "COMPLETE":
        raise ArtifactError("selection pair-output schema/status is not COMPLETE")
    if pair_payload.get("input_sha256") != selection.get("input_sha256") or pair_payload.get("commit") != provenance.get("commit"):
        raise HashMismatchError("selection pair-output input hash/commit differs from the selection manifest")
    fingerprints = provenance["model_state_fingerprints"]
    config = provenance["normalized_model_config"]
    tokenizer = provenance["tokenizer_assets"]
    cache = provenance["immutable_clean_base_cache"]
    environment = provenance["environment"]
    expected_pair_provenance = {
        "invocation_id": provenance.get("invocation_id"),
        "snapshot_provenance_status": "READY",
        "model_state_sha256": fingerprints["before_sweeps"]["sha256"],
        "normalized_config_sha256": config["sha256"],
        "tokenizer_assets_sha256": tokenizer["aggregate_sha256"],
        "clean_base_cache_sha256": cache["before_sweeps"]["sha256"],
        "local_snapshot_revisions_sha256": provenance["model"]["local_snapshot_revisions_sha256"],
        "environment_sha256": environment["sha256"],
    }
    for name, expected in expected_pair_provenance.items():
        if pair_payload.get(name) != expected:
            raise HashMismatchError(f"selection pair-output {name} differs from the fresh selection manifest")
    retained = selection.get("retained_pairs")
    if (
        pair_payload.get("seed") != selection.get("seed")
        or pair_payload.get("retained_pairs") != retained
    ):
        raise HashMismatchError("selection pair-output seed/retained ids differ from the selection manifest")
    pair_records = pair_payload.get("pair_records")
    if not isinstance(retained, list) or not isinstance(pair_records, list) or len(pair_records) != len(retained):
        raise ArtifactError("selection pair-output retained pair records are incomplete")
    for index, (record, pair_id) in enumerate(zip(pair_records, retained)):
        base_pair = record.get("base_pair") if isinstance(record, Mapping) else None
        true_source_pair = record.get("true_source_pair") if isinstance(record, Mapping) else None
        source_a_pair = record.get("source_a_pair") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or type(record.get("pair_id")) is not int
            or record.get("pair_id") != pair_id
            or not isinstance(base_pair, Mapping)
            or not isinstance(true_source_pair, Mapping)
            or not isinstance(source_a_pair, Mapping)
            or type(base_pair.get("pair_index")) is not int
            or type(true_source_pair.get("pair_index")) is not int
            or type(source_a_pair.get("pair_index")) is not int
            or base_pair.get("pair_index") != pair_id
            or true_source_pair != base_pair
            or source_a_pair.get("pair_index") != pair_id
            or source_a_pair.get("family") != "source_A_same_number_different_noun"
        ):
            raise ArtifactError(f"selection pair-output record {index} does not match retained pair {pair_id!r}")
    true_heads = selection.get("true_sweep", {}).get("heads") if isinstance(selection.get("true_sweep"), Mapping) else None
    source_heads = selection.get("source_a_sweep", {}).get("heads") if isinstance(selection.get("source_a_sweep"), Mapping) else None
    if not isinstance(true_heads, list) or len(true_heads) != HEAD_COUNT or not isinstance(source_heads, list) or len(source_heads) != HEAD_COUNT:
        raise ArtifactError("selection fresh sweeps do not both contain 144 head rows")
    if pair_payload.get("true_heads") != true_heads or pair_payload.get("source_a_heads") != source_heads:
        raise HashMismatchError("selection pair-output fresh head rows differ from the selection manifest")
    checkpoint = _read_json(checkpoint_path, "selection checkpoint")
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("status") != "COMPLETE":
        raise ArtifactError("selection checkpoint schema/status is not COMPLETE")
    declared_checkpoint_hash = checkpoint.get("checkpoint_sha256")
    checkpoint_material = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    if not _is_sha256_hex(declared_checkpoint_hash) or declared_checkpoint_hash != sha256_json(checkpoint_material):
        raise HashMismatchError("selection checkpoint canonical self-hash is missing or mismatched")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ArtifactError("selection checkpoint metadata is missing")
    expected_metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "commit": provenance.get("commit"),
        "protocol_sha256": provenance.get("protocol_sha256"),
        "calibration_sha256": provenance.get("calibration_sha256"),
        "stage1_sha256": provenance.get("stage1_sha256"),
        "input_sha256": selection.get("input_sha256"),
        "source": "fresh_true_and_source_A",
        "canonical_head_order": selection.get("canonical_head_order"),
        "invocation_id": provenance.get("invocation_id"),
        "model_state_sha256_before_sweeps": fingerprints["before_sweeps"]["sha256"],
        "normalized_config_sha256": config["sha256"],
        "tokenizer_assets_sha256": tokenizer["aggregate_sha256"],
        "clean_base_cache_sha256": cache["before_sweeps"]["sha256"],
        "local_snapshot_revisions_sha256": provenance["model"]["local_snapshot_revisions_sha256"],
        "environment_sha256": environment["sha256"],
        "resume_policy": "validate_prior_checkpoint_then_restart_both_sweeps_in_new_invocation",
    }
    if sha256_json(dict(metadata)) != sha256_json(expected_metadata):
        raise HashMismatchError("selection checkpoint metadata differs from the exact selection manifest contract")
    completed = checkpoint.get("completed_heads")
    rows = checkpoint.get("head_rows")
    if not isinstance(completed, Mapping) or set(completed) != {"true", "source_a"} or not isinstance(rows, Mapping) or set(rows) != {"true", "source_a"}:
        raise ArtifactError("selection COMPLETE checkpoint lacks exact true/source_a maps")
    expected_completed = [f"L{layer}H{head}" for layer in range(12) for head in range(12)]
    manifest_sweeps = {"true": true_heads, "source_a": source_heads}
    for sweep_name in ("true", "source_a"):
        if completed.get(sweep_name) != expected_completed:
            raise ArtifactError(f"selection COMPLETE checkpoint {sweep_name} heads are not canonical")
        sweep_rows = rows.get(sweep_name)
        if not isinstance(sweep_rows, Mapping) or set(sweep_rows) != set(expected_completed):
            raise ArtifactError(f"selection checkpoint {sweep_name} completed-head and row keys differ")
        manifest_rows = {
            f"L{row['layer']}H{row['head']}": row
            for row in manifest_sweeps[sweep_name]
            if isinstance(row, Mapping) and "layer" in row and "head" in row
        }
        if dict(sweep_rows) != manifest_rows:
            raise HashMismatchError(f"selection checkpoint {sweep_name} rows differ from the selection manifest")
    return {"pair_output_sha256": pair_hash, "checkpoint_sha256": checkpoint_hash}


def _validate_frozen_input_files(provenance: Mapping[str, Any]) -> dict[str, str | None]:
    verified: dict[str, str | None] = {}
    for label in ("calibration",):
        path_value = provenance.get(label)
        hash_value = provenance.get(f"{label}_sha256")
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or not _is_sha256_hex(hash_value)
        ):
            raise HashMismatchError(f"selection provenance {label} path/hash is missing")
        raw_path = Path(path_value).expanduser()
        path = raw_path.resolve(strict=False)
        if sha256_file(path) != hash_value:
            raise HashMismatchError(f"selection provenance {label} file differs from its declared raw hash")
        verified[f"{label}_sha256"] = hash_value
    stage1_hash = provenance.get("stage1_sha256")
    if stage1_hash is not None and not _is_sha256_hex(stage1_hash):
        raise HashMismatchError("recorded nonblocking Stage-1 crosscheck hash is malformed")
    # Historical Stage 1 is never reopened here.  Fresh C remains freezable if
    # that optional external file is absent or changed after selection.
    verified["stage1_sha256"] = stage1_hash
    return verified


def _validate_source_a_snapshot(provenance: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    source_model = provenance.get("model")
    if not isinstance(source_model, Mapping):
        raise ArtifactError("selection provenance.model is missing")
    protocol_revisions = protocol.get("model", {}).get("expected_local_snapshot_revisions", {})
    local_snapshots = source_model.get("local_snapshot_revisions")
    if (
        not isinstance(protocol_revisions, Mapping)
        or not isinstance(local_snapshots, Mapping)
        or set(local_snapshots) != {"gpt2", "sae"}
        or not _is_sha256_hex(source_model.get("local_snapshot_revisions_sha256"))
        or source_model.get("local_snapshot_revisions_sha256") != sha256_json(dict(local_snapshots))
        or source_model.get("sae_loaded") is not False
        or source_model.get("sae_revision_fingerprint_present") is not True
    ):
        raise ArtifactError("selection lacks the exact pinned GPT-2/SAE revision fingerprint structure")
    verified_snapshots: dict[str, dict[str, str]] = {}
    for name in ("gpt2", "sae"):
        entry = local_snapshots.get(name)
        expected_revision = protocol_revisions.get(name)
        if not isinstance(entry, Mapping) or not _is_git_commit(expected_revision):
            raise ArtifactError(f"selection {name} local revision record is missing")
        ref_value = entry.get("refs_main_path")
        snapshot_value = entry.get("snapshot_path")
        if (
            entry.get("expected_revision") != expected_revision
            or entry.get("observed_revision") != expected_revision
            or entry.get("snapshot_present") is not True
            or entry.get("revision_check") != "exact local refs/main and snapshots/<revision> match"
            or not _is_sha256_hex(entry.get("refs_main_sha256"))
            or not isinstance(ref_value, str)
            or not Path(ref_value).is_absolute()
            or not isinstance(snapshot_value, str)
            or not Path(snapshot_value).is_absolute()
        ):
            raise ArtifactError(f"selection {name} local revision/ref metadata differs from the frozen protocol")
        ref_raw = Path(ref_value).expanduser()
        if ref_raw.is_symlink():
            raise ArtifactError(f"selection {name} refs/main must not be a symlink")
        ref_path = ref_raw.resolve(strict=False)
        snapshot_path = Path(snapshot_value).expanduser().resolve(strict=False)
        try:
            ref_bytes = ref_path.read_bytes()
            observed_ref = ref_bytes.decode("utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ArtifactError(f"cannot read selection {name} refs/main: {exc}") from exc
        if (
            not snapshot_path.is_dir()
            or sha256_bytes(ref_bytes) != entry.get("refs_main_sha256")
            or observed_ref != expected_revision
        ):
            raise HashMismatchError(f"selection {name} refs/main no longer points to the pinned revision")
        verified_snapshots[name] = {
            "revision": expected_revision,
            "ref_path": str(ref_path),
            "snapshot_path": str(snapshot_path),
            "ref_sha256": str(entry["refs_main_sha256"]),
        }
    gpt2 = verified_snapshots["gpt2"]
    sae = verified_snapshots["sae"]
    if (
        source_model.get("snapshot_revision_expected") != gpt2["revision"]
        or source_model.get("snapshot_revision_observed") != gpt2["revision"]
        or source_model.get("snapshot_ref_path") != gpt2["ref_path"]
        or source_model.get("local_model_revision") != gpt2["revision"]
        or source_model.get("local_sae_revision") != sae["revision"]
        or provenance.get("local_model_revision") != gpt2["revision"]
        or provenance.get("local_sae_revision") != sae["revision"]
        or provenance.get("activation_dtype") != "float32"
        or provenance.get("runtime_environment_fingerprint")
        != provenance.get("environment", {}).get("sha256")
    ):
        raise ArtifactError("selection direct A7 provenance fields disagree with the joint GPT-2/SAE fingerprint")
    expected_architecture = {"n_layers": 12, "n_heads": 12, "d_model": 768, "d_vocab": 50_257}
    if source_model.get("architecture") != expected_architecture:
        raise ArtifactError("selection source-A model architecture differs from frozen GPT-2-small")
    if (
        source_model.get("device") != "cpu"
        or source_model.get("dtype") != "float32"
        or source_model.get("offline") is not True
        or source_model.get("snapshot_revision_check") != "exact local refs/main match"
    ):
        raise ArtifactError("selection source-A model execution/snapshot provenance is incomplete")
    return {
        "revision": gpt2["revision"],
        "ref_path": gpt2["ref_path"],
        "gpt2": verified_snapshots["gpt2"],
        "sae": verified_snapshots["sae"],
        "local_snapshot_revisions_sha256": source_model["local_snapshot_revisions_sha256"],
        "sae_loaded": False,
        "sae_revision_fingerprint_present": True,
        "status": "EXPECTED_LOCAL_REF_MATCH_RECORDED_NOT_BLOB_HASH",
    }


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise ArtifactError(f"output parent is not a directory: {parent}")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    except OSError as exc:
        raise ArtifactError(f"atomic candidate write failed for {path}: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def freeze_candidate(protocol_path: Path, selection_path: Path, output_path: Path, *, require_status: str = "COMPLETE") -> dict[str, Any]:
    """Validate inputs, construct C, and atomically write immutable candidate JSON."""

    if require_status != "COMPLETE":
        raise ArtifactError("only --require-status COMPLETE is permitted by the frozen workflow")
    raw_output = output_path.expanduser()
    if raw_output.is_symlink():
        raise ArtifactError(f"candidate output leaf must not be a symlink: {raw_output}")
    protocol_path = protocol_path.expanduser().resolve(strict=False)
    selection_path = selection_path.expanduser().resolve(strict=False)
    output_path = raw_output.resolve(strict=False)
    if output_path in {protocol_path, selection_path}:
        raise ArtifactError("candidate output must be disjoint from resolved protocol/selection inputs")
    if output_path.exists() and (output_path.is_symlink() or not output_path.is_file()):
        raise ArtifactError("existing candidate output must be a non-symlink regular file")
    protocol = _read_json(protocol_path, "protocol")
    selection = _read_json(selection_path, "selection")
    protocol_hash = sha256_file(protocol_path)
    selection_hash = sha256_file(selection_path)
    _require_clean_and_complete(selection, require_status)
    declared_protocol_hash = _declared_protocol_hash(selection)
    if declared_protocol_hash != protocol_hash:
        raise HashMismatchError(
            f"selection protocol hash {declared_protocol_hash} does not match supplied protocol {protocol_hash}"
        )
    provenance = selection["provenance"]
    true_sweep, source_a_sweep = _extract_sweeps(selection)
    result: CandidateResult = construct_candidate(
        true_sweep,
        source_a_sweep,
        protocol,
        selection_provenance=provenance,
    )
    derivative_evidence = _validate_selection_derivatives(selection, true_sweep, source_a_sweep, result)
    for label in ("protocol", "calibration", "stage1", "pair_output", "checkpoint"):
        protected_value = provenance.get(label)
        if isinstance(protected_value, str) and output_path == Path(protected_value).expanduser().resolve(strict=False):
            raise ArtifactError(f"candidate output must be disjoint from resolved {label} provenance input")
    model_provenance = provenance.get("model")
    snapshot_ref_path = model_provenance.get("snapshot_ref_path") if isinstance(model_provenance, Mapping) else None
    if isinstance(snapshot_ref_path, str) and output_path == Path(snapshot_ref_path).expanduser().resolve(strict=False):
        raise ArtifactError("candidate output must be disjoint from the resolved model snapshot ref input")
    local_snapshots = model_provenance.get("local_snapshot_revisions") if isinstance(model_provenance, Mapping) else None
    if isinstance(local_snapshots, Mapping):
        for name in ("gpt2", "sae"):
            entry = local_snapshots.get(name)
            ref_value = entry.get("refs_main_path") if isinstance(entry, Mapping) else None
            if isinstance(ref_value, str) and output_path == Path(ref_value).expanduser().resolve(strict=False):
                raise ArtifactError(f"candidate output must be disjoint from the resolved {name} refs/main input")
    if (
        true_sweep.get("model_snapshot_status") != FRESH_SWEEP_SNAPSHOT_STATUS
        or source_a_sweep.get("model_snapshot_status") != FRESH_SWEEP_SNAPSHOT_STATUS
        or true_sweep.get("invocation_id") != provenance.get("invocation_id")
        or source_a_sweep.get("invocation_id") != provenance.get("invocation_id")
    ):
        raise ArtifactError("candidate inputs are not two fresh sweeps from the selection invocation")
    source_a_snapshot = _validate_source_a_snapshot(provenance, protocol)
    input_hash = selection.get("input_sha256")
    if not _is_sha256_hex(input_hash) or provenance.get("input_sha256") != input_hash:
        raise HashMismatchError("selection input_sha256 is missing or disagrees with provenance")
    frozen_input_hashes = _validate_frozen_input_files(provenance)
    linked_hashes = _validate_linked_runtime_artifacts(selection, provenance)
    pair_output_hash = linked_hashes["pair_output_sha256"]
    checkpoint_hash = linked_hashes["checkpoint_sha256"]
    material = result.as_dict()
    # Replace canonical JSON hashes with the raw file hashes that establish the
    # actual command inputs.  The pure constructor still validates protocol
    # contents and records canonical nested-sweep hashes in its evidence.
    material["protocol_sha256"] = protocol_hash
    material["selection_source_a_sha256"] = selection_hash
    material["true_sweep_sha256"] = sha256_json(true_sweep)
    material["source_a_sweep_sha256"] = sha256_json(source_a_sweep)
    material["selection_sha256"] = selection_hash
    material["status"] = "COMPLETE" if result.status == "NONEMPTY" else "COMPLETE_NO_CANDIDATES"
    material["candidate_status"] = result.status
    material["candidate_C"] = [head.as_dict() for head in result.candidate_heads]
    material["selection_derivative_evidence"] = derivative_evidence
    material["shipped_stage1_crosscheck"] = dict(selection["shipped_stage1_crosscheck"])
    material["selection_provenance"] = {
        "commit": provenance["commit"],
        "invocation_id": provenance["invocation_id"],
        "dirty": False,
        "git_status": "clean",
        "clean_tree_scope": provenance["clean_tree_scope"],
        "completion_clean_tree_checked": True,
        "input_sha256": input_hash,
        "pair_output_sha256": pair_output_hash,
        "checkpoint_sha256": checkpoint_hash,
        "calibration_sha256": frozen_input_hashes["calibration_sha256"],
        "stage1_sha256": frozen_input_hashes["stage1_sha256"],
        "snapshot_provenance_status": "READY",
        "model_state_sha256": provenance["model_state_fingerprints"]["before_sweeps"]["sha256"],
        "normalized_config_sha256": provenance["normalized_model_config"]["sha256"],
        "tokenizer_assets_sha256": provenance["tokenizer_assets"]["aggregate_sha256"],
        "clean_base_cache_sha256": provenance["immutable_clean_base_cache"]["before_sweeps"]["sha256"],
        "local_snapshot_revisions_sha256": source_a_snapshot["local_snapshot_revisions_sha256"],
        "local_model_revision": provenance["local_model_revision"],
        "local_sae_revision": provenance["local_sae_revision"],
        "activation_dtype": "float32",
        "environment_sha256": provenance["environment"]["sha256"],
        "source_a_model_snapshot_revision": source_a_snapshot["revision"],
        "source_a_model_snapshot_ref_path": source_a_snapshot["ref_path"],
        "source_a_model_snapshot_status": source_a_snapshot["status"],
        "gpt2_local_snapshot": source_a_snapshot["gpt2"],
        "sae_local_snapshot": source_a_snapshot["sae"],
        "sae_loaded": False,
        "sae_revision_fingerprint_present": True,
        "shipped_stage1_role": "descriptive_non_blocking_only",
    }
    material["immutable"] = True
    material["manual_override"] = False
    material["generated_at"] = _utc_now()
    material["candidate_sha256"] = sha256_json(material)
    _atomic_write_json(output_path, material)
    return material


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze Experiment 05's selection-only head candidate pool.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--require-status", choices=("COMPLETE",), default="COMPLETE")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        freeze_candidate(args.protocol, args.selection, args.output, require_status=args.require_status)
    except (CoreError, OSError, ValueError) as exc:
        print(f"freeze_candidate: ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - execution requires explicit user authorization
    raise SystemExit(main())
