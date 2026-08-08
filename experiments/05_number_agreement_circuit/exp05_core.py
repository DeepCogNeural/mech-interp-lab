"""Pure, model-free primitives for Experiment 05.

This module is deliberately independent of PyTorch, TransformerLens, HuggingFace,
and the experiment runners.  It contains the frozen statistical and provenance
conventions from Amendments 3--5.  Runner artifacts must be converted to the
explicit scalar pair-record schema below before calling these functions:

``exp05.stage_sweep.v1``::

    {
      "schema": "exp05.stage_sweep.v1", "status": "COMPLETE",
      "seed": 20260801, "source": "true" or "source_a",
      "head_count": 144,
      "directions": ["singular_to_plural", "plural_to_singular"],
      "heads": [{"layer": 0, "head": 0,
                 "pair_records": [
                   {"pair_id": "p0", "direction": "singular_to_plural",
                    "effect": 0.1},
                   {"pair_id": "p0", "direction": "plural_to_singular",
                    "effect": 0.2}]}]
    }

``exp05.selection.v1``::

    {"schema": "exp05.selection.v1", "status": "COMPLETE",
     "true_sweep": <stage_sweep.v1>,
     "source_a_sweep": <stage_sweep.v1>}

The pair-record effect is already sign-aligned and is in raw ``Delta d`` units.
No adapter guesses whether an existing array is paired, directed, or aligned.  A
missing field, duplicate direction, incomplete status, or non-finite value raises
one of the typed errors below.  Primary thresholds are constants in this module;
the freezer does not accept threshold overrides from its command line.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np


# Frozen protocol constants.  Keep these at module scope so runners and the
# freezer share one source of truth rather than duplicating command-line flags.
LAYER_COUNT = 12
HEADS_PER_LAYER = 12
HEAD_COUNT = LAYER_COUNT * HEADS_PER_LAYER
PAIR_DIRECTIONS = ("singular_to_plural", "plural_to_singular")
REQUESTED_PAIR_COUNT = 240
MIN_RETAINED_PAIRS = 140
STAGE1_SELECTION_SEED = 20_260_801
STAGE1_TOP_K = 10
MAX_NESTED_SET_SIZE = 8
PAIR_SIGN_CONSISTENCY_MIN = 0.90
SOURCE_A_PERCENTILE = 99.0
BOOTSTRAP_DRAWS = 100_000
CI_DRAWS = 10_000
HOLM_ALPHA = 0.05
Q3_TEST_ID = 301
Q4_TEST_ID = 401
Q4_SUBSET_SIZE = 12
Q4_ACCEPTED_SUBSETS = 100
Q4_MAX_ATTEMPTS = 10_000
AMENDMENT4_TRAINING_PAIRS = 40
AMENDMENT4_EVALUATION_PAIRS = 150
AMENDMENT4_SPLIT_OFFSET = 701
TARGET_LATENT_IDS = (8922, 8952, 13352, 13594, 15165, 17956, 19093, 19955, 21401, 21581, 21805, 23011)
STAGE_SWEEP_SCHEMA = "exp05.stage_sweep.v1"
SELECTION_SCHEMA = "exp05.selection.v1"
PROTOCOL_SCHEMA = "exp05-number-agreement-protocol"
# Stage-2/Stage-3 runners pin this API string so a partially upgraded core cannot
# silently change the estimator used by an experiment run.
CORE_API_VERSION = "exp05-core-v1"
FRESH_SWEEP_SNAPSHOT_STATUS = "FRESH_SAME_INVOCATION_MODEL_STATE_FINGERPRINT_MATCHED"


class CoreError(Exception):
    """Base class for all fail-closed Experiment 05 errors."""


class ProtocolError(CoreError):
    """A frozen protocol value is missing, malformed, or changed."""


class ArtifactError(CoreError):
    """An input artifact does not satisfy its declared schema."""


class IncompleteArtifactError(ArtifactError):
    """An artifact is not COMPLETE and therefore cannot feed an adjudicator."""


class DirtyArtifactError(ArtifactError):
    """An artifact declares uncommitted or otherwise dirty provenance."""


class HashMismatchError(ArtifactError):
    """A declared hash does not match the supplied content."""


class HeadSchemaError(ArtifactError):
    """The canonical 144-head universe is not represented exactly once."""


class PairSchemaError(ArtifactError):
    """Pair records do not contain exactly the two named directions."""


class BlockedError(CoreError):
    """A registered computation is blocked rather than silently altered."""


class BlockStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class SplitRole(str, Enum):
    RANK_TRAINING = "rank_training"
    EVALUATION = "evaluation"
    UNUSED_AFTER_EVAL_CAP = "unused_after_eval_cap"


@dataclass(frozen=True, order=True)
class HeadId:
    """One canonical TransformerLens head coordinate."""

    layer: int
    head: int

    def __post_init__(self) -> None:
        if type(self.layer) is not int or type(self.head) is not int:
            raise HeadSchemaError("layer and head must be plain integers")
        if not (0 <= self.layer < LAYER_COUNT and 0 <= self.head < HEADS_PER_LAYER):
            raise HeadSchemaError(f"head out of canonical range: L{self.layer}H{self.head}")

    @property
    def flat_id(self) -> int:
        return self.layer * HEADS_PER_LAYER + self.head + 1

    def as_dict(self) -> dict[str, int]:
        return {"layer": self.layer, "head": self.head, "flat_id": self.flat_id}


def head_id(layer: int, head: int) -> HeadId:
    """Construct and validate a canonical head coordinate."""

    return HeadId(layer=layer, head=head)


def flat_head_id(layer: int, head: int) -> int:
    """Return the frozen one-based flat test id ``12*layer + head + 1``."""

    return HeadId(layer=layer, head=head).flat_id


def head_from_flat(flat_id: int) -> HeadId:
    """Invert the frozen one-based flat head id."""

    if type(flat_id) is not int or not (1 <= flat_id <= HEAD_COUNT):
        raise HeadSchemaError(f"flat head id must be an integer in 1..{HEAD_COUNT}: {flat_id!r}")
    zero_based = flat_id - 1
    return HeadId(layer=zero_based // HEADS_PER_LAYER, head=zero_based % HEADS_PER_LAYER)


def canonical_heads() -> tuple[HeadId, ...]:
    """Return all 144 heads in canonical flat-id order."""

    return tuple(HeadId(layer, head) for layer in range(LAYER_COUNT) for head in range(HEADS_PER_LAYER))


def _stable_key(value: Hashable) -> tuple[Any, ...]:
    """Sort ids deterministically, using numeric order for integer pair ids."""

    try:
        hash(value)
    except TypeError as exc:
        raise PairSchemaError(f"pair_id must be hashable: {value!r}") from exc
    if type(value) is int:
        return (0, value)
    return (1, type(value).__name__, repr(value))


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON with the repository's hash-stable canonical settings."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"value is not canonical finite JSON: {exc}") from exc


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Hash bytes using SHA-256 and return lowercase hexadecimal."""

    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ArtifactError("sha256_bytes requires a bytes-like value")
    return hashlib.sha256(bytes(value)).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a canonical JSON value."""

    return sha256_bytes(canonical_json_bytes(value))


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: str | Path) -> str:
    """Hash a file's raw bytes without loading it all into memory."""

    file_path = Path(path)
    if not file_path.is_file():
        raise ArtifactError(f"hash input is not a regular file: {file_path}")
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot read hash input {file_path}: {exc}") from exc
    return digest.hexdigest()


def finite_float(value: Any, name: str = "value") -> float:
    """Coerce one real scalar, rejecting bools, strings, NaN, and infinities."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ProtocolError(f"{name} must be a finite numeric scalar, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite, got {value!r}")
    return result


def finite_array(values: Any, name: str = "values", *, ndim: int | None = None) -> np.ndarray:
    """Return a float64 array after rejecting ragged, non-numeric, or non-finite input."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must be numeric and rectangular: {exc}") from exc
    if ndim is not None and array.ndim != ndim:
        raise ProtocolError(f"{name} must have ndim={ndim}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ProtocolError(f"{name} contains NaN or infinity")
    return array


@dataclass(frozen=True)
class PairRecord:
    """A minimal pair with one sign-aligned effect for each named direction."""

    pair_id: Hashable
    effects_by_direction: tuple[tuple[str, float], tuple[str, float]]
    directions: tuple[str, str] = PAIR_DIRECTIONS

    def __post_init__(self) -> None:
        if len(self.directions) != 2 or self.directions[0] == self.directions[1]:
            raise PairSchemaError("exactly two distinct directions are required")
        names = tuple(name for name, _ in self.effects_by_direction)
        if names != self.directions:
            raise PairSchemaError(f"directions must be exactly {self.directions!r}, got {names!r}")
        finite_float(self.effects_by_direction[0][1], "pair effect")
        finite_float(self.effects_by_direction[1][1], "pair effect")
        _stable_key(self.pair_id)

    @property
    def effects(self) -> tuple[float, float]:
        return (self.effects_by_direction[0][1], self.effects_by_direction[1][1])


def _record_effect(record: Mapping[str, Any], effect_key: str) -> float:
    if effect_key not in record:
        raise PairSchemaError(f"pair record missing scalar {effect_key!r}: {record!r}")
    return finite_float(record[effect_key], f"pair record {effect_key}")


def group_pair_records(
    records: Iterable[Mapping[str, Any]],
    *,
    directions: Sequence[str] = PAIR_DIRECTIONS,
    effect_key: str = "effect",
    pair_id_key: str = "pair_id",
    direction_key: str = "direction",
) -> tuple[PairRecord, ...]:
    """Group scalar records and require exactly one of each named direction.

    This function intentionally rejects Stage-1's separate parallel arrays.  A
    runner must materialise one scalar mapping per directed edit first, making the
    pair unit visible in the artifact and preventing accidental resampling of only
    one direction.
    """

    expected = tuple(str(direction) for direction in directions)
    if len(expected) != 2 or expected[0] == expected[1] or any(not name for name in expected):
        raise PairSchemaError("directions must contain exactly two distinct non-empty names")
    grouped: dict[Hashable, dict[str, float]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PairSchemaError(f"pair record {index} is not an object")
        if pair_id_key not in record or direction_key not in record:
            raise PairSchemaError(f"pair record {index} needs {pair_id_key!r} and {direction_key!r}")
        pair_key = record[pair_id_key]
        _stable_key(pair_key)
        direction = record[direction_key]
        if not isinstance(direction, str) or direction not in expected:
            raise PairSchemaError(f"unknown direction {direction!r}; expected {expected!r}")
        by_direction = grouped.setdefault(pair_key, {})
        if direction in by_direction:
            raise PairSchemaError(f"duplicate pair/direction {pair_key!r}/{direction!r}")
        by_direction[direction] = _record_effect(record, effect_key)
    if not grouped:
        raise PairSchemaError("at least one complete pair is required")
    result: list[PairRecord] = []
    for pair_key in sorted(grouped, key=_stable_key):
        by_direction = grouped[pair_key]
        if set(by_direction) != set(expected):
            raise PairSchemaError(
                f"pair {pair_key!r} must contain exactly {expected!r}, got {tuple(sorted(by_direction))!r}"
            )
        result.append(
            PairRecord(
                pair_id=pair_key,
                effects_by_direction=((expected[0], by_direction[expected[0]]), (expected[1], by_direction[expected[1]])),
                directions=expected,
            )
        )
    return tuple(result)


def pair_clusters(values_2d: Any) -> np.ndarray:
    """Validate a pair-cluster matrix with one row per pair and two directions."""

    array = finite_array(values_2d, "pair_clusters")
    if array.ndim == 1:
        if array.size % 2:
            raise PairSchemaError(f"directed pair vector must have an even length, got {array.size}")
        array = array.reshape(-1, 2)
    if array.ndim != 2:
        raise PairSchemaError(f"pair cluster matrix must be 2-D, got shape {array.shape}")
    if array.shape[1] != 2:
        raise PairSchemaError(f"pair cluster matrix must have exactly two columns, got {array.shape}")
    if array.shape[0] < 1:
        raise PairSchemaError("at least one pair cluster is required")
    return array


def paired_effect_array(records: Sequence[PairRecord] | Any) -> np.ndarray:
    """Return an ``N x 2`` float64 matrix from pair records or an existing matrix."""

    if isinstance(records, np.ndarray) or (
        isinstance(records, Sequence) and records and not isinstance(records[0], PairRecord)
    ):
        return pair_clusters(records)
    if not records:
        raise PairSchemaError("at least one pair record is required")
    if not all(isinstance(record, PairRecord) for record in records):
        raise PairSchemaError("records must be PairRecord objects or an N x 2 numeric matrix")
    directions = records[0].directions
    if any(record.directions != directions for record in records):
        raise PairSchemaError("all PairRecord objects must use the same directions")
    return pair_clusters(np.asarray([record.effects for record in records], dtype=np.float64))


def pair_sign_consistency(records_or_values: Sequence[PairRecord] | Any) -> float:
    """Return the fraction of pairs whose *both* direction effects are strictly positive."""

    values = paired_effect_array(records_or_values)
    return float(np.mean(np.logical_and(values[:, 0] > 0.0, values[:, 1] > 0.0)))


def linear_percentile(values: Any, q: float) -> float:
    """NumPy's explicitly frozen linear percentile interpolation."""

    q_value = finite_float(q, "percentile q")
    if not 0.0 <= q_value <= 100.0:
        raise ProtocolError(f"percentile q must be in [0,100], got {q_value}")
    array = finite_array(values, "percentile values").reshape(-1)
    if array.size == 0:
        raise ProtocolError("percentile values cannot be empty")
    try:
        return float(np.percentile(array, q_value, method="linear"))
    except TypeError as exc:  # pragma: no cover - old NumPy should fail closed
        raise ProtocolError("installed NumPy lacks the required method='linear' percentile") from exc


def rng(experiment_seed: int, test_id: int) -> np.random.Generator:
    """Return the only permitted bootstrap/matched-subset RNG stream."""

    if type(experiment_seed) is not int or experiment_seed < 0:
        raise ProtocolError(f"experiment_seed must be a non-negative integer: {experiment_seed!r}")
    if type(test_id) is not int or not (1 <= test_id <= HEAD_COUNT or test_id in {Q3_TEST_ID, Q4_TEST_ID} or test_id >= 1000):
        raise ProtocolError("test_id is not registered; use 1..144, 301, 401, or a declared >=1000 id")
    return np.random.default_rng(experiment_seed * 1000 + test_id)


def _fixed_draws(draws: int, expected: int, name: str) -> int:
    if type(draws) is not int or draws != expected:
        raise ProtocolError(f"{name} is frozen at {expected}; got {draws!r}")
    return draws


@dataclass(frozen=True)
class BootstrapPValue:
    observed_mean: float
    p_value: float
    draws: int
    experiment_seed: int
    test_id: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed_mean": self.observed_mean,
            "p_value": self.p_value,
            "draws": self.draws,
            "experiment_seed": self.experiment_seed,
            "test_id": self.test_id,
            "method": "pair_cluster_two_sided_add_one",
        }


def bootstrap_p_value(
    values_2d: Any,
    experiment_seed: int,
    test_id: int,
    resamples: int = BOOTSTRAP_DRAWS,
) -> float:
    """Two-sided add-one pair-cluster bootstrap p-value for a raw mean.

    The statistic is the mean over both directed columns.  Each resample draws
    pair rows with replacement and keeps both directions together.  The stream is
    ``default_rng(experiment_seed * 1000 + test_id)`` and the exact p-value is
    ``2*min((count(mu_b <= 0)+1)/(B+1), (count(mu_b >= 0)+1)/(B+1))`` clipped to one.
    """

    draws = _fixed_draws(resamples, BOOTSTRAP_DRAWS, "bootstrap resamples")
    values = pair_clusters(values_2d)
    observed = float(values.mean())
    generator = rng(experiment_seed, test_id)
    less_equal = 0
    greater_equal = 0
    # Chunking bounds memory while retaining one RNG stream and therefore exact
    # reproducibility independent of machine memory.
    chunk_size = 512
    for start in range(0, draws, chunk_size):
        size = min(chunk_size, draws - start)
        indices = generator.integers(0, values.shape[0], size=(size, values.shape[0]))
        means = values[indices].mean(axis=(1, 2))
        less_equal += int(np.count_nonzero(means <= 0.0))
        greater_equal += int(np.count_nonzero(means >= 0.0))
    p_value = min(1.0, 2.0 * min((less_equal + 1) / (draws + 1), (greater_equal + 1) / (draws + 1)))
    return float(p_value)


def bootstrap_two_sided_p(
    values_2d: Any,
    *,
    experiment_seed: int,
    test_id: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> BootstrapPValue:
    """Structured counterpart to :func:`bootstrap_p_value`."""

    values = pair_clusters(values_2d)
    p_value = bootstrap_p_value(values, experiment_seed, test_id, draws)
    return BootstrapPValue(float(values.mean()), p_value, draws, experiment_seed, test_id)


def percentile_ci(values: Any, seed: int, test_id: int, resamples: int = CI_DRAWS) -> tuple[float, float]:
    """Pair-cluster percentile 95% CI using the frozen 10,000-draw stream.

    ``values`` is an N x 2 matrix.  The returned interval uses NumPy's linear
    2.5/97.5 percentiles.  This helper is the common Stage-2 interval primitive;
    it does not expose an alpha or interpolation override.
    """

    draws = _fixed_draws(resamples, CI_DRAWS, "percentile-CI resamples")
    array = pair_clusters(values)
    generator = rng(seed, test_id)
    means: list[np.ndarray] = []
    chunk_size = 512
    for start in range(0, draws, chunk_size):
        size = min(chunk_size, draws - start)
        indices = generator.integers(0, array.shape[0], size=(size, array.shape[0]))
        means.append(array[indices].mean(axis=(1, 2)))
    bootstrap_means = np.concatenate(means)
    return (
        linear_percentile(bootstrap_means, 2.5),
        linear_percentile(bootstrap_means, 97.5),
    )


# Compatibility aliases are intentional: runners call these names, while the
# longer names above document the frozen statistical contract.
def percentile(values: Any, q: float) -> float:
    """Alias for the frozen NumPy-linear percentile primitive."""

    return linear_percentile(values, q)


def bootstrap_mean_ci(values: Any, *, seed: int, test_id: int, resamples: int = CI_DRAWS) -> tuple[float, float]:
    """Runner-facing alias for the frozen pair-cluster percentile interval."""

    return percentile_ci(values, seed=seed, test_id=test_id, resamples=resamples)


def rng_for(experiment_seed: int, test_id: int) -> np.random.Generator:
    """Alias for :func:`rng`, retained as the runner-facing spelling."""

    return rng(experiment_seed, test_id)


def paired_difference_ci(
    true_values_2d: Any,
    source_a_values_2d: Any,
    *,
    seed: int,
    test_id: int = Q3_TEST_ID,
    resamples: int = CI_DRAWS,
) -> tuple[float, float]:
    """Percentile CI for paired true-minus-source-A effects, preserving pairs."""

    draws = _fixed_draws(resamples, CI_DRAWS, "paired-difference CI resamples")
    true_values = pair_clusters(true_values_2d)
    source_a_values = pair_clusters(source_a_values_2d)
    if true_values.shape != source_a_values.shape:
        raise PairSchemaError("true and source-A pair matrices must have identical shape")
    difference = true_values - source_a_values
    return percentile_ci(difference, seed, test_id, draws)


def holm_step_down(pvalues: Mapping[Any, Any] | Sequence[Any], alpha: float = HOLM_ALPHA) -> dict[Any, bool] | list[bool]:
    """Holm step-down decisions with stable flat-head tie ordering.

    Keys are normally one-based flat head ids.  ``HeadId`` keys and ``(layer,
    head)`` keys are accepted and returned unchanged.  Equal p-values are ordered
    by canonical flat head id, never by insertion order.  Only the frozen alpha
    ``0.05`` is accepted so a CLI cannot weaken the primary rule.
    """

    alpha_value = finite_float(alpha, "Holm alpha")
    if alpha_value != HOLM_ALPHA:
        raise ProtocolError(f"Holm alpha is frozen at {HOLM_ALPHA}; got {alpha!r}")
    if isinstance(pvalues, Mapping):
        items = list(pvalues.items())
    else:
        items = [(index + 1, value) for index, value in enumerate(pvalues)]
    if not items:
        raise ProtocolError("Holm family cannot be empty")

    # Use a numeric canonical id for heads.  The textual fallback is only for
    # generic non-head mappings and is not used by candidate selection.
    def sort_key(item: tuple[Any, Any]) -> tuple[float, int, str]:
        p_value = finite_float(item[1], f"p-value for {item[0]!r}")
        if not 0.0 <= p_value <= 1.0:
            raise ProtocolError(f"p-value must be in [0,1]: {p_value}")
        if _is_head_key(item[0]):
            return p_value, _coerce_head(item[0]).flat_id, ""
        return p_value, HEAD_COUNT + 1, repr(_stable_key(item[0]))

    ordered = sorted(items, key=sort_key)
    decisions: dict[Any, bool] = {key: False for key, _ in items}
    still_rejecting = True
    family_size = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        p_value = finite_float(value, f"p-value for {key!r}")
        threshold = alpha_value / (family_size - rank)
        if still_rejecting and p_value <= threshold:
            decisions[key] = True
        else:
            still_rejecting = False
            decisions[key] = False
    if isinstance(pvalues, Mapping):
        return decisions
    # Preserve the caller's sequence shape for the runner-facing API.  The
    # stable ordering above still determines each decision; this reconstruction
    # is not insertion-order-dependent for equal p-values.
    return [bool(decisions[index + 1]) for index in range(len(items))]


def _is_head_key(value: Any) -> bool:
    return isinstance(value, HeadId) or (isinstance(value, tuple) and len(value) == 2) or (
        type(value) is int and 1 <= value <= HEAD_COUNT
    )


def _coerce_head(value: Any) -> HeadId:
    if isinstance(value, HeadId):
        return value
    if type(value) is int:
        return head_from_flat(value)
    if isinstance(value, tuple) and len(value) == 2:
        return HeadId(value[0], value[1])
    raise HeadSchemaError(f"not a canonical head key: {value!r}")


@dataclass(frozen=True)
class CandidateEvidence:
    head: HeadId
    stage1_rank: int
    true_mean: float
    source_a_mean: float
    source_a_noise_edge: float
    pair_sign_consistency: float
    bootstrap_p_value: float
    holm_reject: bool
    eligible: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "head": self.head.as_dict(),
            "stage1_rank": self.stage1_rank,
            "true_mean": self.true_mean,
            "source_a_mean": self.source_a_mean,
            "source_a_noise_edge": self.source_a_noise_edge,
            "pair_sign_consistency": self.pair_sign_consistency,
            "bootstrap_p_value": self.bootstrap_p_value,
            "holm_reject": self.holm_reject,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CandidateResult:
    candidate_heads: tuple[HeadId, ...]
    rank_order: tuple[HeadId, ...]
    nested_sets_by_rank: tuple[tuple[HeadId, ...], ...]
    evidence: tuple[CandidateEvidence, ...]
    status: str
    protocol_sha256: str | None = None
    true_sweep_sha256: str | None = None
    source_a_sweep_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "exp05.candidate.v1",
            "status": self.status,
            "candidate_status": self.status,
            "candidate_heads": [head.as_dict() for head in self.candidate_heads],
            "rank_order": [head.as_dict() for head in self.rank_order],
            "nested_sets": [[head.as_dict() for head in subset] for subset in self.nested_sets_by_rank],
            "selection_evidence": [entry.as_dict() for entry in self.evidence],
            "protocol_sha256": self.protocol_sha256,
            "true_sweep_sha256": self.true_sweep_sha256,
            "source_a_sweep_sha256": self.source_a_sweep_sha256,
        }


def _require_complete_sweep(sweep: Mapping[str, Any], source_name: str) -> list[Mapping[str, Any]]:
    if not isinstance(sweep, Mapping):
        raise ArtifactError(f"{source_name} must be an object matching {STAGE_SWEEP_SCHEMA}")
    if sweep.get("schema") != STAGE_SWEEP_SCHEMA:
        raise ArtifactError(f"{source_name} must declare schema {STAGE_SWEEP_SCHEMA!r}")
    if sweep.get("status") != "COMPLETE":
        raise IncompleteArtifactError(f"{source_name} status must be COMPLETE")
    if sweep.get("dirty") is not False:
        raise DirtyArtifactError(f"{source_name} must explicitly declare dirty=false")
    if sweep.get("head_count") != HEAD_COUNT:
        raise HeadSchemaError(f"{source_name} must declare head_count={HEAD_COUNT}")
    directions = tuple(sweep.get("directions", ()))
    if directions != PAIR_DIRECTIONS:
        raise PairSchemaError(f"{source_name} directions must be exactly {PAIR_DIRECTIONS!r}")
    heads = sweep.get("heads")
    if not isinstance(heads, list) or len(heads) != HEAD_COUNT:
        raise HeadSchemaError(f"{source_name} must contain exactly {HEAD_COUNT} head rows")
    return heads


def _require_fresh_sweep_provenance(sweep: Mapping[str, Any], source_name: str) -> dict[str, str]:
    if sweep.get("measurement_origin") != "fresh_same_invocation":
        raise ArtifactError(f"{source_name} must be measured fresh in the selection invocation")
    if sweep.get("model_snapshot_status") != FRESH_SWEEP_SNAPSHOT_STATUS:
        raise ArtifactError(f"{source_name} lacks the frozen fresh-snapshot completion status")
    fields = (
        "invocation_id",
        "model_state_sha256",
        "normalized_config_sha256",
        "tokenizer_assets_sha256",
        "clean_base_cache_sha256",
        "local_snapshot_revisions_sha256",
        "source_cache_sha256",
    )
    result: dict[str, str] = {}
    for field in fields:
        value = sweep.get(field)
        if not _is_sha256_hex(value):
            raise HashMismatchError(f"{source_name}.{field} must be one exact lowercase SHA-256")
        result[field] = value
    return result


def validate_fresh_sweep_bindings(
    selection_true: Mapping[str, Any],
    selection_a: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Bind both fresh sweeps to every validated top-level snapshot hash."""

    if not isinstance(provenance, Mapping):
        raise ArtifactError("selection provenance must be an object")
    true_fields = _require_fresh_sweep_provenance(selection_true, "true_sweep")
    source_a_fields = _require_fresh_sweep_provenance(selection_a, "source_a_sweep")
    invocation_id = provenance.get("invocation_id")
    fingerprints = provenance.get("model_state_fingerprints")
    before_state = fingerprints.get("before_sweeps") if isinstance(fingerprints, Mapping) else None
    after_true_state = fingerprints.get("after_true_sweep") if isinstance(fingerprints, Mapping) else None
    after_a_state = fingerprints.get("after_source_a_sweep") if isinstance(fingerprints, Mapping) else None
    config = provenance.get("normalized_model_config")
    config_checks = provenance.get("normalized_model_config_checks")
    tokenizer = provenance.get("tokenizer_assets")
    tokenizer_checks = provenance.get("tokenizer_asset_checks")
    cache = provenance.get("immutable_clean_base_cache")
    cache_before = cache.get("before_sweeps") if isinstance(cache, Mapping) else None
    model = provenance.get("model")
    revision_checks = provenance.get("local_snapshot_revision_checks")
    environment = provenance.get("environment")
    expected = {
        "invocation_id": invocation_id,
        "model_state_sha256": before_state.get("sha256") if isinstance(before_state, Mapping) else None,
        "normalized_config_sha256": config.get("sha256") if isinstance(config, Mapping) else None,
        "tokenizer_assets_sha256": tokenizer.get("aggregate_sha256") if isinstance(tokenizer, Mapping) else None,
        "clean_base_cache_sha256": cache_before.get("sha256") if isinstance(cache_before, Mapping) else None,
        "local_snapshot_revisions_sha256": model.get("local_snapshot_revisions_sha256") if isinstance(model, Mapping) else None,
    }
    if any(not _is_sha256_hex(value) for value in expected.values()):
        raise HashMismatchError("selection provenance lacks a required canonical snapshot hash")
    if (
        not isinstance(environment, Mapping)
        or not _is_sha256_hex(environment.get("sha256"))
        or provenance.get("runtime_environment_fingerprint") != environment.get("sha256")
    ):
        raise HashMismatchError("selection runtime environment fingerprint is missing or unbound")
    for label, fields in (("true_sweep", true_fields), ("source_a_sweep", source_a_fields)):
        for name, expected_value in expected.items():
            if fields.get(name) != expected_value:
                raise HashMismatchError(f"{label}.{name} is not bound to selection provenance")
    if (
        not isinstance(model, Mapping)
        or model.get("dtype") != "float32"
        or provenance.get("activation_dtype") != "float32"
        or selection_true.get("activation_dtype") != "float32"
        or selection_a.get("activation_dtype") != "float32"
    ):
        raise ArtifactError("fresh sweep activation dtype is not bound to provenance float32")
    local_snapshots = model.get("local_snapshot_revisions")
    if (
        not isinstance(local_snapshots, Mapping)
        or not isinstance(local_snapshots.get("gpt2"), Mapping)
        or not isinstance(local_snapshots.get("sae"), Mapping)
        or set(local_snapshots) != {"gpt2", "sae"}
        or model.get("local_snapshot_revisions_sha256") != sha256_json(dict(local_snapshots))
        or model.get("sae_loaded") is not False
        or model.get("sae_revision_fingerprint_present") is not True
        or provenance.get("local_model_revision") != local_snapshots["gpt2"].get("observed_revision")
        or provenance.get("local_sae_revision") != local_snapshots["sae"].get("observed_revision")
        or model.get("local_model_revision") != provenance.get("local_model_revision")
        or model.get("local_sae_revision") != provenance.get("local_sae_revision")
    ):
        raise HashMismatchError("selection direct GPT-2/SAE revision fields are not bound to the snapshot fingerprint")
    revision_hash = expected["local_snapshot_revisions_sha256"]
    if (
        not isinstance(revision_checks, Mapping)
        or revision_checks.get("all_exact_match") is not True
        or revision_checks.get("before_sweeps_sha256") != revision_hash
        or revision_checks.get("after_true_sweep_sha256") != revision_hash
        or revision_checks.get("after_source_a_sweep_sha256") != revision_hash
    ):
        raise HashMismatchError("selection GPT-2/SAE revision hashes are not stable across both sweeps")
    model_hash = expected["model_state_sha256"]
    if (
        not isinstance(fingerprints, Mapping)
        or fingerprints.get("all_exact_match") is not True
        or not isinstance(after_true_state, Mapping)
        or not isinstance(after_a_state, Mapping)
        or after_true_state.get("sha256") != model_hash
        or after_a_state.get("sha256") != model_hash
    ):
        raise HashMismatchError("selection model-state before/after hashes are not one exact sweep binding")
    config_hash = expected["normalized_config_sha256"]
    if (
        not isinstance(config_checks, Mapping)
        or config_checks.get("all_exact_match") is not True
        or config_checks.get("before_sweeps_sha256") != config_hash
        or config_checks.get("after_true_sweep_sha256") != config_hash
        or config_checks.get("after_source_a_sweep_sha256") != config_hash
    ):
        raise HashMismatchError("selection config before/after hashes are not one exact sweep binding")
    tokenizer_hash = expected["tokenizer_assets_sha256"]
    if (
        not isinstance(tokenizer_checks, Mapping)
        or tokenizer_checks.get("all_exact_match") is not True
        or tokenizer_checks.get("before_sweeps_sha256") != tokenizer_hash
        or tokenizer_checks.get("after_true_sweep_sha256") != tokenizer_hash
        or tokenizer_checks.get("after_source_a_sweep_sha256") != tokenizer_hash
    ):
        raise HashMismatchError("selection tokenizer before/after hashes are not one exact sweep binding")
    cache_hash = expected["clean_base_cache_sha256"]
    if (
        not isinstance(cache, Mapping)
        or cache.get("all_exact_match") is not True
        or cache.get("after_true_sweep_sha256") != cache_hash
        or cache.get("after_source_a_sweep_sha256") != cache_hash
    ):
        raise HashMismatchError("selection clean-cache before/after hashes are not one exact sweep binding")


def _head_rows_by_id(rows: Sequence[Mapping[str, Any]], source_name: str) -> dict[HeadId, Mapping[str, Any]]:
    result: dict[HeadId, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise HeadSchemaError(f"{source_name} head row {index} is not an object")
        try:
            current = HeadId(row["layer"], row["head"])
        except KeyError as exc:
            raise HeadSchemaError(f"{source_name} head row {index} lacks layer/head") from exc
        if current in result:
            raise HeadSchemaError(f"{source_name} repeats {current}")
        result[current] = row
    missing = set(canonical_heads()) - set(result)
    if missing:
        raise HeadSchemaError(f"{source_name} is missing canonical heads: {sorted(missing)}")
    return result


def _row_pairs(
    row: Mapping[str, Any], source_name: str, head: HeadId
) -> tuple[tuple[Hashable, ...], np.ndarray]:
    records = row.get("pair_records")
    if not isinstance(records, list):
        raise PairSchemaError(
            f"{source_name} {head} must expose scalar pair_records; separate parallel arrays are not accepted"
        )
    grouped = group_pair_records(records, directions=PAIR_DIRECTIONS, effect_key="effect")
    pair_ids = tuple(record.pair_id for record in grouped)
    return pair_ids, paired_effect_array(grouped)


def _frozen_equal(observed: Any, expected: Any) -> bool:
    """Compare one protocol value without Python's bool/int coercion."""

    if isinstance(expected, list):
        return isinstance(observed, list) and len(observed) == len(expected) and all(
            _frozen_equal(left, right) for left, right in zip(observed, expected)
        )
    return type(observed) is type(expected) and observed == expected


def _require_nested_fields(observed: Any, expected: Mapping[str, Any], label: str) -> None:
    """Require every registered nested field with bool/int-safe equality."""

    if not isinstance(observed, Mapping):
        raise ProtocolError(f"{label} is missing")
    for name, expected_value in expected.items():
        field_label = f"{label}.{name}"
        observed_value = observed.get(name)
        if isinstance(expected_value, Mapping):
            _require_nested_fields(observed_value, expected_value, field_label)
        elif not _frozen_equal(observed_value, expected_value):
            raise ProtocolError(f"{field_label} must equal frozen value {expected_value!r}")


def _require_protocol(protocol: Mapping[str, Any]) -> str:
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ProtocolError(f"protocol must declare schema {PROTOCOL_SCHEMA!r}")
    if not _frozen_equal(protocol.get("version"), 1):
        raise ProtocolError("protocol version must be exactly 1")
    if protocol.get("status") not in {"designed_not_executed", "FROZEN", "COMPLETE"}:
        raise ProtocolError("protocol status is not an admissible frozen-design status")
    design_freeze = protocol.get("design_freeze")
    _require_nested_fields(
        design_freeze,
        {
            "latest_amendment": 9,
            "preserved_amendments": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        "protocol.design_freeze",
    )
    head_universe = protocol.get("head_universe")
    if not isinstance(head_universe, Mapping):
        raise ProtocolError("protocol.head_universe is missing")
    nested_fixed = (("layer_count", LAYER_COUNT), ("heads_per_layer", HEADS_PER_LAYER), ("total_heads", HEAD_COUNT))
    for name, expected in nested_fixed:
        if not _frozen_equal(head_universe.get(name), expected):
            raise ProtocolError(f"protocol.head_universe.{name} must equal frozen value {expected!r}")
    expected_head_labels = {
        "layer_ids": "0..11",
        "head_ids": "0..11",
        "flat_test_id": "12 * layer + head + 1",
    }
    for name, expected in expected_head_labels.items():
        if not _frozen_equal(head_universe.get(name), expected):
            raise ProtocolError(f"protocol.head_universe.{name} must equal frozen value {expected!r}")
    model = protocol.get("model")
    if not isinstance(model, Mapping):
        raise ProtocolError("protocol.model is missing")
    expected_model = {
        "name": "gpt2-small",
        "mechanism_library": "TransformerLens",
        "activation_dtype": "float32",
        "sae": "res-jb",
        "sae_layer": 8,
        "residual_width": 768,
    }
    for name, expected in expected_model.items():
        if not _frozen_equal(model.get(name), expected):
            raise ProtocolError(f"protocol.model.{name} must equal frozen value {expected!r}")
    revisions = model.get("expected_local_snapshot_revisions")
    expected_revisions = {
        "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "sae": "57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9",
        "interpretation": "expected local snapshot revisions; not verified blob hashes",
    }
    if not isinstance(revisions, Mapping):
        raise ProtocolError("protocol.model.expected_local_snapshot_revisions is missing")
    for name, expected in expected_revisions.items():
        if not _frozen_equal(revisions.get(name), expected):
            raise ProtocolError(
                f"protocol.model.expected_local_snapshot_revisions.{name} must equal frozen value {expected!r}"
            )
    seeds = protocol.get("seeds")
    if not isinstance(seeds, Mapping) or not _frozen_equal(
        seeds.get("stage1_and_source_a_selection"), STAGE1_SELECTION_SEED
    ):
        raise ProtocolError(f"protocol.seeds.stage1_and_source_a_selection must equal {STAGE1_SELECTION_SEED}")
    stimuli = protocol.get("stimuli")
    if not isinstance(stimuli, Mapping):
        raise ProtocolError("protocol.stimuli is missing")
    expected_stimuli = {
        "generator": "experiment_04_templates_and_gate_a",
        "family": "single_flip_minimal_pairs",
        "template": "The {ADJ} {SUBJ} {PREP} the {ATTRACTOR} ___",
        "requested_pairs": REQUESTED_PAIR_COUNT,
        "attractor": "fixed_and_counterbalanced",
        "subject_token_rule": "last_subword_if_multi_token",
        "directed_edits_per_pair": 2,
        "pair_cluster_rule": "both_directed_edits_stay_together",
    }
    for name, expected in expected_stimuli.items():
        if not _frozen_equal(stimuli.get(name), expected):
            raise ProtocolError(f"protocol.stimuli.{name} must equal frozen value {expected!r}")
    readout = protocol.get("readout")
    expected_readout = {
        "name": "d",
        "definition": 'logit(" are") - logit(" is") at the final position',
        "delta_definition": "sign-aligned source-minus-base change in d",
        "model_readout": "native_unembedding",
        "fitted_readout": False,
    }
    if not isinstance(readout, Mapping):
        raise ProtocolError("protocol.readout is missing")
    for name, expected in expected_readout.items():
        if not _frozen_equal(readout.get(name), expected):
            raise ProtocolError(f"protocol.readout.{name} must equal frozen value {expected!r}")
    gate_a = protocol.get("gate_a")
    gate_conditions = gate_a.get("conditions") if isinstance(gate_a, Mapping) else None
    expected_gate_conditions = {
        "both_members_signed_correct_fraction_at_least": 0.6,
        "minimum_retained_pairs": MIN_RETAINED_PAIRS,
        "median_clean_d_gap_at_least": 1.0,
    }
    if not isinstance(gate_conditions, Mapping):
        raise ProtocolError("protocol.gate_a.conditions is missing")
    for name, expected in expected_gate_conditions.items():
        if not _frozen_equal(gate_conditions.get(name), expected):
            raise ProtocolError(f"protocol.gate_a.conditions.{name} must equal frozen value {expected!r}")
    sanity_floor = gate_a.get("sanity_floor") if isinstance(gate_a, Mapping) else None
    expected_sanity_floor = {
        "layer": 8,
        "position_set": "both",
        "full_residual_E_over_d_gap_at_least": 0.5,
        "sign_consistency_at_least": 0.9,
        "role": "stop_condition_only_not_a_verdict",
    }
    if not isinstance(sanity_floor, Mapping) or any(
        not _frozen_equal(sanity_floor.get(name), expected)
        for name, expected in expected_sanity_floor.items()
    ):
        raise ProtocolError("protocol.gate_a.sanity_floor differs from the frozen stop rule")
    pair_rules = protocol.get("pair_rules")
    if not isinstance(pair_rules, Mapping):
        raise ProtocolError("protocol.pair_rules is missing")
    expected_pair_rules = {
        "resampling_unit": "complete_retained_minimal_pair_cluster",
        "directed_edits_kept_together": True,
        "directional_sign_rule": "both sign-aligned directions must be strictly positive; zero is inconsistent",
    }
    for name, expected in expected_pair_rules.items():
        if not _frozen_equal(pair_rules.get(name), expected):
            raise ProtocolError(f"protocol.pair_rules.{name} must equal frozen value {expected!r}")
    sources = protocol.get("sources")
    if not isinstance(sources, Mapping) or not _frozen_equal(
        sources.get("A"), "same-number_different-subject-noun"
    ):
        raise ProtocolError("protocol.sources.A differs from the frozen source-A construction")
    stage1 = protocol.get("stage1")
    if not isinstance(stage1, Mapping):
        raise ProtocolError("protocol.stage1 is missing")
    if not _frozen_equal(stage1.get("seed"), STAGE1_SELECTION_SEED):
        raise ProtocolError(f"protocol.stage1.seed must equal {STAGE1_SELECTION_SEED}")
    true_sweep = stage1.get("true_source_sweep")
    source_a_supplement = stage1.get("source_a_selection_supplement")
    expected_sweep = {"head_count": HEAD_COUNT, "hook": "hook_z", "position": "final"}
    for label, sweep in (("true_source_sweep", true_sweep), ("source_a_selection_supplement", source_a_supplement)):
        if not isinstance(sweep, Mapping):
            raise ProtocolError(f"protocol.stage1.{label} is missing")
        for name, expected in expected_sweep.items():
            if not _frozen_equal(sweep.get(name), expected):
                raise ProtocolError(f"protocol.stage1.{label}.{name} must equal frozen value {expected!r}")
    if source_a_supplement.get("same_retained_base_pairs") is not True:
        raise ProtocolError("protocol.stage1.source_a_selection_supplement.same_retained_base_pairs must be true")
    selection = protocol.get("selection")
    expected_selection = {
        "candidate_source": "fresh_true_source_and_fresh_source_A_same_snapshot_only",
        "required_sweeps": {
            "fresh_true_source": {
                "head_count": HEAD_COUNT,
                "hook": "hook_z",
                "position": "final",
                "source": "true number-flip",
                "status_required": "COMPLETE",
            },
            "fresh_source_A": {
                "head_count": HEAD_COUNT,
                "hook": "hook_z",
                "position": "final",
                "source": "source-A same-number different-noun",
                "status_required": "COMPLETE",
            },
        },
        "same_snapshot_requirements": {
            "same_runner_invocation": True,
            "same_in_memory_model": True,
            "same_model_config": True,
            "same_tokenizer": True,
            "same_activation_dtype": True,
            "same_immutable_clean_base_cache": True,
            "same_retained_directed_edits_and_head_order": True,
        },
        "state_dict_fingerprint": {
            "before_after_required": True,
            "before_after_sha256_must_match": True,
            "scheme": "lexicographic state_dict keys; key/dtype/shape JSON plus uncast contiguous tensor bytes; uint64 length framing",
            "uncast_bytes": True,
            "record_fields": [
                "state_dict_sha256_before",
                "state_dict_sha256_after",
                "state_dict_key_count",
                "state_dict_entry_metadata",
            ],
            "missing_or_changed_status": "SELECTION_SNAPSHOT_EXECUTION_INCOMPLETE",
        },
        "required_snapshot_fingerprints": {
            "normalized_model_config_sha256": True,
            "tokenizer_asset_hashes": True,
            "local_model_and_sae_revision": True,
            "repository_commit": True,
            "runtime_environment_fingerprint": True,
            "activation_dtype": True,
            "immutable_clean_base_cache_fingerprint": True,
            "record_fields": [
                "normalized_model_config_sha256",
                "tokenizer_asset_hashes",
                "local_model_revision",
                "local_sae_revision",
                "repository_commit",
                "runtime_environment_fingerprint",
                "activation_dtype",
                "immutable_clean_base_cache_sha256",
            ],
        },
        "historical_stage1_crosscheck": {
            "role": "descriptive_non_blocking_only",
            "top10_membership_and_order_recorded": True,
            "fresh_top10_membership_and_order_recorded": True,
            "membership_and_order_overlap_recorded": True,
            "record_fields": [
                "fresh_top10_membership",
                "fresh_top10_order",
                "shipped_stage1_top10_membership",
                "shipped_stage1_top10_order",
                "membership_overlap",
                "order_overlap",
            ],
            "divergence_blocks_fresh_selection": False,
            "may_contribute_to_C": False,
            "additional_logical_forward_equivalents": 0,
            "snapshot_status": "UNVERIFIED_FROM_SHIPPED_STAGE1_ARTIFACT",
        },
        "logical_forward_equivalents": {
            "total": 291,
            "clean": 1,
            "fresh_true_source_cache": 1,
            "fresh_source_A_cache": 1,
            "fresh_true_source_144_heads": 144,
            "fresh_source_A_144_heads": 144,
            "historical_stage1_crosscheck": 0,
            "definition": "1 clean + 1 true cache + 1 source-A cache + 144 fresh true heads + 144 fresh source-A heads",
        },
        "status_codes": {
            "ready": FRESH_SWEEP_SNAPSHOT_STATUS,
            "provenance_ready": "READY",
            "technical_incomplete": "SELECTION_SNAPSHOT_EXECUTION_INCOMPLETE",
            "candidate_not_constructed": "C_NOT_CONSTRUCTED_SAME_SNAPSHOT",
            "q1_no_verdict": "BLOCKED",
            "q2_q3_no_verdict": "NOT_INSTANTIATED_UNRESOLVED_C",
        },
        "failure_boundary": {
            "missing_or_changed_fingerprint": "technical_incomplete",
            "either_fresh_sweep_incomplete": "technical_incomplete",
            "C_constructed_only_when": "same snapshot status ready and both fresh sweeps complete",
            "q4_independent": True,
        },
    }
    _require_nested_fields(selection, expected_selection, "protocol.selection")
    bootstrap = protocol.get("bootstrap_and_multiple_testing")
    if not isinstance(bootstrap, Mapping):
        raise ProtocolError("protocol.bootstrap_and_multiple_testing is missing")
    if not _frozen_equal(bootstrap.get("per_head_resamples"), BOOTSTRAP_DRAWS) or not _frozen_equal(
        bootstrap.get("other_stage2_interval_resamples"), CI_DRAWS
    ):
        raise ProtocolError("protocol bootstrap draw counts differ from the frozen 100000/10000 values")
    expected_bootstrap = {
        "interval_type": "two-sided_percentile",
        "percentiles": [2.5, 97.5],
        "percentile_method": 'numpy.percentile(method="linear")',
        "resampling_unit": "retained_minimal_pair_cluster",
        "directed_edits_and_arms_resampled_together": True,
        "per_head_p_value": "min(1, 2 * min((n_mu_bootstrap_le_0 + 1)/(B + 1), (n_mu_bootstrap_ge_0 + 1)/(B + 1)))",
    }
    for name, expected in expected_bootstrap.items():
        if not _frozen_equal(bootstrap.get(name), expected):
            raise ProtocolError(f"protocol.bootstrap_and_multiple_testing.{name} differs from the frozen rule")
    holm = bootstrap.get("holm")
    expected_holm = {
        "alpha": HOLM_ALPHA,
        "method": "Holm step-down",
        "family": "144 heads within each seed",
        "raw_statistic": "head raw mean delta_d; E_all division is effect-size reporting only",
    }
    if not isinstance(holm, Mapping) or any(
        not _frozen_equal(holm.get(name), expected) for name, expected in expected_holm.items()
    ):
        raise ProtocolError("protocol Holm method/family/alpha/statistic differs from the frozen rule")
    rng_spec = protocol.get("rng")
    expected_rng = {
        "generator": "numpy.random.default_rng",
        "seed_formula": "experiment_seed * 1000 + test_id",
        "iteration_order_derived_seed_forbidden": True,
    }
    if not isinstance(rng_spec, Mapping) or any(
        not _frozen_equal(rng_spec.get(name), expected) for name, expected in expected_rng.items()
    ):
        raise ProtocolError("protocol RNG generator/formula/order rule differs from the frozen rule")
    test_ids = protocol.get("test_ids")
    head_test_ids = test_ids.get("head_sweep") if isinstance(test_ids, Mapping) else None
    if not isinstance(head_test_ids, Mapping) or not _frozen_equal(
        head_test_ids.get("ids"), "1..144"
    ) or not _frozen_equal(head_test_ids.get("mapping"), "12 * layer + head + 1"):
        raise ProtocolError("protocol head-sweep test-id mapping differs from the frozen rule")
    candidate_rule = protocol.get("stage2", {}).get("candidate_rule") if isinstance(protocol.get("stage2"), Mapping) else None
    if not isinstance(candidate_rule, Mapping):
        raise ProtocolError("protocol.stage2.candidate_rule is missing")
    expected_candidate_rule = {
        "source_seed": STAGE1_SELECTION_SEED,
        "rank_source": "fresh_true_source signed mean delta_d descending from same-snapshot selection invocation",
        "top_rank_count": STAGE1_TOP_K,
        "tie_break": "layer then head ascending",
        "requires_source_a_noise_edge": True,
        "requires_true_source_holm_distinguishability": True,
        "requires_pair_level_sign_consistency_at_least": PAIR_SIGN_CONSISTENCY_MIN,
        "pool": "C = fresh same-snapshot true/source-A top-10 heads satisfying all registered requirements",
    }
    for name, expected in expected_candidate_rule.items():
        if not _frozen_equal(candidate_rule.get(name), expected):
            raise ProtocolError(f"protocol.stage2.candidate_rule.{name} differs from the frozen rule")
    noise_edge = candidate_rule.get("source_a_noise_edge")
    if (
        not isinstance(noise_edge, Mapping)
        or not _frozen_equal(noise_edge.get("values"), "absolute source-A mean delta_d for all 144 heads")
        or not _frozen_equal(noise_edge.get("quantile"), 0.99)
        or not _frozen_equal(noise_edge.get("method"), 'numpy.percentile(values, 99, method="linear")')
    ):
        raise ProtocolError("protocol source-A noise edge differs from frozen linear P99 rule")
    if not isinstance(test_ids, Mapping) or not _frozen_equal(
        test_ids.get("q3_true_minus_source_a_interval"), Q3_TEST_ID
    ) or not _frozen_equal(test_ids.get("q4_matched_subset_draw"), Q4_TEST_ID):
        raise ProtocolError("protocol Q3/Q4 test ids differ from frozen 301/401 values")
    q1 = protocol.get("q1")
    nested = q1.get("nested_sets") if isinstance(q1, Mapping) else None
    expected_nested = {
        "order": "fresh same-snapshot true-source signed rank order within C",
        "sizes": "1..min(8, |C|)",
        "membership_fixed_before_joint_stage2_effects": True,
    }
    if not isinstance(nested, Mapping) or any(
        not _frozen_equal(nested.get(name), expected) for name, expected in expected_nested.items()
    ):
        raise ProtocolError("protocol Q1 nested-set construction differs from the frozen candidate rule")
    q4 = protocol.get("q4")
    statistics = q4.get("statistics") if isinstance(q4, Mapping) else None
    guard = statistics.get("denominator_guard") if isinstance(statistics, Mapping) else None
    if not isinstance(guard, Mapping):
        raise ProtocolError("protocol.q4.statistics.denominator_guard is missing")
    expected_guard = {
        "D": "E(delta)",
        "M": "max(1, E|delta|, E|delta_span|, E|delta_comp|)",
        "tau": "sqrt(float64_eps) * M",
        "condition": "abs(D) <= tau",
        "status": "NON_ESTIMABLE_DENOMINATOR",
    }
    for name, expected in expected_guard.items():
        if not _frozen_equal(guard.get(name), expected):
            raise ProtocolError(f"protocol denominator guard field {name!r} differs from Amendment-5")
    return sha256_json(protocol)


def validate_protocol(protocol: Mapping[str, Any]) -> str:
    """Fail closed unless every operative frozen protocol field is exact."""

    return _require_protocol(protocol)


def nested_sets(rank_order: Sequence[HeadId], max_size: int = MAX_NESTED_SET_SIZE) -> tuple[tuple[HeadId, ...], ...]:
    """Return rank-ordered prefixes ``S_1 ... S_min(8, |C|)``."""

    if type(max_size) is not int or max_size != MAX_NESTED_SET_SIZE:
        raise ProtocolError(f"nested-set maximum is frozen at {MAX_NESTED_SET_SIZE}")
    order = tuple(rank_order)
    if any(not isinstance(head, HeadId) for head in order) or len(set(order)) != len(order):
        raise HeadSchemaError("nested-set rank order must contain unique HeadId objects")
    return tuple(order[:index] for index in range(1, min(MAX_NESTED_SET_SIZE, len(order)) + 1))


def construct_candidate(
    selection_true: Mapping[str, Any],
    selection_a: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    selection_provenance: Mapping[str, Any] | None = None,
) -> CandidateResult:
    """Construct C mechanically from two COMPLETE canonical 144-head sweeps.

    ``selection_true`` and ``selection_a`` are separate ``stage_sweep.v1``
    objects.  The caller must not pass the shipped Stage-1 artifact directly: it
    lacks scalar ``pair_records`` and the required source-A sweep, so this function
    rejects it rather than inferring or fabricating the missing artifact.
    """

    protocol_hash = _require_protocol(protocol)
    true_rows = _head_rows_by_id(_require_complete_sweep(selection_true, "true_sweep"), "true_sweep")
    a_rows = _head_rows_by_id(_require_complete_sweep(selection_a, "source_a_sweep"), "source_a_sweep")
    true_provenance = _require_fresh_sweep_provenance(selection_true, "true_sweep")
    source_a_provenance = _require_fresh_sweep_provenance(selection_a, "source_a_sweep")
    if selection_provenance is not None:
        validate_fresh_sweep_bindings(selection_true, selection_a, selection_provenance)
    for field in (
        "invocation_id",
        "model_state_sha256",
        "normalized_config_sha256",
        "tokenizer_assets_sha256",
        "clean_base_cache_sha256",
        "local_snapshot_revisions_sha256",
    ):
        if true_provenance[field] != source_a_provenance[field]:
            raise HashMismatchError(f"fresh true/source-A sweeps disagree on {field}")
    if selection_true.get("seed") != STAGE1_SELECTION_SEED or selection_a.get("seed") != STAGE1_SELECTION_SEED:
        raise ProtocolError(f"both selection sweeps must use seed {STAGE1_SELECTION_SEED}")
    if selection_true.get("source") != "true" or selection_a.get("source") != "source_a":
        raise ArtifactError("sweep sources must be exactly 'true' and 'source_a'")

    true_values: dict[HeadId, np.ndarray] = {}
    source_a_values: dict[HeadId, np.ndarray] = {}
    true_means: dict[HeadId, float] = {}
    source_a_means: dict[HeadId, float] = {}
    pair_scores: dict[HeadId, float] = {}
    p_values: dict[int, float] = {}
    frozen_pair_ids: tuple[Hashable, ...] | None = None
    for head in canonical_heads():
        true_pair_ids, true_array = _row_pairs(true_rows[head], "true_sweep", head)
        source_a_pair_ids, a_array = _row_pairs(a_rows[head], "source_a_sweep", head)
        if true_pair_ids != source_a_pair_ids:
            raise PairSchemaError(f"true/source-A pair ids differ for {head}")
        if frozen_pair_ids is None:
            frozen_pair_ids = true_pair_ids
        elif true_pair_ids != frozen_pair_ids:
            raise PairSchemaError(
                f"{head} does not use the same retained pair ids as the other 144-head sweep rows"
            )
        if true_array.shape != a_array.shape:
            raise PairSchemaError(f"true/source-A pair shapes differ for {head}: {true_array.shape} vs {a_array.shape}")
        true_values[head] = true_array
        source_a_values[head] = a_array
        true_means[head] = float(true_array.mean())
        source_a_means[head] = float(a_array.mean())
        pair_scores[head] = pair_sign_consistency(true_array)
        p_values[head.flat_id] = bootstrap_p_value(true_array, STAGE1_SELECTION_SEED, head.flat_id)

    if frozen_pair_ids is None or len(frozen_pair_ids) < MIN_RETAINED_PAIRS:
        observed = 0 if frozen_pair_ids is None else len(frozen_pair_ids)
        raise PairSchemaError(
            f"selection sweeps contain {observed} retained pairs; protocol requires at least {MIN_RETAINED_PAIRS}"
        )
    if any(type(pair_id) is not int or not 0 <= pair_id < REQUESTED_PAIR_COUNT for pair_id in frozen_pair_ids):
        raise PairSchemaError(f"selection pair ids must be plain integers in 0..{REQUESTED_PAIR_COUNT - 1}")

    source_a_edge = linear_percentile(np.abs(np.asarray(tuple(source_a_means.values()), dtype=np.float64)), SOURCE_A_PERCENTILE)
    holm = holm_step_down(p_values, HOLM_ALPHA)
    ranked = tuple(sorted(canonical_heads(), key=lambda head: (-true_means[head], head.layer, head.head)))
    top_ranked = ranked[:STAGE1_TOP_K]
    evidence: list[CandidateEvidence] = []
    eligible: list[HeadId] = []
    for rank, head in enumerate(top_ranked, start=1):
        reasons: list[str] = []
        if pair_scores[head] < PAIR_SIGN_CONSISTENCY_MIN:
            reasons.append("pair_sign_consistency_below_0.90")
        if abs(true_means[head]) <= source_a_edge:
            reasons.append("true_abs_mean_not_above_source_a_p99_edge")
        if not holm[head.flat_id]:
            reasons.append("holm_not_rejected")
        is_eligible = not reasons
        if is_eligible:
            eligible.append(head)
        evidence.append(
            CandidateEvidence(
                head=head,
                stage1_rank=rank,
                true_mean=true_means[head],
                source_a_mean=source_a_means[head],
                source_a_noise_edge=source_a_edge,
                pair_sign_consistency=pair_scores[head],
                bootstrap_p_value=p_values[head.flat_id],
                holm_reject=holm[head.flat_id],
                eligible=is_eligible,
                reasons=tuple(reasons),
            )
        )
    rank_order = tuple(eligible)
    candidate_heads = tuple(sorted(eligible, key=lambda head: head.flat_id))
    return CandidateResult(
        candidate_heads=candidate_heads,
        rank_order=rank_order,
        nested_sets_by_rank=nested_sets(rank_order),
        evidence=tuple(evidence),
        status="NONEMPTY" if candidate_heads else "EMPTY_UNDER_FROZEN_RULE",
        protocol_sha256=protocol_hash,
        true_sweep_sha256=sha256_json(selection_true),
        source_a_sweep_sha256=sha256_json(selection_a),
    )


@dataclass(frozen=True)
class Q3Result:
    true_mean: float
    source_a_mean: float
    difference_mean: float
    difference_ci: tuple[float, float]
    subject_value_transport_shown: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "E_right_path": self.true_mean,
            "F_path_source_A": self.source_a_mean,
            "D_path": self.difference_mean,
            "D_path_percentile_95_ci": list(self.difference_ci),
            "subject_value_transport_shown": self.subject_value_transport_shown,
            "test_id": Q3_TEST_ID,
        }


def q3_true_minus_a(
    true_values_2d: Any,
    source_a_values_2d: Any,
    *,
    seed: int,
    test_id: int = Q3_TEST_ID,
    resamples: int = CI_DRAWS,
) -> Q3Result:
    """Apply Amendment-3's signed true-minus-source-A Q3 rule."""

    if test_id != Q3_TEST_ID:
        raise ProtocolError(f"Q3 test id is frozen at {Q3_TEST_ID}")
    true_values = pair_clusters(true_values_2d)
    source_a_values = pair_clusters(source_a_values_2d)
    if true_values.shape != source_a_values.shape:
        raise PairSchemaError("Q3 true and source-A arrays must share pair ids and shape")
    difference = true_values - source_a_values
    difference_mean = float(difference.mean())
    ci = percentile_ci(difference, seed, Q3_TEST_ID, resamples)
    return Q3Result(
        true_mean=float(true_values.mean()),
        source_a_mean=float(source_a_values.mean()),
        difference_mean=difference_mean,
        difference_ci=ci,
        subject_value_transport_shown=bool(difference_mean > 0.0 and ci[0] > 0.0),
    )


@dataclass(frozen=True)
class DecoderRowProjector:
    """Float64 thin-SVD projector onto twelve decoder rows."""

    right_singular_rows: np.ndarray
    rank: int
    tolerance: float
    input_shape: tuple[int, int]

    def project(self, delta: Any) -> np.ndarray:
        vector = finite_array(delta, "decoder delta").reshape(-1)
        if vector.size != self.input_shape[1]:
            raise ProtocolError(f"delta must have width {self.input_shape[1]}, got {vector.size}")
        return self.right_singular_rows.T @ (self.right_singular_rows @ vector)

    def complement(self, delta: Any) -> np.ndarray:
        vector = finite_array(delta, "decoder delta").reshape(-1)
        return vector - self.project(vector)


def decoder_row_projector(decoder_rows: Any) -> DecoderRowProjector:
    """Construct the Amendment-3 float64 projector and require numerical rank 12."""

    rows = finite_array(decoder_rows, "decoder rows", ndim=2)
    if rows.shape[0] != Q4_SUBSET_SIZE:
        raise ProtocolError(f"target decoder rows must have exactly {Q4_SUBSET_SIZE} rows, got {rows.shape}")
    if rows.shape[1] < Q4_SUBSET_SIZE:
        raise ProtocolError("decoder row width must be at least the target rank")
    _, singular_values, vh = np.linalg.svd(rows, full_matrices=False)
    s_max = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(rows.shape) * np.finfo(np.float64).eps * s_max
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != Q4_SUBSET_SIZE:
        raise BlockedError(f"target decoder rows have rank {rank}, expected {Q4_SUBSET_SIZE}")
    basis = np.asarray(vh[:rank, :], dtype=np.float64)
    basis.setflags(write=False)
    return DecoderRowProjector(basis, rank, tolerance, (int(rows.shape[0]), int(rows.shape[1])))


@dataclass(frozen=True)
class Q4DenominatorGate:
    status: str
    denominator: float
    scale: float
    tolerance: float
    ratio_of_means_allowed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "D": self.denominator,
            "M": self.scale,
            "tau": self.tolerance,
            "ratio_of_means_allowed": self.ratio_of_means_allowed,
        }


def q4_denominator_gate(full_effects: Any, span_effects: Any, complement_effects: Any) -> Q4DenominatorGate:
    """Apply Advisor-frozen Amendment-5 denominator gate.

    ``D`` is the mean full-delta per-pair effect.  Let ``M`` be the maximum of one
    and the three mean absolute effect magnitudes; ``tau = sqrt(eps_float64)*M``.
    If ``abs(D) <= tau`` the ratios are non-estimable and no regularization or
    alternate denominator is permitted.  Otherwise the registered statistic is
    ratio-of-means.
    """

    full = finite_array(full_effects, "full effects").reshape(-1)
    span = finite_array(span_effects, "span effects").reshape(-1)
    complement = finite_array(complement_effects, "complement effects").reshape(-1)
    if not (full.size == span.size == complement.size and full.size > 0):
        raise ProtocolError("Q4 full/span/complement effect arrays must share a non-empty length")
    denominator = float(full.mean())
    scale = max(1.0, float(np.abs(full).mean()), float(np.abs(span).mean()), float(np.abs(complement).mean()))
    tolerance = math.sqrt(np.finfo(np.float64).eps) * scale
    estimable = abs(denominator) > tolerance
    return Q4DenominatorGate(
        status="ESTIMABLE" if estimable else "NON_ESTIMABLE_DENOMINATOR",
        denominator=denominator,
        scale=scale,
        tolerance=tolerance,
        ratio_of_means_allowed=estimable,
    )


def q4_ratio_of_means(full_effects: Any, span_effects: Any, complement_effects: Any) -> tuple[float, float, Q4DenominatorGate]:
    """Return ``(R_span, R_comp, gate)`` or block on a non-estimable denominator."""

    gate = q4_denominator_gate(full_effects, span_effects, complement_effects)
    if not gate.ratio_of_means_allowed:
        raise BlockedError("Q4 denominator is NON_ESTIMABLE_DENOMINATOR; no ratio is reported")
    span = finite_array(span_effects, "span effects").reshape(-1)
    complement = finite_array(complement_effects, "complement effects").reshape(-1)
    return float(span.mean() / gate.denominator), float(complement.mean() / gate.denominator), gate


@dataclass(frozen=True)
class EightSeedAggregation:
    """Three-way aggregate with an explicit execution-completeness gate."""

    verdict: str
    pass_count: int
    completed_fail_count: int
    scientific_unresolved_count: int
    execution_incomplete_count: int
    lower_pass_count: int | None
    upper_pass_count: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "pass_count": self.pass_count,
            "completed_fail_count": self.completed_fail_count,
            "scientific_unresolved_count": self.scientific_unresolved_count,
            "execution_incomplete_count": self.execution_incomplete_count,
            "lower_pass_count": self.lower_pass_count,
            "upper_pass_count": self.upper_pass_count,
        }


def aggregate_eight_seed_statuses(statuses: Sequence[str | Mapping[str, Any]]) -> EightSeedAggregation:
    """Aggregate exactly eight per-seed statuses without converting missing data to FAIL.

    The only accepted classes are the four names frozen in Amendment 5.  A
    complete aggregate is ``POSITIVE`` when the lower pass count is at least six,
    ``NEGATIVE`` when the upper pass count is below six, and
    ``INCONCLUSIVE_UNRESOLVED_SEEDS`` otherwise.
    """

    if len(statuses) != 8:
        raise ProtocolError(f"eight-seed aggregation requires exactly 8 statuses, got {len(statuses)}")
    normalized: list[str] = []
    for index, status in enumerate(statuses):
        value = status.get("status") if isinstance(status, Mapping) else status
        if value not in {
            "PASS",
            "COMPLETED_FAIL",
            "SCIENTIFIC_UNRESOLVED",
            "EXECUTION_INCOMPLETE",
        }:
            raise ProtocolError(f"seed {index} has unknown aggregation status {value!r}")
        normalized.append(str(value))
    incomplete = sum(value == "EXECUTION_INCOMPLETE" for value in normalized)
    passes = sum(value == "PASS" for value in normalized)
    completed_failures = sum(value == "COMPLETED_FAIL" for value in normalized)
    unresolved = sum(value == "SCIENTIFIC_UNRESOLVED" for value in normalized)
    if passes + completed_failures + unresolved + incomplete != 8:
        raise ProtocolError("aggregation statuses do not partition all eight seeds")
    if incomplete:
        verdict = "BLOCKED_EXECUTION_INCOMPLETE"
        lower = upper = None
    else:
        lower = passes
        upper = passes + unresolved
        verdict = (
            "POSITIVE"
            if lower >= 6
            else "NEGATIVE"
            if upper < 6
            else "INCONCLUSIVE_UNRESOLVED_SEEDS"
        )
    return EightSeedAggregation(
        verdict=verdict,
        pass_count=passes,
        completed_fail_count=completed_failures,
        scientific_unresolved_count=unresolved,
        execution_incomplete_count=incomplete,
        lower_pass_count=lower,
        upper_pass_count=upper,
    )


@dataclass(frozen=True)
class SplitResult:
    seed: int
    status: str
    assignments: tuple[tuple[Hashable, str], ...]
    rank_training_ids: tuple[Hashable, ...]
    evaluation_ids: tuple[Hashable, ...]
    unused_after_eval_cap_ids: tuple[Hashable, ...]
    block_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.status,
            "assignments": [{"pair_id": pair_id, "role": role} for pair_id, role in self.assignments],
            "rank_training_pair_ids": list(self.rank_training_ids),
            "evaluation_pair_ids": list(self.evaluation_ids),
            "unused_after_eval_cap_pair_ids": list(self.unused_after_eval_cap_ids),
            "block_reason": self.block_reason,
        }


def amendment4_split(pair_ids: Sequence[Hashable], seed: int) -> SplitResult:
    """Create the item-disjoint 40/150 Amendment-4 split.

    Input IDs are sorted canonically before ``random.Random(seed + 701).shuffle``.
    The frozen role table is: ``N<40`` → all training and
    ``BLOCKED_INSUFFICIENT_TRAIN``; ``40<=N<80`` → 40 training and the remainder
    evaluation with ``BLOCKED_INSUFFICIENT_EVAL``; ``80<=N<=190`` → 40 training
    and the remainder evaluation READY; ``N>190`` → 40 training, 150 evaluation,
    and the rest ``unused_after_eval_cap`` READY.  Every input id receives a role,
    including blocked cases.
    """

    if type(seed) is not int or seed < 0:
        raise ProtocolError("split seed must be a non-negative integer")
    ids = list(pair_ids)
    keys = [_stable_key(pair_id) for pair_id in ids]
    if len(set(keys)) != len(ids):
        raise PairSchemaError("Amendment-4 pair ids must be unique")
    ordered = sorted(ids, key=_stable_key)
    generator = random.Random(seed + AMENDMENT4_SPLIT_OFFSET)
    generator.shuffle(ordered)
    count = len(ordered)
    training = tuple(ordered[: min(AMENDMENT4_TRAINING_PAIRS, count)])
    remainder = tuple(ordered[len(training) :])
    evaluation = tuple(remainder[:AMENDMENT4_EVALUATION_PAIRS])
    unused = tuple(remainder[AMENDMENT4_EVALUATION_PAIRS:])
    if count < AMENDMENT4_TRAINING_PAIRS:
        status = "BLOCKED_INSUFFICIENT_TRAIN"
        reason = f"only {count} retained pairs; rank-training requires {AMENDMENT4_TRAINING_PAIRS}"
    elif count < 80:
        status = "BLOCKED_INSUFFICIENT_EVAL"
        reason = f"only {count - AMENDMENT4_TRAINING_PAIRS} held-out pairs; evaluation requires at least 40"
    else:
        status = BlockStatus.READY.value
        reason = None
    assignments = tuple(
        [(pair_id, SplitRole.RANK_TRAINING.value) for pair_id in training]
        + [(pair_id, SplitRole.EVALUATION.value) for pair_id in evaluation]
        + [(pair_id, SplitRole.UNUSED_AFTER_EVAL_CAP.value) for pair_id in unused]
    )
    return SplitResult(seed, status, assignments, training, evaluation, unused, reason)


@dataclass(frozen=True)
class MatchedSubsetResult:
    status: str
    eligible_ids: tuple[int, ...]
    accepted_subsets: tuple[tuple[int, ...], ...]
    rejected_rank_deficient: tuple[dict[str, Any], ...]
    attempts: int
    block_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "eligible_pool_ids": list(self.eligible_ids),
            "accepted_subsets": [list(subset) for subset in self.accepted_subsets],
            "rejected_rank_deficient": list(self.rejected_rank_deficient),
            "attempts": self.attempts,
            "block_reason": self.block_reason,
        }


def _matrix_rank_with_frozen_tolerance(rows: np.ndarray) -> tuple[int, float]:
    _, singular_values, _ = np.linalg.svd(rows, full_matrices=False)
    s_max = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(rows.shape) * np.finfo(np.float64).eps * s_max
    return int(np.count_nonzero(singular_values > tolerance)), tolerance


def sample_matched_subsets(
    candidate_ids: Sequence[int],
    decoder_rows: Mapping[int, Any],
    target_ids: Iterable[int],
    *,
    seed: int,
    subset_size: int = Q4_SUBSET_SIZE,
    accepted_count: int = Q4_ACCEPTED_SUBSETS,
    max_attempts: int = Q4_MAX_ATTEMPTS,
) -> MatchedSubsetResult:
    """Draw 100 full-rank twelve-row matched subsets under Amendment 3/4.

    ``candidate_ids`` must be the already sorted 128-candidate causal pool.  Target
    ids are excluded before sampling.  Every rejected rank-deficient draw is
    recorded, and failure to obtain all 100 within 10,000 attempts is BLOCKED.
    """

    if type(seed) is not int or seed < 0:
        raise ProtocolError("matched-subset seed must be a non-negative integer")
    if subset_size != Q4_SUBSET_SIZE or accepted_count != Q4_ACCEPTED_SUBSETS or max_attempts != Q4_MAX_ATTEMPTS:
        raise ProtocolError("matched-subset size, accepted count, and attempt cap are frozen at 12/100/10000")
    ids = list(candidate_ids)
    if len(ids) != 128 or len(set(ids)) != len(ids):
        raise ProtocolError("matched latent candidate pool must contain exactly 128 unique ids")
    if any(type(latent_id) is not int for latent_id in ids):
        raise ProtocolError("matched latent ids must be plain integers")
    ids = sorted(ids)
    target = set(target_ids)
    if any(type(latent_id) is not int for latent_id in target):
        raise ProtocolError("target latent ids must be plain integers")
    eligible = tuple(latent_id for latent_id in ids if latent_id not in target)
    if len(eligible) < subset_size:
        return MatchedSubsetResult(
            status=BlockStatus.BLOCKED.value,
            eligible_ids=eligible,
            accepted_subsets=tuple(),
            rejected_rank_deficient=tuple(),
            attempts=0,
            block_reason=f"only {len(eligible)} eligible ids remain after target exclusion",
        )
    if any(latent_id not in decoder_rows for latent_id in eligible):
        missing = [latent_id for latent_id in eligible if latent_id not in decoder_rows]
        raise ArtifactError(f"decoder rows missing matched-pool ids: {missing[:8]}")
    row_arrays = {latent_id: finite_array(decoder_rows[latent_id], f"decoder row {latent_id}").reshape(-1) for latent_id in eligible}
    widths = {array.size for array in row_arrays.values()}
    if len(widths) != 1:
        raise ProtocolError("matched decoder rows must have one common width")
    generator = rng(seed, Q4_TEST_ID)
    accepted: list[tuple[int, ...]] = []
    rejected: list[dict[str, Any]] = []
    attempts = 0
    while len(accepted) < accepted_count and attempts < max_attempts:
        attempts += 1
        choices = generator.choice(len(eligible), size=subset_size, replace=False)
        subset = tuple(sorted((eligible[int(index)] for index in choices)))
        rows = np.asarray([row_arrays[latent_id] for latent_id in subset], dtype=np.float64)
        rank, tolerance = _matrix_rank_with_frozen_tolerance(rows)
        if rank < subset_size:
            rejected.append({"attempt": attempts, "ids": list(subset), "rank": rank, "tolerance": tolerance})
            continue
        accepted.append(subset)
    if len(accepted) < accepted_count:
        return MatchedSubsetResult(
            status=BlockStatus.BLOCKED.value,
            eligible_ids=eligible,
            accepted_subsets=tuple(accepted),
            rejected_rank_deficient=tuple(rejected),
            attempts=attempts,
            block_reason=f"accepted {len(accepted)}/{accepted_count} full-rank subsets within {max_attempts} attempts",
        )
    return MatchedSubsetResult(
        status=BlockStatus.READY.value,
        eligible_ids=eligible,
        accepted_subsets=tuple(accepted),
        rejected_rank_deficient=tuple(rejected),
        attempts=attempts,
        block_reason=None,
    )


def second_largest_edge(values: Any) -> float:
    """Return the second-largest of exactly 100 finite matched ``R_span`` values."""

    array = finite_array(values, "matched R_span values").reshape(-1)
    if array.size != Q4_ACCEPTED_SUBSETS:
        raise ProtocolError(f"matched edge requires exactly {Q4_ACCEPTED_SUBSETS} values, got {array.size}")
    return float(np.sort(array)[-2])


__all__ = [
    "AMENDMENT4_EVALUATION_PAIRS",
    "AMENDMENT4_SPLIT_OFFSET",
    "AMENDMENT4_TRAINING_PAIRS",
    "CORE_API_VERSION",
    "ArtifactError",
    "BlockStatus",
    "BlockedError",
    "BOOTSTRAP_DRAWS",
    "CI_DRAWS",
    "CandidateEvidence",
    "CandidateResult",
    "CoreError",
    "DecoderRowProjector",
    "DirtyArtifactError",
    "HashMismatchError",
    "HeadId",
    "HeadSchemaError",
    "HOLM_ALPHA",
    "EightSeedAggregation",
    "FRESH_SWEEP_SNAPSHOT_STATUS",
    "IncompleteArtifactError",
    "MatchedSubsetResult",
    "PAIR_DIRECTIONS",
    "PAIR_SIGN_CONSISTENCY_MIN",
    "MIN_RETAINED_PAIRS",
    "PairRecord",
    "PairSchemaError",
    "PROTOCOL_SCHEMA",
    "ProtocolError",
    "Q3Result",
    "Q4DenominatorGate",
    "Q4_ACCEPTED_SUBSETS",
    "Q4_MAX_ATTEMPTS",
    "Q4_SUBSET_SIZE",
    "SELECTION_SCHEMA",
    "SOURCE_A_PERCENTILE",
    "REQUESTED_PAIR_COUNT",
    "SplitResult",
    "SplitRole",
    "STAGE1_SELECTION_SEED",
    "STAGE1_TOP_K",
    "STAGE_SWEEP_SCHEMA",
    "TARGET_LATENT_IDS",
    "amendment4_split",
    "bootstrap_p_value",
    "bootstrap_mean_ci",
    "bootstrap_two_sided_p",
    "canonical_heads",
    "canonical_json_bytes",
    "construct_candidate",
    "decoder_row_projector",
    "finite_array",
    "finite_float",
    "flat_head_id",
    "group_pair_records",
    "head_from_flat",
    "head_id",
    "holm_step_down",
    "linear_percentile",
    "nested_sets",
    "pair_clusters",
    "pair_sign_consistency",
    "paired_difference_ci",
    "paired_effect_array",
    "percentile_ci",
    "percentile",
    "q3_true_minus_a",
    "q4_denominator_gate",
    "q4_ratio_of_means",
    "rng",
    "rng_for",
    "aggregate_eight_seed_statuses",
    "sample_matched_subsets",
    "second_largest_edge",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "validate_protocol",
    "validate_fresh_sweep_bindings",
]
