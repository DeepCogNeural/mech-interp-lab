"""Offline contract tests for Experiment 05's execution pipeline.

These tests deliberately exercise only pure artifact/statistical helpers and
the Stage-2 terminal state machine with a tiny import stub.  No model, network,
experiment runner, or large artifact is loaded.
"""

from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exp05_core as core  # noqa: E402
import freeze_candidate  # noqa: E402
import stage3  # noqa: E402


PROTOCOL_PATH = HERE / "protocol_v1.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
HEX_A = "a" * 64
HEX_B = "b" * 64


def _load_stage2_without_model_stack() -> types.ModuleType:
    """Import stage2 with tiny dependency stubs, without torch/model imports.

    Stage2's empty-C branch needs only JSON/hash/path helpers and its pure
    candidate schema parser.  Keeping the runner import isolated makes this
    regression test runnable in the repository's light Python environment.
    """

    module_name = "_exp05_stage2_contract_test"
    if module_name in sys.modules:
        return sys.modules[module_name]

    torch_stub = types.ModuleType("torch")
    torch_stub.__version__ = "contract-test"
    torch_stub.Tensor = type("Tensor", (), {})

    calibrate_stub = types.ModuleType("calibrate")
    calibrate_stub.gate_a = lambda *args, **kwargs: None
    calibrate_stub.make_source_a = lambda *args, **kwargs: None
    calibrate_stub.make_source_b = lambda *args, **kwargs: None
    calibrate_stub.make_source_c_relative_clause = lambda *args, **kwargs: None

    pilot_stub = types.ModuleType("pilot")
    pilot_stub.CleanPass = type("CleanPass", (), {})
    pilot_stub.build_stimuli = lambda *args, **kwargs: None
    pilot_stub.directed_indices = lambda *args, **kwargs: None
    pilot_stub.load_model = lambda *args, **kwargs: None
    pilot_stub.positions_for_kind = lambda *args, **kwargs: None
    pilot_stub.require_one_token = lambda *args, **kwargs: None
    pilot_stub.set_determinism = lambda *args, **kwargs: None
    pilot_stub.AttentionPatchRunner = type("AttentionPatchRunner", (), {})
    pilot_stub._patch_hook = lambda *args, **kwargs: None
    pilot_stub._source_values = lambda *args, **kwargs: None
    pilot_stub.assert_hook_z_layout = lambda *args, **kwargs: None
    pilot_stub.cached_stage1_clean_pass = lambda *args, **kwargs: None
    pilot_stub.clean_readout_microbatched = lambda *args, **kwargs: None

    stage1_stub = types.ModuleType("stage1")
    stage1_stub.HEADS_PER_LAYER = 12
    stage1_stub.LAYER_COUNT = 12
    stage1_stub.PATCH_BATCH_SIZE = 32
    stage1_stub.HOOK_Z = "hook_z"
    stage1_stub.AttentionPatchRunner = type("AttentionPatchRunner", (), {})
    stage1_stub._patch_hook = lambda *args, **kwargs: None
    stage1_stub._source_values = lambda *args, **kwargs: None
    stage1_stub.assert_hook_z_layout = lambda *args, **kwargs: None
    stage1_stub.cached_stage1_clean_pass = lambda *args, **kwargs: None
    stage1_stub.clean_readout_microbatched = lambda *args, **kwargs: None

    spec = importlib.util.spec_from_file_location(module_name, HERE / "stage2.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create Stage-2 contract-test import spec")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"torch": torch_stub, "calibrate": calibrate_stub, "pilot": pilot_stub, "stage1": stage1_stub},
    ):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


stage2 = _load_stage2_without_model_stack()


def _synthetic_selection_sweeps(
    *, true_value: float = 1.0, source_a_value: float = 1.0, pair_count: int = 140
) -> tuple[dict[str, object], dict[str, object]]:
    """Build canonical fresh-sweep artifacts with scalar pair records."""

    pair_records_true = [
        {"pair_id": pair_id, "direction": direction, "effect": true_value}
        for pair_id in range(pair_count)
        for direction in core.PAIR_DIRECTIONS
    ]
    pair_records_a = [
        {"pair_id": pair_id, "direction": direction, "effect": source_a_value}
        for pair_id in range(pair_count)
        for direction in core.PAIR_DIRECTIONS
    ]
    fields = {
        "schema": core.STAGE_SWEEP_SCHEMA,
        "status": "COMPLETE",
        "dirty": False,
        "seed": core.STAGE1_SELECTION_SEED,
        "head_count": core.HEAD_COUNT,
        "directions": list(core.PAIR_DIRECTIONS),
        "measurement_origin": "fresh_same_invocation",
        "model_snapshot_status": core.FRESH_SWEEP_SNAPSHOT_STATUS,
        "invocation_id": HEX_A,
        "model_state_sha256": HEX_B,
        "normalized_config_sha256": HEX_A,
        "tokenizer_assets_sha256": HEX_B,
        "clean_base_cache_sha256": HEX_A,
        "local_snapshot_revisions_sha256": HEX_B,
        "source_cache_sha256": HEX_A,
        "activation_dtype": "float32",
    }
    true_sweep = {
        **fields,
        "source": "true",
        "heads": [
            {"layer": layer, "head": head, "pair_records": copy.deepcopy(pair_records_true)}
            for layer in range(core.LAYER_COUNT)
            for head in range(core.HEADS_PER_LAYER)
        ],
    }
    source_a_sweep = {
        **fields,
        "source": "source_a",
        "heads": [
            {"layer": layer, "head": head, "pair_records": copy.deepcopy(pair_records_a)}
            for layer in range(core.LAYER_COUNT)
            for head in range(core.HEADS_PER_LAYER)
        ],
    }
    return true_sweep, source_a_sweep


def _empty_candidate_payload(*, protocol_hash: str = HEX_B) -> dict[str, object]:
    material: dict[str, object] = {
        "schema": stage2.CANDIDATE_SCHEMA,
        "status": "COMPLETE_NO_CANDIDATES",
        "candidate_status": "EMPTY_UNDER_FROZEN_RULE",
        "immutable": True,
        "manual_override": False,
        "candidate_heads": [],
        "rank_order": [],
        "candidate_C": [],
        "selection_source_a_sha256": HEX_A,
        "selection_sha256": HEX_A,
        "protocol_sha256": protocol_hash,
        "true_sweep_sha256": HEX_A,
        "source_a_sweep_sha256": HEX_B,
    }
    material["candidate_sha256"] = stage2._sha256_bytes(stage2._json_bytes(material))
    return material


class ProtocolAndCoreContracts(unittest.TestCase):
    def test_protocol_parses_to_canonical_hash_and_rejects_operational_mutation(self) -> None:
        declared_hash = core.validate_protocol(PROTOCOL)
        self.assertEqual(declared_hash, core.sha256_json(PROTOCOL))
        self.assertEqual(PROTOCOL["design_freeze"]["latest_amendment"], 9)
        self.assertEqual(PROTOCOL["head_universe"]["total_heads"], core.HEAD_COUNT)
        self.assertEqual(PROTOCOL["selection"]["logical_forward_equivalents"]["total"], 291)
        self.assertEqual(PROTOCOL["q4"]["statistics"]["denominator_guard"]["status"], "NON_ESTIMABLE_DENOMINATOR")

        mutated = copy.deepcopy(PROTOCOL)
        mutated["head_universe"]["total_heads"] = 143
        with self.assertRaises(core.ProtocolError):
            core.validate_protocol(mutated)

        bool_mutated = copy.deepcopy(PROTOCOL)
        bool_mutated["version"] = True
        with self.assertRaises(core.ProtocolError):
            core.validate_protocol(bool_mutated)

    def test_amendment4_assigns_every_n_over_190_id_and_records_unused_tail(self) -> None:
        pair_ids = list(range(200))
        result = core.amendment4_split(pair_ids, seed=20260802)
        self.assertEqual(result.status, core.BlockStatus.READY.value)
        self.assertEqual(len(result.rank_training_ids), core.AMENDMENT4_TRAINING_PAIRS)
        self.assertEqual(len(result.evaluation_ids), core.AMENDMENT4_EVALUATION_PAIRS)
        self.assertEqual(len(result.unused_after_eval_cap_ids), 10)
        self.assertEqual(len(result.assignments), 200)
        self.assertEqual({pair_id for pair_id, _ in result.assignments}, set(pair_ids))
        self.assertEqual(
            set(result.unused_after_eval_cap_ids),
            set(pair_ids) - set(result.rank_training_ids) - set(result.evaluation_ids),
        )
        self.assertTrue(
            all(role == core.SplitRole.UNUSED_AFTER_EVAL_CAP.value for _, role in result.assignments if _ in result.unused_after_eval_cap_ids)
        )

    def test_candidate_rejects_true_source_a_pair_population_drift(self) -> None:
        true_sweep, source_a_sweep = _synthetic_selection_sweeps()
        source_a_sweep["heads"][0]["pair_records"][-2]["pair_id"] = 140  # type: ignore[index]
        with mock.patch.object(core, "bootstrap_p_value", return_value=1e-12):
            with self.assertRaises(core.PairSchemaError):
                core.construct_candidate(true_sweep, source_a_sweep, PROTOCOL)

    def test_candidate_empty_c_is_terminal_frozen_outcome(self) -> None:
        true_sweep, source_a_sweep = _synthetic_selection_sweeps(true_value=1.0, source_a_value=1.0)
        with mock.patch.object(core, "bootstrap_p_value", return_value=1e-12):
            result = core.construct_candidate(true_sweep, source_a_sweep, PROTOCOL)
        self.assertEqual(result.status, "EMPTY_UNDER_FROZEN_RULE")
        self.assertEqual(result.candidate_heads, ())
        self.assertEqual(result.rank_order, ())
        self.assertEqual(result.nested_sets_by_rank, ())
        self.assertTrue(all(not evidence.eligible for evidence in result.evidence))
        self.assertIn("true_abs_mean_not_above_source_a_p99_edge", result.evidence[0].reasons)


class Q4Contracts(unittest.TestCase):
    def test_q4_denominator_gate_is_fail_closed_and_ratio_uses_means(self) -> None:
        gate = core.q4_denominator_gate([1.0, 2.0], [0.5, 1.0], [0.5, 1.0])
        self.assertEqual(gate.status, "ESTIMABLE")
        ratio_span, ratio_comp, _ = core.q4_ratio_of_means([1.0, 2.0], [0.5, 1.0], [0.5, 1.0])
        self.assertAlmostEqual(ratio_span, 0.5)
        self.assertAlmostEqual(ratio_comp, 0.5)

        non_estimable = core.q4_denominator_gate([1.0, -1.0], [0.2, 0.2], [0.8, 0.8])
        self.assertEqual(non_estimable.status, "NON_ESTIMABLE_DENOMINATOR")
        self.assertFalse(non_estimable.ratio_of_means_allowed)
        with self.assertRaises(core.BlockedError):
            core.q4_ratio_of_means([1.0, -1.0], [0.2, 0.2], [0.8, 0.8])

    def test_decoder_projector_requires_rank_twelve_and_projects_in_float64(self) -> None:
        rows = np.eye(core.Q4_SUBSET_SIZE, dtype=np.float64)
        projector = core.decoder_row_projector(rows)
        delta = np.arange(core.Q4_SUBSET_SIZE, dtype=np.float64)
        np.testing.assert_allclose(projector.project(delta), delta)
        np.testing.assert_allclose(projector.complement(delta), np.zeros_like(delta))
        self.assertEqual(projector.rank, core.Q4_SUBSET_SIZE)

        deficient = np.ones((core.Q4_SUBSET_SIZE, core.Q4_SUBSET_SIZE), dtype=np.float64)
        with self.assertRaises(core.BlockedError):
            core.decoder_row_projector(deficient)

    def test_matched_draws_exclude_targets_and_bind_exactly_100_full_rank_subsets(self) -> None:
        target_ids = tuple(core.TARGET_LATENT_IDS)
        extra_ids = list(range(100_000, 100_116))
        candidate_ids = list(target_ids) + extra_ids
        eligible_ids = sorted(set(candidate_ids) - set(target_ids))
        decoder_rows = {
            latent_id: np.eye(len(eligible_ids), dtype=np.float64)[index]
            for index, latent_id in enumerate(eligible_ids)
        }
        first = core.sample_matched_subsets(candidate_ids, decoder_rows, target_ids, seed=20260806)
        second = core.sample_matched_subsets(candidate_ids, decoder_rows, target_ids, seed=20260806)
        self.assertEqual(first.status, core.BlockStatus.READY.value)
        self.assertEqual(first.eligible_ids, tuple(eligible_ids))
        self.assertEqual(len(first.accepted_subsets), core.Q4_ACCEPTED_SUBSETS)
        self.assertEqual(first.attempts, core.Q4_ACCEPTED_SUBSETS)
        self.assertEqual(first.accepted_subsets, second.accepted_subsets)
        self.assertTrue(all(len(subset) == core.Q4_SUBSET_SIZE for subset in first.accepted_subsets))
        self.assertTrue(all(set(subset).isdisjoint(target_ids) for subset in first.accepted_subsets))


class ArtifactAndRunnerContracts(unittest.TestCase):
    def test_stage2_joint_patch_hook_accepts_transformerlens_hook_keyword(self) -> None:
        class Sliceable:
            shape = (1,)

            def __getitem__(self, _key: object) -> "Sliceable":
                return self

        class FakeValue:
            def detach(self) -> "FakeValue":
                return self

            def float(self) -> "FakeValue":
                return self

            def cpu(self) -> "FakeValue":
                return self

        fake_torch = types.SimpleNamespace(
            no_grad=contextlib.nullcontext,
            cat=lambda values: tuple(values),
        )

        class FakeModel:
            tokenizer = object()
            cfg = types.SimpleNamespace(n_heads=12, d_head=64)

            def run_with_hooks(self, _tokens: object, *, fwd_hooks: object, return_type: str) -> str:
                self.assert_return_type = return_type
                for _name, callback in fwd_hooks:  # type: ignore[assignment]
                    callback("activation", hook="transformer-lens-hook")
                return "logits"

        pilot_stub = types.ModuleType("pilot")
        pilot_stub.logit_difference = lambda *args, **kwargs: FakeValue()
        with (
            mock.patch.object(stage2, "torch", fake_torch),
            mock.patch.object(stage2, "require_one_token", side_effect=[1, 2]),
            mock.patch.object(stage2, "_patch_hook", return_value="patched") as patch_hook,
            mock.patch.object(stage2, "_check_runtime"),
            mock.patch.dict(sys.modules, {"pilot": pilot_stub}),
        ):
            result = stage2._run_selected_z(
                FakeModel(),
                Sliceable(),
                Sliceable(),
                [0],
                {(0, 0): Sliceable()},
                object(),
                object(),
                [{"layer": 0, "head": 0}],
                "contract",
                started=0.0,
                cap=1.0,
            )

        self.assertEqual(len(result), 1)
        patch_hook.assert_called_once()

    def test_stage2_requires_complete_selection_state_fingerprints(self) -> None:
        complete = {
            "schema": "exp05.model_state_fingerprint.v1",
            "scheme": "lexicographic state_dict keys; key/dtype/shape JSON plus uncast contiguous tensor bytes; uint64 length framing",
            "encoding_detail": "canonical JSON metadata; unsigned uint64 big-endian metadata and raw-byte lengths",
            "entries": [
                {
                    "key": "weight",
                    "dtype": "torch.float32",
                    "shape": [1],
                    "byte_length": 4,
                    "bytes_sha256": HEX_A,
                }
            ],
            "key_count": 1,
            "sha256": HEX_B,
            "exact_match_before": True,
        }
        self.assertEqual(
            stage2._validate_selection_state_fingerprint(complete, label="after_true_sweep"),
            HEX_B,
        )

        compact = {
            "key_count": 1,
            "sha256": HEX_B,
            "exact_match_before": True,
        }
        with self.assertRaises(stage2.Stage2Stop):
            stage2._validate_selection_state_fingerprint(compact, label="after_true_sweep")

    def test_stage3_self_hash_and_runtime_alias_guards(self) -> None:
        body = {"schema": "contract", "status": "RUNNING", "value": 7}
        bound = stage3._with_self_hash(body)
        self.assertEqual(bound["self_sha256"], stage3._self_hash(bound))
        changed = dict(bound, value=8)
        self.assertNotEqual(changed["self_sha256"], stage3._self_hash(changed))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            immutable = root / "input.json"
            immutable.write_text("{}", encoding="utf-8")
            with self.assertRaises(stage3.Stage3Stop):
                stage3._validate_runtime_paths(
                    immutable={"input": immutable},
                    runtime={"output": immutable},
                )
            with self.assertRaises(stage3.Stage3Stop):
                stage3._validate_runtime_paths(
                    immutable={"input": immutable},
                    runtime={"output": root / "same.json", "draw_csv": root / "same.json"},
                )

    def test_freeze_candidate_rejects_output_alias_before_reading_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            selection_path = Path(temp) / "selection.json"
            with self.assertRaises(core.ArtifactError):
                freeze_candidate.freeze_candidate(PROTOCOL_PATH, selection_path, PROTOCOL_PATH)

    def test_stage2_empty_c_writes_valid_terminal_manifest_and_no_science(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path = root / "protocol.json"
            calibration_path = root / "calibration.json"
            selection_path = root / "selection.json"
            candidate_path = root / "candidate.json"
            output_path = root / "stage2.json"
            pair_output_path = root / "pairs.csv"
            for path in (protocol_path, calibration_path, selection_path):
                path.write_text("{}\n", encoding="utf-8")
            candidate_path.write_text(
                json.dumps(
                    _empty_candidate_payload(protocol_hash=stage2._sha256_file(protocol_path)),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            output_path.write_text(json.dumps({"status": "COMPLETE", "scientific_verdict_emitted": True}), encoding="utf-8")
            args = argparse.Namespace(
                protocol=str(protocol_path),
                calibration=str(calibration_path),
                selection=str(selection_path),
                candidate=str(candidate_path),
                expected_git_commit="0" * 40,
                require_clean_tree=True,
                max_wall_seconds=60.0,
                output=str(output_path),
                pair_output=str(pair_output_path),
                checkpoint=None,
            )
            with (
                mock.patch.object(stage2, "_validate_protocol"),
                mock.patch.object(stage2, "_validate_calibration", return_value={"A": 0.2, "C": 0.26983}),
                mock.patch.object(stage2, "_validate_selection_dependency", return_value={"selection_provenance_sha256": HEX_A}),
                mock.patch.object(stage2, "_git", return_value=args.expected_git_commit),
                mock.patch.object(stage2, "_require_clean_tree_except_runtime_artifacts", return_value=[]),
                mock.patch.object(stage2, "CoreAdapter", return_value=object()),
            ):
                manifest, code = stage2.run(args)
            self.assertEqual(code, 0)
            self.assertEqual(manifest["status"], "NOT_INSTANTIATED_VALID_EMPTY_C")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "NOT_INSTANTIATED_VALID_EMPTY_C")
            self.assertEqual(payload["status_code"], "NOT_INSTANTIATED_VALID_EMPTY_C")
            self.assertEqual(payload["q1"]["status"], "COMPLETE_NO_CANDIDATES")
            self.assertEqual(payload["q2"]["status"], "NOT_INSTANTIATED_NO_TESTED_SET")
            self.assertEqual(payload["q3"]["status"], "NOT_INSTANTIATED_NO_TESTED_SET")
            self.assertFalse(payload["scientific_verdict_emitted"])
            self.assertFalse(payload["supplies_published_science"])
            self.assertTrue(payload["stale_pair_output_is_not_bound"])
            self.assertFalse(pair_output_path.exists())

    def test_runner_cli_and_source_invariants_do_not_offer_stage1_or_candidate_q4_path(self) -> None:
        parser = stage2.build_parser()
        parsed = parser.parse_args(
            [
                "--protocol",
                "p",
                "--calibration",
                "c",
                "--selection",
                "s",
                "--candidate",
                "d",
                "--expected-git-commit",
                "0" * 40,
                "--require-clean-tree",
                "--max-wall-seconds",
                "1",
                "--output",
                "o",
                "--pair-output",
                "r",
            ]
        )
        self.assertEqual(parsed.candidate, "d")
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parser.parse_args(["--stage1", "stage1_results.json"])

        stage3_parser = stage3._parser()
        split_args = stage3_parser.parse_args(
            [
                "materialize-splits",
                "--protocol",
                "p",
                "--gate-cache",
                "g",
                "--split-manifest",
                "s",
                "--split-csv",
                "c",
                "--prepare-manifest",
                "m",
            ]
        )
        self.assertEqual(split_args.command, "materialize-splits")
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                stage3_parser.parse_args(["materialize-splits", "--candidate", "candidate.json"])

        source = (HERE / "stage2.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertNotIn("--stage1", source)
        self.assertIn("NOT_INSTANTIATED_VALID_EMPTY_C", source)
        atomic_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_atomic_write_json"
        ]
        self.assertGreaterEqual(len(atomic_calls), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
