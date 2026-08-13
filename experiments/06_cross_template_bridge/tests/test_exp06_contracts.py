"""Model-free contract and corruption tests for Experiment 06.

The synthetic artifacts exercise the complete compact evidence lifecycle. No
model, SAE, tokenizer, network call, or scientific experiment is loaded here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
EXP05 = HERE.parent / "05_number_agreement_circuit"


def _load(name: str, path: Path, bindings: dict[str, object] | None = None) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, bindings or {}):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


bridge = _load("_exp06_bridge_contract", EXP05 / "bridge_rescue.py")
runner = _load("_exp06_runner_contract", HERE / "run_experiment.py", {"bridge_rescue": bridge})
packet = _load("_exp06_packet_contract", HERE / "make_public_results.py", {"run_experiment": runner})


TARGET_IDS = list(range(12))
Q4_SEEDS = list(range(1000, 1008))


def _canonical_protocol_sha() -> str:
    encoded = json.dumps(
        runner.FROZEN_PROTOCOL,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _asset_provenance() -> dict[str, object]:
    contract_sha = hashlib.sha256(
        json.dumps(
            runner.ASSET_CONTRACT,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    model = {
        "schema": "exp06-model-state-fingerprint-v1",
        "sha256": "c" * 64,
        "key_count": 123,
        "scheme": runner.ASSET_CONTRACT["state_fingerprint"]["model_scheme"],
    }
    sae = {
        "schema": "exp06-sae-decoder-fingerprint-v1",
        "sha256": "d" * 64,
        "dtype": "torch.float32",
        "shape": [24_576, runner.RESIDUAL_WIDTH],
        "scheme": runner.ASSET_CONTRACT["state_fingerprint"]["sae_decoder_scheme"],
    }
    return {
        "contract_sha256": contract_sha,
        "repositories": {
            group: {
                "repo_id": runner.ASSET_CONTRACT[group]["repo_id"],
                "revision": runner.ASSET_CONTRACT[group]["revision"],
                "files": copy.deepcopy(runner.ASSET_CONTRACT[group]["files"]),
            }
            for group in ("gpt2", "sae")
        },
        "runtime_versions": copy.deepcopy(runner.ASSET_CONTRACT["runtime_versions"]),
        "loader_boundary": (
            "loaders consumed private copies written from the exact hash-validated cache byte buffers; "
            "TransformerLens config and weights were constructed from the staged local directory "
            "without a model-name lookup"
        ),
        "model_state": {"before": model, "after": copy.deepcopy(model), "exact_match": True},
        "sae_decoder": {"before": sae, "after": copy.deepcopy(sae), "exact_match": True},
    }


def _q4_payload() -> dict[str, object]:
    rows = []
    for seed in reversed(Q4_SEEDS):
        accepted = []
        for draw in range(runner.MATCHED_COUNT):
            start = 100 + draw * runner.TARGET_COUNT
            accepted.append({"draw_index": draw, "latent_ids": list(range(start, start + runner.TARGET_COUNT))})
        rows.append(
            {
                "seed": seed,
                "projector": {"target_latent_ids": TARGET_IDS},
                "matched_draws": {"accepted": accepted},
            }
        )
    return {
        "schema": "exp05-number-agreement-stage3-v1; frozen Q4",
        "status": "COMPLETE",
        "seed_results": rows,
    }


def _gate(passed: bool) -> dict[str, object]:
    retained = 150 if passed else 130
    return {
        "generated_pairs": runner.REQUESTED_PAIRS,
        "retained_pair_ids": list(range(retained)),
        "retained_pairs": retained,
        "fraction": retained / runner.REQUESTED_PAIRS,
        "median_gap": 2.0 if passed else 0.5,
        "thresholds": {
            "fraction_at_least": 0.6,
            "retained_at_least": 140,
            "median_gap_at_least": 1.0,
        },
        "passed": passed,
    }


def _branch_values(branch: str) -> tuple[float, float]:
    if branch == "MECHANISM_NEGATIVE":
        return 0.01, 0.20
    if branch == "SPAN_NEGATIVE":
        return 0.10, 0.05
    return 0.10, 0.20


def _raw_result(branch: str) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    q4 = _q4_payload()
    frozen = bridge.parse_q4_frozen_sets(q4)
    seed_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    direct, target = _branch_values(branch)

    for ordinal, (seed, frozen_row) in enumerate(zip(runner.FRESH_SEEDS, frozen["ordinal_sets"])):
        gate_invalid = branch == "NON_ESTIMABLE" and ordinal == 0
        gate = _gate(not gate_invalid)
        if gate_invalid:
            seed_rows.append(
                {
                    "seed": seed,
                    "source_q4_seed": int(frozen_row["source_q4_seed"]),
                    "status": "GATE_A_POPULATION_INVALID",
                    "reason": "GATE_A_FAIL",
                    "gate_a": gate,
                }
            )
            continue

        effects = [0.001 * draw for draw in range(runner.MATCHED_COUNT)]
        effects[-1] = 9.0  # one extreme maximum must never become the adjudicating edge
        edge, maximum = runner._matched_edge_summary(effects)
        advantage = target - edge
        complement = 0.01
        closure = target + complement
        estimands = {
            "D_s": direct,
            "full_signed_effect": direct,
            "T_s": target,
            "E_s_second_largest_matched": edge,
            "M_s_max_descriptive": maximum,
            "A_s": advantage,
            "complement_signed_effect": complement,
            "target_plus_complement_signed_effect_sum": closure,
            "target_plus_complement_minus_full_signed_effect": closure - direct,
            "target_plus_complement_over_full_ratio_descriptive": closure / direct,
            "target_over_direct_ratio_descriptive": target / direct,
            "complement_over_direct_ratio_descriptive": complement / direct,
        }
        seed_rows.append(
            {
                "seed": seed,
                "source_q4_seed": int(frozen_row["source_q4_seed"]),
                "status": "ADJUDICABLE",
                "gate_a": gate,
                "evaluation_pair_ids": list(range(runner.MAX_EVAL_PAIRS)),
                "evaluation_item_count": 2 * runner.MAX_EVAL_PAIRS,
                "target_latent_ids": TARGET_IDS,
                "target_projector": {"rank": runner.TARGET_COUNT, "arithmetic": "float64"},
                "identity_diagnostics": {
                    "selected_positions_max_abs": 0.0,
                    "non_final_positions_max_abs": 0.0,
                    "non_final_tolerance": runner.NON_FINAL_TOLERANCE,
                    "full_vs_true_final_logit_max_abs": 0.0,
                    "full_vs_true_final_logit_tolerance": runner.FULL_LOGIT_TOLERANCE,
                },
                "estimands": estimands,
            }
        )
        for draw, (draw_binding, effect) in enumerate(zip(frozen_row["matched"], effects)):
            matched_rows.append(
                {
                    "seed": seed,
                    "source_q4_seed": int(frozen_row["source_q4_seed"]),
                    "draw_index": draw,
                    "latent_ids": list(draw_binding["latent_ids"]),
                    "M_sj": effect,
                    "descriptive_ratio": effect / direct,
                    "projector": {"rank": runner.TARGET_COUNT, "arithmetic": "float64"},
                }
            )

    adjudication = runner.adjudicate(seed_rows)
    self_verdict = str(adjudication["verdict"])
    if self_verdict != branch:
        raise AssertionError(f"synthetic branch {branch} produced {self_verdict}")
    git = {
        "commit": "a" * 40,
        "expected_commit": "a" * 40,
        "require_clean_tree": True,
        "status_porcelain": "",
        "source_tree_sha256": "b" * 64,
    }
    raw: dict[str, object] = {
        "schema": runner.SCHEMA,
        "status": "COMPLETE",
        "verdict": branch,
        "scientific_verdict_emitted": True,
        "inputs": {
            "protocol_sha256": _canonical_protocol_sha(),
            "q4_raw_sha256": runner.Q4_RAW_SHA256,
            "asset_contract_sha256": _asset_provenance()["contract_sha256"],
            "registered_seeds": list(runner.FRESH_SEEDS),
            "q4_seed_ordinals": Q4_SEEDS,
            "target_latent_ids": TARGET_IDS,
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
        "seed_results": seed_rows,
        "matched_rows": matched_rows,
        "claim_boundary": copy.deepcopy(runner.FROZEN_PROTOCOL["claim_boundary"]),
        "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "asset_provenance": _asset_provenance(),
        "runtime": {
            "wall_clock_seconds": 1.0,
            "batch_size": 32,
            "adjudicable_seed_count": sum(row["status"] == "ADJUDICABLE" for row in seed_rows),
            "matched_row_count": len(matched_rows),
        },
        "git": git,
        "git_final": copy.deepcopy(git),
    }
    return raw, q4, frozen


def _simple_seed_rows(direct: list[float], target: list[float], advantage: list[float]) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "status": "ADJUDICABLE",
            "estimands": {"D_s": d_value, "T_s": t_value, "A_s": a_value},
        }
        for seed, d_value, t_value, a_value in zip(runner.FRESH_SEEDS, direct, target, advantage)
    ]


class FrozenProtocolAndDecisionContracts(unittest.TestCase):
    def test_staged_loader_never_uses_model_name_or_named_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staged = (Path(temporary) / "gpt2").resolve()
            staged.mkdir()
            calls: list[tuple[str, str]] = []

            class FakeConfig:
                @staticmethod
                def to_dict() -> dict[str, object]:
                    return {"architectures": ["GPT2LMHeadModel"], "vocab_size": 50_257}

            hf_model = types.SimpleNamespace(config=FakeConfig())
            fake_dtype = object()
            tokenizer = object()

            class PoisonedNamedLoader:
                @staticmethod
                def get_pretrained_model_config(path: str, **kwargs: object) -> object:
                    self.assertEqual(Path(path), staged)
                    self.assertEqual(
                        kwargs["hf_cfg"],
                        {"architectures": ["GPT2LMHeadModel"], "vocab_size": 50_257},
                    )
                    self.assertIs(kwargs["dtype"], fake_dtype)
                    self.assertIs(kwargs["local_files_only"], True)
                    self.assertIs(kwargs["fold_ln"], True)
                    self.assertEqual(kwargs["device"], "cpu")
                    self.assertEqual(kwargs["n_devices"], 1)
                    calls.append(("config", path))
                    return types.SimpleNamespace(device="cpu")

                @staticmethod
                def get_pretrained_state_dict(path: str, _cfg: object, **kwargs: object) -> dict[str, object]:
                    self.assertEqual(Path(path), staged)
                    self.assertIs(kwargs["hf_model"], hf_model)
                    self.assertIs(kwargs["dtype"], fake_dtype)
                    self.assertIs(kwargs["local_files_only"], True)
                    calls.append(("weights", path))
                    return {"staged": object()}

            class PoisonedHookedTransformer:
                @classmethod
                def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:  # pragma: no cover
                    raise AssertionError("named-cache loader must never be called")

                def __init__(self, cfg: object, tokenizer: object, move_to_device: bool) -> None:
                    self.cfg = cfg
                    self.tokenizer = tokenizer
                    self.move_to_device = move_to_device
                    self.loaded: dict[str, object] | None = None
                    self.moved = False

                def load_and_process_state_dict(self, state: dict[str, object], **kwargs: object) -> None:
                    self.loaded = state
                    self.assertions = kwargs

                def move_model_modules_to_device(self) -> None:
                    self.moved = True

            model = runner._build_hooked_transformer_from_staged(
                staged,
                hf_model,
                tokenizer,
                torch_module=types.SimpleNamespace(float32=fake_dtype),
                hooked_transformer_cls=PoisonedHookedTransformer,
                loading_module=PoisonedNamedLoader,
            )
            self.assertEqual(calls, [("config", str(staged)), ("weights", str(staged))])
            self.assertIsNotNone(model.loaded)
            self.assertEqual(set(model.loaded), {"staged"})
            self.assertIs(model.tokenizer, tokenizer)
            self.assertFalse(model.move_to_device)
            self.assertTrue(model.moved)
            self.assertEqual(
                model.assertions,
                {
                    "fold_ln": True,
                    "center_writing_weights": True,
                    "center_unembed": True,
                    "fold_value_biases": True,
                    "refactor_factored_attn_matrices": False,
                },
            )

    def test_protocol_json_exactly_matches_literal_and_deep_drift_fails(self) -> None:
        protocol = json.loads((HERE / "protocol_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol, runner.FROZEN_PROTOCOL)
        runner.validate_protocol(protocol)
        drifted = copy.deepcopy(protocol)
        drifted["decision_rule"]["positive"]["all_required"].pop()
        with self.assertRaises(runner.BridgeStop) as caught:
            runner.validate_protocol(drifted)
        self.assertEqual(caught.exception.gate, "protocol_contract")

    def test_only_canonical_protocol_path_is_accepted(self) -> None:
        self.assertEqual(runner.resolve_path("@exp06/protocol_v1.json"), (HERE / "protocol_v1.json").resolve())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "protocol.json"
            external.write_text(json.dumps(runner.FROZEN_PROTOCOL), encoding="utf-8")
            output = root / "result.json"
            args = argparse.Namespace(
                output=str(output),
                q4_results=str(root / "q4.json"),
                protocol=str(external),
                expected_git_commit="a" * 40,
                require_clean_tree=True,
                max_wall_seconds=60.0,
                batch_size=1,
            )
            with self.assertRaises(runner.BridgeStop) as caught:
                runner.run(args)
            self.assertEqual(caught.exception.gate, "protocol_path")
            self.assertFalse(output.exists())

    def test_source_a_cycle_is_exact_stable_and_preserves_registered_fields(self) -> None:
        records = []
        for index, (singular, plural) in enumerate(runner.SOURCE_A_LEMMA_ORDER):
            records.append(
                {
                    "template": "The {SUBJ} that the {ATTRACTOR} often {RELVERB}",
                    "subject_singular": singular,
                    "subject_plural": plural,
                    "attractor": f"attractor-{index}",
                    "relative_verb": f"verb-{index}",
                    "subject_token_index": 1,
                }
            )
        base = types.SimpleNamespace(pair_records=records)

        def build(_tokenizer: object, *, family: str, rows: list[dict[str, object]]) -> object:
            return types.SimpleNamespace(family=family, pair_records=rows)

        first = runner.make_relative_source_a(None, base, nouns=runner.SOURCE_A_LEMMA_ORDER, build_stimuli_from_rows=build)
        second = runner.make_relative_source_a(None, base, nouns=runner.SOURCE_A_LEMMA_ORDER, build_stimuli_from_rows=build)
        self.assertEqual(first.pair_records, second.pair_records)
        for index, row in enumerate(first.pair_records):
            expected = runner.SOURCE_A_LEMMA_ORDER[(index + 1) % len(runner.SOURCE_A_LEMMA_ORDER)]
            self.assertEqual((row["subject_singular"], row["subject_plural"]), expected)
            self.assertNotEqual(expected, runner.SOURCE_A_LEMMA_ORDER[index])
            self.assertEqual(row["attractor"], f"attractor-{index}")
            self.assertEqual(row["relative_verb"], f"verb-{index}")
            self.assertEqual(row["subject_token_index"], 1)
        with self.assertRaises(runner.BridgeStop) as caught:
            runner.make_relative_source_a(
                None,
                base,
                nouns=tuple(reversed(runner.SOURCE_A_LEMMA_ORDER)),
                build_stimuli_from_rows=build,
            )
        self.assertEqual(caught.exception.gate, "source_A_mapping")

    def test_four_branches_and_exact_eight_seed_execution(self) -> None:
        for branch in ("POSITIVE", "MECHANISM_NEGATIVE", "SPAN_NEGATIVE", "NON_ESTIMABLE"):
            raw, _q4, _frozen = _raw_result(branch)
            self.assertEqual(runner.adjudicate(raw["seed_results"])["verdict"], branch)
            self.assertEqual(packet._reaggregate_adjudication(raw["seed_results"])["verdict"], branch)

        raw, _q4, _frozen = _raw_result("POSITIVE")
        for broken in (
            raw["seed_results"][:-1],
            [*raw["seed_results"][:-1], copy.deepcopy(raw["seed_results"][0])],
            [*raw["seed_results"][:-1], {**raw["seed_results"][-1], "status": "UNKNOWN"}],
        ):
            with self.assertRaises(runner.BridgeStop) as caught:
                runner.adjudicate(broken)
            self.assertEqual(caught.exception.gate, "seed_execution")

    def test_joint_six_of_eight_requires_the_same_seed_intersection(self) -> None:
        direct = [0.10] * 8
        target = [0.10] * 6 + [-0.001] * 2
        advantage = [-0.001] * 2 + [0.10] * 6
        result = runner.adjudicate(_simple_seed_rows(direct, target, advantage))
        self.assertEqual(result["verdict"], "SPAN_NEGATIVE")
        conditions = result["span_transfer"]["conditions"]
        self.assertTrue(conditions["target_lower_bound_above_zero"])
        self.assertTrue(conditions["advantage_lower_bound_above_zero"])
        self.assertEqual(conditions["joint_target_and_advantage_positive_registered_seed_count"], 4)

    def test_second_largest_is_not_max_and_rejects_bad_population(self) -> None:
        effects = [float(index) for index in range(100)]
        effects[-1] = 1_000.0
        edge, maximum = runner._matched_edge_summary(effects)
        self.assertEqual(edge, 98.0)
        self.assertEqual(maximum, 1_000.0)
        with self.assertRaises(runner.BridgeStop):
            runner._matched_edge_summary(effects[:-1])
        broken = list(effects)
        broken[0] = math.nan
        with self.assertRaises(runner.BridgeStop) as caught:
            runner._matched_edge_summary(broken)
        self.assertEqual(caught.exception.gate, "nonfinite_statistic")


class PacketIntegrityContracts(unittest.TestCase):
    def test_all_four_synthetic_branches_pass_independent_validation(self) -> None:
        for branch in ("POSITIVE", "MECHANISM_NEGATIVE", "SPAN_NEGATIVE", "NON_ESTIMABLE"):
            raw, _q4, frozen = _raw_result(branch)
            packet._validate(raw, "same", "same", frozen)

    def test_q4_binding_and_compact_corruptions_fail_closed(self) -> None:
        raw, _q4, frozen = _raw_result("POSITIVE")

        def corrupt(path: str, mutate: object) -> dict[str, object]:
            value = copy.deepcopy(raw)
            if path == "ordinal":
                value["inputs"]["q4_seed_ordinals"][0] = 999
            elif path == "target":
                value["inputs"]["target_latent_ids"][0] = 999
            elif path == "matched_ids":
                value["matched_rows"][0]["latent_ids"][0] += 1
            elif path == "edge":
                value["seed_results"][0]["estimands"]["E_s_second_largest_matched"] = value["seed_results"][0]["estimands"]["M_s_max_descriptive"]
            elif path == "advantage":
                value["seed_results"][0]["estimands"]["A_s"] += 0.1
            elif path == "ratio":
                value["matched_rows"][0]["descriptive_ratio"] += 0.1
            elif path == "closure":
                value["seed_results"][0]["estimands"]["target_plus_complement_signed_effect_sum"] += 0.1
            elif path == "git":
                value["git_final"]["commit"] = "c" * 40
            elif path == "offline":
                value["offline_env"]["TRANSFORMERS_OFFLINE"] = "0"
            elif path == "asset_revision":
                value["asset_provenance"]["repositories"]["sae"]["revision"] = "0" * 40
            elif path == "asset_file":
                value["asset_provenance"]["repositories"]["gpt2"]["files"]["model.safetensors"] = "0" * 64
            elif path == "model_state":
                value["asset_provenance"]["model_state"]["after"]["sha256"] = "0" * 64
            elif path == "sae_state":
                value["asset_provenance"]["sae_decoder"]["after"]["sha256"] = "0" * 64
            elif path == "adjudication":
                value["adjudication"]["verdict"] = "SPAN_NEGATIVE"
            else:  # pragma: no cover
                raise AssertionError(path)
            return value

        for label in (
            "ordinal",
            "target",
            "matched_ids",
            "edge",
            "advantage",
            "ratio",
            "closure",
            "git",
            "offline",
            "asset_revision",
            "asset_file",
            "model_state",
            "sae_state",
            "adjudication",
        ):
            with self.subTest(label=label):
                with self.assertRaises(packet.PacketStop):
                    packet._validate(corrupt(label, None), "same", "same", frozen)

    def test_nonfinite_json_and_identity_values_fail_closed(self) -> None:
        with self.assertRaises(runner.BridgeStop):
            runner._json_object_from_bytes(b'{"value":NaN}', "synthetic runner input")
        with self.assertRaises(packet.PacketStop):
            packet._json_object_from_bytes(b'{"value":Infinity}', "synthetic packet input")

        raw, _q4, frozen = _raw_result("POSITIVE")
        for field, value in (
            ("selected_positions_max_abs", math.nan),
            ("non_final_positions_max_abs", math.inf),
            ("full_vs_true_final_logit_max_abs", -math.inf),
        ):
            broken = copy.deepcopy(raw)
            broken["seed_results"][0]["identity_diagnostics"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(packet.PacketStop):
                    packet._validate(broken, "same", "same", frozen)

    def test_guard_failure_is_stopped_but_gate_failure_is_non_estimable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "stopped.json"
            args = argparse.Namespace(
                output=str(output),
                q4_results=str(root / "q4.json"),
                protocol=str(HERE / "protocol_v1.json"),
                expected_git_commit="a" * 40,
                require_clean_tree=True,
                max_wall_seconds=60.0,
                batch_size=1,
            )
            with mock.patch.object(
                bridge,
                "_git_provenance",
                side_effect=runner.BridgeStop("dirty_tree", "synthetic dirty tree"),
            ):
                result, code = runner.run(args)
            self.assertEqual(code, 2)
            self.assertEqual(result["status"], "STOPPED")
            self.assertIs(result["verdict"], None)
            self.assertIs(result["scientific_verdict_emitted"], False)
            self.assertEqual(result["reason"]["gate"], "dirty_tree")
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "STOPPED")

        raw, _q4, _frozen = _raw_result("NON_ESTIMABLE")
        self.assertEqual(runner.adjudicate(raw["seed_results"])["verdict"], "NON_ESTIMABLE")

    def test_runner_and_packet_refuse_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reserved = root / "reserved.json"
            runner._reserve_output(reserved)
            before = reserved.read_bytes()
            with self.assertRaises(runner.BridgeStop) as caught:
                runner._reserve_output(reserved)
            self.assertEqual(caught.exception.gate, "output_exists")
            self.assertEqual(reserved.read_bytes(), before)

            raw, q4, _frozen = _raw_result("POSITIVE")
            q4_path = root / "q4.json"
            q4_path.write_text(json.dumps(q4, sort_keys=True), encoding="utf-8")
            q4_hash = packet._sha256_file(q4_path)
            raw["inputs"]["q4_raw_sha256"] = q4_hash
            raw_path = root / "raw.json"
            raw_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
            raw_hash = packet._sha256_file(raw_path)
            output = root / "packet"
            with mock.patch.object(runner, "Q4_RAW_SHA256", q4_hash):
                built = packet.build_packet(raw_path, q4_path, output, raw_hash)
                self.assertEqual(built["verdict"], "POSITIVE")
                artifact_hashes = {
                    path.name: packet._sha256_file(path)
                    for path in output.iterdir()
                    if path.is_file()
                }
                with self.assertRaises(packet.PacketStop):
                    packet.build_packet(raw_path, q4_path, output, raw_hash)
                self.assertEqual(
                    artifact_hashes,
                    {
                        path.name: packet._sha256_file(path)
                        for path in output.iterdir()
                        if path.is_file()
                    },
                )

    def test_packet_build_failure_never_exposes_a_partial_final_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, q4, _frozen = _raw_result("POSITIVE")
            q4_path = root / "q4.json"
            q4_path.write_text(json.dumps(q4, sort_keys=True), encoding="utf-8")
            q4_hash = packet._sha256_file(q4_path)
            raw["inputs"]["q4_raw_sha256"] = q4_hash
            raw_path = root / "raw.json"
            raw_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
            raw_hash = packet._sha256_file(raw_path)
            output = root / "packet"
            original = packet._atomic_text
            calls = 0

            def fail_after_one(path: Path, text: str) -> None:
                nonlocal calls
                calls += 1
                original(path, text)
                if calls == 2:
                    raise OSError("synthetic staging failure")

            with mock.patch.object(runner, "Q4_RAW_SHA256", q4_hash), mock.patch.object(
                packet,
                "_atomic_text",
                side_effect=fail_after_one,
            ):
                with self.assertRaises(OSError):
                    packet.build_packet(raw_path, q4_path, output, raw_hash)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(f".{output.name}.publish.lock").exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
