"""Offline contracts for the fresh-seed bridge-rescue runner.

These tests exercise only model-free estimators and artifact parsing.  They do
not load GPT-2, an SAE, or the Q4 41 MB result into the test process.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bridge_rescue_contract", HERE / "bridge_rescue.py")
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


class BridgePureContracts(unittest.TestCase):
    def test_ratio_of_directed_means_and_t7_summary(self) -> None:
        self.assertAlmostEqual(bridge.mean_ratio([1.0, -1.0], [3.0, -3.0], [1.0, -1.0], [5.0, -5.0]), 0.5)
        summary = bridge._t7_summary([1.0] * 8)
        self.assertEqual(summary["degrees_of_freedom"], 7)
        self.assertAlmostEqual(summary["mean"], 1.0)

    def test_rank_twelve_projector_is_float64_and_fail_closed(self) -> None:
        rows = torch.zeros((12, 768), dtype=torch.float32)
        rows[:, :12] = torch.eye(12)
        basis, metadata = bridge.decoder_row_projector(rows)
        self.assertEqual(metadata["rank"], 12)
        self.assertEqual(basis.dtype, torch.float64)
        value = torch.arange(768, dtype=torch.float64)
        projected = bridge.project_float64(value, basis)
        self.assertTrue(torch.allclose(projected[12:], torch.zeros(756, dtype=torch.float64)))
        with self.assertRaises(bridge.BridgeStop):
            bridge.decoder_row_projector(torch.ones((12, 768)))

    def test_reader_projection_reports_coefficient_and_cosine(self) -> None:
        baseline = torch.zeros((2, 4), dtype=torch.float32)
        full = torch.ones((2, 4), dtype=torch.float32)
        self.assertAlmostEqual(bridge.reader_projection(full, baseline, full)["coefficient"], 1.0)
        self.assertAlmostEqual(bridge.reader_projection(full, baseline, full)["cosine"], 1.0)

    def test_q4_parser_binds_target_exclusion_and_ordinal_draws(self) -> None:
        target = list(range(12))
        seed_rows = []
        for q4_seed in range(8):
            accepted = [{"draw_index": index, "latent_ids": list(range(100 + index * 12, 112 + index * 12))} for index in range(100)]
            seed_rows.append({"seed": 1000 + q4_seed, "projector": {"target_latent_ids": target}, "matched_draws": {"accepted": accepted}})
        payload = {"schema": "exp05-number-agreement-stage3-v1; frozen Q4", "status": "COMPLETE", "seed_results": seed_rows}
        parsed = bridge.parse_q4_frozen_sets(payload)
        self.assertEqual(parsed["q4_seeds"], list(range(1000, 1008)))
        self.assertEqual(len(parsed["ordinal_sets"][0]["matched"]), 100)
        self.assertTrue(set(parsed["ordinal_sets"][0]["matched"][0]["latent_ids"]).isdisjoint(target))
        reordered = {**payload, "seed_results": list(reversed(seed_rows))}
        reordered_parsed = bridge.parse_q4_frozen_sets(reordered)
        self.assertEqual(
            [item["source_q4_seed"] for item in reordered_parsed["ordinal_sets"]],
            list(range(1000, 1008)),
        )
        duplicate_seed = {**payload, "seed_results": [*seed_rows[:-1], {**seed_rows[-1], "seed": seed_rows[0]["seed"]}]}
        with self.assertRaises(bridge.BridgeStop):
            bridge.parse_q4_frozen_sets(duplicate_seed)
        broken = {**payload, "seed_results": [*seed_rows[:-1], {**seed_rows[-1], "matched_draws": {"accepted": seed_rows[-1]["matched_draws"]["accepted"][:-1]}}]}
        with self.assertRaises(bridge.BridgeStop):
            bridge.parse_q4_frozen_sets(broken)

    def test_path_aliases_and_runtime_alias_guard(self) -> None:
        self.assertEqual(bridge.resolve_path_alias("@exp05/bridge_rescue.py"), (HERE / "bridge_rescue.py").resolve())
        self.assertEqual(bridge.resolve_path_alias("@repo/README.md"), (bridge.SOURCE_WORKTREE / "README.md").resolve())
        with self.assertRaises(bridge.BridgeStop):
            bridge.resolve_path_alias("@unknown/file")
        with self.assertRaises(bridge.BridgeStop):
            bridge._validate_runtime_paths(HERE / "out.json", [HERE / "out.json"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
