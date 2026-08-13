"""Model-free contracts for Experiment 02's shipped-row reaggregation.

These tests never train the toy autoencoder or fit a probe. They protect the
algebraic claim, supported Student-t intervals, seed-level aggregation, and
the exact completed-run row grid used by ``REAGGREGATE_ONLY``.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp02_readout_contract", HERE / "readout.py")
assert SPEC is not None and SPEC.loader is not None
readout = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = readout
SPEC.loader.exec_module(readout)


def _shipped_rows() -> tuple[list[list[object]], list[list[object]], list[list[object]]]:
    payload = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    return payload["xor"], payload["enum"], payload["status"]


class Exp02Contracts(unittest.TestCase):
    def test_additive_xor_identity_and_constructive_witness(self) -> None:
        constant = 0.37
        weight_i, weight_j = 1.1, -0.7
        positive_i, positive_j = 0.23, 0.81
        score = lambda a, b: constant + weight_i * a + weight_j * b
        self.assertAlmostEqual(
            score(0.0, positive_j) + score(positive_i, 0.0),
            score(0.0, 0.0) + score(positive_i, positive_j),
        )

        # Use unequal signed magnitudes from each quadrant, then apply the
        # experiment's actual coordinate-wise ReLU representation. The raw
        # rule x_i+x_j>0 would not be a valid 0.75 witness for these points.
        corners = ((-0.2, -0.9), (-0.8, 0.1), (0.3, -0.7), (0.4, 0.6))
        xor = (0, 1, 1, 0)
        predictions = tuple(int(max(a, 0.0) + max(b, 0.0) > 0) for a, b in corners)
        self.assertEqual(predictions, (0, 1, 1, 1))
        self.assertEqual(sum(int(pred == label) for pred, label in zip(predictions, xor)) / 4, 0.75)

        payload = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
        boundary = payload["analysis"]["coordinatewise_math"]
        self.assertEqual(boundary["theorem"], "additive linear scores cannot perfectly separate XOR")
        self.assertIs(boundary["chance_ceiling"], False)
        self.assertEqual(boundary["constructive_witness_accuracy"], 0.75)
        self.assertIn("ReLU(x_i)+ReLU(x_j)>0", boundary["witness"])

    def test_ci95_uses_registered_student_t_and_rejects_unregistered_n(self) -> None:
        self.assertEqual(readout._T975, {2: 4.302652730, 7: 2.364624252})
        self.assertAlmostEqual(readout.ci95([0.0, 1.0, 2.0]), 2.484137711895, places=10)
        self.assertAlmostEqual(readout.ci95(list(range(8))), 2.047824672637, places=10)
        self.assertEqual(readout.ci95([1.0]), 0.0)
        with self.assertRaisesRegex(ValueError, "no supported t critical value"):
            readout.ci95([0.0, 1.0, 2.0, 3.0])

    def test_pooled_contrasts_average_s_within_each_seed(self) -> None:
        rows: list[tuple[object, ...]] = []
        for sparsity, seed, difference in ((0.0, 0, 1.0), (0.9, 0, 3.0), (0.0, 1, 10.0), (0.9, 1, 14.0)):
            rows.append((8, sparsity, seed, 0.0, "random", 0.0, 0.0))
            rows.append((8, sparsity, seed, 0.0, "superposition", difference, 0.0))
            rows.append((8, sparsity, seed, 0.0, "monosemantic", -difference, 0.0))

        self.assertEqual(
            readout.paired_diff(rows, 8, 0.0),
            [(0.0, 0, 1.0), (0.0, 1, 10.0), (0.9, 0, 3.0), (0.9, 1, 14.0)],
        )
        self.assertEqual(readout.pooled_seed_diffs(rows, 8, 0.0), [2.0, 12.0])
        self.assertEqual(
            readout.pooled_seed_arm_diffs(rows, 8, 0.0, "superposition", "monosemantic"),
            [4.0, 24.0],
        )

    def test_shipped_raw_grid_is_exact_complete_unique_and_finite(self) -> None:
        xor_rows, enum_rows, status_rows = _shipped_rows()
        readout.validate_shipped_raw_rows(xor_rows, enum_rows, status_rows)
        self.assertEqual((len(xor_rows), len(enum_rows), len(status_rows)), (600, 360, 120))
        self.assertEqual(readout._canonical_sha256(xor_rows), readout.SHIPPED_RAW_SHA256["xor"])
        self.assertEqual(
            readout._canonical_sha256({"xor": xor_rows, "enum": enum_rows, "status": status_rows}),
            readout.SHIPPED_RAW_SHA256["combined"],
        )

        missing = copy.deepcopy(xor_rows)
        missing.pop()
        with self.assertRaisesRegex(ValueError, "xor grid drift"):
            readout.validate_shipped_raw_rows(missing, enum_rows, status_rows)

        duplicate = copy.deepcopy(xor_rows)
        duplicate[-1] = copy.deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            readout.validate_shipped_raw_rows(duplicate, enum_rows, status_rows)

        nonfinite = copy.deepcopy(xor_rows)
        nonfinite[0][5] = float("nan")
        with self.assertRaisesRegex(ValueError, "not finite"):
            readout.validate_shipped_raw_rows(nonfinite, enum_rows, status_rows)

    def test_reaggregate_is_model_free_and_preserves_raw_rows(self) -> None:
        source = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
        original_rows = {key: copy.deepcopy(source[key]) for key in ("xor", "enum", "status")}

        def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("training or probe code was called during reaggregation")

        with tempfile.TemporaryDirectory() as temporary:
            temp_dir = Path(temporary)
            figures = temp_dir / "figures"
            figures.mkdir()
            (temp_dir / "results.json").write_text(json.dumps(source), encoding="utf-8")
            with (
                mock.patch.object(readout, "HERE", str(temp_dir)),
                mock.patch.object(readout, "FIGDIR", str(figures)),
                mock.patch.object(readout, "train_ae", forbidden),
                mock.patch.object(readout, "probe", forbidden),
                mock.patch.object(readout, "xor_acc", forbidden),
                mock.patch.object(readout, "enum_acc", forbidden),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                readout.reaggregate_existing()

            regenerated = json.loads((temp_dir / "results.json").read_text(encoding="utf-8"))
            self.assertEqual({key: regenerated[key] for key in original_rows}, original_rows)
            self.assertEqual(
                regenerated["analysis"],
                readout.analysis_summary([tuple(row) for row in original_rows["xor"]], 8, [0.0, 0.05, 0.1]),
            )
            self.assertEqual(
                regenerated["artifact_status"],
                {
                    "raw_rows": "loaded unchanged from the existing completed run",
                    "analysis": "recomputed from shipped xor rows",
                    "figures": "regenerated from shipped rows without fitting a model or probe",
                    "mode": "CHECKED_IN_REAGGREGATION",
                    "public_artifact": True,
                    "output_root": "canonical experiment directory",
                },
            )
            for name in (readout.HEADLINE_FIGURE, "02_superposition_vs_random_paired.png"):
                self.assertGreater((figures / name).stat().st_size, 0)

    def test_smoke_output_is_separate_and_cannot_reaggregate_canonical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(readout._output_root(str(root), True), str(root / "smoke_output"))
            self.assertEqual(readout._output_root(str(root), False), str(root))
        with mock.patch.object(readout, "SMOKE", True), mock.patch.object(
            readout,
            "REAGGREGATE_ONLY",
            True,
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
                readout.run()

    def test_figure_failure_cannot_publish_a_new_result_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            figures = root / "figures"
            figures.mkdir()
            result = root / "results.json"
            result.write_bytes(b"old-result\n")
            with mock.patch.object(readout, "FIGDIR", str(figures)), mock.patch.object(
                readout,
                "plot_and_summarize",
                side_effect=RuntimeError("synthetic figure failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic figure failure"):
                    readout._publish_payload_and_figures(
                        {"new": True},
                        str(result),
                        [],
                        [],
                        [],
                        20,
                        8,
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
            self.assertEqual(result.read_bytes(), b"old-result\n")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
