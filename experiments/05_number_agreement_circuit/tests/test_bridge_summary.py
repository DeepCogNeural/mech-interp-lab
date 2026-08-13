"""Model-free contracts for Exp05 bridge presentation reaggregation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp05_bridge_summary_contract", HERE / "make_bridge_summary.py")
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


class CheckedInBridgeReaggregation(unittest.TestCase):
    def test_exact_compact_rows_recompute_published_statistics(self) -> None:
        metrics, matched, packet = summary._checked_in_rows(summary.DEFAULT_OUTPUT)
        self.assertEqual((len(metrics), len(matched)), (8, 800))
        self.assertAlmostEqual(packet["metrics"]["R_target"]["mean"], 0.6786390188122342)
        self.assertAlmostEqual(packet["metrics"]["R_target_clamped"]["mean"], 0.6740468735483219)
        self.assertEqual(packet["target_exceeds_matched_max_count"], 8)
        self.assertIs(packet["review_receipt"], None)
        self.assertEqual(packet["regeneration"]["mode"], "checked_in_hash_bound")
        self.assertIs(packet["regeneration"]["raw_result_revalidated_this_invocation"], False)
        self.assertNotIn("gate_a", packet["integrity"])
        historical = packet["integrity"]["historical_raw_dependent_report"]
        self.assertEqual(historical["status"], "REPORTED_BY_ORIGINAL_PACKET_NOT_REVALIDATED")
        self.assertEqual(
            historical["authority_receipt"]["sha256"],
            summary.EXPECTED_HISTORICAL_RECEIPT_SHA256,
        )

    def test_checked_in_build_preserves_compact_bytes_and_refreshes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "packet"
            summary.build_checked_in(output)
            self.assertEqual(
                summary.sha256_file(output / "bridge_seed_metrics.csv"),
                summary.EXPECTED_SEED_CSV_SHA256,
            )
            self.assertEqual(
                summary.sha256_file(output / "bridge_matched_ratios.csv"),
                summary.EXPECTED_MATCHED_CSV_SHA256,
            )
            packet = summary.read_json(output / "bridge_result_summary.json")
            manifest = summary.read_json(output / "bridge_figure_manifest.json")
            index = summary.read_json(output / "artifact_index.json")
            self.assertEqual(packet["regeneration"]["mode"], "checked_in_hash_bound")
            self.assertEqual(manifest["regeneration"]["mode"], "checked_in_hash_bound")
            self.assertEqual(index["bridge_followup"]["regeneration"]["mode"], "checked_in_hash_bound")
            self.assertGreater((output / "figure_bridge_rescue.svg").stat().st_size, 0)
            self.assertGreater((output / "figure_bridge_rescue.png").stat().st_size, 0)

            checksum_lines = (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(checksum_lines), 20)
            for line in checksum_lines:
                expected, relative = line.split("  ", 1)
                self.assertEqual(summary.sha256_file(output / relative), expected)

    def test_compact_byte_drift_is_rejected_before_reaggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "packet"
            summary._base_package(output)
            path = output / "bridge_seed_metrics.csv"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed CSV SHA mismatch"):
                summary._checked_in_rows(output)

    def test_historical_receipt_drift_is_rejected_before_reaggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "packet"
            summary._base_package(output)
            path = output / summary.HISTORICAL_RECEIPT_NAME
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipt["reported_integrity"]["gate_a"]["passed_count"] = 7
            path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance receipt SHA mismatch"):
                summary._checked_in_rows(output)

    def test_publication_failure_after_directory_swap_restores_prior_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "packet"
            summary._base_package(output)
            canonical_output = output.resolve()
            before = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            original_fsync = summary._fsync_path
            injected = False

            def fail_once_after_swap(path: Path) -> None:
                nonlocal injected
                if path.resolve() == canonical_output and not injected:
                    injected = True
                    raise OSError("synthetic post-swap durability failure")
                original_fsync(path)

            with mock.patch.object(summary, "_fsync_path", side_effect=fail_once_after_swap):
                with self.assertRaisesRegex(OSError, "synthetic post-swap"):
                    summary.build_checked_in(output)
            after = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)
            self.assertFalse(output.with_name(f".{output.name}.bridge-publish.lock").exists())
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.previous.*")),
                [],
            )

    def test_partial_old_generation_cleanup_failure_keeps_committed_new_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "packet"
            summary._base_package(output)
            (output / "RESULTS.md").write_text("obsolete generation\n", encoding="utf-8")
            original_rmtree = summary.shutil.rmtree
            injected = False

            def partially_delete_old_then_fail(path: Path, *args: object, **kwargs: object) -> None:
                nonlocal injected
                candidate = Path(path)
                if ".previous." in candidate.name and not injected:
                    injected = True
                    (candidate / "packet" / "RESULTS.md").unlink()
                    raise OSError("synthetic partial old-generation cleanup failure")
                original_rmtree(candidate, *args, **kwargs)

            with mock.patch.object(
                summary.shutil,
                "rmtree",
                side_effect=partially_delete_old_then_fail,
            ):
                summary.build_checked_in(output)
            self.assertTrue(injected)
            self.assertNotEqual(
                (output / "RESULTS.md").read_text(encoding="utf-8"),
                "obsolete generation\n",
            )
            for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines():
                expected, relative = line.split("  ", 1)
                self.assertEqual(summary.sha256_file(output / relative), expected)
            self.assertFalse(output.with_name(f".{output.name}.bridge-publish.lock").exists())

    def test_cli_requires_exactly_one_evidence_source(self) -> None:
        checked = summary.parse_args(["--reaggregate-checked-in"])
        self.assertTrue(checked.reaggregate_checked_in)
        raw = summary.parse_args(["--bridge-results", "/tmp/bridge_results.json"])
        self.assertEqual(raw.bridge_results, "/tmp/bridge_results.json")
        with self.assertRaises(SystemExit):
            summary.parse_args([])
        with self.assertRaises(SystemExit):
            summary.parse_args(
                ["--reaggregate-checked-in", "--bridge-results", "/tmp/bridge_results.json"]
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
