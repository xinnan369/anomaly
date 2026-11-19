"""Check that the formal table is bound to a completed evaluation and checkpoint."""

import tempfile
import unittest
from pathlib import Path

from run_baseline import write_json
from summarize_baselines import row


class SummaryTest(unittest.TestCase):
    def setUp(self):
        sandbox = tempfile.TemporaryDirectory(prefix="vicad-test-summary-")
        self.addCleanup(sandbox.cleanup)
        root = Path(sandbox.name)
        self.runs, self.evaluations = root / "runs", root / "evaluations"
        self.train = self.runs / "dinomaly" / "Car"
        self.test = self.evaluations / "dinomaly" / "Car"
        self.train.mkdir(parents=True)
        write_json(self.train / "status.json", {
            "status": "PASS", "mode": "train", "epochs_completed": 100, "optimization_steps": 30_000,
            "final_checkpoint": {"sha256": "fixture"},
        })

    def setup_evaluation(self, status, checksum="fixture"):
        self.test.mkdir(parents=True)
        write_json(self.test / "status.json", {"status": status, "evaluation_checkpoint_sha256": checksum})
        write_json(self.test / "synergy.json", {
            "epoch": 100, "model": "dinomaly", "class": "Car", "checkpoint_sha256": checksum,
            "metrics_pct": {"AUROC_obj_sp": 90, "AUROC_view_sp": 80, "AUPRO_view": 50},
        })
        for path in ("predictions.json", "diagnostics.csv", "gpu.csv"):
            (self.test / path).touch()

    def summary(self):
        return row(self.runs, self.evaluations, "dinomaly", "Car")

    def test_running_evaluation_keeps_metrics_empty(self):
        self.setup_evaluation("RUNNING")
        result = self.summary()
        self.assertEqual(result["status"], "EVALUATING")
        self.assertIsNone(result["O-AUROC_pct"])

    def test_passed_evaluation_reports_bound_metrics(self):
        self.setup_evaluation("PASS")
        result = self.summary()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["iterations"], 30_000)
        self.assertEqual(result["I-AUROC_pct"], 80)

    def test_mismatched_checkpoint_is_rejected(self):
        self.setup_evaluation("PASS", checksum="mismatched")
        with self.assertRaisesRegex(ValueError, "do not match the training checkpoint"):
            self.summary()


if __name__ == "__main__":
    unittest.main()
