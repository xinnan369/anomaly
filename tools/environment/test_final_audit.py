"""Regression checks for probes and the final baseline evidence audit."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from baseline_metrics import save_arrays
from summarize_baselines import CLEAN_CONTINUOUS, FAILED, RECOVERED_DEVIATION, row
from training_diagnostics import NonFiniteTrainingError, TrainingDiagnostics, nonfinite_names
from verify_baselines import audited_files, check_repeat, compare_labels, IDENTITY
from run_baseline import read_record, write_json


class DiagnosticTest(unittest.TestCase):
    def test_tensor_reductions_detect_nan_and_infinity(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            with self.subTest(device=device):
                bad = nonfinite_names([
                    ("finite", torch.tensor([-1., 2.], device=device)),
                    ("nan", torch.tensor([0., float("nan"), 1.], device=device)),
                    ("inf", torch.tensor([float("-inf"), 0.], dtype=torch.float64, device=device)),
                    ("no_grad", None),
                ])
                self.assertEqual(set(bad), {"nan", "inf"})

    def test_adam_probe_leaves_parameters_gradients_and_rng_identical(self):
        def step(observe):
            param = torch.nn.Parameter(torch.tensor([1., -2.]))
            net = torch.nn.ParameterList([param])
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            trainer = SimpleNamespace(net=net, optim=optimizer)
            observer = TrainingDiagnostics(trainer, None, ".")
            hook = optimizer.register_step_post_hook(observer.after_optimizer) if observe else None
            state = torch.random.get_rng_state()
            for _ in range(3):
                optimizer.zero_grad()
                (param.square().sum()).backward()
                if observe:
                    observer.gradients("gradients_before_clip")
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if observe:
                    observer.gradients("gradients_after_clip")
                optimizer.step()
            if hook:
                hook.remove()
                self.assertEqual(observer.successful_updates, 3)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            return param.detach(), param.grad, optimizer.state[param]["exp_avg"]

        self.assertTrue(all(torch.equal(old, observed) for old, observed in zip(step(False), step(True))))

    def test_first_nonfinite_keeps_stage_iteration_and_inputs(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / "rng").mkdir()
            net = torch.nn.Linear(1, 1)
            trainer = SimpleNamespace(net=net, optim=torch.optim.Adam(net.parameters(), lr=0.01), iter=7, epoch=0)
            observer = TrainingDiagnostics(trainer, root, ".")
            observer.paths = ["Doll/Good/01/image.png"]
            with self.assertRaisesRegex(NonFiniteTrainingError, "non-finite loss at iteration 7"):
                observer.check("loss", [("loss", torch.tensor(float("nan")))])
            record = read_record(root / "first-nonfinite.json")
            self.assertEqual((record["stage"], record["iteration"], record["batch_paths"]),
                             ("loss", 7, observer.paths))
            self.assertTrue((root / "rng/failure.pt").is_file())


class EvidenceTest(unittest.TestCase):
    def test_rejects_repeat_metric_or_checkpoint_change(self):
        original = {"checkpoint_sha256": "sha", "metrics_pct": {"AUPRO_view": 90.0}}
        self.assertEqual(check_repeat(original, original), {"AUPRO_view": 0.0})
        with self.assertRaisesRegex(ValueError, "repeat disagrees"):
            check_repeat(original, {**original, "metrics_pct": {"AUPRO_view": 90.01}})
        with self.assertRaisesRegex(ValueError, "changed checkpoint"):
            check_repeat(original, {**original, "checkpoint_sha256": "other"})

    def test_rejects_array_value_tampering_and_different_masks(self):
        with tempfile.TemporaryDirectory() as path:
            folder = Path(path) / "arrays"
            evidence = save_arrays(folder, {"masks": np.zeros((2, 2), dtype=np.uint8)})
            self.assertEqual(audited_files(folder, evidence, {"masks": (2, 2)})["masks"].sum(), 0)
            np.save(folder / "masks.npy", np.ones((2, 2), dtype=np.uint8), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "prediction mismatch"):
                audited_files(folder, evidence, {"masks": (2, 2)})
        evidence = {name + ".npy": {"sha256": "match"} for name in (*IDENTITY, "imgs_masks")}
        compare_labels(evidence, evidence)
        with self.assertRaisesRegex(ValueError, "labels, or masks differ"):
            compare_labels(evidence, {**evidence, "imgs_masks.npy": {"sha256": "new"}})

    def test_failed_clean_attempt_never_falls_back_to_historical_result(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            runs, evaluations = root / "runs", root / "evaluations"
            history = runs / "dinomaly/Doll"
            history.mkdir(parents=True)
            write_json(history / "status.json", {"status": "PASS", "mode": "train"})
            self.assertEqual(row(runs, evaluations, "dinomaly", "Doll")["training_provenance"], CLEAN_CONTINUOUS)
            write_json(history / "recovery.json", {"rng_and_loader_state_available": False})
            self.assertEqual(row(runs, evaluations, "dinomaly", "Doll")["training_provenance"], RECOVERED_DEVIATION)
            attempt = root / "clean"
            (attempt / "train").mkdir(parents=True)
            write_json(attempt / "train/status.json", {"status": "FAILED", "mode": "train"})
            record = row(runs, evaluations, "dinomaly", "Doll", attempt)
            self.assertEqual(record["training_provenance"], FAILED)
            self.assertIsNone(record["AUPRO_pct"])


if __name__ == "__main__":
    unittest.main()
