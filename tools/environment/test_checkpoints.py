"""Exercise the frozen-checkpoint path and provenance guards without running a model."""

import tempfile
import unittest
from pathlib import Path

import torch

from run_baseline import CONFIGS, REVISION, RELEASE_REVISION, WEIGHT_SHA, digest, read_record, validate_checkpoint, write_json

from stabilized_attention import metadata


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        sandbox = tempfile.TemporaryDirectory(prefix="vicad-test-checkpoint-")
        self.addCleanup(sandbox.cleanup)
        root = Path(sandbox.name)
        self.upstream = root / "upstream"
        self.run = root / "run"
        self.config = CONFIGS["dinomaly"]
        config_file = self.upstream / self.config
        config_file.parent.mkdir(parents=True)
        config_file.write_text("# synthetic config\n", encoding="utf-8")
        self.checkpoint = self.run / "checkpoints" / "DinomalyTrainer_benchmark" / "['Car']_ckpt.pth"
        self.checkpoint.parent.mkdir(parents=True)
        write_json(self.run / "config.json", {
            "class": "Car", "cfg_path": self.config, "cfg_sha256": digest(config_file),
            "upstream_revision": REVISION, "release_revision": RELEASE_REVISION,
            "weight_sha256": WEIGHT_SHA, "mode": "train", "test_during_training": False,
            "protocol": "M2AD-Synergy", "seed": 42, "epoch": 100, "batch_train": 12,
        })
        self.record(100, 30_000)

    def record(self, epoch, iteration):
        torch.save({"epoch": epoch, "iter": iteration, "net": {"param": torch.ones(1)}}, self.checkpoint)
        write_json(self.run / "status.json", {
            "status": "PASS", "mode": "train", "model": "dinomaly", "class": "Car",
            "epochs_completed": 100, "optimization_steps": 30_000,
            "final_checkpoint": {"path": self.checkpoint.relative_to(self.run).as_posix(),
                                 "bytes": self.checkpoint.stat().st_size, "sha256": digest(self.checkpoint)},
        })

    def test_exact_final_checkpoint(self):
        self.assertEqual(validate_checkpoint(self.checkpoint, "dinomaly", "Car", self.upstream),
                         digest(self.checkpoint))

    def test_stability_mode_and_patch_hash_cannot_cross(self):
        require_args = (self.checkpoint, "dinomaly", "Car", self.upstream)
        with self.assertRaisesRegex(ValueError, "attention mode"):
            validate_checkpoint(*require_args, attention_mode="stabilized")
        for name in ("config.json", "status.json"):
            file = self.run / name
            write_json(file, {**read_record(file), **metadata("stabilized")})
        self.assertEqual(validate_checkpoint(*require_args, attention_mode="stabilized"), digest(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "attention mode"):
            validate_checkpoint(*require_args)
        file = self.run / "status.json"
        write_json(file, {**read_record(file), "stability_patch_sha256": "foreign"})
        with self.assertRaisesRegex(ValueError, "stability patch"):
            validate_checkpoint(*require_args, attention_mode="stabilized")

    def test_unsafe_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected official checkpoint path"):
            validate_checkpoint(self.checkpoint.with_name("other_ckpt.pth"), "dinomaly", "Car", self.upstream)

    def test_early_epoch_and_iteration_are_rejected_even_with_matching_hash(self):
        for epoch, iteration in ((99, 30_000), (100, 29_999)):
            with self.subTest(epoch=epoch, iteration=iteration):
                self.record(epoch, iteration)
                with self.assertRaisesRegex(ValueError, "epoch 100 and iteration 30000"):
                    validate_checkpoint(self.checkpoint, "dinomaly", "Car", self.upstream)

    def test_mismatched_category_config_and_bytes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected official checkpoint path"):
            validate_checkpoint(self.checkpoint, "dinomaly", "Cube", self.upstream)
        source = self.upstream / self.config
        old = source.read_bytes()
        source.write_bytes(b"altered config")
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_checkpoint(self.checkpoint, "dinomaly", "Car", self.upstream)
        source.write_bytes(old)
        with self.checkpoint.open("ab") as stream:
            stream.write(b"altered")
        with self.assertRaisesRegex(ValueError, "recorded final training artifact"):
            validate_checkpoint(self.checkpoint, "dinomaly", "Car", self.upstream)


if __name__ == "__main__":
    unittest.main()
