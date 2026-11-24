from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from unittest.mock import patch
import torch

from tools.comprehensive.ablation import ABLATION_SOURCES, PROTOCOL, ablation_fields, canonical_scores, evaluation_metadata
from vicad_syn.data.manifest import Group
from vicad_syn.evaluation.synergy import official_inputs
from tools.comprehensive.historical_intervals import frozen_labels
from vicad_syn.experiment import digest, read_json

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/environment"))
from run_baseline import expected_iterations, official_counts


class AblationTests(unittest.TestCase):
    def test_freeze_anchor_and_leaf_baseline(self):
        plan = read_json(PROTOCOL)
        self.assertEqual(plan["ablation"]["B6"], ["D_combined", "patch_stability"])
        self.assertEqual(ABLATION_SOURCES["B6"], "selected")
        normal = torch.arange(4 * 10 * 3 * 2).reshape(4, 10, 3, 2)
        self.assertTrue(torch.equal(ablation_fields({}, normal)["C_leaf"], normal[:, :, 2]))

    def test_canonical_scores_preserve_good_bad_with_same_display_names(self):
        groups = [Group("one", "000", ({"cls_name": "Bird", "object_name": "003", "object_anomaly": 0},)),
                  Group("two", "000", ({"cls_name": "Bird", "object_name": "003", "object_anomaly": 1},))]
        canonical = canonical_scores(groups, [False, False], np.array([[.25] * 10, [.75] * 10]))
        self.assertEqual([x["identity"] for x in canonical["objects"]],
                         [["Bird", "003", 0], ["Bird", "003", 1]])
        self.assertEqual([x["score"] for x in canonical["views"]], [.25, .75])
        self.assertEqual([x["label"] for x in canonical["views"]], [0, 0])

    def test_official_exception_counts_for_dice_and_holder(self):
        self.assertEqual(official_counts("Dice"), (3480, 8400, 70, 290, 700))
        self.assertEqual(official_counts("Holder"), (3600, 8280, 69, 300, 690))
        self.assertEqual(expected_iterations("Dice"), 29_000)
        self.assertEqual(expected_iterations("Holder"), 30_000)


    def test_frozen_masks_bind_a_reused_good_and_ng_name(self):
        groups = [Group("bad", "000", ({"object_name": "003", "object_anomaly": 1},)),
                  Group("good", "000", ({"object_name": "003", "object_anomaly": 0},))]
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            receipt = {"scores": {}}
            for setting, masks in [("selected", [np.zeros((2, 2)), np.ones((2, 2))]),
                                   ("v1_A4", [np.ones((2, 2)), np.zeros((2, 2))])]:
                folder = root / setting
                folder.mkdir()
                np.save(folder / "view_instance_names.npy", ["003_000", "003_000"])
                np.save(folder / "view_instance_masks.npy", masks)
                receipt["scores"][setting] = {"fused_and_patch_inputs": {
                    file.name: {"sha256": digest(file)} for file in folder.glob("*.npy")}}
            self.assertEqual(frozen_labels(groups, root, receipt).tolist(), [True, False])


    def test_mask_cache_reuses_paths_and_preserves_official_resize(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            Image.fromarray(np.array([[0, 255], [0, 255]], dtype=np.uint8)).save(root / "defect.png")
            row = {"object_name": "003", "cls_name": "Bird", "illumination": "01",
                   "view": "000", "object_anomaly": 1, "image_anomaly": 1, "mask_path": "defect.png"}
            groups = [Group("id", "000", (row, {**row, "illumination": "02"}))]
            original = official_inputs(groups, root, np.zeros((2, 1, 1)))
            with patch("tools.comprehensive.ablation.Image.open", wraps=Image.open) as source:
                checked = evaluation_metadata(groups, root)
            source.assert_called_once_with(root / "defect.png")
            for key, value in checked.items():
                self.assertTrue(np.array_equal(value, original[key]), key)
