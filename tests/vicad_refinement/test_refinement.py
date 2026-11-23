import unittest
from pathlib import Path

import torch

from vicad_refinement.development import select, split_development
from vicad_refinement.model import estimate_prototypes, patch_weights, residuals, weighted_lights
from vicad_refinement.synthetic import perturb
from vicad_syn.data.manifest import Group
from vicad_syn.experiment import read_json
from vicad_syn.fusion.weights import light_weights
from vicad_syn.models.predictor import Predictor


REGISTRY = read_json(Path(__file__).resolve().parents[2] / "vicad_syn/refinement/protocol.json")
torch.set_num_threads(1)


class RefinementTests(unittest.TestCase):
    def test_leave_target_out_each_context(self):
        from vicad_refinement.model import context_predictions
        torch.manual_seed(2)
        model = Predictor(channels=8, heads=3, hidden=4).eval()
        x = torch.nn.functional.normalize(torch.randn(1, 10, 4, 8), dim=-1)
        reference = x.clone()
        changed = x.clone()
        changed[:, 2] = 25
        first = context_predictions(x, model, reference)
        second = context_predictions(changed, model, reference)
        for key in first:
            torch.testing.assert_close(first[key][:, 2], second[key][:, 2], atol=0, rtol=0)

    def test_layer_collation_and_scores(self):
        torch.manual_seed(4)
        x = torch.randn(2, 10, 3, 4, 8)
        model = Predictor(channels=8, heads=3, hidden=4).eval()
        proto = estimate_prototypes(x, torch.tensor([0, 0]), view_count=1)
        scores, s, e, layers = residuals(x, model, proto)
        self.assertEqual(layers.shape, (2, 10, 3, 4))
        self.assertEqual(s.shape, (2, 10, 4))
        self.assertTrue(torch.isfinite(e).all())
        torch.testing.assert_close(scores["D_combined"], (scores["B_multilayer"] + scores["C_normal"]) / 2)

    def test_weight_finiteness_and_zero_when_all_invalid(self):
        residual = torch.ones((2, 10, 2))
        signals = torch.zeros_like(residual)
        residual[0, :, 0] = float("nan")
        signals[1, 0, 0] = float("inf")
        w = patch_weights(residual, (signals,))
        self.assertTrue(torch.isfinite(w).all() and (w >= 0).all())
        self.assertEqual(w[1, 0, 0].item(), 0)
        torch.testing.assert_close(w.sum(dim=1), torch.tensor([[0., 1.], [1., 1.]]))
        for mode in REGISTRY["fusion_modes"]:
            self.assertTrue(weighted_lights(residual, signals, signals, mode, 0.25).isfinite().all())

    def test_local_protection_has_equal_fusion_floor(self):
        residual = torch.zeros((1, 10, 2))
        signal = residual.clone()
        residual[0, 3, 0] = 3.0
        signal[0, 3, 0] = 1.0
        original = weighted_lights(residual, signal, signal, "equal", .25).mean(1)
        protected = weighted_lights(residual, signal, signal, "protected_both", .25).mean(1)
        self.assertTrue((protected >= .25 * original).all())

    def test_v1_global_exact(self):
        torch.manual_seed(1)
        x, s, e = torch.rand(3, 10, 5), torch.rand(3, 10, 5), torch.rand(3, 10, 5)
        legacy = 10 * light_weights(x, (s, e))[:, :, None] * x
        torch.testing.assert_close(weighted_lights(x, s, e, "v1_global", .25), legacy, atol=0, rtol=0)

    def test_preserve_normal_object_isolation(self):
        groups = [Group(str(i), "000", ()) for i in range(30)]
        front, back = split_development(groups, REGISTRY)
        self.assertEqual(len(front), 24)
        self.assertEqual(len(back), 6)
        self.assertFalse(front & back)

    def test_synthetic_case_cardinality_and_determinism(self):
        images = torch.zeros((10, 3, 224, 224))
        for scenario, count in zip(REGISTRY["synthetic"], (0, 1, 4, 10, 10, 1)):
            first, ids, mask = perturb(images, scenario, 15, REGISTRY)
            second, again, check = perturb(images, scenario, 15, REGISTRY)
            self.assertEqual(len(ids), count)
            self.assertEqual(again, ids)
            self.assertEqual(bool(mask.any()), scenario not in ("clean", "light_only"))
            torch.testing.assert_close(first, second)
            torch.testing.assert_close(mask, check)

    def test_predeclared_selection_order_and_missing_guard(self):
        rows = [{"candidate": candidate, "fusion": fusion, "category": category,
                 "scenario": scenario, "pixel_ap": .2}
                for candidate in REGISTRY["candidates"] for fusion in REGISTRY["fusion_modes"]
                for category in REGISTRY["development_categories"] for scenario in REGISTRY["synthetic"][1:-1]]
        best, ranking = select(rows, REGISTRY)
        self.assertEqual((best["candidate"], best["fusion"]), ("A_final", "equal"))
        self.assertEqual(len(ranking), 24)
        with self.assertRaises(ValueError):
            select(rows[:-1], REGISTRY)
