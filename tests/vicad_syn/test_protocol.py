import unittest

import torch

from vicad_syn.data.manifest import LIGHTS, VIEWS, groups, split_ids
from vicad_syn.fusion.weights import fuse_lights, light_weights
from vicad_syn.models.predictor import Predictor
from vicad_syn.uncertainty.signals import context_instability, epistemic_uncertainty


def manifest(number, bucket, view, light, label):
    return dict(img_path=f"Car/{bucket}/{number}/A{view}_I{light}.png", cls_name="Car",
                object_name=number, view=view, illumination=light, image_anomaly=label, object_anomaly=label)


class TestData(unittest.TestCase):
    def setUp(self):
        self.rows = [manifest("001", bucket, view, light, label)
                     for bucket, label in (("Good", 0), ("NG", 1)) for view in VIEWS for light in reversed(LIGHTS)]

    def test_same_view_ten_ordered_lights_and_duplicate_bare_object_names(self):
        found = groups(self.rows, "Car")
        self.assertEqual(len(found), 24)
        for group in found:
            self.assertEqual(tuple(row["illumination"] for row in group.records), LIGHTS)
            self.assertEqual({row["view"] for row in group.records}, {group.view})
        self.assertEqual(len({group.identity for group in found}), 2)

    def test_groups_ignore_ground_truth_for_inference_identity(self):
        def keys(rows):
            return [(group.identity, group.view, group.paths) for group in groups(rows, "Car")]
        self.assertEqual(keys(self.rows), keys([{**row, "object_anomaly": 1-row["object_anomaly"],
                                               "image_anomaly": 1-row["image_anomaly"]} for row in self.rows]))

    def test_no_partial_or_duplicate_group(self):
        with self.assertRaisesRegex(ValueError, "incomplete view"):
            groups(self.rows[:-1], "Car")
        with self.assertRaisesRegex(ValueError, "duplicate illumination"):
            groups(self.rows + [self.rows[0]], "Car")

    def test_internal_split_is_repeatable_and_grouped_by_object(self):
        grouped = groups(self.rows, "Car")
        fit, valid = split_ids(grouped, 42, 1)
        self.assertEqual((fit, valid), split_ids(grouped[::-1], 42, 1))
        self.assertTrue(fit.isdisjoint(valid))
        self.assertEqual(sum(group.identity in fit for group in grouped), 12)


class TestPredictor(unittest.TestCase):
    def test_target_light_cannot_reach_any_of_its_head_outputs(self):
        torch.manual_seed(42)
        for conditioned in (False, True):
            model = Predictor(8, 7, 3, conditioned)
            with torch.no_grad():
                for head in model.heads:
                    head.output[-1].weight.normal_()
            x = torch.randn(2, 10, 7, 8)
            y = x.clone()
            for target in range(10):
                y.copy_(x)
                y[:, target] = torch.nan
                for subset in ("full", "first", "last"):
                    original = model(x, subset)[:, :, target]
                    changed = model(y, subset)[:, :, target]
                    self.assertTrue(torch.equal(original, changed))
            self.assertEqual(model(x).shape, (3, 2, 10, 7, 8))

    def test_target_feature_gradient_is_exactly_zero(self):
        model = Predictor(8, 7, 3)
        x = torch.randn(1, 10, 5, 8, requires_grad=True)
        gradient = torch.autograd.grad(model(x)[:, :, 4].sum(), x)[0]
        self.assertEqual(gradient[:, 4].count_nonzero(), 0)
        self.assertGreater(gradient[:, :4].count_nonzero(), 0)

    def test_uncertainty_distinguishes_variation_sources(self):
        a = torch.zeros(3, 1, 10, 4, 2)
        high_context = context_instability(a, a + 2)
        self.assertTrue((high_context > 0).all())
        self.assertFalse(context_instability(a, a).any())
        self.assertFalse(epistemic_uncertainty(a + 2).any())
        changed = a.clone()
        changed[1] += 2
        self.assertTrue((epistemic_uncertainty(changed) > 0).all())
        self.assertFalse(context_instability(changed, changed).any())


class TestFusion(unittest.TestCase):
    def test_uniform_fusion_matches_view_mean(self):
        residual = torch.arange(60, dtype=torch.float32).reshape(2, 10, 3)
        weights = light_weights(residual, [])
        torch.testing.assert_close(weights, torch.full_like(weights, 0.1))
        torch.testing.assert_close(fuse_lights(residual, weights), residual.mean(1))

    def test_failed_light_and_all_invalid_are_finite(self):
        residual = torch.ones(2, 10, 3)
        residual[0, 3] = float("nan")
        residual[1] = float("inf")
        signal = torch.zeros_like(residual)
        signal[0, 2] = float("nan")
        weights = light_weights(residual, [signal])
        torch.testing.assert_close(weights.sum(1), torch.tensor([1.,0.]))
        self.assertEqual(weights[0, 2].item(), 0)
        self.assertEqual(weights[0, 3].item(), 0)
        self.assertTrue(torch.isfinite(fuse_lights(residual, weights)).all())

    def test_confidence_decreases_under_higher_disagreement(self):
        residual = torch.ones(1, 10, 3)
        signal = torch.zeros((1,10,3))
        signal[:, 4] = torch.finfo(torch.float32).max
        weights = light_weights(residual, [signal])
        self.assertTrue((weights >= 0).all())
        self.assertLess(weights[0, 4], weights[0, 0])
        self.assertTrue(torch.isfinite(fuse_lights(residual * signal.max(), weights)).all())


if __name__ == "__main__":
    unittest.main()
