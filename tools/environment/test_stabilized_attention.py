"""Regression checks for the pre-fixed denominator guard."""

from contextlib import ExitStack
import unittest

import torch

from stabilized_attention import EPSILON, GuardCounts, install, metadata, stabilized_reciprocal


class LinearAttention2(torch.nn.Module):
    def forward(self, q, k):
        z = 1.0 / torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
        return z


class StabilizedAttentionTest(unittest.TestCase):
    def test_reciprocal_and_gradients_equal_above_epsilon(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([EPSILON * 1.1, 0.01, 5., 1e6], dtype=dtype, requires_grad=True)
            original = 1. / x
            patched = stabilized_reciprocal(x)
            self.assertTrue(torch.equal(original, patched))
            self.assertTrue(torch.equal(torch.autograd.grad(original.sum(), x)[0],
                                        torch.autograd.grad(patched.sum(), x)[0]))

    def test_zero_denominator_produces_finite_forward_and_gradient(self):
        q = torch.zeros((1, 1, 1, 3), requires_grad=True)
        k = torch.ones((1, 1, 2, 3), requires_grad=True)
        denom = torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
        output = stabilized_reciprocal(denom) * q.sum()
        gradients = torch.autograd.grad(output.sum(), (q, k))
        self.assertTrue(output.isfinite().all())
        self.assertTrue(all(gradient.isfinite().all() for gradient in gradients))

    def test_runtime_patch_counts_position_and_restores_original(self):
        net = torch.nn.ModuleList(LinearAttention2() for _ in range(8))
        counter = GuardCounts(2, "cpu")
        counter.bind(net, LinearAttention2)
        q = torch.zeros((1, 1, 1, 2))
        k = torch.ones_like(q)
        with ExitStack() as scope:
            install(LinearAttention2, counter, scope)
            for layer in net:
                self.assertTrue(layer(q, k).isfinite().all())
            counter.assert_valid_batch()
            for layer in net:
                self.assertTrue(layer(torch.ones_like(q), k).isfinite().all())
            counter.assert_valid_batch()
        self.assertTrue(net[0](q, k).isinf().all())
        record = counter.record()
        self.assertEqual((record["protected_positions"], record["zero_positions"]), (8, 8))
        self.assertEqual([event["batch"] for event in record["batches_with_protection"]], [1])

    def test_guard_rejects_invalid_original_denominator(self):
        net = torch.nn.ModuleList(LinearAttention2() for _ in range(8))
        counter = GuardCounts(1, "cpu")
        counter.bind(net, LinearAttention2)
        with ExitStack() as scope:
            install(LinearAttention2, counter, scope)
            for layer in net:
                layer(torch.full((1, 1, 1, 2), -1.), torch.ones((1, 1, 1, 2)))
            with self.assertRaisesRegex(ValueError, "negative original"):
                counter.assert_valid_batch()

    def test_mode_identity(self):
        self.assertEqual(metadata("official"), {"attention_mode": "official"})
        self.assertEqual(metadata("stabilized")["model_identity"], "Dinomaly-stabilized")
        self.assertEqual(metadata("stabilized")["epsilon"], 1e-6)
