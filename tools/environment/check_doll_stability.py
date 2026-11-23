"""Gate the formal variant on observed rounding and on the actual upstream module."""

import argparse
import collections
from collections.abc import Iterable
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

import torch

from run_baseline import REVISION, check_upstream, digest, read_record, require
from stabilized_attention import EPSILON, GuardCounts, UPSTREAM_ATTENTION, install, source_sha256


def prove(upstream, evidence):
    check_upstream(upstream)
    observed = read_record(evidence)
    state = observed["pre_update_state"]
    layer = observed["attention_denominator"][6]
    case, = layer["zero_origin"]
    require(observed["stage_matches_clean_failure"] and observed["matching_batches"] == 141
            and observed["target"] == 11241 and observed["upstream_revision"] == REVISION
            and observed["diagnostic_script_sha256"] == digest(Path(__file__).with_name("diagnose_doll_failure.py"))
            and observed["attention_source_sha256"] == digest(upstream / UPSTREAM_ATTENTION),
            "forensic replay does not match the pinned fault and diagnostic source")
    require(not any(state[key] for key in ("parameters_nonfinite", "previous_gradients_nonfinite", "adam_nonfinite"))
            and state["input"]["nonfinite"] == 0 and state["gradient_tensors"] > 0 and state["adam_states"] > 0,
            "pre-failure tensors were not proven finite")
    require(layer["attention"] == "model.decoder.6.attn" and layer["zeros"] == layer["reciprocal_nonfinite"] == 1
            and case["coordinate"] == [10, 4, 112] and case["q_before_elu_dtype"] == "torch.float32"
            and case["q_after_elu_plus_one"]["zeros"] > 0 and case["summed_k"]["zeros"] > 0
            and case["q_before_elu"]["nonfinite"] == case["k_after_elu_plus_one"]["nonfinite"] == 0
            and 0 < case["float64_reference_denom"] < EPSILON,
            "zero denominator not explained by finite-input ELU rounding")
    return case


def check_module(upstream):
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    if not hasattr(collections, "Iterable"):
        collections.Iterable = Iterable
    from model.dinomaly_utils.vision_transformer import LinearAttention2

    def run(module, data):
        x = data.clone().requires_grad_()
        output = module(x)[0]
        grads = torch.autograd.grad(output.sum(), (x, *module.parameters()))
        return output.detach(), grads

    torch.manual_seed(42)
    net = torch.nn.ModuleList(LinearAttention2(dim=8, num_heads=2, qkv_bias=True) for _ in range(8)).cuda()
    datos = torch.linspace(-0.2, 0.2, 24, device="cuda").reshape(1, 3, 8)
    ref = [run(layer, datos) for layer in net]
    counters = GuardCounts(2, "cuda")
    counters.bind(net, LinearAttention2)
    with ExitStack() as stack:
        install(LinearAttention2, counters, stack)
        for layer, (expected, grads) in zip(net, ref, strict=True):
            actual, got = run(layer, datos)
            require(torch.equal(actual, expected) and all(torch.equal(a, b) for a, b in zip(got, grads, strict=True)),
                    "safe forward or gradient changed")
        counters.assert_valid_batch()
        with torch.no_grad():
            for layer in net:
                layer.qkv.weight[:8].zero_()
                layer.qkv.bias[:8].fill_(-24)
        for layer in net:
            output, gradients = run(layer, datos)
            require(bool(output.isfinite().all()) and all(bool(grad.isfinite().all()) for grad in gradients),
                    "zero-denominator backward or forward is not finite")
        counters.assert_valid_batch()
    observed = counters.record()
    require(observed["batches_with_protection"][0]["batch"] == 2
            and observed["zero_positions"] == observed["protected_positions"] == 48,
            "the real upstream modules did not exercise the intended guard")
    return {"ordinary_output_and_gradients_bitwise_equal": True, "zero_forward_and_gradients_finite": True,
            "guard_counts": observed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=lambda path: Path(path).resolve(), required=True)
    parser.add_argument("--proof", type=lambda path: Path(path).resolve(), required=True)
    args = parser.parse_args()
    causal = prove(args.upstream, args.proof)
    reference = check_module(args.upstream)
    print(json.dumps({"status": "PASS", "upstream_revision": REVISION, "epsilon": EPSILON,
                      "patch_sha256": source_sha256(), "proof_sha256": digest(args.proof),
                      "observed_zero": causal, "upstream_module_test": reference}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
