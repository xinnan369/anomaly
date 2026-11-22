"""Read-only probes around the pinned Dinomaly trainer's existing operations."""

from collections import defaultdict
import json
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch


class NonFiniteTrainingError(RuntimeError):
    """The first monitored training stage with a non-finite tensor."""


@torch.no_grad()
def nonfinite_names(named_tensors):
    """Reduce on the tensor device, with one host transfer per device/dtype group."""
    groups = defaultdict(list)
    for name, tensor in named_tensors:
        if tensor is not None and tensor.numel() and tensor.is_floating_point():
            groups[tensor.device, tensor.dtype].append((name, tensor.detach()))
    invalid = []
    for group in groups.values():
        names, tensors = zip(*group)
        maxima = torch._foreach_norm(tensors, float("inf"))
        finite = torch.stack(maxima).isfinite().tolist()
        invalid.extend(name for name, ok in zip(names, finite) if not ok)
    return invalid


class TrainingDiagnostics:
    def __init__(self, trainer, output, dataset_root):
        self.trainer = trainer
        self.output = output
        self.dataset_root = Path(dataset_root)
        self.paths = []
        self.successful_updates = 0
        self.stage_checks = defaultdict(int)
        self.params = [(name, param) for name, param in trainer.net.named_parameters()
                       if param.requires_grad]
        self.names = {param: name for name, param in self.params}

    def save_rng(self, name):
        """Capture parent RNGs at a boundary; transforms in this configuration are deterministic."""
        state = {
            "epoch": self.trainer.epoch,
            "iteration": self.trainer.iter,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        }
        torch.save(state, self.output / "rng" / f"{name}.pt")

    def check(self, stage, named_tensors):
        self.stage_checks[stage] += 1
        invalid = nonfinite_names(named_tensors)
        if invalid:
            self.save_rng("failure")
            self.write_record("first-nonfinite.json", {
                "stage": stage,
                "tensor_names": invalid,
                "epoch_completed": self.trainer.epoch,
                "iteration": self.trainer.iter,
                "last_successful_update": self.successful_updates,
                "batch_paths": self.paths,
                "learning_rates": [group["lr"] for group in self.trainer.optim.param_groups],
                "stage_check_counts": dict(self.stage_checks),
            })
            raise NonFiniteTrainingError(f"non-finite {stage} at iteration {self.trainer.iter}: {invalid[:6]}")

    def write_record(self, name, record):
        path = self.output / name
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)

    def batch(self, inputs, stream):
        self.paths = [Path(path).relative_to(self.dataset_root).as_posix() for path in inputs["img_path"]]
        stream.write(json.dumps({"iteration": self.trainer.iter, "paths": self.paths},
                                separators=(",", ":")) + "\n")
        self.check("preprocessed_input", [("img", inputs["img"])])

    def gradients(self, stage):
        self.check(stage, ((name, param.grad) for name, param in self.params))

    def after_optimizer(self, optimizer, args, kwargs):
        self.check("adam_moments", (
            (f"{self.names[param]}.{key}", value)
            for param, state in optimizer.state.items()
            for key, value in state.items()
            if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
        ))
        self.check("updated_parameters", self.params)
        self.successful_updates += 1

    def install(self, upstream, stack):
        trainer = self.trainer
        loader = trainer.train_loader
        if (trainer.iter or trainer.epoch or trainer.cfg.trainer.resume_dir
                or trainer.cfg.model.kwargs["checkpoint_path"] or trainer.loss_scaler
                or trainer.cfg.loss.clip_grad is None or type(trainer.optim) is not torch.optim.Adam
                or loader.persistent_workers or loader.generator is not None
                or type(loader.sampler) is not torch.utils.data.RandomSampler):
            raise ValueError("Doll diagnostics require a fresh pinned Dinomaly trainer")
        (self.output / "rng").mkdir()
        stream = stack.enter_context((self.output / "batch-order.jsonl").open(
            "w", encoding="utf-8", buffering=1))
        set_input, forward, compute_loss = trainer.set_input, trainer.forward, trainer.compute_loss
        save, clip = trainer.save_checkpoint, upstream.dispatch_clip_grad

        def checked_input(inputs, *args, **kwargs):
            self.batch(inputs, stream)
            return set_input(inputs, *args, **kwargs)

        def checked_forward(*args, **kwargs):
            result = forward(*args, **kwargs)
            self.check("encoder_features", ((f"en.{i}", tensor) for i, tensor in enumerate(trainer.en)))
            self.check("decoder_features", ((f"de.{i}", tensor) for i, tensor in enumerate(trainer.de)))
            return result

        def checked_loss(*args, **kwargs):
            value, logs = compute_loss(*args, **kwargs)
            self.check("loss", [("dinomaly_loss", value)])
            return value, logs

        def checked_clip(*args, **kwargs):
            self.gradients("gradients_before_clip")
            result = clip(*args, **kwargs)
            self.gradients("gradients_after_clip")
            return result

        def save_with_rng():
            result = save()
            self.save_rng(f"epoch-{trainer.epoch:03d}")
            return result

        trainer.set_input, trainer.forward, trainer.compute_loss = checked_input, checked_forward, checked_loss
        trainer.save_checkpoint = save_with_rng
        upstream.dispatch_clip_grad = checked_clip
        stack.callback(setattr, upstream, "dispatch_clip_grad", clip)
        handle = trainer.optim.register_step_post_hook(self.after_optimizer)
        stack.callback(handle.remove)
        self.check("initial_parameters", trainer.net.named_parameters())
        self.save_rng("initial")
        self.write_record("diagnostic-environment.json", {
            "python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "workers": loader.num_workers, "persistent_workers": loader.persistent_workers,
            "loader_sampler": type(loader.sampler).__name__, "sampler_uses_default_generator": True,
            "training_transform": repr(loader.dataset.transform),
        })
        with (self.output / "dependency-freeze.txt").open("w", encoding="utf-8") as target:
            subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], stdout=target, check=True)

    def finish(self):
        self.write_record("finite-diagnostics.json", {
            "status": "PASS" if self.successful_updates == 30_000 else "INCOMPLETE",
            "successful_updates": self.successful_updates,
            "stage_check_counts": dict(self.stage_checks),
        })
