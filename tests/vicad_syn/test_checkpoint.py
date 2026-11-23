"""Exercise the normal-training epoch journal across an interrupted process."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from vicad_syn.training import train


class CheckpointResumeTest(unittest.TestCase):
    def test_interrupted_epoch_resumes_to_identical_predictor(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        features = torch.randn(30,10,2,768)
        groups = [SimpleNamespace(identity=f"obj-{i:02d}") for i in range(30)]
        protocol = {"config_sha256":"test-config","config":{
            "seed":42,"epochs":2,"dev_epochs":1,"dev_objects":24,"batch_groups":5,
            "hidden":5,"heads":2,"learning_rate":0.001,"weight_decay":0.0001}}
        cache = {"sha256":"test-cache"}
        original_save = train.save_checkpoint

        def checkpoint_then_interrupt(path, state):
            original_save(path,state)
            if path.name == "latest.pt" and len(state["epochs"]) == 1:
                raise InterruptedError("after durable epoch-1 checkpoint")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fresh = train.fit(root/"fresh",features,groups,cache,protocol,True,"formal")
            target = root/"resume"
            with patch.object(train,"save_checkpoint",side_effect=checkpoint_then_interrupt):
                with self.assertRaises(InterruptedError):
                    train.fit(target,features,groups,cache,protocol,True,"formal")
            self.assertTrue((target/"latest.pt").is_file())
            recovered = train.fit(target,features,groups,cache,protocol,True,"formal")
            left = torch.load(root/"fresh"/"checkpoint.pt",weights_only=True)["model"]
            right = torch.load(target/"checkpoint.pt",weights_only=True)["model"]
            self.assertEqual(fresh["steps"],recovered["steps"])
            self.assertTrue(all(torch.equal(left[name],value) for name,value in right.items()))


if __name__ == "__main__":
    unittest.main()
