"""Execute TASK-004 in gated development and formal evaluation phases."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess

import torch

from vicad_refinement.data import encode, load_features, load_groups
from vicad_refinement.development import csv_write, select, split_development, study
from vicad_refinement.model import MultiLayerBackbone
from vicad_refinement.training import fit
from vicad_syn.experiment import check, digest, pinned_upstream, protocol, read_json, write_json


REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "vicad_syn/refinement/protocol.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def source_binding():
    return {path.relative_to(REPO).as_posix(): digest(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def fixed_protocol(registry):
    return {"registered_sha256": digest(REGISTRY), "inherited_v1": protocol(), "versions": source_binding(),
            "heldout": registry["confirmatory_order"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "freeze", "confirm", "explore"))
    for key in ("inputs", "output"):
        parser.add_argument(f"--{key}", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--holdout", type=lambda p: Path(p).resolve())
    parser.add_argument("--category")
    args = parser.parse_args()
    check(not args.output.is_relative_to(REPO), "keep experiment data outside Git")
    args.output.mkdir(parents=True, exist_ok=True)
    registry = read_json(REGISTRY)
    frozen = fixed_protocol(registry)
    base = frozen["inherited_v1"]["config"]
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "freeze":
            seal(args.output, registry, frozen)
            return
        check(torch.cuda.is_available() and torch.cuda.device_count() == 1, "requires one CUDA device")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        with pinned_upstream(args.inputs / "inputs/upstream", base["upstream_revision"]):
            encoder = MultiLayerBackbone(args.inputs / "inputs/weights" / base["backbone"],
                                         base, registry["feature_blocks_one_based"]).to(device)
            check(not encoder.training and all(not p.requires_grad for p in encoder.parameters()), "encoder can train")
            if args.phase == "develop":
                develop(args, registry, frozen, base, encoder, device)
                return
            ticket = read_json(args.output / "freeze.json")
            check(ticket and ticket["binding"] == frozen, "freeze normal-only selection before evaluation")
            from vicad_refinement.formal import evaluate_category
            evaluate_category(args, registry, frozen, base, encoder, device, ticket)


def develop(args, registry, frozen, base, encoder, device):
    root = args.output / "development"
    root.mkdir(exist_ok=True)
    binding = root / "binding.json"
    if binding.exists():
        check(read_json(binding) == frozen, "development code or protocol changed")
    else:
        write_json(binding, frozen)
    all_rows = []
    receipt = {}
    for category in registry["development_categories"]:
        folder = root / category
        groups = load_groups(args.inputs, category, base, (30, 70), "train")
        ids, held = split_development(groups, registry)
        cache = encode(args.inputs, groups, folder / "train-features", encoder,
                       {"revision": base["data_revision"], "encoder": frozen["versions"]["vicad_refinement/model.py"],
                        "weights": base["backbone_sha256"]}, device)
        features = load_features(folder / "train-features")
        model, proto, trained = fit(folder / "fit", cache, groups, features, ids,
                                    registry["seed"], base, frozen, device)
        result = study(category, groups, held, features, args.inputs, folder,
                       encoder, model, proto, registry, device)
        check(set(trained["binding"]["training_objects"]).isdisjoint(result["normal_validation_objects"]),
              "development instance leak")
        receipt[category] = {**result, "features": cache, "training": trained}
        with (folder / "proxy-rows.csv").open(newline="", encoding="utf-8") as handle:
            all_rows.extend(csv.DictReader(handle))
    winner, ranked = select(all_rows, registry)
    csv_write(root / "selection.csv", ranked)
    write_json(root / "development.json",
               {"status": "PASS", "completed_utc": now(), "binding": frozen, "categories": receipt,
                "winner": winner, "selection_sha256": digest(root / "selection.csv")})
    print(f"Normal-only candidate: {winner}", flush=True)


def seal(output, registry, frozen):
    development = read_json(output / "development/development.json")
    check(development and development["binding"] == frozen and development["status"] == "PASS",
          "current registered development must finish first")
    check(digest(output / "development/selection.csv") == development["selection_sha256"], "changed proxy ranking")
    seal_path = output / "freeze.json"
    check(not seal_path.exists(), "preserve the one-time freeze receipt")
    path_list = [REGISTRY.relative_to(REPO).as_posix(), *frozen["versions"]]
    status = subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--", *path_list], text=True)
    check(not status.strip(), "commit registered evaluation and model code before the freeze")
    final_sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    write_json(seal_path, {"sealed_utc": now(), "git_sha": final_sha, "binding": frozen,
                          "winner": development["winner"], "development_sha256": digest(output / "development/development.json"),
                          "selection_sha256": development["selection_sha256"],
                          "confirmatory_order": registry["confirmatory_order"],
                          "seeds": registry["confirmatory_seeds"]})
    print(f"Frozen at {final_sha}: {development['winner']}", flush=True)


if __name__ == "__main__":
    main()
