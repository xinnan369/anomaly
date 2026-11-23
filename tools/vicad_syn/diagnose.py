"""Post-hoc score and visibility diagnostics on the full verified test runs."""

import argparse
from pathlib import Path

import numpy as np

from tools.vicad_syn.verify import write_csv
from vicad_syn.experiment import read_json


def pixel_means(maps, masks, positive_rows):
    fg_sum = bg_sum = 0.
    fg_count = bg_count = 0
    for start in range(0, len(positive_rows), 32):
        subset = positive_rows[start:start+32]
        score = maps[subset]
        truth = masks[subset].reshape(score.shape).astype(bool)
        fg_sum += np.sum(score, where=truth, dtype=np.float64)
        bg_sum += np.sum(score, where=~truth, dtype=np.float64)
        fg_count += int(truth.sum())
        bg_count += truth.size - int(truth.sum())
    return {"positive_samples": len(positive_rows), "positive_pixels": fg_count,
            "negative_pixels_in_positive_samples": bg_count,
            "mask_pixel_mean": fg_sum/fg_count if fg_count else None,
            "other_pixel_mean": bg_sum/bg_count if bg_count else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "verified", "output"):
        ap.add_argument(f"--{key}", type=Path, required=True)
    args = ap.parse_args()
    verified = read_json(args.verified)
    rows = []
    for cls in verified["categories"]:
        source = args.run / cls / "conditioned/evaluation"
        raw = source / "raw"
        masks = np.load(raw/"imgs_masks.npy", allow_pickle=False, mmap_mode="r")
        maps = np.load(raw/"anomaly_maps.npy", allow_pickle=False, mmap_mode="r")
        positive = np.concatenate([np.flatnonzero(masks[i:i+64].reshape(-1,224*224).any(axis=1))+i
                                   for i in range(0,len(masks),64)])
        visible = np.bincount(positive//10,minlength=len(masks)//10)
        rows.append({"class":cls,"setting":"unfused",**pixel_means(maps,masks,positive),
                     "views_with_visible_mask":int((visible>0).sum()),
                     "median_lights_with_visible_mask":float(np.median(visible[visible>0]))})
        for mode in ("A1","A4"):
            folder = source / mode / "fused"
            mask = np.load(folder/"view_instance_masks.npy", allow_pickle=False, mmap_mode="r")
            heatmap = np.load(folder/"view_instance_anomaly_maps.npy", allow_pickle=False, mmap_mode="r")
            idx = np.flatnonzero(mask.reshape(len(mask),-1).any(axis=1))
            rows.append({"class":cls,"setting":mode,**pixel_means(heatmap,mask,idx),
                         "views_with_visible_mask":len(idx),
                         "median_lights_with_visible_mask":float(np.median(visible[visible>0]))})
    if rows:
        write_csv(args.output,rows)


if __name__ == "__main__":
    main()
