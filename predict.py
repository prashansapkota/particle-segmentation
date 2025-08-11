#!/usr/bin/env python3
"""
predict.py — simple inference script to count particles on a new image.

- Picks the LAST epoch checkpoint (model_epoch_*.pth) from --run_dir
  (falls back to Final_Model.pth if no epoch files exist; NEVER uses best_model.pth)
- Runs sliding-window inference (uses your train_patch.sliding_window_inference)
- Postprocesses to instances, measures, saves outputs

Usage:
  python predict.py --image data/new_images/sample.tif \
                    --run_dir experiment/run_20250805_104636 \
                    --out_dir predictions_sample \
                    --thresh 0.25 --min_size 5 --morph_ksize 3 --watershed \
                    --min_distance 5 --min_circularity 0.3
"""

import os
import re
import cv2
import json
import glob
import math
import argparse
import numpy as np
import torch
import pandas as pd

from model import pretrained_Unet
from train_patch import sliding_window_inference
from scipy import ndimage as ndi
from skimage import morphology, measure, segmentation, feature, color
import matplotlib.pyplot as plt
from scipy.spatial import distance


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def imread_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def pick_last_epoch_checkpoint(run_dir):
    """Pick the numerically last model_epoch_*.pth in run_dir; fallback to Final_Model.pth."""
    epoch_ckpts = glob.glob(os.path.join(run_dir, "model_epoch_*.pth"))
    if epoch_ckpts:
        def epoch_num(s):
            m = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(s))
            return int(m.group(1)) if m else -1
        epoch_ckpts.sort(key=epoch_num)
        return epoch_ckpts[-1]
    # fallback
    final = os.path.join(run_dir, "Final_Model.pth")
    if os.path.exists(final):
        return final
    raise FileNotFoundError(f"No model_epoch_*.pth or Final_Model.pth found in {run_dir}")

def postprocess(prob_map, thresh=0.25, min_size=5, ksize=3):
    """Threshold -> morph open -> remove small objects."""
    bin_mask = (prob_map > thresh).astype(np.uint8)
    if ksize > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    cleaned = morphology.remove_small_objects(bin_mask.astype(bool), min_size=min_size)
    return cleaned.astype(np.uint8)

def watershed_instances(binary, min_distance=5):
    """Split touching blobs; return label image."""
    if binary.dtype != bool:
        binary = binary.astype(bool)
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.int32)
    distance = ndi.distance_transform_edt(binary)
    peaks = feature.peak_local_max(
        distance, labels=binary, min_distance=min_distance, footprint=np.ones((3,3))
    )
    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (py, px) in enumerate(peaks, start=1):
        markers[py, px] = i
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return labels

def compute_circularity(region):
    per = region.perimeter
    if per == 0:
        return 0.0
    return float(4 * math.pi * region.area / (per ** 2))

def measure_instances(labels, min_size=5, min_circularity=0.0):
    """Return list of dicts per instance: label_id,x,y,area,equiv_diameter,circularity"""
    out = []
    for r in measure.regionprops(labels):
        if r.area < min_size:
            continue
        circ = compute_circularity(r)
        if circ < min_circularity:
            continue
        y, x = r.centroid
        out.append({
            "label_id": int(r.label),
            "x": float(x),
            "y": float(y),
            "area": float(r.area),
            "equiv_diameter": float(r.equivalent_diameter),
            "circularity": float(circ),
        })
    return out


# ----------------------------
# Core prediction
# ----------------------------
def predict_image(
    image_path,
    run_dir,
    out_dir,
    thresh=0.25,
    min_size=5,
    morph_ksize=3,
    watershed=False,
    min_distance=5,
    min_circularity=0.0,
    patch_size=256,
    stride=128,
    device=None,
    crop_bottom=100,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    ckpt_path = pick_last_epoch_checkpoint(run_dir)
    print(f"[Model] Loading checkpoint: {ckpt_path}")
    model = pretrained_Unet(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Read image
    base = os.path.splitext(os.path.basename(image_path))[0]

    # load grayscale
    gray = imread_gray(image_path)

    # crop from the bottom (if requested)
    if crop_bottom and crop_bottom > 0:
        cut = min(crop_bottom, gray.shape[0] - 1)  # guard against oversize crop
        gray = gray[:-cut, :]

    # convert to RGB *after* cropping so shapes match everywhere
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Inference (your function already normalizes internally)
    with torch.no_grad():
        prob_map = sliding_window_inference(
            model=model,
            full_image=rgb,
            device=device,
            transform_fn=None,
            patch_size=patch_size,
            stride=stride
        )

    # Save probability map
    prob_vis = np.clip((prob_map * 255).astype(np.uint8), 0, 255)
    cv2.imwrite(os.path.join(out_dir, f"{base}_prob.png"), prob_vis)

    # Postprocess -> instances
    binary = postprocess(prob_map, thresh=thresh, min_size=min_size, ksize=morph_ksize)
    cv2.imwrite(os.path.join(out_dir, f"{base}_bin.png"), (binary*255).astype(np.uint8))

    if watershed:
        labels = watershed_instances(binary, min_distance=min_distance)
    else:
        labels = measure.label(binary, connectivity=2)

    # Save colored labels
    labels_rgb = (color.label2rgb(labels, image=gray, bg_label=0) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, f"{base}_labels.png"), cv2.cvtColor(labels_rgb, cv2.COLOR_RGB2BGR))

    # Measure
    props = measure_instances(labels, min_size=min_size, min_circularity=min_circularity)

    # Compute distances between all particle centroids
    centroids = np.array([[p["x"], p["y"]] for p in props])
    if len(centroids) > 1:
        dists = distance.pdist(centroids)  # condensed pairwise distances
        plt.figure()
        plt.hist(dists, bins=30, edgecolor="black")
        plt.xlabel("Distance between particles (pixels)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Inter-Particle Distances")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_distance_hist.png"))
        plt.close()

    # Save per-particle CSV
    df = pd.DataFrame(props, columns=["label_id","x","y","area","equiv_diameter","circularity"])
    csv_path = os.path.join(out_dir, f"{base}_particles.csv")
    df.to_csv(csv_path, index=False)

    # Save overlay of predicted centers (blue)
    overlay = np.stack([gray]*3, axis=-1)
    for p in props:
        cv2.circle(overlay, (int(round(p["x"])), int(round(p["y"]))), 2, (0,0,255), -1)
    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Save summary
    summary = {
        "image": os.path.basename(image_path),
        "count": int(len(props)),
        "mean_area": float(np.mean([p["area"] for p in props])) if props else 0.0,
        "median_area": float(np.median([p["area"] for p in props])) if props else 0.0,
        "mean_equiv_diameter": float(np.mean([p["equiv_diameter"] for p in props])) if props else 0.0,
        "median_equiv_diameter": float(np.median([p["equiv_diameter"] for p in props])) if props else 0.0,
        "mean_circularity": float(np.mean([p["circularity"] for p in props])) if props else 0.0,
        "median_circularity": float(np.median([p["circularity"] for p in props])) if props else 0.0,
        "params": {
            "thresh": float(thresh),
            "min_size": int(min_size),
            "morph_ksize": int(morph_ksize),
            "watershed": bool(watershed),
            "min_distance": int(min_distance),
            "min_circularity": float(min_circularity),
            "patch_size": int(patch_size),
            "stride": int(stride),
            "crop_bottom": int(crop_bottom),
        }
    }
    with open(os.path.join(out_dir, f"{base}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] {base}: count={summary['count']} | saved to {out_dir}")
    return summary


# ----
# CLI
# ----
def main():
    parser = argparse.ArgumentParser(description="Predict particle count on a new image (last-epoch checkpoint).")
    parser.add_argument("--image", required=True, help="Path to a single input image (tif/png/jpg).")
    parser.add_argument("--run_dir", default="experiment/run_20250805_104636", help="Run directory with checkpoints.")
    parser.add_argument("--out_dir", default="predictions", help="Output directory.")
    parser.add_argument("--thresh", type=float, default=0.25, help="Probability threshold.")
    parser.add_argument("--min_size", type=int, default=5, help="Min particle size (pixels).")
    parser.add_argument("--morph_ksize", type=int, default=3, help="Morph open kernel size.")
    parser.add_argument("--watershed", action="store_true", help="Split touching particles via watershed.")
    parser.add_argument("--min_distance", type=int, default=5, help="Min peak distance (watershed).")
    parser.add_argument("--min_circularity", type=float, default=0.0, help="Filter by circularity (0..1).")
    parser.add_argument("--patch_size", type=int, default=256, help="Sliding window patch size.")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride.")
    parser.add_argument("--crop_bottom", type=int, default=100, help="Pixels to remove from the bottom before inference.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    predict_image(
        image_path=args.image,
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        thresh=args.thresh,
        min_size=args.min_size,
        morph_ksize=args.morph_ksize,
        watershed=args.watershed,
        min_distance=args.min_distance,
        min_circularity=args.min_circularity,
        patch_size=args.patch_size,
        stride=args.stride,
        device=device,
        crop_bottom=args.crop_bottom
    )

if __name__ == "__main__":
    main()
