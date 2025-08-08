import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import pretrained_Unet
from train import sliding_window_inference
from skimage import morphology, feature, draw, segmentation, measure
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import math

def postprocess(prob_map, thresh=0.25, min_size=5, ksize=3):
    bin_mask = (prob_map > thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    cleaned = morphology.remove_small_objects(opened.astype(bool), min_size=min_size)
    return cleaned.astype(np.uint8)

def apply_watershed(prob_map, threshold=0.25, min_distance=5):
    binary = prob_map > threshold
    distance = ndi.distance_transform_edt(binary)
    local_maxi = feature.peak_local_max(
        distance, min_distance=min_distance, labels=binary, footprint=np.ones((3, 3))
    )
    marker_mask = np.zeros_like(distance, dtype=bool)
    marker_mask[tuple(local_maxi.T)] = True
    markers = measure.label(marker_mask)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return labels

def compute_circularity(region):
    if region.perimeter == 0:
        return 0
    return 4 * np.pi * region.area / (region.perimeter ** 2)

def hungarian_particle_matching(pred_labeled, true_binary, max_distance=10, min_size=2, circularity_thresh=0.3):
    true_labeled = measure.label(true_binary, connectivity=2)

    pred_particles = [
        p for p in measure.regionprops(pred_labeled)
        if p.area >= min_size and compute_circularity(p) >= circularity_thresh
    ]
    true_particles = [p for p in measure.regionprops(true_labeled) if p.area >= min_size]

    pred_centers = np.array([p.centroid for p in pred_particles])
    true_centers = np.array([p.centroid for p in true_particles])
    n_pred = len(pred_centers)
    n_true = len(true_centers)

    if n_pred == 0 or n_true == 0:
        return {
            'n_true': n_true, 'n_predicted': n_pred, 'matched': 0,
            'detection_rate': 0, 'precision': 0, 'f1': 0, 'matches': [],
            'tp': 0, 'fp': n_pred, 'fn': n_true,
            'pred_centers': pred_centers, 'true_centers': true_centers
        }

    cost = np.linalg.norm(pred_centers[:, None, :] - true_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost[i, j] <= max_distance]

    matched_pred = {i for i, _ in matches}
    matched_true = {j for _, j in matches}
    tp = len(matches)
    fp = n_pred - len(matched_pred)
    fn = n_true - len(matched_true)

    detection_rate = tp / n_true if n_true else 0
    precision = tp / n_pred if n_pred else 0
    f1 = 2 * detection_rate * precision / (detection_rate + precision) if (detection_rate + precision) else 0

    return {
        'n_true': n_true,
        'n_predicted': n_pred,
        'matched': tp,
        'detection_rate': detection_rate,
        'precision': precision,
        'f1': f1,
        'matches': matches,
        'pred_centers': pred_centers,
        'true_centers': true_centers,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def visualize_distance_histogram(pred_centers, true_centers, matches, filename, epoch, save_dir):
    """
    Draws a histogram of pixel distances for each matched pair (pred↔GT).
    - pred_centers, true_centers: arrays of shape [N,2]
    - matches: list of (i,j) index pairs
    """
    # compute distances
    dists = []
    for i, j in matches:
        py, px = pred_centers[i]
        ty, tx = true_centers[j]
        d = math.hypot(py - ty, px - tx)
        dists.append(d)

    # nothing matched? skip
    if len(dists) == 0:
        return

    # plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(dists, bins=20, edgecolor='black')
    plt.title(f"{filename} – Epoch {epoch}\nMatching distances")
    plt.xlabel("Distance (pixels)")
    plt.ylabel("Count")
    plt.tight_layout()

    # save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{filename}_epoch{epoch}_dist_hist.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Saved distance histogram: {out_path}")

def visualize_predictions(image, prob_map, gt_mask, epoch, filename, save_dir, match_stats):
    """
    Top row: original / probability / TP-FP-FN overlay (as before).
    Bottom-left: Hungarian matching (green=GT, blue=Pred, red lines=matches).
    Bottom-middle: unused (hidden). Bottom-right: metrics text.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Build TP/FP/FN overlay
    tp_fp_fn_vis = np.stack([image] * 3, axis=-1)
    matched_pred = {i for i, _ in match_stats['matches']}
    matched_true = {j for _, j in match_stats['matches']}
    # mark preds
    for i, (y, x) in enumerate(np.round(match_stats['pred_centers']).astype(int)):
        color = [0,255,0] if i in matched_pred else [0,0,255]
        tp_fp_fn_vis[y-1:y+2, x-1:x+2] = color
    # mark fns
    for j, (y, x) in enumerate(np.round(match_stats['true_centers']).astype(int)):
        if j not in matched_true:
            tp_fp_fn_vis[y-1:y+2, x-1:x+2] = [255,0,0]

    # Build Hungarian matching viz
    match_vis = np.stack([image] * 3, axis=-1)
    # draw lines
    for i, j in match_stats['matches']:
        ty, tx = np.round(match_stats['true_centers'][j]).astype(int)
        py, px = np.round(match_stats['pred_centers'][i]).astype(int)
        rr, cc = draw.line(ty, tx, py, px)
        match_vis[rr, cc] = [255, 0, 0]
    # draw centers
    for ty, tx in np.round(match_stats['true_centers']).astype(int):
        match_vis[ty-1:ty+2, tx-1:tx+2] = [0, 255, 0]
    for py, px in np.round(match_stats['pred_centers']).astype(int):
        match_vis[py-1:py+2, px-1:px+2] = [0, 0, 255]

    # layout
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs[0,0].imshow(image, cmap='gray');       axs[0,0].set_title("Original Image");    axs[0,0].axis('off')
    axs[0,1].imshow(prob_map, cmap='hot', vmin=0, vmax=1); axs[0,1].set_title("Probability Map"); axs[0,1].axis('off')
    axs[0,2].imshow(tp_fp_fn_vis);            axs[0,2].set_title("TP/FP/FN Overlay"); axs[0,2].axis('off')

    axs[1,0].imshow(match_vis);                axs[1,0].set_title("Hungarian Matching -- green=GT, blue=Preds"); axs[1,0].axis('off')
    axs[1,1].axis('off')  # unused

    metrics = (
        f"TP: {match_stats['tp']}\n"
        f"FP: {match_stats['fp']}\n"
        f"FN: {match_stats['fn']}\n"
        f"Precision: {match_stats['precision']:.2f}\n"
        f"Recall: {match_stats['detection_rate']:.2f}\n"
        f"F1: {match_stats['f1']:.2f}"
    )
    axs[1,2].text(0.1, 0.5, metrics, fontsize=12)
    axs[1,2].axis('off')

    plt.suptitle(f"{filename} - Epoch {epoch}")
    save_path = os.path.join(save_dir, f"{filename}_epoch{epoch}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {save_path}")

def evaluate_checkpoints(
    checkpoints_dir,
    image_dir,
    mask_dir,
    save_dir="evaluation_outputs",
    patch_size=256,
    stride=128,
    start_epoch=150,
    end_epoch=1950,
    step=150,
    thresh=0.25,
    dilation_iters=0,
    device=None
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = lambda img: torch.from_numpy(img.transpose(2,0,1)).float() / 255

    filenames = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, end_epoch+1, step):
        ckpt = os.path.join(checkpoints_dir, f"model_epoch_{epoch}.pth")
        if not os.path.exists(ckpt):
            print(f"❌ Skipping missing checkpoint: {ckpt}")
            continue

        print(f"\n🔍 Evaluating checkpoint: {ckpt}")
        model = pretrained_Unet(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        for fname in filenames:
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(mask_dir, fname.replace(".tif","_mask.png"))

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt = (mask>127).astype(np.uint8)
            if dilation_iters>0:
                gt = cv2.dilate(gt, np.ones((3,3),np.uint8), iterations=dilation_iters)

            with torch.no_grad():
                prob_map = sliding_window_inference(
                    model=model, full_image=rgb,
                    patch_size=patch_size, stride=stride,
                    device=device, transform_fn=transform
                )

            binary = postprocess(prob_map, thresh=thresh)
            labels_ws = apply_watershed(prob_map, threshold=thresh)
            stats = hungarian_particle_matching(labels_ws, gt)

            print(
                f"{fname} @ Epoch {epoch} | "
                f"TP: {stats['tp']} | FP: {stats['fp']} | FN: {stats['fn']} | "
                f"Prec: {stats['precision']:.2f} | Rec: {stats['detection_rate']:.2f} | F1: {stats['f1']:.2f}"
            )

            visualize_predictions(
                image=img,
                prob_map=prob_map,
                gt_mask=gt,
                epoch=epoch,
                filename=fname.replace(".tif",""),
                save_dir=save_dir,
                match_stats=stats
            )
            visualize_distance_histogram(
                stats['pred_centers'],
                stats['true_centers'],
                stats['matches'],
                fname.replace(".tif", ""),
                epoch,
                save_dir
            )

if __name__ == "__main__":
    evaluate_checkpoints(
        checkpoints_dir="experiment/run_20250805_131647",
        image_dir="data/cropped_images",
        mask_dir="data/cropped_masks",
        save_dir="evaluation_outputs_hungarian_focal_10",
        patch_size=256,
        stride=128,
        start_epoch=150,
        end_epoch=1950,
        step=150,
        thresh=0.25,
        dilation_iters=0

    )
