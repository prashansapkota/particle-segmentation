import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import CroppedSegmentationDataset
from model import pretrained_Unet
from utils import dice_coeff_with_threshold
from torch.utils.data import DataLoader
import os
from datetime import datetime
from skimage.measure import label, regionprops
import pathlib
from scipy.optimize import linear_sum_assignment
import wandb  # <---- Added for logging

from skimage import morphology

def postprocess(prob_map, thresh=0.5, min_size=5, ksize=3):
    bin_mask = (prob_map > thresh).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened   = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    cleaned  = morphology.remove_small_objects(opened.astype(bool), min_size=min_size)
    return cleaned.astype(np.uint8)

def hungarian_particle_matching(pred_binary, true_binary, max_distance=10, min_size=5):
    pred_labeled = label(pred_binary, connectivity=2)
    true_labeled = label(true_binary, connectivity=2)
    pred_particles = [p for p in regionprops(pred_labeled) if p.area >= min_size]
    true_particles = [p for p in regionprops(true_labeled) if p.area >= min_size]
    n_predicted = len(pred_particles)
    n_true = len(true_particles)
    if n_predicted == 0 or n_true == 0:
        return {
            'n_true': n_true,
            'n_predicted': n_predicted,
            'matched': 0,
            'detection_rate': 0,
            'precision': 0,
        }
    pred_centers = np.array([p.centroid for p in pred_particles])
    true_centers = np.array([p.centroid for p in true_particles])
    cost_matrix = np.linalg.norm(pred_centers[:, None, :] - true_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] <= max_distance]
    matched = len(matches)
    detection_rate = matched / n_true if n_true > 0 else 0
    precision = matched / n_predicted if n_predicted > 0 else 0
    return {
        'n_true': n_true,
        'n_predicted': n_predicted,
        'matched': matched,
        'detection_rate': detection_rate,
        'precision': precision,
    }

def calculate_confusion_matrix(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    tn = np.sum((pred == 0) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    return tp, fp, tn, fn

def test_model_simple(
    model_path,
    image_dir="data/cropped_images", 
    mask_dir="data/cropped_masks",
    num_samples=10,
    device=None,
    wandb_project="particle-seg-eval",  # Set your project name here
    wandb_run_name=None,
    log_per_sample=True,   # If True, logs sample images to wandb
):
    # ----- WANDB INIT -----
    wandb.init(
        project=wandb_project, 
        name=wandb_run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        reinit=True
    )
    # ----------------------

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"simple_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/probability_maps", exist_ok=True)
    print(f"📁 Saving results to: {results_dir}")
    
    model = pretrained_Unet(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    thresh_path = pathlib.Path(model_path).with_name("best_thresh.txt")
    if thresh_path.exists():
        best_thresh = float(thresh_path.read_text())
    else:
        best_thresh = 0.8
    print(f"Using threshold {best_thresh:.2f}")

    pos_weight_path = pathlib.Path(model_path).with_name("best_pos_weight.txt")
    if pos_weight_path.exists():
        best_pos_weight = float(pos_weight_path.read_text())
        print(f"Using pos_weight {best_pos_weight:.2f}")
    else:
        best_pos_weight = 2.0
        print(f"Using default pos_weight {best_pos_weight:.2f}")
    
    dataset = CroppedSegmentationDataset(image_dir, mask_dir, augment=False, use_original_mask=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    all_results = []
    print("Testing samples...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= num_samples:
                break
            print(f"Processing sample {i+1}/{num_samples}")
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            img_np = images[0].cpu().permute(1, 2, 0).numpy()
            mask_np = masks[0, 0].cpu().numpy()
            probs = torch.sigmoid(outputs[0, 0]).cpu().numpy()
            plt.figure(figsize=(8, 6))
            plt.imshow(probs, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(label='Probability')
            plt.title(f'Sample {i} - Probability Map')
            plt.axis('off')
            plt.savefig(f'{results_dir}/probability_maps/sample_{i:03d}_probabilities.png', dpi=150, bbox_inches='tight')
            plt.close()
            pred_binary = (probs > best_thresh).astype(np.uint8)
            tp, fp, tn, fn = calculate_confusion_matrix(pred_binary, mask_np)
            particle_stats = hungarian_particle_matching(pred_binary, mask_np, max_distance=10)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            dice = dice_coeff_with_threshold(outputs, masks, threshold=best_thresh).item()
            sample_result = {
                'sample': i,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'precision': precision, 'recall': recall, 'dice': dice,
                'particles_true': particle_stats['n_true'],
                'particles_found': particle_stats['n_predicted'],
                'particles_matched': particle_stats['matched'],
                'detection_rate': particle_stats['detection_rate'],
                'particle_precision': particle_stats['precision']
            }
            all_results.append(sample_result)
            
            # Optional: Log individual sample to wandb
            if log_per_sample:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes[0, 0].imshow(img_np[:,:,0], cmap='gray')
                axes[0, 0].set_title('Input Image')
                axes[0, 0].axis('off')
                axes[0, 1].imshow(mask_np, cmap='gray')
                axes[0, 1].set_title('Ground Truth')
                axes[0, 1].axis('off')
                axes[0, 2].imshow(pred_binary, cmap='gray')
                axes[0, 2].set_title('Prediction')
                axes[0, 2].axis('off')
                im1 = axes[1, 0].imshow(probs, cmap='hot', vmin=0, vmax=1)
                axes[1, 0].set_title('Probability Heat Map')
                axes[1, 0].axis('off')
                plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)
                error_map = np.zeros((*pred_binary.shape, 3))
                tp_mask = (pred_binary == 1) & (mask_np == 1)
                error_map[tp_mask] = [0, 1, 0]  # Green
                fp_mask = (pred_binary == 1) & (mask_np == 0)
                error_map[fp_mask] = [1, 0, 0]  # Red
                fn_mask = (pred_binary == 0) & (mask_np == 1)
                error_map[fn_mask] = [0, 0, 1]  # Blue
                axes[1, 1].imshow(error_map)
                axes[1, 1].set_title('Error Analysis\n🟢TP 🔴FP 🔵FN')
                axes[1, 1].axis('off')
                metrics_text = f"""TP: {tp}
FP: {fp}
TN: {tn}
FN: {fn}

Precision: {precision:.3f}
Recall: {recall:.3f}
Dice: {dice:.3f}

PARTICLE DETECTION:
Real: {particle_stats['n_true']}
Found: {particle_stats['n_predicted']}
Matched: {particle_stats['matched']}
Detection: {particle_stats['detection_rate']:.3f}
P-Precision: {particle_stats['precision']:.3f}"""
                axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                                fontsize=12, verticalalignment='top', fontfamily='monospace')
                axes[1, 2].set_title('Metrics')
                axes[1, 2].axis('off')
                plt.suptitle(f'Sample {i} Analysis', fontsize=16)
                plt.tight_layout()
                sample_img_path = f'{results_dir}/sample_{i:03d}.png'
                plt.savefig(sample_img_path, dpi=150, bbox_inches='tight')
                plt.close()
                wandb.log({f"sample_{i:03d}": wandb.Image(sample_img_path)})

    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    avg_tp = np.mean([r['tp'] for r in all_results])
    avg_fp = np.mean([r['fp'] for r in all_results])
    avg_tn = np.mean([r['tn'] for r in all_results])
    avg_fn = np.mean([r['fn'] for r in all_results])
    avg_precision = np.mean([r['precision'] for r in all_results])
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_dice = np.mean([r['dice'] for r in all_results])
    avg_particles_true = np.mean([r['particles_true'] for r in all_results])
    avg_particles_found = np.mean([r['particles_found'] for r in all_results])
    avg_particles_matched = np.mean([r['particles_matched'] for r in all_results])
    avg_detection_rate = np.mean([r['detection_rate'] for r in all_results])
    avg_particle_precision = np.mean([r['particle_precision'] for r in all_results])
    print(f"Average TP: {avg_tp:.1f}")
    print(f"Average FP: {avg_fp:.1f}")
    print(f"Average TN: {avg_tn:.1f}")
    print(f"Average FN: {avg_fn:.1f}")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average Dice: {avg_dice:.3f}")
    print(f"\nPARTICLE DETECTION AVERAGES:")
    print(f"Real Particles/Image: {avg_particles_true:.1f}")
    print(f"Found Particles/Image: {avg_particles_found:.1f}")
    print(f"Matched Particles/Image: {avg_particles_matched:.1f}")
    print(f"Detection Rate: {avg_detection_rate:.3f} ({avg_detection_rate*100:.1f}%)")
    print(f"Particle Precision: {avg_particle_precision:.3f} ({avg_particle_precision*100:.1f}%)")

    # Save summary plots and log to wandb
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    confusion_data = [avg_tp, avg_fp, avg_fn, avg_tn]
    confusion_labels = ['TP', 'FP', 'FN', 'TN']
    colors = ['green', 'red', 'blue', 'gray']
    ax1.bar(confusion_labels, confusion_data, color=colors, alpha=0.7)
    ax1.set_title('Average Confusion Matrix Values')
    ax1.set_ylabel('Count')
    metrics = ['Precision', 'Recall', 'Dice']
    values = [avg_precision, avg_recall, avg_dice]
    ax2.bar(metrics, values, color=['orange', 'purple', 'brown'], alpha=0.7)
    ax2.set_title('Average Performance Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    summary_path = f'{results_dir}/summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    # ---- WANDB LOGGING -----
    wandb.log({
        "avg_tp": avg_tp,
        "avg_fp": avg_fp,
        "avg_fn": avg_fn,
        "avg_tn": avg_tn,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_dice": avg_dice,
        "avg_particles_true": avg_particles_true,
        "avg_particles_found": avg_particles_found,
        "avg_particles_matched": avg_particles_matched,
        "avg_detection_rate": avg_detection_rate,
        "avg_particle_precision": avg_particle_precision,
        "summary_plot": wandb.Image(summary_path),
    })
    wandb.finish()
    print(f"\n Results saved to: {results_dir}")
    print(f" Contains:")
    print(f"    sample_XXX.png - Individual analysis")
    print(f"    summary.png - Overall performance")
    return all_results, results_dir

if __name__ == "__main__":
    test_model_simple(
        model_path="experiment/best_model.pth",
        image_dir="data/cropped_images",
        mask_dir="data/cropped_masks",
        num_samples=10,
        wandb_project="particle-seg-eval",  # <- set your project here!
        wandb_run_name=None,                 # <- can set a name if you want
        log_per_sample=True                  # <- Set to True to log each sample to wandb
    )






# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from dataset import CroppedSegmentationDataset
# from model import pretrained_Unet
# from utils import dice_coeff_with_threshold
# from torch.utils.data import DataLoader
# import os
# from datetime import datetime
# from skimage.measure import label, regionprops
# import pathlib
# from scipy.optimize import linear_sum_assignment
# from scipy.ndimage import binary_dilation

# from skimage import morphology

# # GRACE_RADIUS = 1
# def postprocess(prob_map, thresh=0.5, min_size=10, ksize=3):
#     """High threshold + morph-open + remove tiny blobs."""
#     bin_mask = (prob_map > thresh).astype(np.uint8)
#     kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
#     opened   = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
#     cleaned  = morphology.remove_small_objects(opened.astype(bool), min_size=min_size)
#     return cleaned.astype(np.uint8)

# def hungarian_particle_matching(pred_binary, true_binary, max_distance=10, min_size=5):
#     """
#     Matches predicted particles to ground truth using Hungarian algorithm.
#     Returns matched count, unmatched predictions (FP), unmatched GT (FN), etc.
#     """
#     # Connected components
#     pred_labeled = label(pred_binary, connectivity=2)
#     true_labeled = label(true_binary, connectivity=2)
#     pred_particles = [p for p in regionprops(pred_labeled) if p.area >= min_size]
#     true_particles = [p for p in regionprops(true_labeled) if p.area >= min_size]

#     n_predicted = len(pred_particles)
#     n_true = len(true_particles)
#     if n_predicted == 0 or n_true == 0:
#         return {
#             'n_true': n_true,
#             'n_predicted': n_predicted,
#             'matched': 0,
#             'detection_rate': 0,
#             'precision': 0,
#         }

#     pred_centers = np.array([p.centroid for p in pred_particles])
#     true_centers = np.array([p.centroid for p in true_particles])

#     # Compute cost matrix (Euclidean distance)
#     cost_matrix = np.linalg.norm(pred_centers[:, None, :] - true_centers[None, :, :], axis=2)

#     # Apply Hungarian algorithm
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)

#     # Filter by distance
#     matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] <= max_distance]
#     matched = len(matches)

#     # Compute detection rate & precision
#     detection_rate = matched / n_true if n_true > 0 else 0
#     precision = matched / n_predicted if n_predicted > 0 else 0

#     return {
#         'n_true': n_true,
#         'n_predicted': n_predicted,
#         'matched': matched,
#         'detection_rate': detection_rate,
#         'precision': precision,
#     }

# def calculate_confusion_matrix(pred, target):
#     """Calculate TP, FP, TN, FN"""
#     pred = pred.flatten()
#     target = target.flatten()
    
#     tp = np.sum((pred == 1) & (target == 1))
#     fp = np.sum((pred == 1) & (target == 0)) 
#     tn = np.sum((pred == 0) & (target == 0))
#     fn = np.sum((pred == 0) & (target == 1))
    
#     return tp, fp, tn, fn

# def test_model_simple(
#     model_path,
#     image_dir="data/cropped_images", 
#     mask_dir="data/cropped_masks",
#     num_samples=10,
#     device=None
# ):
#     """
#     Simple model testing with TP/FP/TN/FN and heat maps
#     """
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Create simple results directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_dir = f"simple_results_{timestamp}"
#     os.makedirs(results_dir, exist_ok=True)
#     os.makedirs(f"{results_dir}/probability_maps", exist_ok=True)  # Add this back as PNG folder
    
#     print(f"📁 Saving results to: {results_dir}")
    
#     # Load model
#     model = pretrained_Unet(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     # ── load best threshold chosen during training ──
#     thresh_path = pathlib.Path(model_path).with_name("best_thresh.txt")
#     if thresh_path.exists():
#         best_thresh = float(thresh_path.read_text())
#     else:
#         best_thresh = 0.8          # fallback if file missing
#     print(f"Using threshold {best_thresh:.2f}")

#     # Load best pos_weight chosen during training
#     pos_weight_path = pathlib.Path(model_path).with_name("best_pos_weight.txt")
#     if pos_weight_path.exists():
#         best_pos_weight = float(pos_weight_path.read_text())
#         print(f"Using pos_weight {best_pos_weight:.2f}")
#     else:
#         best_pos_weight = 2.0   # fallback
#         print(f"Using default pos_weight {best_pos_weight:.2f}")
    
#     # Load dataset
#     dataset = CroppedSegmentationDataset(image_dir, mask_dir, augment=False, use_original_mask=True)
#     loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
#     # Storage for results
#     all_results = []
    
#     print("Testing samples...")
#     with torch.no_grad():
#         for i, (images, masks) in enumerate(loader):
#             if i >= num_samples:
#                 break
                
#             print(f"Processing sample {i+1}/{num_samples}")
                
#             images, masks = images.to(device), masks.to(device)
#             outputs = model(images)
            
#             # Get numpy arrays
#             img_np = images[0].cpu().permute(1, 2, 0).numpy()
#             mask_np = masks[0, 0].cpu().numpy()

#             # # Apply grace radius to ground truth mask 
#             # mask_np = binary_dilation(mask_np, iterations=GRACE_RADIUS).astype(np.float32)

#             probs = torch.sigmoid(outputs[0, 0]).cpu().numpy()
            
#             # Save probability map as PNG (easy to view)
#             plt.figure(figsize=(8, 6))
#             plt.imshow(probs, cmap='hot', vmin=0, vmax=1)
#             plt.colorbar(label='Probability')
#             plt.title(f'Sample {i} - Probability Map')
#             plt.axis('off')
#             plt.savefig(f'{results_dir}/probability_maps/sample_{i:03d}_probabilities.png', 
#                        dpi=150, bbox_inches='tight')
#             plt.close()
            
#             # # Test threshold 0.5
#             pred_binary = (probs > best_thresh).astype(np.uint8)





#             # pred_binary = postprocess(probs, thresh=best_thresh, min_size=10)




#             # Calculate confusion matrix
#             tp, fp, tn, fn = calculate_confusion_matrix(pred_binary, mask_np)
            
#             # Calculate particle detection metrics (NEW!)
#             particle_stats = hungarian_particle_matching(pred_binary, mask_np, max_distance=10)

#             # Calculate metrics
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#             recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#             # dice = dice_coeff_with_threshold(outputs, masks, threshold=0.5).item()


#             dice = dice_coeff_with_threshold(outputs, masks, threshold=best_thresh).item()



            
#             # Store results
#             sample_result = {
#                 'sample': i,
#                 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
#                 'precision': precision, 'recall': recall, 'dice': dice,
#                 # Particle detection results
#                 'particles_true': particle_stats['n_true'],
#                 'particles_found': particle_stats['n_predicted'],
#                 'particles_matched': particle_stats['matched'],
#                 'detection_rate': particle_stats['detection_rate'],
#                 'particle_precision': particle_stats['precision']
#             }
#             all_results.append(sample_result)
            
#             # Create visualization
#             fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
#             # Top row
#             axes[0, 0].imshow(img_np[:,:,0], cmap='gray')
#             axes[0, 0].set_title('Input Image')
#             axes[0, 0].axis('off')
            
#             axes[0, 1].imshow(mask_np, cmap='gray')
#             axes[0, 1].set_title('Ground Truth')
#             axes[0, 1].axis('off')
            
#             axes[0, 2].imshow(pred_binary, cmap='gray')
#             axes[0, 2].set_title('Prediction')
#             axes[0, 2].axis('off')
            
#             # Bottom row - Heat maps and analysis
#             # Probability heat map
#             im1 = axes[1, 0].imshow(probs, cmap='hot', vmin=0, vmax=1)
#             axes[1, 0].set_title('Probability Heat Map')
#             axes[1, 0].axis('off')
#             plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)
            
#             # Error analysis heat map (FP=red, FN=blue, TP=green)
#             error_map = np.zeros((*pred_binary.shape, 3))
            
#             # True Positives (green)
#             tp_mask = (pred_binary == 1) & (mask_np == 1)
#             error_map[tp_mask] = [0, 1, 0]  # Green
            
#             # False Positives (red)  
#             fp_mask = (pred_binary == 1) & (mask_np == 0)
#             error_map[fp_mask] = [1, 0, 0]  # Red
            
#             # False Negatives (blue)
#             fn_mask = (pred_binary == 0) & (mask_np == 1) 
#             error_map[fn_mask] = [0, 0, 1]  # Blue
            
#             axes[1, 1].imshow(error_map)
#             axes[1, 1].set_title('Error Analysis\n🟢TP 🔴FP 🔵FN')
#             axes[1, 1].axis('off')
            
#             # Metrics text
#             metrics_text = f"""TP: {tp}
# FP: {fp}
# TN: {tn}
# FN: {fn}

# Precision: {precision:.3f}
# Recall: {recall:.3f}
# Dice: {dice:.3f}

# PARTICLE DETECTION:
# Real: {particle_stats['n_true']}
# Found: {particle_stats['n_predicted']}
# Matched: {particle_stats['matched']}
# Detection: {particle_stats['detection_rate']:.3f}
# P-Precision: {particle_stats['precision']:.3f}"""
            
#             axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
#                            fontsize=12, verticalalignment='top', fontfamily='monospace')
#             axes[1, 2].set_title('Metrics')
#             axes[1, 2].axis('off')
            
#             plt.suptitle(f'Sample {i} Analysis', fontsize=16)
#             plt.tight_layout()
#             plt.savefig(f'{results_dir}/sample_{i:03d}.png', dpi=150, bbox_inches='tight')
#             plt.close()
    
#     # Create summary
#     print("\n" + "="*60)
#     print("SUMMARY RESULTS")
#     print("="*60)
    
#     # Calculate averages
#     avg_tp = np.mean([r['tp'] for r in all_results])
#     avg_fp = np.mean([r['fp'] for r in all_results])
#     avg_tn = np.mean([r['tn'] for r in all_results])
#     avg_fn = np.mean([r['fn'] for r in all_results])
#     avg_precision = np.mean([r['precision'] for r in all_results])
#     avg_recall = np.mean([r['recall'] for r in all_results])
#     avg_dice = np.mean([r['dice'] for r in all_results])
    
#     # Particle detection averages
#     avg_particles_true = np.mean([r['particles_true'] for r in all_results])
#     avg_particles_found = np.mean([r['particles_found'] for r in all_results])
#     avg_particles_matched = np.mean([r['particles_matched'] for r in all_results])
#     avg_detection_rate = np.mean([r['detection_rate'] for r in all_results])
#     avg_particle_precision = np.mean([r['particle_precision'] for r in all_results])
    
#     print(f"Average TP: {avg_tp:.1f}")
#     print(f"Average FP: {avg_fp:.1f}")
#     print(f"Average TN: {avg_tn:.1f}")
#     print(f"Average FN: {avg_fn:.1f}")
#     print(f"Average Precision: {avg_precision:.3f}")
#     print(f"Average Recall: {avg_recall:.3f}")
#     print(f"Average Dice: {avg_dice:.3f}")
    
#     print(f"\nPARTICLE DETECTION AVERAGES:")
#     print(f"Real Particles/Image: {avg_particles_true:.1f}")
#     print(f"Found Particles/Image: {avg_particles_found:.1f}")
#     print(f"Matched Particles/Image: {avg_particles_matched:.1f}")
#     print(f"Detection Rate: {avg_detection_rate:.3f} ({avg_detection_rate*100:.1f}%)")
#     print(f"Particle Precision: {avg_particle_precision:.3f} ({avg_particle_precision*100:.1f}%)")
    
#     # Create summary plot
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Confusion matrix summary
#     confusion_data = [avg_tp, avg_fp, avg_fn, avg_tn]
#     confusion_labels = ['TP', 'FP', 'FN', 'TN']
#     colors = ['green', 'red', 'blue', 'gray']
    
#     ax1.bar(confusion_labels, confusion_data, color=colors, alpha=0.7)
#     ax1.set_title('Average Confusion Matrix Values')
#     ax1.set_ylabel('Count')
    
#     # Metrics
#     metrics = ['Precision', 'Recall', 'Dice']
#     values = [avg_precision, avg_recall, avg_dice]
    
#     ax2.bar(metrics, values, color=['orange', 'purple', 'brown'], alpha=0.7)
#     ax2.set_title('Average Performance Metrics')
#     ax2.set_ylabel('Score')
#     ax2.set_ylim(0, 1)
    
#     plt.tight_layout()
#     plt.savefig(f'{results_dir}/summary.png', dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"\n Results saved to: {results_dir}")
#     print(f" Contains:")
#     print(f"    sample_XXX.png - Individual analysis")
#     print(f"    summary.png - Overall performance")
    
#     return all_results, results_dir

# if __name__ == "__main__":
#     # Example usage
#     test_model_simple(
#         model_path="experiment/best_model.pth",  # Update this path
#         image_dir="data/cropped_images",
#         mask_dir="data/cropped_masks",
#         num_samples=10
#     )    
   
