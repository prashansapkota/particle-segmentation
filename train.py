# train.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SlidingWindowSegmentationDataset
from utils import dice_score, set_seed, extract_sliding_patches, FocalLoss, WeightedFocalDiceLoss
import wandb
from scipy.ndimage import binary_dilation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    if preds.ndim == 4:
        preds = preds.squeeze(1)  # shape: [B, H, W]
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    batch_size = preds.size(0)
    scores = []

    for i in range(batch_size):
        pred = (preds[i] > threshold).float().view(-1)
        target = targets[i].float().view(-1)
        intersection = (pred * target).sum()

        union = pred.sum() + target.sum()
        dice = (2. * intersection + eps) / (union + eps)
        scores.append(dice.item())

    return sum(scores) / len(scores)

def get_loaders(image_dir, mask_dir, batch_size, dilation_iters, use_original_mask, val_split=0.2, seed=42):
    all_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    train_filenames, val_filenames = train_test_split(all_filenames, test_size=val_split, random_state=seed)

    train_set = SlidingWindowSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        filenames=train_filenames,
        augment=True,
        use_original_mask=use_original_mask,
        dilation_iters=dilation_iters,
        patch_size=256,              # Use patch_size instead of crop_size
        stride=128,         # Use the correct CLI arg
        seed=seed
    )
    val_set = SlidingWindowSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        filenames=val_filenames,
        augment=False,
        use_original_mask=use_original_mask,
        dilation_iters=dilation_iters,
        patch_size=256,              # Use patch_size instead of crop_size
        stride=128,         # Use the correct CLI arg
        seed=seed
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def sliding_window_inference(model, full_image, device, transform_fn, patch_size=256, stride=128):
    height, width = full_image.shape[:2]
    prob_map = np.zeros((height, width), np.float32)
    cover_map = np.zeros((height, width), np.float32)

    # Compute top and left positions similar to training logic
    top_positions = list(range(0, height - patch_size + 1, stride))
    if (height - patch_size) not in top_positions:
        top_positions.append(height - patch_size)

    left_positions = list(range(0, width - patch_size + 1, stride))
    if (width - patch_size) not in left_positions:
        left_positions.append(width - patch_size)
        
    transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    for top in top_positions:
        for left in left_positions:
            patch = full_image[top:top + patch_size, left:left + patch_size]
            if full_image.ndim == 2:
                patch = np.stack([patch] * 3, axis=-1)  # Convert to 3 channels if grayscale
            
            tensor = transform(image=patch)['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits)
                probs_np = probs.squeeze().cpu().numpy()

            prob_map[top:top + patch_size, left:left + patch_size] += probs_np
            cover_map[top:top + patch_size, left:left + patch_size] += 1

    cover_map[cover_map == 0] = 1  # avoid division by zero
    return prob_map / cover_map


def train_model(args, model, device):
    print(f"Training with batch size: {args.batch_size}")
    set_seed(args.seed)
    image_dir = args.images_dir
    mask_dir = args.masks_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    out_dir = args.experiment_name
    erosion_freq = args.erosion_freq
    erosion_iters = args.erosion_iters
    dilation_iters = args.dilation_iters
    use_original_mask = False
    best_val_dice = -1
    wandb_initialized = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader = None, None
    prev_dilation_iters, prev_use_original_mask = None, None

    for epoch in range(epochs):
        if epoch % erosion_freq == 0 and epoch > 0 and not use_original_mask:
            dilation_iters = max(dilation_iters - erosion_iters, 0)
            print(f"\n[Erosion] Epoch {epoch+1}: Updated dilation_iters to {dilation_iters}")

        if dilation_iters == 0 and not use_original_mask:
            use_original_mask = True

        if (dilation_iters != prev_dilation_iters) or (use_original_mask != prev_use_original_mask) or (train_loader is None):
            train_loader, val_loader = get_loaders(
                image_dir, mask_dir, batch_size,
                dilation_iters=dilation_iters,
                use_original_mask=use_original_mask,
                val_split=args.val_split,
                seed=args.seed
            )
            prev_dilation_iters = dilation_iters
            prev_use_original_mask = use_original_mask

            n_pos, n_pixels = 0, 0
            for images, masks, _ in train_loader:
                n_pos += masks.sum().item()
                n_pixels += masks.numel()
            n_neg = n_pixels - n_pos
            alpha = n_pos / (n_pos + n_neg + 1e-6)
            print(f"Alpha: {alpha:.3f}")
            loss_fn = BCEWithLogitsLoss(alpha=alpha, gamma=2.0, focal_weight=0.7, dice_weight=0.3).to(device)
            loss_fn_val = BCEWithLogitsLoss(alpha=alpha, gamma=2.0, focal_weight=0.7, dice_weight=0.3).to(device)

        model.train()
        total_loss = 0.0
        for images, masks, _ in tqdm(train_loader, desc="Train"):
            images, masks = images.to(device), masks.to(device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_dice_sum = 0.0
        val_loss_sum = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device).float()
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                outputs = model(images)
                val_loss = loss_fn_val(outputs, masks)
                val_loss_sum += val_loss.item() * images.size(0)

                pred_probs = torch.sigmoid(outputs)
                pred_mask = (pred_probs > 0.5).float()
                curr_dice = dice_score(pred_mask, masks)
                val_dice_sum += curr_dice * images.size(0)
                total_samples += images.size(0)

        avg_val_dice = val_dice_sum / total_samples
        avg_val_loss = val_loss_sum / total_samples

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}")

        if args.wandb and not wandb_initialized:
            wandb.init(project="particle-segmentation", name=args.experiment_name)
            wandb_initialized = True

        if args.wandb:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_dice": avg_val_dice,
                "dilation_iters": dilation_iters,
                "alpha": alpha,
            })

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            print("✅ Saved new best model!")

        if (epoch + 1) % 150 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_epoch_{epoch+1}.pth"))
            print(f"📦 Saved model at epoch {epoch+1}")

    torch.save(model.state_dict(), os.path.join(out_dir, "Final_Model.pth"))
    print(f"Training complete! Model weights saved to {out_dir}")