import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from albumentations.core.transforms_interface import DualTransform
import albumentations as A

def set_seed(seed):
    """Set random seed for complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    """
    Compute the Dice coefficient between predicted and target masks.

    Args:
        preds (torch.Tensor): Predicted masks of shape [B, 1, H, W] or [B, H, W].
        targets (torch.Tensor): Ground truth masks of same shape as preds.
        threshold (float): Probability threshold for binarization. Default is 0.5.
        eps (float): Small value to avoid division by zero. Default is 1e-6.

    Returns:
        float: Mean Dice score across the batch.
    """
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


def extract_sliding_patches(image, mask, patch_size, stride):
    """
    Extract sliding window patches from image and mask.
    Args:
        image: np.array [H, W, 3]
        mask: np.array [H, W]
        patch_size: int
        stride: int
    Returns:
        list of (patch_image, patch_mask)
    """
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_img = image[y:y + patch_size, x:x + patch_size]
            patch_mask = mask[y:y + patch_size, x:x + patch_size]
            patches.append((patch_img, patch_mask))
    return patches


# class WeightedFocalDiceLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, focal_weight=0.7, dice_weight=0.3, eps=1e-6):
#         super(WeightedFocalDiceLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.focal_weight = focal_weight
#         self.dice_weight = dice_weight
#         self.eps = eps

#     def forward(self, inputs, targets):
#         if inputs.ndim == 4:
#             inputs = inputs.squeeze(1)
#         if targets.ndim == 4:
#             targets = targets.squeeze(1)

#         # === Focal Loss ===
#         probs = torch.sigmoid(inputs)
#         targets = targets.float()
#         bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce)
#         focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce).mean()

#         # === Dice Loss ===
#         probs_flat = probs.view(probs.size(0), -1)
#         targets_flat = targets.view(targets.size(0), -1)
#         intersection = (probs_flat * targets_flat).sum(dim=1)
#         union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
#         dice_score = (2. * intersection + self.eps) / (union + self.eps)
#         dice_loss = 1 - dice_score.mean()

#         return self.focal_weight * focal_loss + self.dice_weight * dice_loss