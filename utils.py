import torch

import pathlib
import torch.nn.functional as F

def dice_coeff(pred, target, smooth=1e-8):
    """
    Compute Dice coefficient between predicted and target masks
    Args:
        pred: Tensor of shape [B, 1, H, W] with binary values (0 or 1)
        target: Tensor of shape [B, 1, H, W] with binary values
        smooth: Smoothing factor to avoid division by zero
    Returns:
        dice: Dice coefficient (float)
    """
    # Flatten tensors
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    return (2. * intersection + smooth) / (union + smooth)

def dice_coeff_with_threshold(pred_logits, target, threshold=0.5, smooth=1e-8):
    """
    Compute Dice coefficient with automatic thresholding
    Args:
        pred_logits: Tensor of shape [B, 1, H, W] with raw logits or probabilities
        target: Tensor of shape [B, 1, H, W] with binary values
        threshold: Threshold for converting probabilities to binary predictions
        smooth: Smoothing factor to avoid division by zero
    Returns:
        dice: Dice coefficient (float)
    """
    # Convert to probabilities if logits, then threshold
    if pred_logits.max() > 1.0 or pred_logits.min() < 0.0:
        pred_probs = torch.sigmoid(pred_logits)
    else:
        pred_probs = pred_logits
    
    pred_binary = (pred_probs > threshold).float()
    
    return dice_coeff(pred_binary, target, smooth)





def bce_dice_loss(inputs, targets, pos_weight=None, smooth=1.0):
    """
    BCE-With-Logits + Dice loss (single function, no class).
    """
    bce  = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
    probs        = torch.sigmoid(inputs)
    intersection = (probs * targets).sum()
    dice         = (2 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    return bce + (1 - dice)