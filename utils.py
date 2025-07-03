def dice_coeff(pred, target, smooth=1e-8):
    """
    Compute Dice coefficient between predicted and target masks
    Args:
        pred: Tensor of shape [B, 1, H, W] with probabilities
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