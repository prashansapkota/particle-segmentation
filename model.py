import torch
import segmentation_models_pytorch as smp
import torch.nn as nn


def pretrained_Unet(device):
    """
    Returns a U-Net model for binary segmentation using SMP.

    Args:
        device (str or torch.device): Device for model ('cuda' or 'cpu')

    Returns:
        torch.nn.Module: U-Net model on correct device
    """
    model = smp.Unet(
        encoder_name='resnet101',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
        activation='sigmoid'
    )
    # so loss functions like BCEWithLogitsLoss can be used instead
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])

    return model.to(device)