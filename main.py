

import argparse
import os
import torch
from train import train_model
from model import pretrained_Unet
from utils import set_seed
from datetime import datetime


def main(args):
    """
    Orchestrates a single training run.

    Args:
        args (argparse.Namespace):
            Command-line configuration including:
                - images_dir (str): Path to training images.
                - masks_dir (str): Path to binary masks matching the images.
                - image_resize (int): Side length for (square) resizing.
                - val_split (float): Fraction of data used for validation.
                - epochs (int): Number of training epochs.
                - batch_size (int): Batch size.
                - lr (float): Learning rate.
                - experiment_name (str): Base output directory for runs.
                - dilation_iters (int): Initial mask dilation iterations (curriculum).
                - seed (int): Random seed.
                - erosion_freq (int): How often to reduce dilation during training.
                - erosion_iters (int): Amount of dilation reduction each erosion step.
                - wandb (bool): Whether to log to Weights & Biases.
                - use_original_mask (bool): If True, skip curriculum and use original masks.
                - patch_stride (int): Stride for sliding-window patch extraction.

    Returns:
        None
            Side effects:
                - Creates a timestamped run directory.
                - Trains the model and saves checkpoints/logs into the run directory.
    """
    # Set seed at the very beginning
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")

    # Create a unique experiment directory using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_name = os.path.join(args.experiment_name, f"run_{timestamp}")
    os.makedirs(args.experiment_name, exist_ok=True)

    model = pretrained_Unet(device)
    train_model(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nano-particle U-Net segmentation training")
    parser.add_argument("--images_dir", type=str, default="data/cropped_images", help="Directory with cropped images")
    parser.add_argument("--masks_dir", type=str, default="data/cropped_masks", help="Directory with cropped masks")
    parser.add_argument('--image_resize', type=int, default=256, help='Input image size (default: 256)')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio (e.g., 0.2 for 20%)')
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Base folder for experiment outputs")
    parser.add_argument("--dilation_iters", type=int, default=10, help="Dilation iterations for mask (start, curriculum begins here)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--erosion_freq", type=int, default=150, help="Epochs between each mask erosion (dilation reduction)")
    parser.add_argument("--erosion_iters", type=int, default=1, help="How much to reduce dilation each erosion")
    parser.add_argument("--wandb", action="store_true", help="If set, log to Weights & Biases")
    parser.add_argument("--use_original_mask", action="store_true", help="Always use the original (undilated) mask")
    parser.add_argument("--patch_stride", type=int, default=128, help="Stride for patch extraction (sliding window style)")
    args = parser.parse_args()
    main(args)
