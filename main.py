import argparse
import os
import torch
from train import train_model
from model import pretrained_Unet

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.experiment_name, exist_ok=True)
    
    train_model(
        image_dir=args.images_dir,
        mask_dir=args.masks_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        out_dir=args.experiment_name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nano-particle U-Net segmentation training")
    parser.add_argument("--images_dir", type=str, default="data/cropped_images", help="Directory with cropped images")
    parser.add_argument("--masks_dir", type=str, default="data/cropped_masks", help="Directory with croppedmasks")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Experiment folder for outputs")
    args = parser.parse_args()
    main(args)