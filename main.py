import argparse
import os
import torch
from train import train_model
from model import pretrained_Unet

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.experiment_name, exist_ok=True)
    model = pretrained_Unet(device)
    train_model(args, model, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nano-particle U-Net segmentation training")
    parser.add_argument("--images_dir", type=str, default="data/cropped_images", help="Directory with cropped images")
    parser.add_argument("--masks_dir", type=str, default="data/cropped_masks", help="Directory with cropped masks")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="Experiment folder for outputs")
    parser.add_argument("--dilation_iters", type=int, default=5, help="Dilation iterations for mask (start, curriculum begins here)")
    parser.add_argument("--erosion_freq", type=int, default=30, help="Epochs between each mask erosion (dilation reduction)")
    parser.add_argument("--erosion_iters", type=int, default=1, help="How much to reduce dilation each erosion")
    parser.add_argument("--wandb", action="store_true", help="If set, log to Weights & Biases")
    parser.add_argument("--use_original_mask", action="store_true", help="Always use the original (undilated) mask")
    args = parser.parse_args()
    main(args)





# import argparse
# import os
# import torch
# from train import train_model
# from model import pretrained_Unet

# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     os.makedirs(args.experiment_name, exist_ok=True)
    
#     train_model(
#         image_dir=args.images_dir,
#         mask_dir=args.masks_dir,

#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         device=device,
#         out_dir=args.experiment_name,
#         erosion_freq=args.erosion_freq,
#         erosion_iters=args.erosion_iters,
#         dilation_start=args.dilation_start,
#         dilation_min=args.dilation_min,
        
#     )

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Nano-particle U-Net segmentation training")
#     parser.add_argument("--images_dir", type=str, default="data/cropped_images", help="Directory with cropped images")
#     parser.add_argument("--masks_dir", type=str, default="data/cropped_masks", help="Directory with croppedmasks")
#     parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
#     parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
#     parser.add_argument("--experiment_name", type=str, default="experiment", help="Experiment folder for outputs")
#     parser.add_argument("--erosion_freq", type=int, default=30, help="Epochs between each mask erosion (dilation reduction)")
#     parser.add_argument("--erosion_iters", type=int, default=1, help="How much to reduce dilation each erosion")
#     parser.add_argument("--dilation_start", type=int, default=10, help="Starting dilation for mask curriculum")
#     parser.add_argument("--dilation_min", type=int, default=3, help="Minimum allowed dilation for masks")
#     args = parser.parse_args()
#     main(args)