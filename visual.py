import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import CroppedSegmentationDataset
from model import pretrained_Unet

def predict_single_sample(model_path, sample_idx, image_dir, mask_dir, device=None, threshold=0.5):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = pretrained_Unet(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataset = CroppedSegmentationDataset(image_dir, mask_dir, augment=False)
    img, mask, mask_name = dataset[sample_idx]
    img_batch = img.unsqueeze(0).to(device)
    mask_np = mask.numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    with torch.no_grad():
        output = model(img_batch)
        probs = torch.sigmoid(output[0,0]).cpu().numpy()
        pred_bin = (probs > threshold).astype(np.uint8)
    img_np = img[0].numpy() if img.shape[0] == 1 else img.permute(1,2,0).numpy()
    return img_np, mask_np, probs, pred_bin

def get_sorted_checkpoints(folder='experiment', pattern='model_epoch_*.pth'):
    files = glob.glob(os.path.join(folder, pattern))
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return files

def plot_evolution(all_preds, sample_idx):
    num_epochs = len(all_preds)
    fig, axes = plt.subplots(4, num_epochs, figsize=(3*num_epochs, 12))
    if num_epochs == 1:
        axes = axes[:, np.newaxis]
    for i, pred in enumerate(all_preds):
        axes[0, i].imshow(pred['img'], cmap='gray')
        axes[0, i].set_title(f"Epoch {pred['epoch']}\nInput")
        axes[0, i].axis('off')
        axes[1, i].imshow(pred['mask'], cmap='gray')
        axes[1, i].set_title("GT Mask")
        axes[1, i].axis('off')
        im2 = axes[2, i].imshow(pred['probs'], cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title("Prob. Map")
        axes[2, i].axis('off')
        axes[3, i].imshow(pred['pred_bin'], cmap='gray')
        axes[3, i].set_title("Prediction")
        axes[3, i].axis('off')
    plt.tight_layout()
    filename = f'evolution_sample_{sample_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def main():
    NUM_SAMPLES = 10  
    image_dir = "data/cropped_images"
    mask_dir = "data/cropped_masks"
    threshold = 0.75  # Adjust as needed

    checkpoint_files = get_sorted_checkpoints('experiment', 'model_epoch_*.pth')

    for sample_idx in range(NUM_SAMPLES):
        print(f"\n=== Visualizing Sample {sample_idx} ===")
        all_preds = []
        for ckpt in checkpoint_files:
            img, mask, probs, pred_bin = predict_single_sample(
                ckpt, sample_idx, image_dir, mask_dir, threshold=threshold
            )
            epoch = int(ckpt.split('_')[-1].split('.')[0])
            all_preds.append({
                'epoch': epoch,
                'img': img,
                'mask': mask,
                'probs': probs,
                'pred_bin': pred_bin
            })
        plot_evolution(all_preds, sample_idx)

if __name__ == "__main__":
    main()
