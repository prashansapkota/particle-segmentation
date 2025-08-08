import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import CroppedSegmentationDataset
from tqdm import tqdm
import csv
import random

def get_loaders(image_dir, mask_dir, batch_size=8, val_split=0.2, augment=True, dilation_iters=5, use_original_mask=False):
    dataset = CroppedSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        augment=augment,
        use_original_mask=use_original_mask,
        dilation_iters=dilation_iters
    )
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def dice_score(pred, target, threshold=0.7):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    dice = (2 * intersection + 1e-8) / (union + 1e-8)
    return dice.item()

def load_mask_pos_weights(csv_path):
    mask_pos_weights = {}
    with open(csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mask_pos_weights[row["mask_name"]] = float(row["pos_weight"])
    return mask_pos_weights

mask_pos_weights = load_mask_pos_weights("mask_pos_weights.csv")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(args, model, device):
    import torch.nn as nn
    import os
    from tqdm import tqdm

    # === SET SEED FOR REPRODUCIBILITY ===
    seed = getattr(args, 'seed', 42)
    set_seed(seed)

    image_dir = args.images_dir
    mask_dir = args.masks_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    out_dir = args.experiment_name
    dilation_iters = args.dilation_iters
    use_original_mask = args.use_original_mask
    use_wandb = getattr(args, "wandb", False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(out_dir, exist_ok=True)
    best_val_dice = -1

    if use_wandb:
        import wandb
        wandb.init(
            project="particle-segmentation",
            name=args.experiment_name
        )

    # === CREATE DATA LOADERS ONCE ===
    train_loader, val_loader = get_loaders(
        image_dir, mask_dir, batch_size,
        dilation_iters=dilation_iters,
        use_original_mask=use_original_mask
    )

    # === CALCULATE POS_WEIGHT ONCE (using a few batches for stability) ===
    # n_pos, n_pixels = 0, 0 # This block is no longer needed as pos_weights are loaded
    # sample_batches = 0
    # for images, masks in train_loader:
    #     n_pos += masks.sum().item()
    #     n_pixels += masks.numel()
    #     sample_batches += 1
    #     if sample_batches >= 5:  # Use up to 5 batches for a stable estimate
    #         break
    # n_neg = n_pixels - n_pos
    # pos_weight = n_neg / max(n_pos, 1)
    # print(f"Initial pos_weight: {pos_weight:.3f}")

    # === CALCULATE LOSS_WEIGHT ONCE ===
    # image_size = args.image_resize ** 2 # This block is no longer needed
    # n_dilated = 1 + 2 * dilation_iters * (dilation_iters + 1)
    # weight_ratio = (
    #     (image_size * 100 / n_dilated)
    #     / (image_size * 100 / (image_size - n_dilated))
    # )
    # loss_fn = nn.BCEWithLogitsLoss(
    #     pos_weight=torch.tensor(weight_ratio).to(device)
    # )
    # print(f"Loss weight: {weight_ratio:.4f}")

    for epoch in range(epochs):
        # === TRAINING ===
        model.train()
        total_loss = 0.0
        for images, masks, mask_names in tqdm(train_loader, desc="Train"):
            images, masks = images.to(device), masks.to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            optimizer.zero_grad()
            # Get pos_weight for each sample in the batch
            pos_weights = torch.tensor([mask_pos_weights[name] for name in mask_names], device=device)
            # Expand pos_weights to match mask shape
            pos_weights = pos_weights.view(-1, 1, 1, 1).expand_as(masks)
            outputs = model(images)
            loss = nn.functional.binary_cross_entropy_with_logits(outputs, masks, pos_weight=pos_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # === VALIDATION ===
        model.eval()
        total_val_loss = 0.0
        total_val_dice = 0.0
        with torch.no_grad():
            for images, masks, mask_names in tqdm(val_loader, desc="Val"):
                images, masks = images.to(device), masks.to(device)
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                outputs = model(images)
                # Get pos_weight for each sample in the batch
                pos_weights = torch.tensor([mask_pos_weights[name] for name in mask_names], device=device)
                pos_weights = pos_weights.view(-1, 1, 1, 1).expand_as(masks)
                loss = nn.functional.binary_cross_entropy_with_logits(outputs, masks, pos_weight=pos_weights)
                total_val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                dice = dice_score(probs, masks)
                total_val_dice += dice * images.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_dice = total_val_dice / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_dice": avg_val_dice,
                "dilation_iters": dilation_iters,
                # "pos_weight": pos_weight, # This line is no longer needed
            })

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            print("✅ Saved new best model!")

    torch.save(model.state_dict(), os.path.join(out_dir, "Final_Model.pth"))
    print(f"Training complete! Model weights saved to {out_dir}")



