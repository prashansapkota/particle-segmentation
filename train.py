import os
import torch
import time
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CroppedSegmentationDataset
from utils import dice_coeff_with_threshold, bce_dice_loss
from model import pretrained_Unet

from functools import partial #NEW
import numpy as np

def train_model(
        image_dir,
        mask_dir,
        epochs=30,
        batch_size=2,
        lr=1e-3,
        device=None,
        out_dir="Experiment"
):
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    model = pretrained_Unet(device)
    print(f"moved")

    # load dataset and split
    dataset = CroppedSegmentationDataset(image_dir, mask_dir, augment=True)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loader
    num_workers = min(4, os.cpu_count())
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(2, num_workers),
        pin_memory=False
    )

    # Simple weighted BCE - adjust pos_weight for your particle density
    # pos_weight = torch.tensor([5.0]).to(device)  # Start here, adjust if needed
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    # pos_weight = torch.tensor([4.0]).to(device)  # Start here, adjust if needed
    # criterion = partial(bce_dice_loss, pos_weight=pos_weight)

    # ── adaptive pos-weight tensor & loss fn ───────────────────
    pos_weight_t = torch.tensor([2.0], device=device)    # start at 2.0

    def loss_fn(logits, targets):
        return bce_dice_loss(logits, targets, pos_weight=pos_weight_t)

    best_thresh = 0.80   # will be updated each epoch

    # Updates the model weights to make it better after each batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Reducing the learning rate if validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )

    # Track best model
    best_val_loss = float('inf')
    best_val_dice = -float('inf')
    best_pos_weight = pos_weight_t.item()
    best_thresh_for_dice = best_thresh
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    # Adaptive tuning with patience
    fp_over_fn_patience = 0
    fn_over_fp_patience = 0
    ADAPT_PATIENCE = 2   # epochs before changing pos_weight

    pos_weight_history = []
    threshold_history = []

    print(f"Training for {epochs} epochs...")
    start_time = time.time()

    # Training Loops
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            # loss = criterion(outputs, masks)

            loss = loss_fn(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
        avg_train_loss = train_loss / len(train_dataset)

        # Validation loops
        model.eval()
        val_loss = 0.0
        val_dice = 0.0


        
        all_fp = all_fn = all_tp = 0
        probs_list, masks_list = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)




                probs = torch.sigmoid(outputs).cpu()
                probs_list.append(probs)
                masks_list.append(masks.cpu())

                pb = (probs > 0.8).float()  # quick fixed-thr counts
                tp = ((pb==1)&(masks.cpu()==1)).sum()
                fp = ((pb==1)&(masks.cpu()==0)).sum()
                fn = ((pb==0)&(masks.cpu()==1)).sum()
                all_tp += tp.item(); all_fp += fp.item(); all_fn += fn.item()

                
                # loss calculation
                # val_loss += criterion(outputs, masks).item() * images.size(0)


                val_loss += loss_fn(outputs, masks).item() * images.size(0)



                # FIXED: dice calculation with proper thresholding
                # val_dice += dice_coeff_with_threshold(outputs, masks, threshold=0.5)
                val_dice += dice_coeff_with_threshold(outputs, masks, threshold=best_thresh)



                
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_dice = val_dice / len(val_dataset)



        # ── A) adapt pos_weight ───────────────────────────────
        fp_rate = all_fp / max(all_fp + all_tp, 1)
        fn_rate = all_fn / max(all_fn + all_tp, 1)
        if fp_rate > fn_rate * 1.2 and pos_weight_t.item() > 1.0:
            fp_over_fn_patience += 1
            fn_over_fp_patience = 0
            if fp_over_fn_patience >= ADAPT_PATIENCE:
                pos_weight_t *= 0.9
                fp_over_fn_patience = 0
        elif fn_rate > fp_rate * 1.2 and pos_weight_t.item() < 8.0:
            fn_over_fp_patience += 1
            fp_over_fn_patience = 0
            if fn_over_fp_patience >= ADAPT_PATIENCE:
                pos_weight_t *= 1.1
                fn_over_fp_patience = 0
        else:
            fp_over_fn_patience = 0
            fn_over_fp_patience = 0
        pos_weight_t.clamp_(2.0, 7.0)


        # ── B) find best Dice threshold ───────────────────────
        probs_cat = torch.cat(probs_list)
        masks_cat = torch.cat(masks_list)
        th_grid = torch.linspace(0.2, 0.95, steps=31)
        dices = [dice_coeff_with_threshold(probs_cat, masks_cat, t).item() for t in th_grid]
        best_thresh = th_grid[int(np.argmax(dices))].item()
        epoch_val_dice = max(dices)

        pos_weight_history.append(pos_weight_t.item())
        threshold_history.append(best_thresh)

        # Save best model by best Dice
        if epoch_val_dice > best_val_dice:
            best_val_dice = epoch_val_dice
            best_pos_weight = pos_weight_t.item()
            best_thresh_for_dice = best_thresh
            torch.save(model.state_dict(), f"{out_dir}/best_model.pth")
            with open(f"{out_dir}/best_thresh.txt", "w") as f:
                f.write(str(best_thresh_for_dice))
            with open(f"{out_dir}/best_pos_weight.txt", "w") as f:
                f.write(str(best_pos_weight))

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)

        # Update scheduler
        scheduler.step(avg_val_loss)

        # # Save best model
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), f"{out_dir}/best_model.pth")

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Dice: {avg_val_dice:.4f} | "
              f"pos_w {pos_weight_t.item():.2f} | thr {best_thresh:.2f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")
        
    # Save model weights
    torch.save(history, f"{out_dir}/history.pt")
    torch.save(model.state_dict(), f"{out_dir}/Final_Model.pth")
    print(f"Training complete! Model weights saved to {out_dir}")