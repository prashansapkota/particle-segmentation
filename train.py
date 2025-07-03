import os
import torch
import time
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import CroppedSegmentationDataset
from utils import dice_coeff
from model import pretrained_Unet

def train_model(
        image_dir,
        mask_dir,
        epochs=1,
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
    dataset = CroppedSegmentationDataset(image_dir, mask_dir)
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

    # Loss function to check the model's prediction to the ground truth
    criterion = torch.nn.BCEWithLogitsLoss()

    # Updates the model weights to make it better after each batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Reducing the learning rate if validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )

    # Track best model
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
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
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
        avg_train_loss = train_loss / len(train_dataset)

        # Validation loops
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                # loss calculation
                val_loss += criterion(outputs, masks).item() * images.size(0)
                # dice calculation
                pred_probs = torch.sigmoid(outputs)
                val_dice += dice_coeff(pred_probs, masks)
                
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_dice = val_dice / len(val_dataset)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{out_dir}/best_model.pth")

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Dice: {avg_val_dice:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")
        
    # Save model weights
    torch.save(history, f"{out_dir}/history.pt")
    torch.save(model.state_dict(), f"{out_dir}/Final_Model.pth")
    print(f"Training complete! Model weights saved to {out_dir}")