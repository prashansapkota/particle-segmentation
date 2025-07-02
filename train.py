import torch
from torch.utils.data import DataLoader, random_split
from dataset import CroppedSegmentationDataset
from model import pretrained_Unet

def train_model(
        image_dir,
        mask_dir,
        epochs=30,
        batch_size=2,
        lr=1e-3,
        device=None,
        out_dir="Experiment"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and split
    dataset = CroppedSegmentationDataset(image_dir, mask_dir)
    val_size = max(1, int(0.2 % len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model
    model = pretrained_Unet(device)

    # Loss function to check the model's prediction to the ground truth
    criterion = torch.nn.BCEWithLogitsLoss()

    # Updates the model weights to make it better after each batch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loops
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation loops
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save model weights
        torch.save(model.state_dict(), f"{out_dir}/unet_particles.pth")
        print(f"Training complete! Model weights saved to {out_dir}/unet_particles.pth")
















    


    