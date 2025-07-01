from dataset import CroppedSegmentationDataset
from torch.utils.data import DataLoader

dataset = CroppedSegmentationDataset('data/cropped_images', 'data/cropped_masks')
loader = DataLoader(dataset, batch_size=2, shuffle=True)


for images, masks in loader:
    print("Image batch shape:", images.shape)  # [batch, 3, H, W]
    print("Mask batch shape:", masks.shape)    # [batch, 1, H, W]
    break