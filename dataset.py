import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class CroppedSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        mask_name = os.path.splitext(img_name)[0] + "_mask.png"
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if image.ndim == 2:  # grayscale tif
            image = np.stack([image] * 3, axis=-1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image.transpose(2, 0, 1))  # [3, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)          # [1, H, W]

        return image, mask







