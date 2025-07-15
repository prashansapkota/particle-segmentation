import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
import inspect
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation

#GRACE_RADIUS = 1

class CroppedSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = os.listdir(image_dir)  
        self.augment = augment

        has_var_limit = "var_limit" in inspect.signature(A.GaussNoise).parameters
        gauss = (
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3)
            if has_var_limit else
            A.GaussNoise(var_limit_x=(5.0, 20.0), var_limit_y=(5.0, 20.0), p=0.3)
        )

        if self.augment:
            self.transform = A.Compose([
                A.PadIfNeeded(
                    min_height=1024,
                    min_width=1024,
                    border_mode=cv2.BORDER_REFLECT101,
                    p=1.0
                ),
                A.RandomCrop(
                    height=256,
                    width=256,
                    p=1.0
                ),
                A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT101, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.RandomGamma(gamma_limit=(80,120), p=0.3),

                gauss,
                A.ElasticTransform(p=0.2),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.PadIfNeeded(
                    min_height=1024,
                    min_width=1024,
                    border_mode=cv2.BORDER_REFLECT101,
                    p=1.0
                ),
                A.CenterCrop(height=256, width=256, p=1.0),
                ToTensorV2()
            ])
            

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

        elif image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Handle different bit depths properly
        image = image.astype(np.float32)
        if image.max() > 1.0:
            # Normalize based on the actual bit depth
            if image.max() > 65535:  # 32-bit
                image = image / image.max()
            elif image.max() > 255:  # 16-bit
                image = image / 65535.0
            else:  # 8-bit
                image = image / 255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        # Grace radius dilation
        #mask = binary_dilation(mask, iterations=GRACE_RADIUS).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']      
        mask = augmented['mask'].unsqueeze(0) 


        return image, mask
