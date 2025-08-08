import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
import inspect
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation

class CroppedSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=True, use_original_mask=False, dilation_iters=5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = os.listdir(image_dir)  
        self.augment = augment
        self.use_original_mask = use_original_mask
        self.dilation_iters = dilation_iters

        # has_var_limit = "var_limit" in inspect.signature(A.GaussNoise).parameters
        # gauss = (
        #     A.GaussNoise(var_limit=(1.0, 4.0), p=0.05)
        #     if has_var_limit else
        #     A.GaussNoise(var_limit_x=(1.0, 4.0), var_limit_y=(1.0, 4.0), p=0.05)
        # )

        if self.augment:
            self.transform = A.Compose([
                A.RandomCrop(
                    height=256,
                    width=256,
                    p=1.0
                ),
                A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT101, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, affine_alpha=10, p=0.1),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
                A.RandomGamma(gamma_limit=(80,120), p=0.7),
                # gauss,
                A.ElasticTransform(p=0.2),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                # A.PadIfNeeded(
                #     min_height=1024,
                #     min_width=1024,
                #     border_mode=cv2.BORDER_REFLECT101,
                #     p=1.0
                # ),
                # A.CenterCrop(height=256, width=256, p=1.0),
                ToTensorV2()
            ])
            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        base, ext = os.path.splitext(img_name)
        if base.endswith('_mask'):
            mask_name = base + '.png'  
        else:
            mask_name = base + '_mask.png'
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # --- Error handling for missing files ---
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # Normalize as before
        if image.max() > 1.0:
            if image.max() > 65535:
                image = image / image.max()
            elif image.max() > 255:
                image = image / 65535.0
            else:
                image = image / 255.0
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        # --- Dilation logic ---
        if not self.use_original_mask and self.dilation_iters > 0:
            mask = binary_dilation(mask, iterations=self.dilation_iters).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']  # [1, H, W]
        mask = augmented['mask']    # [1, H, W]

        return image, mask
# Updated CroppedSegmentationDataset with Sliding Window Support


import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation, gaussian_filter
from utils_patch import set_seed, extract_sliding_patches


class SlidingWindowSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames=None, patch_size=256, stride=128, 
                 augment=True, normalization=True, use_original_mask=False, 
                 dilation_iters=0, seed=42):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(filenames) if filenames else sorted(os.listdir(image_dir))
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.normalization = normalization
        self.use_original_mask = use_original_mask
        self.dilation_iters = dilation_iters
        self.seed = seed

        set_seed(seed)

        # ImageNet normalization for pretrained ResNet101
        normalization = A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        )

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.8, 1.2), border_mode=cv2.BORDER_REFLECT, translate_percent=0.1, rotate=(-15, 15), p=0.7),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(16, 16), p=0.7),
            A.RandomGamma(gamma_limit=(40, 160), p=0.7),
            normalization,
            ToTensorV2()
        ]) if augment else A.Compose([
            normalization,
            ToTensorV2()
        ])

        self.patches = []  # (image_idx, top, left)
        self.images = []
        self.masks = []
        # self.filenames = sorted(os.listdir(image_dir))

        for i, fname in enumerate(self.filenames):
            base_name, _ = os.path.splitext(fname)
            image_path = os.path.join(image_dir, fname)
            mask_filename = base_name + '_mask.png' if not base_name.endswith('_mask') else base_name + '.png'
            mask_path = os.path.join(mask_dir, mask_filename)

            image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if image_gray is None or mask_gray is None:
                continue

            image = np.repeat(image_gray[..., None], 3, axis=2)
            mask = (mask_gray > 127).astype(np.float32)

            if not self.use_original_mask and self.dilation_iters > 0:
                mask = binary_dilation(mask, iterations=self.dilation_iters).astype(np.float32)

            self.images.append(image)
            self.masks.append(mask)

            H, W = image.shape[:2]
            for top in range(0, H - patch_size + 1, stride):
                for left in range(0, W - patch_size + 1, stride):
                    self.patches.append((i, top, left))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image_idx, top, left = self.patches[idx]
        image = self.images[image_idx][top:top + self.patch_size, left:left + self.patch_size]
        mask = self.masks[image_idx][top:top + self.patch_size, left:left + self.patch_size]
        if not self.use_original_mask:
            mask = gaussian_filter(mask.astype(np.float32), sigma=1.0) 

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']  # [3, H, W]
        mask_tensor = augmented['mask'].unsqueeze(0).float()  # [1, H, W]
        filename = self.filenames[image_idx]

        return image_tensor, mask_tensor, filename
