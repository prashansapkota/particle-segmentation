import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation, gaussian_filter
from utils import set_seed, extract_sliding_patches


class SlidingWindowSegmentationDataset(Dataset):
    """
    Dataset for segmentation tasks using a sliding window approach.

    Loads grayscale images and binary masks, converts images to 3-channel RGB,
    applies optional mask dilation and Gaussian smoothing, extracts overlapping
    patches, and performs optional augmentation and normalization.

    """
    def __init__(self, image_dir, mask_dir, filenames=None, patch_size=256, stride=128, 
                 augment=True, normalization=True, use_original_mask=False, 
                 dilation_iters=0, seed=42):
        """Initialize dataset, load images and masks, prepare patch index list, and set transforms.
        
        Args:
            image_dir (str): Directory containing input images.
            mask_dir (str): Directory containing corresponding masks.
            filenames (list): Specific list of image filenames. Defaults to all in `image_dir`.
            patch_size (int): Size of square patches. Default is 256.
            stride (int): Step size for sliding window. Default is 128.
            augment (bool): Apply augmentation. Default is True.
            normalization (bool): Normalize with ImageNet stats. Default is True.
            use_original_mask (bool): Use masks as-is without post-processing. Default is False.
            dilation_iters (int): Number of dilation iterations for masks. Default is 0.
            seed (int): Random seed for reproducibility. Default is 42.
        """

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
        """Return the total number of extracted patches."""
        return len(self.patches)

    def __getitem__(self, idx):
        """
        Get a patch and its corresponding mask.

        Args:
            idx (int): Patch index.

        Returns:
            tuple: (image_tensor [3,H,W], mask_tensor [1,H,W], source filename)
        """
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

