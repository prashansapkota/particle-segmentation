# Particle Segmentation

This repo contains a pipeline for converting scientific images and annotations into training data, cropping, patch extraction, augmentation, and training a U-Net model for particle detection.

## Steps

1. Convert CSV to masks:
    ```bash
    python scripts/csv_to_mask.py
    ```

2. Crop 100px from bottom:
    ```bash
    python scripts/crop_bottom.py
    ```

3. Make 256x256 patches:
    ```bash
    python scripts/make_patches.py
    ```

4. Train:
    ```bash
    python train.py
    ```

Data folders should be under `data/`.

...