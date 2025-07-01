from utils import safe_makedir, list_images, get_mask_path, open_image
from PIL import Image
import os

CROPPED_IMAGE_DIR = 'data/cropped_images'
CROPPED_MASK_DIR = 'data/cropped_masks'
PATCH_IMAGE_DIR = 'data/patches/images'
PATCH_MASK_DIR = 'data/patches/masks'
PATCH_SIZE = 256

safe_makedir(PATCH_IMAGE_DIR)
safe_makedir(PATCH_MASK_DIR)

img_files = list_images(CROPPED_IMAGE_DIR)

for fname in img_files:
    img_path = os.path.join(CROPPED_IMAGE_DIR, fname)
    mask_path = get_mask_path(fname, CROPPED_MASK_DIR)
    if not os.path.exists(mask_path):
        continue
    img = open_image(img_path, mode = 'RGB')
    mask = open_image(mask_path, mode = 'L')

    w, h = img.size
    for i in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
        for j in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
            img_patch = img.crop(i, j, i + PATCH_SIZE, j + PATCH_SIZE)
            mask_patch = img.crop(i, j, i + PATCH_SIZE, j + PATCH_SIZE)
            img_patch.save()









