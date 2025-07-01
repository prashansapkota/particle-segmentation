from utils import safe_makedir, list_images, get_mask_path, open_image, crop_image_bottom
from PIL import Image
import os

# Setting file paths
RAW_IMAGE_DIR = 'data/images'
RAW_MASK_DIR = 'data/masks'
CROPPED_IMAGE_DIR = 'data/cropped_images'
CROPPED_MASK_DIR = 'data/cropped_masks'
CROP_PX = 100

# Creating output folders 
safe_makedir(CROPPED_IMAGE_DIR)
safe_makedir(CROPPED_MASK_DIR)

img_files = list_images(RAW_IMAGE_DIR)

for fname in img_files:
    img_path = os.path.join(RAW_IMAGE_DIR, fname)
    mask_path = get_mask_path(fname, RAW_MASK_DIR)

    # if the images doesn't have matching masks it skips
    if not os.path.exists(mask_path):
        continue
    
    # Load the image file located at the path
    img = open_image(img_path)
    mask = open_image(mask_path)

    # Crops the img and mask by 100 px from the bottom
    img_cropped = crop_image_bottom((img, CROP_PX))
    mask_cropped = crop_image_bottom((img, CROP_PX))

    # Saves the cropped img and mask 
    img_cropped.save(os.path.join(CROPPED_IMAGE_DIR, fname))
    mask_cropped.save(get_mask_path(fname, CROPPED_MASK_DIR))








