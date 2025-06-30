from PIL import Image
import os

# Setting file paths
RAW_IMAGE_DIR = 'data/images'
RAW_MASK_DIR = 'data/masks'
CROPPED_IMAGE_DIR = 'data/cropped_images'
CROPPED_MASK_DIR = 'data/cropped_masks'

# Creating output folders
os.makedirs(CROPPED_IMAGE_DIR, exist_ok=True)
os.makedirs(CROPPED_MASK_DIR, exist_ok=True)

for fname in os.listdir(RAW_IMAGE_DIR):
    # if the file is not an image it skips
    if not (fname.endswith(".tif") or fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".tiff")):
        continue

    img_path = os.path.join(RAW_IMAGE_DIR, fname)
    mask_path = os.path.join(RAW_MASK_DIR, os.path.splitext(fname)[0] + '_mask.png')

    # if the images doesn't have matching masks it skips
    if not os.path.exists(mask_path):
        continue
    
    # Load the image file located at the path
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    w, h = img.size

    # Crops the img and mask by 100 px from the bottom
    img_cropped = img.crop((0, 0, w, h - 100))
    mask_cropped = mask.crop((0, 0, w, h - 100 ))

    # Saves the cropped img and mask 
    img_cropped.save(os.path.join(CROPPED_IMAGE_DIR, fname))
    mask_cropped.save(os.path.join(CROPPED_MASK_DIR, os.path.splitext(fname)[0] + '_mask.png'))








