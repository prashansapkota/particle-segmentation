import os
from PIL import Image

def safe_makedir(path):
    os.makedirs(path, exist_ok=True)

def list_images(folder, extensions=('.tif', '.png', '.jpg', '.jpeg', '.tiff')):
    """List all files in folder with allowed extensions (case-insensitive)."""
    return [fname for fname in os.listdir(folder) if fname.lower().endswith(extensions)]

def get_mask_path(image_fname, mask_dir, mask_suffix='_mask.png'):
    """Given an image filename and mask directory, return full mask path."""
    base = os.path.splitext(image_fname)[0]
    return os.path.join(mask_dir, f"{base}{mask_suffix}")

def open_image(path, mode=None):
    """Open an image and convert to specified mode if given."""
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return img

def crop_image_bottom(img, crop_px):
    """Crop given number of pixels from bottom of image."""
    w, h = img.size
    return img.crop((0, 0, w, h - crop_px))