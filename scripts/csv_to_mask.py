import pandas as pd
import numpy as np
import cv2 
import os

# Setting file paths and creating output folder
CSV_PATH = "data/merged_input_images.csv"
RAW_MASK_DIR = "data/masks"
os.makedirs(RAW_MASK_DIR, exist_ok=True)

# Reading the csv file
df = pd.read_csv(CSV_PATH)

# Loading the image with the necessary features
for index, row in df.iterrows():
    image_name = row['image_name']
    width = int(row['image_width'])
    height = int(row['image_height'])
    n_landmarks = int(row['n_landmarks'])
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Annotating the created mask with circles
    for i in range(1, n_landmarks + 1):
        x_col = f'landmark_{i}_x'
        y_col = f'landmark_{i}_y'
        
        # Check to see if x or y coordinate is missing
        if pd.isna(row.get(x_col)) or pd.isna(row.get(y_col)):
            continue

        x = int(round(row[x_col]))
        y = int(round(row[y_col]))
        cv2.circle(mask, (x,y), radius=1, color=1, thickness=-1)
    
    mask_path = os.path.join(RAW_MASK_DIR, os.path.splitext(image_name)[0] + '_mask.png')
    cv2.imwrite(mask_path, mask * 255)


        













