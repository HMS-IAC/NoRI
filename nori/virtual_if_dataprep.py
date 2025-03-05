# This is a Python file to extract protein and lipid channels and save them as uint-16

import os
import cv2 as cv
import tifffile
from tqdm import tqdm

# Define the path to the input and output directories
DATA_FOLDER = "../../_DATA/ALL"
OUTPUT_FOLDER = "../../analysis/all_images(processed)/virtual_IF"

# Input files
files = os.listdir(DATA_FOLDER)

# Load all images and extract first two channels and create a new image
for file in tqdm(files):
    if file.endswith(".tif"):
        # Load the image
        img = tifffile.imread(os.path.join(DATA_FOLDER, file))

        # Extract the first two channels
        if img.shape[0] == 6:
            img = img[:2, :, :]

            # Save the image
            tifffile.imwrite(os.path.join(OUTPUT_FOLDER, file), img)

