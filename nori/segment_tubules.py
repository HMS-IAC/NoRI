"""
This script segments tubules in a set of images using the SAM model.
The images are loaded from a folder, processed, and saved in an output folder.
"""

import os
import numpy as np
import cv2 as cv
from typing import List, Tuple
from nori.data_loader import read_tiff_and_extract_channels
from nori.image_processing import combine_input_channels, normalize_intensity_levels
from nori.segmentation import sam_segmentation_tiled
from nori.utils import read_file_names, transpose_input_image, pad_image


# Constants
ROOT_FOLDER: str = "_DATA"
OUT_FOLDER: str = os.path.join(ROOT_FOLDER, "segmentations")
CHECKPOINT_PATH: str = "sam_model/sam_vit_h_4b8939.pth"
TIFF_FILES: List[str] = read_file_names(root_folder=f"{ROOT_FOLDER}/nori_images")
TILE_SHAPE: Tuple[int, int] = (1970, 2000)
STRIDE: int = 500
DEVICE: str = 'cpu'  # 'cuda' for NVIDIA GPU

def main() -> None:
    for tiff_file in TIFF_FILES:
        image_name: str = os.path.basename(tiff_file).split('.')[0]
        print(f'Loading image {image_name}')
        
        try:
            image = load_and_prepare_image(tiff_file)
            image, image_transposed = transpose_input_image(image)
            
            # SAM segmentation
            print('Tubule segmentation...')
            process_and_save_image(image, image_name, image_transposed)
        
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

def load_and_prepare_image(tiff_file: str) -> np.ndarray:
    """Load and preprocess the TIFF file."""
    ch1, ch2, _, _, _, _ = read_tiff_and_extract_channels(tiff_file)

    # convert to 8-bit image
    ch1 = normalize_intensity_levels(ch1)
    ch2 = normalize_intensity_levels(ch2)
    
    # equalize histogram
    ch1 = cv.equalizeHist(ch1)
    ch2 = cv.equalizeHist(ch2)

    # image = cv.imread(tiff_file)
    # if image is None:
    #     raise ValueError(f"Failed to load image {tiff_file}")
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # image[:, :, 2] = combine_input_channels(image[:, :, 0], image[:, :, 1])

    ch3 = combine_input_channels(ch1, ch2)
    image = np.stack([ch1, ch2, ch3], axis=-1)

    return image

def process_and_save_image(image: np.ndarray, image_name: str, image_transposed: bool) -> None:
    """Process the image and save the segmentation result."""
    padded_image = pad_image(image, tile_shape=TILE_SHAPE, stride=STRIDE)
    
    n_tiles: int = int(np.ceil((padded_image.shape[1] - TILE_SHAPE[1]) / STRIDE)) + 1
    mask: np.ndarray = np.zeros((padded_image.shape[0], padded_image.shape[1]), dtype='uint8')

    tiles = np.zeros((*TILE_SHAPE, 3, n_tiles), dtype='uint8')
    for i in range(n_tiles):
        s_start = i * STRIDE
        s_end = s_start + TILE_SHAPE[1]
        tiles[:, :, :, i] = padded_image[:, s_start:s_end, :]

    segmented_tile = sam_segmentation_tiled(tiles=tiles, checkpoint_path=CHECKPOINT_PATH, points_per_side=64, box_nms_thresh=0.3, device=DEVICE)

    for i in range(n_tiles):
        s_start = i * STRIDE
        s_end = s_start + TILE_SHAPE[1]
        mask[:, s_start:s_end] = np.logical_or(mask[:, s_start:s_end], segmented_tile[:, :, i])

    final_mask = mask[:image.shape[0], :image.shape[1]]
    if image_transposed:
        final_mask = np.transpose(final_mask, (1, 0))

    print("Saving tubule mask...")
    cv.imwrite(os.path.join(OUT_FOLDER, f"tubule/{image_name}.tif"), normalize_intensity_levels(final_mask))

if __name__ == '__main__':
    main()