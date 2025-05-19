import os
import tifffile
import cv2 as cv
import numpy as np

PADDING=10

TEST = True

if TEST:
    DATA_PATH = {
    "raw_images": "assets/test_data/raw_images",
    "processed_images": "assets/test_data/processed_images",
    "tubule_labels": "assets/test_data/tubule_masks",
    "nuclei_masks": "assets/test_data/nuclei_masks",
    "bb_masks": "assets/test_data/brushborder_masks",
    "lumen_masks": "assets/test_data/lumen_masks",
    "glomeruli_masks": "assets/test_data/glomeruli_masks",
    "csv_files": "assets/test_data/csv_files"
    }
else:
    DATA_PATH = {
        "raw_images": "../../_DATA/ALL",
        "processed_images": "../../analysis/all_images(processed)",
        "tubule_labels": "../../analysis/all_images(processed)/analyzed/out/labeled_tubule_mask",
        "nuclei_masks": "../../analysis/all_images(processed)/analyzed/masks/nuclei_cleaned",
        "bb_masks": "../../analysis/all_images(processed)/analyzed/masks/bb_cleaned",
        "lumen_masks": "../../analysis/all_images(processed)/analyzed/masks/lumen_cleaned",
        "glomeruli_masks": "../../analysis/all_images(processed)/analyzed/masks/glomeruli",
        "csv_files": "../../analysis/all_images(processed)/analyzed/out/csv"
    }


def read_file_names(root_folder: str, file_type=0):
    """
    Returns a list of full paths for TIFF files within a given directory.

    Args:
        root_folder (str): Root directory to search for TIFF files.

    Returns:
        List[str]: List of file paths.
    """
    if file_type==0:
        image_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_folder)
            for file in files if file.lower().endswith(('.tif', '.tiff'))
        ]
        image_files = [file.split('/')[-1].split('.')[0] for file in image_files]

    elif file_type==1:
        image_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_folder)
            for file in files if file.lower().endswith(('.png'))
        ]
        image_files = [file.split('/')[-1].split('.')[0] for file in image_files]

    return image_files



def read_tiff_and_extract_channels(file_path, separate_channels=True):
    """
    Reads a TIFF file, extracts protein and lipid channels, and returns 2 image stacks with image patches.

    Args:
        file_path (str): Path to the TIFF file.

    Returns:
        list: List containing two image channels.
    """
    try:
        # Read TIFF file. Image is read in (c x h x w) format
        image = tifffile.imread(file_path)


        if separate_channels:
            # Check the number of channels in the image
            if image.shape[0] == 7:
                print("Image has 7 channels")
                protein_channel = image[1, :, :]
                lipid_channel = image[2, :, :]
                endomucine_channel = image[3, :, :]
                ch4 = image[4, :, :]
                ch5 = image[5, :, :]
                ch6 = image[6, :, :]

                return [protein_channel, lipid_channel, endomucine_channel, ch4, ch5, ch6]
            elif image.shape[0] == 6:
                print("Image has 6 channels")
                protein_channel = image[0, :, :]
                lipid_channel = image[1, :, :]
                endomucine_channel = image[2, :, :]                
                ch4 = image[3, :, :]
                ch5 = image[4, :, :]
                ch6 = image[5, :, :]

                return [protein_channel, lipid_channel, endomucine_channel, ch4, ch5, ch6]
            elif image.shape[0] == 3:
                print("Image has 3 channels. Assigning Ch1 as Protein and Ch2 as Lipid")
                protein_channel = image[0, :, :]
                lipid_channel = image[1, :, :]
                endomucine_channel = image[2, :, :]

                return [protein_channel, lipid_channel, endomucine_channel]
            else:
                print(f"Image has {image.shape[0]} channels")
                return None
        else:
            return image

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    


def _create_mask_outlines(masks, thickness=2):
    outlined_image = np.zeros_like(masks)
    contours, _ = cv.findContours(masks.astype('uint8'), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(outlined_image, contours, -1, 255, thickness=thickness)

    return outlined_image

def extract_tubule(tubule_mask, idx):
    
    # Create mask with idx
    mask = 255*(tubule_mask==idx).astype(np.uint8)
    # crop individual channels
    contour, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cell_mask = np.zeros_like(tubule_mask, dtype='uint8')
    
    cv.drawContours(cell_mask, [contour[0]], -1, 255, thickness=cv.FILLED)
    k = create_circular_se(5)
    cell_mask = cv.dilate(cell_mask, kernel=k)

    # Crop the region from the original image using the mask
    # cell_cropped = cv.bitwise_and(original_image, original_image, mask=cell_mask)

    # Get the bounding box of the contour to determine the filename
    x, y, w, h = cv.boundingRect(contour[0])

    x = np.max((0,x-PADDING))
    y = np.max((0,y-PADDING))
    w+=2*PADDING
    h+=2*PADDING

    return cell_mask[y:y + h, x:x + w], x, y, w, h


def create_circular_se(radius):

    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    

