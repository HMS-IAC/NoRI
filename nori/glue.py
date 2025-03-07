"""
This script takes individual masks for tubules, nuclei, brushborder, and lumen and combines them to create the cytoplasm area.
It then measures intensities of tubules, cytoplasm and substrucures, and saves the results in a CSV file.
"""

import os
import cv2 as cv
import numpy as np
import pandas as pd
from typing import List

from nori.data_loader import read_tiff_and_extract_channels
from nori.utils import read_file_names, extract_tubule, extract_cyto_only_mask, get_centroid
from nori.image_processing import remove_border_tubules, image_opening
from nori.measure import measure_intensity, measure_content, measure_nuclei_intensity





ROOT_FOLDER = "_DATA"

FOLDERS = {
'OUT' : '_DATA/out',
'TUBULES' : "_DATA/segmentations/tubule",
'NUCLEI' : "_DATA/segmentations/nucleus",
'BRUSHBORDER' : "_DATA/segmentations/brush_border",
'LUMEN' : "_DATA/segmentations/lumen",
'TUBULE_CLASS' : "_DATA/tubule_class"
}



CONST = {
    'BSA' : 1.3643,
    'DOPC' : 1.0101
}



def process_image(file_path: str, FOLDERS: dict, constants: dict) -> dict:
    """
    Process a single image to extract and measure various features.
    
    Parameters:
        file_path (str): Path to the image file.
        FOLDERS (dict): Dictionary containing various folder paths.
        thresholds (dict): Dictionary containing threshold values for channel classification.
        constants (dict): Dictionary containing constant values for measurement calculations.

    Returns:
        dict: Processed data for the image.
    """
    image_name = file_path.split('/')[-1].split('.')[0]
    print(image_name)

    try:
        protein_channel, lipid_channel, ch3, ch4, ch5, ch6 = read_tiff_and_extract_channels(file_path)
    except Exception as e:
        print(f'Cannot open {image_name}: {e}')
        return {}

    try:
        print(f'Reading tubule masks...')
        tubule = cv.imread(f'{FOLDERS["TUBULES"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)
        tubule = remove_border_tubules(tubule)
        
        print('Reading substructure masks...')
        nuclei = cv.imread(f'{FOLDERS["NUCLEI"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)
        brushborder = cv.imread(f'{FOLDERS["BRUSHBORDER"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)
        lumen = cv.imread(f'{FOLDERS["LUMEN"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)


        if 'KIM' not in image_name:
            tubule_class_image = cv.imread(f'{FOLDERS["TUBULE_CLASS"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)
            brushborder = cv.imread(f'{FOLDERS["BRUSHBORDER"]}/{image_name}.png', cv.IMREAD_GRAYSCALE)
            kim=False
        else:
            tubule_class_image = cv.imread(f'{FOLDERS["TUBULE_CLASS"]}/KIM/{image_name}.png', cv.IMREAD_GRAYSCALE)
            brushborder = np.zeros_like(nuclei)
            kim=True


        contours, _ = cv.findContours(tubule, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        data_list = []
        
        # cyto_only = np.zeros_like(tubule)

        for idx, contour in enumerate(contours, start=1):
            data = process_contour(idx=idx,
                                   contour=contour,
                                   protein_channel=protein_channel,
                                   lipid_channel=lipid_channel,
                                   tubule=tubule,
                                   nuclei=nuclei,
                                   brushborder=brushborder,
                                   lumen=lumen,
                                   ch3=ch3,
                                   ch4=ch4,
                                   ch5=ch5,
                                   ch6=ch6,
                                   tubule_class_image=tubule_class_image,
                                   image_name=image_name,
                                   kim=kim)
            if data:
                data_list.append(data)
                # update_classification_image(tubule_class_image, contour, data)

        save_results(data_list, FOLDERS['OUT'], image_name)
    except Exception as e:
        print(f'Cannot find mask for {image_name}: {e}')

def process_contour(idx: int, contour: np.ndarray, protein_channel: np.ndarray, lipid_channel: np.ndarray, tubule: np.ndarray, nuclei: np.ndarray,
                    brushborder: np.ndarray, lumen: np.ndarray, ch3: np.ndarray, ch4: np.ndarray, ch5: np.ndarray, ch6: np.ndarray, tubule_class_image: np.ndarray, 
                    image_name: str, kim=False) -> dict:
    """
    Process a single tubule contour to measure various features.
    
    Parameters:
        idx (int): Index of the contour.
        contour (np.ndarray): Contour array.
        protein_channel (np.ndarray): Protein channel image.
        lipid_channel (np.ndarray): Lipid channel image.
        tubule (np.ndarray): Tubule mask image.
        nuclei (np.ndarray): Nuclei mask image.
        brushborder (np.ndarray): Brushborder mask image.
        lumen (np.ndarray): Lumen mask image.
        thresholds (dict): Dictionary containing threshold values for channel classification.
        constants (dict): Dictionary containing constant values for measurement calculations.

    Returns:
        dict: Processed data for the contour.
    """
    LTL, Umod, AQP2, KIM, unlabeled = False, False, False, False, False
    tubule_ch1, tubule_mask_ch1, x, y, w, h = extract_tubule(original_image=protein_channel, contour=contour, binary_mask=tubule)
    tubule_ch2, _, _, _, _, _ = extract_tubule(original_image=lipid_channel, contour=contour, binary_mask=tubule)
    
    nuclei_temp = nuclei[y:y+h, x:x+w]
    nuclei_temp = np.logical_and(tubule_mask_ch1, nuclei_temp)

    _, mean_nuclei_protein_intensity, std_nuclei_protein_intensity, nuclei_count = measure_nuclei_intensity(tubule_ch1, nuclei_temp)
    _, mean_nuclei_lipid_intensity, std_nuclei_lipid_intensity, _ = measure_nuclei_intensity(tubule_ch2, nuclei_temp)


    total_nuclei_protein, mean_nuclei_protein, std_nuclei_protein = measure_content(tubule_ch1, nuclei_temp, CONST['BSA'])
    total_nuclei_lipid, mean_nuclei_lipid, std_nuclei_lipid = measure_content(tubule_ch2, nuclei_temp, CONST['DOPC'])

    

    brushborder_temp = brushborder[y:y+h, x:x+w]
    brushborder_temp = np.logical_and(tubule_mask_ch1, brushborder_temp)


    if brushborder_temp.sum()!=0:
        bb_present = True
        bb_protein_intensity = np.round(tubule_ch1[brushborder_temp].mean(), 4)
        bb_lipid_intensity = np.round(tubule_ch2[brushborder_temp].mean(), 4)

        total_bb_protein, mean_bb_protein, std_bb_protein = measure_content(tubule_ch1, brushborder_temp, CONST['BSA'])
        total_bb_lipid, mean_bb_lipid, std_bb_lipid = measure_content(tubule_ch2, brushborder_temp, CONST['DOPC'])

    else:
        bb_present = False
        bb_protein_intensity = 0
        bb_lipid_intensity = 0

        total_bb_protein, mean_bb_protein, std_bb_protein = 0, 0, 0
        total_bb_lipid, mean_bb_lipid, std_bb_lipid = 0, 0, 0

    lumen_temp = lumen[y:y+h, x:x+w]
    lumen_temp = np.logical_and(tubule_mask_ch1, lumen_temp)

    if lumen_temp.sum()!=0:
        lumen_present = True
        lumen_protein_intensity = np.round(tubule_ch1[lumen_temp].mean(), 4)
        lumen_lipid_intensity = np.round(tubule_ch2[lumen_temp].mean(), 4)

        total_lumen_protein, mean_lumen_protein, std_lumen_protein = measure_content(tubule_ch1, lumen_temp, CONST['BSA'])
        total_lumen_lipid, mean_lumen_lipid, std_lumen_lipid = measure_content(tubule_ch2, lumen_temp, CONST['DOPC'])

    else:
        lumen_present = False
        lumen_protein_intensity = 0
        lumen_lipid_intensity = 0

        total_lumen_protein, mean_lumen_protein, std_lumen_protein = 0, 0, 0
        total_lumen_lipid, mean_lumen_lipid, std_lumen_lipid = 0, 0, 0



    cyto_only_temp = extract_cyto_only_mask(tubule_mask_ch1, nuclei_mask=nuclei_temp,
                                    bb_mask=brushborder_temp, lumen_mask=lumen_temp)
    
    cyto_only_temp = image_opening(255*cyto_only_temp.astype(np.uint8), radius=7, iterations=1)
    
    total_cyto_area = cyto_only_temp.sum()
    

    total_protein_intensity, mean_protein_intensity, std_protein_intensity = measure_intensity(tubule_ch1, cyto_only_temp)
    total_lipid_intensity, mean_lipid_intensity, std_lipid_intensity = measure_intensity(tubule_ch2, cyto_only_temp)

    total_protein, mean_protein, std_protein = measure_content(tubule_ch1, cyto_only_temp, CONST['BSA'])
    total_lipid, mean_lipid, std_lipid = measure_content(tubule_ch2, cyto_only_temp, CONST['DOPC'])

    

    # Other channels
    ch3_temp = ch3[y:y+h, x:x+w]
    _, mean_ch3_intensity, _ = measure_intensity(ch3_temp, tubule_mask_ch1)

    ch4_temp = ch4[y:y+h, x:x+w]
    _, mean_ch4_intensity, _ = measure_intensity(ch4_temp, tubule_mask_ch1)

    ch5_temp = ch5[y:y+h, x:x+w]
    _, mean_ch5_intensity, _ = measure_intensity(ch5_temp, tubule_mask_ch1)

    ch6_temp = ch6[y:y+h, x:x+w]
    _, mean_ch6_intensity, _ = measure_intensity(ch6_temp, tubule_mask_ch1)

    tubule_class_temp, _, _, _, _, _ = extract_tubule(original_image=tubule_class_image, contour=contour, binary_mask=tubule)
    loc = tubule_class_temp.max()

    if kim:
        # 1: LTL, 2: Umod, 3: AQP2, 4: Unlabeled
        if loc==1:
            LTL = True
            tubule_class = 'LTL'

        elif loc==2:
            Umod = True
            tubule_class = 'Umod'

        elif loc==3:
            AQP2 = True
            tubule_class = 'AQP2'

        else:
            unlabeled = True
            tubule_class = 'Unlabeled'

    else:
        tubule_class = 'not_KIM'
        if loc==1:
            KIM = True
            tubule_class = 'KIM'

    if cyto_only_temp.sum() == 0:
        return {}

    total_cyto_area = cyto_only_temp.sum()
    cx, cy = get_centroid(contour)

    # check if the folder exists
    if not os.path.exists(f'{FOLDERS["OUT"]}/tubule_masks/{image_name}'):
        os.makedirs(f'{FOLDERS["OUT"]}/tubule_masks/{image_name}')

    cv.imwrite(f'{FOLDERS["OUT"]}/tubule_masks/{image_name}/{idx}.png', tubule_mask_ch1)

    return {
            'id': idx,
            'x': cx,
            'y': cy,
            'total_cyto_area': total_cyto_area,
            'total_protein': total_protein,
            'mean_protein': mean_protein,
            'std_protein': std_protein,
            'total_lipid': total_lipid,
            'mean_lipid': mean_lipid,
            'std_lipid': std_lipid,
            'total_protein_intensity': total_protein_intensity,
            'mean_protein_intensity': mean_protein_intensity,
            'std_protein_intensity': std_protein_intensity,
            'total_lipid_intensity': total_lipid_intensity,
            'std_lipid_intensity': std_lipid_intensity,
            'mean_lipid_intensity': mean_lipid_intensity,
            'mean_ch3_intensity': mean_ch3_intensity,
            'mean_ch4_intensity': mean_ch4_intensity,
            'mean_ch5_intensity': mean_ch5_intensity,
            'mean_ch6_intensity': mean_ch6_intensity,
            'LTL': LTL,
            'Umod': Umod,
            'AQP2': AQP2,
            'tubule_class': tubule_class,
            'nuclei_count': nuclei_count,
            'total_nuclei_protein' : total_nuclei_protein,
            'mean_nuclei_protein' : mean_nuclei_protein,
            'std_nuclei_protein' : std_nuclei_protein,
            'mean_nuclei_protein_intensity': mean_nuclei_protein_intensity,
            'std_nuclei_protein_intensity': std_nuclei_protein_intensity,
            'total_nuclei_lipid' : total_nuclei_lipid,
            'mean_nuclei_lipid' : mean_nuclei_lipid,
            'std_nuclei_lipid' : std_nuclei_lipid,
            'mean_nuclei_lipid_intensity': mean_nuclei_lipid_intensity,
            'std_nuclei_lipid_intensity': std_nuclei_lipid_intensity,
            'bb_exists': bb_present,
            'bb_protein_intensity': bb_protein_intensity,
            'bb_lipid_intensity': bb_lipid_intensity,
            'total_bb_protein' : total_bb_protein,
            'mean_bb_protein' : mean_bb_protein,
            'std_bb_protein' : std_bb_protein,
            'total_bb_lipid' : total_bb_lipid,
            'mean_bb_lipid' : mean_bb_lipid,
            'std_bb_lipid' : std_bb_lipid,
            'lumen_exists': lumen_present,
            'lumen_protein_intensity': lumen_protein_intensity,
            'lumen_lipid_intensity': lumen_lipid_intensity,
            'total_lumen_protein' : total_lumen_protein,
            'mean_lumen_protein' : mean_lumen_protein,
            'std_lumen_protein' : std_lumen_lipid,
            'total_lumen_lipid' : total_lumen_lipid,
            'mean_lumen_lipid' : mean_lumen_lipid,
            'std_lumen_lipid' : std_lumen_lipid
        }


def save_results(data_list: List[dict], out_folder: str, image_name: str, tubule_class_image: np.ndarray=None):
    """
    Save the results of the image processing.

    Parameters:
        data_list (List[dict]): List of processed data for the image.
        tubule_class_image (np.ndarray): The classification image.
        out_folder (str): Output folder path.
        image_name (str): Name of the image.
    """
    if not data_list:
        return
    d = pd.DataFrame(data_list)
    if tubule_class_image is not None:
        cv.imwrite(f'{out_folder}/tubule_class_mask/{image_name}.png', tubule_class_image)
    d.to_csv(f'{out_folder}/csv/{image_name}.csv')










def main():
    """
    Main function to process all images in the specified folder.
    """
    file_paths = read_file_names(root_folder=f'{ROOT_FOLDER}/nori_images', file_type=0)
    for file_path in file_paths:
        process_image(file_path, FOLDERS, CONST)

if __name__ == '__main__':
    main()
