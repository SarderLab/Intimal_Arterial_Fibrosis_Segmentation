import os
import csv
import pandas as pd
import numpy as np
import cv2
import skimage.morphology as skm
from itseg.config import ARTERIES_FEATURE_FILE, PREDICTIONS_FOLDER_NAME, RESULT_XLSX_FILENAME, INTIMA_XLSX_COLNAMES
from sklearn.metrics import DistanceMetric
from resources.Dataset import IntimaDataset
from girder_client import GirderClient


# Function to evaluate the predictions and generate the results csv
def evaluateCSV(gc: GirderClient, item_id: str,csv_path: str, svs_file_path:str , dataset_path: str) -> None:
    """This function is used to generate the results of the Intimal Segmentation with the calculated results based on arteries_Features.xlsx

    Args:
        gc (GirderClient): A girder client object to interact with the Girder API
        item_id (str): The itemId of the svs image we are working with
        csv_path (str): The csv file which contains details about the Original Image Dimensions(BBox values)
        svs_file_path (str): The local path to a copy of the svs image 
        dataset_path (str): The primary local filepath from where we can access several related files required for the computations

    Raises:
        e: Propogate the exception
    """
    idx = 0
    logs = []
    pred_path = os.path.join(dataset_path, PREDICTIONS_FOLDER_NAME)
    item_name = os.path.basename(svs_file_path)
    base_name = os.path.splitext(item_name)[0]
    arteries_feature_file_path = os.path.join(dataset_path, ARTERIES_FEATURE_FILE)
    try:
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            data = list(reader)
            for row in data:
                if if_prediction_exists(pred_path, base_name, int(row[1])):
                    pred, cropped_item_name = load_prediction(pred_path, base_name, int(row[1]))
                    (x1, y1, x2, y2), compartment_id = get_featurefile_info(row, arteries_feature_file_path)
                    intimal_area, luminal_area, is_mask_complete = get_intimal_area(pred, row, arteries_feature_file_path, compartment_id, child_radius_threshold = 20, parent_area_threshold = 3000)
                    mean_thickness = calc_thickness(pred) if intimal_area else 0
                    row.append(mean_thickness)
                    luminal_radius = get_luminal_radius(luminal_area)
                    stenosis_ratio = get_stenosis_area(intimal_area, luminal_area)
                    row.append(stenosis_ratio)
                    logs.append([idx, item_name, cropped_item_name, int(row[1]), x1, y1, x2, y2, compartment_id, luminal_area, luminal_radius, intimal_area if intimal_area else "N/A", mean_thickness, (luminal_radius + mean_thickness), is_mask_complete, stenosis_ratio])
                
        log_df = pd.DataFrame(logs, columns=INTIMA_XLSX_COLNAMES)
        save_path = os.path.join(dataset_path, RESULT_XLSX_FILENAME)
        log_df.to_excel(save_path, index=False)
        gc.uploadFileToItem(itemId= item_id, filepath=save_path)
    except Exception as e:
        raise e
        
# Function to get the stenosis area
def get_stenosis_area(intimal_area, luminal_area):
    if intimal_area is not None and luminal_area is not None and (intimal_area + luminal_area) > 0:
        return round((intimal_area / (intimal_area + luminal_area)) * 100, 4)
    else:
        return "N/A"

# Function to get the luminal radius
def get_luminal_radius(area):
    return round(np.sqrt(area / np.pi), 4) if area is not None else 0

# Function to get the luminal area from the feature file
def get_luminal_area_from_feature_file(row, feature_file_path, compartment_id):
    if compartment_id is not None and feature_file_path is not None:
        item = row[0]
        if os.path.exists(feature_file_path):
            df_mf = pd.read_excel(feature_file_path, sheet_name="Morphological Features")
            luminal_area = df_mf[df_mf['compartment_ids'] == compartment_id]['Luminal Space Area'].values[0]       
            return luminal_area
        else:
            print(f"Can't get Luminal Area from features file for item {item}")
            return 0
    else:
        return 0
    
# Function to get the feature file info
def get_featurefile_info(row, feature_filepath, distance_type='euclidean'):
    item, xmin, ymin, xmax, ymax = str(row[0]), int(row[2]), int(row[3]), int(row[4]), int(row[5])
    if os.path.exists(feature_filepath):
        df_bb = pd.read_excel(feature_filepath, sheet_name="Bounding Boxes")
        distance_fn = DistanceMetric.get_metric(distance_type)
        (x1, y1, x2, y2), compartment_id = get_closest_compartment_id(distance_fn, df_bb, xmin, ymin, xmax, ymax)
        return (x1, y1, x2, y2), compartment_id
    else:
        print(f"Features File not found for item {item}")
        return 0

# Function to get the closest compartment id
def get_closest_compartment_id(distance_fn, df, xmin, ymin, xmax, ymax):
    distances = []
    try:
        for index, row in df.iterrows():
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            dist = distance_fn.pairwise([[xmin, ymin, xmax, ymax]], [[x1, y1, x2, y2]])[0][0]
            distances.append(dist)
        distances = np.array(distances)
        min_idx = np.argmin(distances)
        x1, y1, x2, y2 = df.iloc[min_idx]['x1'], df.iloc[min_idx]['y1'], df.iloc[min_idx]['x2'], df.iloc[min_idx]['y2']
        compartment_id = df.iloc[min_idx]['compartment_ids']
        return (x1, y1, x2, y2), compartment_id
    except Exception as e:
        print(f"Error: {e}")

# Function to check if the prediction exists
def if_prediction_exists(pred_path, basename, point_index):
    if os.path.exists(os.path.join(pred_path, basename + f"_point{point_index}_cropped.png")) or any([basename in item for item in os.listdir(pred_path)]):
        return True
    return False        

# Function to load the prediction
def load_prediction(pred_path, basename, point_index):
    item_png_name = basename + f"_point{point_index}_cropped.png"
    pred_path = os.path.join(pred_path, item_png_name)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.resize(pred, (IntimaDataset.WIDTH, IntimaDataset.HEIGHT), interpolation = cv2.INTER_AREA)
    pred = pred / 255
    pred = pred.astype(np.uint8)
    return pred, item_png_name

# Function to get the intimal area
def get_intimal_area(pred, row, feature_path, compartment_id, child_radius_threshold = 20, parent_area_threshold = 3000) -> tuple:
    contour_img = pred.copy()
    is_mask_complete = True
    try:
        contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            hierarchy = hierarchy[0]
            areas = np.array([cv2.contourArea(cntr) for cntr in contours])
            threshold_area = np.pi * (child_radius_threshold ** 2)

            largest_area = np.max(areas)
            largest_idx = np.argmax(areas)
            inner_areas = np.array([area for area, hier in zip(areas, hierarchy) if hier[-1] == largest_idx and area > threshold_area]) 
            if len(inner_areas) > 0:
                luminal_area = np.max(inner_areas)
                intimal_area = largest_area - luminal_area
                return intimal_area, luminal_area, is_mask_complete
            else:
                intimal_area = sum([area for area, hier in zip(areas, hierarchy) if hier[-1] == -1 and area > parent_area_threshold])
                luminal_area = get_luminal_area_from_feature_file(row, feature_path, compartment_id)
                is_mask_complete = False
                return intimal_area, luminal_area, is_mask_complete
        else:
            
            is_mask_complete = False
            luminal_area = get_luminal_area_from_feature_file(row, feature_path, compartment_id)
            return None, luminal_area, is_mask_complete
    except Exception as e:
        print(f"Error: {e}")

# Function to calculate the thickness
def calc_thickness(img):
    distance = img.copy()
    distance = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    
    binary = img.copy()
    binary = img.astype(np.float32) / 255
    skeleton = skm.skeletonize(img).astype(np.float32)
    
    non_zero_pix = np.where(skeleton != 0)
    
    thickness_vals = cv2.multiply(distance, skeleton)
    thickness_vals = thickness_vals[skeleton != 0]
    
    return round(2 * np.mean(thickness_vals), 4) if len(thickness_vals) > 0 else 0
