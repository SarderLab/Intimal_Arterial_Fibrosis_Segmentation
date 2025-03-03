import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from bin.utils import load_checkpoint_from_uid, get_pred, create_dir
from resources.Models import UNETWithAttention
from resources.Dataset import IntimaDataset
from itseg.utils import uploadIntemaAnnotations


# Function to get the device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to get the model and optimizer
def get_model(device, learning_rate, file_path):
    model = UNETWithAttention(1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if torch.cuda.is_available():
        model, optimizer, metrics = load_checkpoint_from_uid(model, optimizer, file_path=file_path)
    else:
        model, optimizer, metrics = load_checkpoint_from_uid(model, optimizer, file_path=file_path, map_location=torch.device('cpu'))

    return model, optimizer

# Function to evaluate a certain dataset of images and masks
def evaluate(gc,item_id, dataset_path, model_file_path, box_coords_dict, save_flag=True, method='basic'):
    time_taken = []
    eval_count = 0
    device = get_device()
    model, optim = get_model(device, learning_rate=1e-4, file_path=model_file_path)
    model.eval()

    dataset = IntimaDataset(os.path.join(dataset_path, 'images'))
    print(f"Evaluation dataset {dataset.uid} loaded with {len(dataset)} images")
    with torch.no_grad():

        for idx, (image, name) in enumerate(dataset):
            image = image.to(device).float().unsqueeze(0)

            st = time.time()
            logit = F.sigmoid(model(image))
            time_taken.append(time.time() - st)
            pred = get_pred(logit, method)
            if save_flag:
                save_path = save_prediction(pred, name, os.path.join(dataset_path, 'predictions'), box_coords_dict)
                eval_count += 1
    
    print(f"Time taken to evaluate {eval_count} images: {np.mean(time_taken):0.4f} seconds")

    uploadIntemaAnnotations(gc,item_id, dataset_path, box_coords_dict)
    
    if save_flag:
        print(f"Predictions saved at {save_path}")
        generate_overlays(dataset_path)
    
def generate_overlays(dataset_path):
    create_dir(os.path.join(dataset_path, 'overlays'))
    if len(os.listdir(os.path.join(dataset_path, 'images'))) != len(os.listdir(os.path.join(dataset_path, 'predictions'))):
        print("Images and predictions count mismatch. Skipping overlay generation...")
        return
    
    image_list = os.listdir(os.path.join(dataset_path, 'images'))
    try:
        for file in image_list:
            image = cv2.imread(os.path.join(dataset_path, 'images', file), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (IntimaDataset.WIDTH, IntimaDataset.HEIGHT), interpolation = cv2.INTER_AREA)
            pred = cv2.imread(os.path.join(dataset_path, 'predictions', file), cv2.IMREAD_GRAYSCALE)
            pred = cv2.resize(pred, (IntimaDataset.WIDTH, IntimaDataset.HEIGHT), interpolation = cv2.INTER_AREA)

            overlay = create_overlay(image, pred)
            cv2.imwrite(os.path.join(dataset_path, 'overlays', file), overlay)
        print(f"Overlays generated at {os.path.join(dataset_path, 'overlays')}")

    except Exception as e:
        print(f"Error generating overlays: {e}")
    
def create_overlay(image, mask, alpha=0.5):
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(image, alpha, mask, (1 - alpha), 0)
    return overlay

# Function to save the prediction as image
def save_prediction(logit, name, save_path,box_coords_dict:dict):
    logit = logit.cpu().detach().numpy()
    logit = logit * 255
    logit = logit.astype(np.uint8)
    logit = logit.reshape(logit.shape[2], logit.shape[3])
    bbox = box_coords_dict[name]
    original_width = bbox["xmax"] - bbox["xmin"]
    original_height = bbox["ymax"] - bbox["ymin"]
    resized_logit = cv2.resize(logit, (original_width,original_height), interpolation=cv2.INTER_NEAREST)
    create_dir(save_path)
    try:
        cv2.imwrite(os.path.join(save_path, name), resized_logit)
        return save_path
    
    except Exception as e:
        print(f"Error saving prediction {name}: {e}")
