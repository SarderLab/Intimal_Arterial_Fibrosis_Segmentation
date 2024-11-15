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
from tqdm.notebook import tqdm


# Function to get the device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to get the model and optimizer
def get_model(device, learning_rate, checkpoint_params, method='basic'):
    model = UNETWithAttention(1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_name = f"{model._get_name()}{checkpoint_params['uid']}"
    for checkpoint in os.listdir(os.path.join(checkpoint_params['path'], model_name)):
        if str(checkpoint_params['val']) in checkpoint:
            checkpoint_params['epoch'] = int(checkpoint.split('_')[-1].split('.')[0])
            break
        
    model, optimizer, metrics = load_checkpoint_from_uid(model, optimizer, checkpoint_params['uid'], checkpoint_params['epoch'], checkpoint_params['val'], checkpoint_params['path'])
    return model, optimizer

# Function to evaluate a certain dataset of images and masks
def evaluate(dataset_path, checkpoint_params, save_flag=True, method='basic'):
    time_taken = []
    eval_count = 0
    device = get_device()
    model, optim = get_model(device, 1e-4, checkpoint_params, method)
    model.eval()

    dataset = IntimaDataset(os.path.join(dataset_path, 'images'))
    print(f"Evaluation dataset {dataset.uid} loaded with {len(dataset)} images")

    loop = tqdm(dataset, desc=f"Evaluating Images", unit='image')
    with torch.no_grad():
        for idx, (image, name) in enumerate(loop):
            image = image.to(device).float().unsqueeze(0)

            st = time.time()
            logit = F.sigmoid(model(image))
            time_taken.append(time.time() - st)
            pred = get_pred(logit, method)
            if save_flag:
                save_path = save_prediction(pred, name, dataset, model.uid, os.path.join(dataset_path, 'predictions'))
                eval_count += 1
    
    print(f"Time taken to evaluate {eval_count} images: {np.mean(time_taken):0.4f} seconds")
    if save_flag:
        print(f"Predictions saved at {save_path}")
        return save_path

# Function to save the prediction as image
def save_prediction(logit, name, save_path):
    logit = logit.cpu().detach().numpy()
    logit = logit * 255
    logit = logit.astype(np.uint8)
    logit = logit.reshape(logit.shape[2], logit.shape[3])

    create_dir(save_path)
    try:
        cv2.imwrite(os.path.join(save_path, name), logit)
        return save_path
    
    except Exception as e:
        print(f"Error saving prediction {name}: {e}")
