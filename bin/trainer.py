import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import cv2

from bin.utils import get_pred, calc_dice, calculate_metrics, save_logit
from tqdm.notebook import tqdm


# Function to train a model on given dataloader
def train(epoch, model, loader, loss_fn, optim, num_epochs, device, method = 'basic'):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    adc = 0.0
    loop = tqdm(loader, desc=f"Training {epoch + 1}/{num_epochs}", unit='batch')
    for idx, (images, masks, names) in enumerate(loop):
        images, masks = images.to(device).float(), masks.to(device).float()
        

        logits = F.sigmoid(model(images))
        loss = loss_fn(logits, masks)
        running_loss += loss.item()
 
        pred = get_pred(logits, method)
        
        masks = masks.long()
        adc += calc_dice(masks, pred)
        correct += (pred == masks).sum()
        total += np.array(masks.shape).prod()

        optim.zero_grad()
        loss.backward()
        optim.step()
    
    running_loss /= len(loader)
    adc /= len(loader)
    accuracy = correct / total * 100
    print(f"Train Loss {running_loss:.4f} | Train Accuracy {accuracy:.2f}% | Training ADC {adc:.2f}")
    return running_loss, accuracy.cpu().item(), adc

# Function to validate on given data loader
def validate(epoch, model, loader, loss_fn, optim, num_epochs, device, method = 'basic'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    adc = 0.0

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Validating {epoch + 1}/{num_epochs}", unit='batch')
        for idx, (images, masks, names) in enumerate(loop):
            images, masks = images.to(device).float(), masks.to(device).float()

            logits = F.sigmoid(model(images))
            loss = loss_fn(logits, masks)

            running_loss += loss.item()
            pred = get_pred(logits, method)
            masks = masks.long()
            adc += calc_dice(masks, pred)
            correct += (pred == masks).sum()
            total += np.array(masks.shape).prod()
        
        running_loss /= len(loader)
        adc /= len(loader)
        accuracy = correct / total * 100
    print(f"Validation Loss {running_loss:.4f} | Validation Accuracy {accuracy:.2f}% | Validation ADC {adc:.2f}")
    return running_loss, accuracy.cpu().item(), adc

# Function to evaluate a certain dataset of images and masks
def evaluate(model, epoch, dataset, device, set_type='', method='basic', save=False, dirt = './results/'):
    results = []
    time_taken = []
    saved_path = None
    model.eval()
    loop = tqdm(dataset, desc=f"Evaluating {set_type} Set", unit='image')
    with torch.no_grad():
        for idx, (image, mask, name) in enumerate(loop):
            image, mask = image.to(device).float().unsqueeze(0), mask.to(device).float().unsqueeze(0)

            st = time.time()
            logit = F.sigmoid(model(image))
            time_taken.append(time.time() - st)
            pred = get_pred(logit, method)
            if save:
                saved_path = save_logit(pred, name, dataset, set_type)
                
            mask = mask.long()
            
            dc = calc_dice(mask, pred)
            accuracy, precision, recall, f1, jaccard = calculate_metrics(mask, pred)
            accuracy, precision, recall, f1, jaccard = round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2), round(jaccard, 2)
            results.append([name, accuracy, precision, recall, f1, jaccard, dc])

    scores = np.array([s[1:] for s in results if not (s[1] == 0 or s[2] == 0 or s[3] == 0 or s[4] == 0 or s[5] == 0 or s[6] == 0)])
    mean_score = np.mean(scores, axis=0)
    
    print(f"\tAccuracy: {mean_score[0]:0.4f}")
    print(f"\tPrecision: {mean_score[1]:0.4f}")
    print(f"\tRecall: {mean_score[2]:0.4f}")
    print(f"\tF1 score: {mean_score[3]:0.4f}")
    print(f"\tJaccard/mIoU: {mean_score[4]:0.4f}")
    print(f"\tAvg Dice Score: {mean_score[5]:0.4f}")

    if save:
        saved_path = os.path.join('./predictions/', f"{set_type.lower()}_{dataset.uid}")
        print(f"Predictions saved at {saved_path}")
        
    df = pd.DataFrame(results, columns = ['Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard Score/IoU', 'Dice Score'])
    file_path = dirt + f"{model._get_name()}{model.uid}_{set_type.lower()}_{epoch}_results.csv"
    df.to_csv(file_path)
    print(f"Results saved at {file_path}")

    return time_taken, saved_path