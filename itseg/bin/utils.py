import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from skimage import filters
import shutil

# Model saving, loading and keeping best models
def keep_best_models(model_name, best_models, dirt='checkpoints/'):
    model_path = os.path.join(dirt, model_name)
    for key, val in best_models.items():
        for cp in os.listdir(model_path):
            if key in cp and str(val) not in cp:
                os.remove(os.path.join(model_path, cp))

def save_checkpoint(epoch, model, optim, val=None, metrics = None, dirt='checkpoints/'):
    file_dir = dirt + f"{model._get_name()}{model.uid}/"
    if not os.path.exists(file_dir):
        create_dir(file_dir)    
    file_path = file_dir + f"best_model_by_{val}_{epoch}.pth" if val else file_dir + f"model_at_{epoch}.pth"
    
    if metrics:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'metrics': metrics
        }, file_path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
        }, file_path)
    if val:
        print(f"Saved model at {file_path} at {epoch} according to best {val}.")
    else:
        print(f"Saved model at {file_path} at {epoch}.")

def load_checkpoint(model, optimizer, epoch, val=None, dirt='checkpoints/'):
    file_path = dirt + f"{model._get_name()}{model.uid}/"
    file_path += f"best_model_by_{val}_{epoch}.pth" if val else f"model_at_{epoch}.pth"
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {file_path}")
    epoch = checkpoint['epoch']
    if checkpoint['metrics']:
        metrics = checkpoint['metrics']
        return model, optimizer, metrics
    else:
        return model, optimizer
    
def load_checkpoint_from_uid(model, optimizer, uid, epoch, val=None, dirt='checkpoints/'):
    file_path = os.path.join(dirt, f"{model._get_name()}{uid}/")
    file_path += f"best_model_by_{val}_{epoch}.pth" if val else f"model_at_{epoch}.pth"
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {file_path}")
    model.uid = uid
    epoch = checkpoint['epoch']
    if checkpoint['metrics']:
        metrics = checkpoint['metrics']
        return model, optimizer, metrics
    else:
        return model, optimizer, None

# Metrics saving, loading and plotting
def load_metrics(model, epoch, val=None, dirt='checkpoints/'):
    file_path = dirt + f"{model._get_name()}{model.uid}/"
    file_path += f"model_by_{val}_{epoch}.pth" if val else f"model_at_{epoch}.pth"
    checkpoint = torch.load(file_path)
    metrics = checkpoint['metrics']
    return metrics
    
def plot_performance(td, vd, file_path, file_name):
    if not td.keys() == vd.keys():
        raise KeyError(f"Keys Mismatch")
        
    keys = list(td.keys())
    fig, ax = plt.subplots(len(keys), 2, figsize=(20, 6))

    for idx, key in enumerate(keys):
        ax[idx, 0].plot(range(1, len(td[key]) + 1), td[key], label='Training')
        ax[idx, 0].set_title(f'Training {key.title()}')
        ax[idx, 0].set_xlabel('Epoch')
        ax[idx, 0].set_ylabel(key.title())
        ax[idx, 0].legend()

        ax[idx, 1].plot(range(1, len(vd[key]) + 1), vd[key], label='Validation', color='orange')
        ax[idx, 1].set_title(f'Validation {key.title()}')
        ax[idx, 1].set_xlabel('Epoch')
        ax[idx, 1].set_ylabel(key.title())
        ax[idx, 1].legend()
    
    plt.savefig(os.path.join(file_path,(file_name + '.png')))
    plt.tight_layout()
    plt.show()
    
def get_metrics(train_loss, train_acc, val_loss, val_acc):
    metrics = {
        'train': {
            'loss': train_loss,
            'acc': train_acc
        },
        'val': {
            'loss': val_loss,
            'acc': val_acc
        }
    }
    return metrics

# Data visualization
def get_mem_info():
    print(f'''
        Memory allocated: {(torch.cuda.memory_allocated() / 1e6):.3f} MBs
        Maximum Memory Aloocated: {(torch.cuda.max_memory_allocated() / 1e6):.3f} MBs
        Memory Reserved: {(torch.cuda.memory_reserved() / 1e6):.3f} MBs
        Maximum Memory Reserved: {(torch.cuda.max_memory_reserved() / 1e6):.3f} MBs
    ''')
    
def visualize(dataset, method, save_path, set_type = 'Test', dataset_path="./dataset/", top=1.0):
    if save_path.split('/')[-1][-8:] != dataset.uid or set_type.lower() != '_'.join(save_path.split('/')[-1].split('_')[:-1]):
        raise Exception("Incorrect UID or dataset type given")
    
    image_files = sorted(os.listdir(os.path.join(dataset_path, set_type.lower(), "images")))
    pred_files = sorted(os.listdir(save_path))
    if image_files != pred_files:
        raise ValueError("Number of images and predictions do not match")
        
    for idx, file in enumerate(image_files):
        image = cv2.imread(os.path.join(dataset_path, set_type.lower(), "images", file), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (dataset.WIDTH, dataset.HEIGHT), interpolation = cv2.INTER_AREA)
        mask = cv2.imread(os.path.join(dataset_path, set_type.lower(), "masks", file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (dataset.WIDTH, dataset.HEIGHT), interpolation = cv2.INTER_AREA)
        pred = cv2.imread(os.path.join(save_path, file), cv2.IMREAD_GRAYSCALE)
        pred = cv2.resize(pred, (dataset.WIDTH, dataset.HEIGHT), interpolation = cv2.INTER_AREA)
        
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        fig.suptitle(file)
        fig.tight_layout()
        fig.subplots_adjust(top=top)
        ax[0].set_title("Image")
        ax[0].imshow(image)
        ax[1].set_title("Mask")
        ax[1].imshow(mask, cmap='gray')
        ax[2].set_title("Prediction")
        ax[2].imshow(pred, cmap='gray')

# Saving predictions
def save_logit(logit, name, dataset, set_type='', dirt = "./predictions/"):
    path = os.path.join(dirt, f"{set_type.lower()}_{dataset.uid}")
    logit = logit.cpu().detach().numpy()
    logit = logit * 255
    logit = logit.astype(np.uint8)
    logit = logit.reshape(logit.shape[2], logit.shape[3])
    
    if not os.path.exists(path):
        create_dir(path)
        
    cv2.imwrite(os.path.join(path, name), logit)
    return path

# Metrics calculation
def calculate_ss(true_y, pred_y):
    # Calculate true positives, false positives, true negatives, false negatives
    tp = np.sum((true_y == 1) & (pred_y == 1))
    tn = np.sum((true_y == 0) & (pred_y == 0))
    fp = np.sum((true_y == 0) & (pred_y == 1))
    fn = np.sum((true_y == 1) & (pred_y == 0))
    
    return tp, tn, fp, fn

def calculate_metrics(true_y, pred_y):
    if all([type(true_y) == torch.Tensor, type(pred_y) == torch.Tensor]):
        true_y = true_y.view(-1).long().cpu().detach().numpy()
        pred_y = pred_y.view(-1).long().cpu().detach().numpy()
    elif all([type(true_y) != torch.Tensor, type(pred_y) != torch.Tensor]):
        true_y = true_y.flatten()
        pred_y = pred_y.flatten()

    tp, tn, fp, fn = calculate_ss(true_y, pred_y)
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Calculate precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    # Calculate recall
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    # Calculate Jaccard index (IoU)
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
    return accuracy, precision, recall, f1, jaccard

def calc_dice(true_y, pred_y, epsilon = 1):
    if all([type(true_y) == torch.Tensor and type(pred_y) == torch.Tensor]):
        if not true_y.size(0) == pred_y.size(0) and not true_y.ndims == pred_y.ndims:
           raise ValueError(f"Incorrect Input size or shape {true_y.shape} and {pred_y.shape}")
    
        true_y = true_y.contiguous().view(-1)
        pred_y = pred_y.contiguous().view(-1)
        
        intersection = torch.sum(pred_y * true_y).detach().cpu().numpy()
        union = torch.sum(pred_y + true_y).detach().cpu().numpy()
    elif all([type(true_y) != torch.Tensor and type(pred_y) != torch.Tensor]):
        true_y = true_y.flatten()
        pred_y = pred_y.flatten()
        
        intersection = np.sum(true_y * pred_y)
        union = np.sum(true_y) + np.sum(pred_y)

    return ((2 * intersection) + epsilon) / (union + epsilon)

# File operations
def clear_dir(path, uid=None):
    if uid:
        for file in os.listdir(path):
            if uid in file:
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                shutil.rmtree(file_path)
    else:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.remove(file_path)
                shutil.rmtree(file_path)
                
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# Get predictions
def get_pred(logits, method='basic', threshold = None):
    if threshold:
        return (logits > threshold)
    else:
        if method == 'basic':
            threshold = 0.5
            if type(logits) == torch.Tensor:
                return (logits > threshold).long()
            else:
                return (logits > threshold).astype(np.int32)
        elif method == 'li':
            pred = []
            for logit in logits:
                threshold = filters.threshold_li(logit.detach().cpu().numpy())
                pred.append((logit > threshold).long())
            return torch.stack(pred)

# Reset trainable parameters and clear variables
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            
def clear_variables(*args):
    for arg in args:
        del arg