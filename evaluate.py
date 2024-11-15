import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

import warnings
warnings.filterwarnings("ignore")

from bin.utils import create_dir, load_checkpoint_from_uid
from bin.trainer import evaluate
from resources.Models import UNETWithAttention
from resources.Dataset import IntimaDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchsummary import summary
from tqdm.notebook import tqdm


def main(device, uid, dataset_path, checkpoint_path, results_path, method, save_predictions=False):
    best_models = ['loss', 'accuracy', 'adc']

    dataset = IntimaDataset(os.path.join(dataset_path, 'test'))
    print(f"Test Set Size {len(dataset)}")

    unet = UNETWithAttention(1).to(device)
    optim = torch.optim.Adam(unet.parameters(), lr=1e-4)
    model_name = f"{unet._get_name()}{uid}"
    print(f"Model Name: {model_name}")
    checkpoints = os.listdir(os.path.join(checkpoint_path, model_name))
    for model in best_models:
        for checkpoint in checkpoints:
            if model in checkpoint:
                epoch = int(checkpoint.split('_')[-1].split('.')[0])
                checkpoints.remove(checkpoint)
                print(f"Loading {checkpoint} for {model_name} at epoch {epoch}")

                unet, optim, metrics = load_checkpoint_from_uid(unet, optim, uid, epoch, model, checkpoint_path)
                time_taken, saved_path = evaluate(unet, epoch, dataset, device, "Test", method, save_predictions, results_path)
                print(f"Took {sum(time_taken):.4f} to predict {len(dataset)} number of images")
                if saved_path:
                    print(f"Results saved at {saved_path}")

    best_epoch = 0
    for cp in checkpoints:
        epoch = int(cp.split('_')[-1].split('.')[0])
        if epoch > best_epoch:
            best_epoch = epoch
    
    unet, optim, metrics = load_checkpoint_from_uid(unet, optim, uid, best_epoch, None, checkpoint_path)
    time_taken, saved_path = evaluate(unet, best_epoch, dataset, device, "Test", method, save_predictions, results_path)
    print(f"Took {sum(time_taken):.4f} to predict {len(dataset)} number of images")
    if saved_path:
        print(f"Results saved at {saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Seed for reproducability", required=False, default=42)
    parser.add_argument("--uid", help="Unique ID for model", required=True)

    
    args = parser.parse_args()
    os.environ['PYTHONSEED'] = str(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    dataset_path = './dataset/'
    checkpoint_path = "./checkpoints/"
    results_path = "./results/"
    # predictions_path = "./predictions/"

    create_dir(checkpoint_path)
    create_dir(results_path)
    # create_dir(predictions_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device

    uid = str(args.uid)
    method = 'basic'
    main(device, uid, dataset_path, checkpoint_path, results_path, method)
