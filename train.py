# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import argparse

from bin.utils import create_dir, get_metrics, save_checkpoint, plot_performance, keep_best_models
from bin.trainer import train, validate
from bin.create_dataset import create_dataset
from resources.Models import UNETWithAttention
from resources.Dataset import IntimaDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from tqdm import tqdm


def main(device, dataset_path, checkpoint_path, files_path, thresholding_method , BATCH_SIZE, K_FOLDS, LEARNING_RATE, NUM_EPOCHS):
    # Data set initialization
    dataset = IntimaDataset(os.path.join(dataset_path, 'train'))
    print(f"Train Dataset Size: {len(dataset)}")

    # K=5 Fold Cross Validation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True)
    fold_ids = list(kfold.split(dataset))

    # Random Subset Samplers for each fold
    train_samplers = [SubsetRandomSampler(ids[0]) for ids in fold_ids]
    valid_samplers = [SubsetRandomSampler(ids[1]) for ids in fold_ids]

    # Dictionary for storing the average results across all folds
    train_results = {'loss': [], 'acc': [], 'adc':[]}
    val_results = {'loss': [], 'acc': [], 'adc':[]}

    # Loss Function
    bce = nn.BCEWithLogitsLoss()
    def loss_fn(logits, masks):
        return bce(logits, masks)

    method = thresholding_method
    torch.cuda.empty_cache()

    # Base Model Initialization
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
    unet = UNETWithAttention(1).to(device)
    model_name = f"{unet._get_name()}{unet.uid}"
    optim = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)

    # Lists for storing intermediate results
    main_losses, main_accs, main_adcs = [], [], []

    # Best Accuracy and Loss initializtion
    best_loss = float('inf')
    best_acc = 0.0
    best_adc = 0.0
    
    best_models = dict(
        loss=None,
        adc=None,
        accuracy=None
    )

    # K-Fold Model, Optimizer and Scheduler Initialization
    models = [UNETWithAttention(1).to(device) for _ in range(K_FOLDS)]
    optimizers = [torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE) for unet in models]
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min') for optim in optimizers]

    # Model Run
    for epoch in range(NUM_EPOCHS):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_adcs = []
        val_adcs = []
        for fold, (train_sampler, valid_sampler, fold_model, fold_optim, fold_scheduler) in enumerate(zip(train_samplers, valid_samplers, models, optimizers, schedulers)):
            file_name = f"model_{fold_model._get_name()}{fold_model.uid}_"
            print(f"Epoch {epoch+1}\tFold: {fold + 1}\t Model: {file_name}")
            print("---------------------------------------------------------------------------")
            
            # Data Loader Initialization
            train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler=train_sampler)
            valid_loader = DataLoader(dataset, batch_size = BATCH_SIZE, sampler=valid_sampler)

            # Training Each Fold Model
            train_loss, train_acc, train_adc = train(epoch, fold_model, train_loader, loss_fn, fold_optim, NUM_EPOCHS, device, method)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            train_adcs.append(train_adc)

            # Validating Each Fold Model
            val_loss, val_acc, val_adc = validate(epoch, fold_model, valid_loader, loss_fn, fold_optim, NUM_EPOCHS, device, method)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_adcs.append(val_adc)
            
            fold_scheduler.step(val_loss)
        
            print("---------------------------------------------------------------------------")

        train_loss = np.array(train_losses).mean()
        train_acc = np.array(train_accs).mean()
        train_adc = np.array(train_adcs).mean()
        # Average Training Results
        train_results['loss'].append(train_loss)
        train_results['acc'].append(train_acc)
        train_results['adc'].append(train_adc)
        print(f"Average Train Results for Fold Models\nTrain Loss: {train_loss:.3f}\tTrain Accuracy: {train_acc:.2f}%\tAverage Training Dice Score[ADC]: {train_adc:.2f}")
        val_loss = np.array(val_losses).mean()
        val_acc = np.array(val_accs).mean()
        val_adc = np.array(val_adcs).mean()
        # Average Validation Results
        val_results['loss'].append(val_loss)
        val_results['acc'].append(val_acc)
        val_results['adc'].append(val_adc)
        
        print(f"Average Validation Results for Fold Models\nValidation Loss: {val_loss:.3f}\tValidation Accuracy: {val_acc:.2f}%\tAverage Validation Dice Score[ADC]: {val_adc:.2f}")
        print("---------------------------------------------------------------------------")
        
        print(f"Training Combination Model model_{model_name}_")

        # Train Base Model
        loss, acc, adc = train(epoch, unet, loader, loss_fn, optim, NUM_EPOCHS, device, method)
        main_losses.append(loss)
        main_accs.append(acc)
        main_adcs.append(adc)
        
        if epoch % 50 == 0:
            metrics = get_metrics(train_loss, train_acc, val_loss, val_acc)
            save_checkpoint(epoch, unet, optim, None, metrics, checkpoint_path)
        
        if val_loss < best_loss:
            best_loss = val_loss
            metrics = get_metrics(train_loss, train_acc, val_loss, val_acc)
            # Save model by best loss
            save_checkpoint(epoch, unet, optim, 'loss', metrics, checkpoint_path)
            best_models['loss'] = epoch
            
        elif val_acc > best_acc:
            best_acc = val_acc
            metrics = get_metrics(train_loss, train_acc, val_loss, val_acc)
            # Save model by best accuracy
            save_checkpoint(epoch, unet, optim, 'accuracy', metrics, checkpoint_path)
            best_models['accuracy'] = epoch
            
        elif val_adc > best_adc:
            best_adc = val_adc
            metrics = get_metrics(train_loss, train_acc, val_loss, val_acc)
            save_checkpoint(epoch, unet, optim, 'adc', metrics, checkpoint_path)
            best_models['adc'] = epoch

        print("---------------------------------------------------------------------------")


    # Plot main model metrics
    fig, ax = plt.subplots(3, 1, figsize=(15, 5))
    ax[0].plot(main_losses, label='Loss')
    ax[0].set_title('Main Model Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(main_accs, label='Accuracy')
    ax[1].set_title('Main Model Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[2].plot(main_adcs, label='Average Dice Score[ADC]')
    ax[2].set_title('Main Model ADC')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('ADC')
    ax[2].legend()
    plt.savefig(os.path.join(files_path, f"{model_name}_{NUM_EPOCHS}_main_metrics.png"))

    # Plot average metrics for 100 epochs
    plot_performance(train_results, val_results, f"{K_FOLDS}_{model_name}_{NUM_EPOCHS}")
    
    # Remove all other saved models
    keep_best_models(model_name, best_models, checkpoint_path)


# Code
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="Number of epochs to run", required=True)
    parser.add_argument("--folds", help="Number of fold to split the dataset on", required=True)
    parser.add_argument("--batch_size", help="The batch size used while training", required=True)
    parser.add_argument("--lr", help="Learning rate for the model", required=False, default=1e-4)
    parser.add_argument("--seed", help="Seed for reproducability", required=False, default=42)
    parser.add_argument("--split", help="Split size for dataset creation", required=False, default=0.15)
    
    args = parser.parse_args()
    seed = int(args.seed)
    os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    thresholding_method = 'basic'
    warnings.filterwarnings("ignore")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataset_path = './dataset/'
    checkpoint_path = "./checkpoints/"
    files_path = "./files/"
    results_path = "./results/"
    predictions_path = "./predictions/"

    create_dir(checkpoint_path)
    create_dir(files_path)
    create_dir(results_path)
    create_dir(predictions_path)

    create_dataset(os.path.join(os.getcwd(), 'data'), dataset_path, float(args.split), seed)
    
    if args.epochs and args.folds and args.batch_size:
        main(device, dataset_path, checkpoint_path, files_path, thresholding_method, int(args.batch_size), int(args.folds), float(args.lr), int(args.epochs))
    else:
        print("Please provide all the required arguments")
        parser.print_help()

    