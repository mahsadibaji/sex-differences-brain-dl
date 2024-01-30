import torch
import torch.nn as nn
import argparse

from data_loader import *
from training import train
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import *
import wandb
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size,  number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--results_dir', type=str, default ="./results/", help='results directory')
    parser.add_argument('--source_train_csv', type=str, help='path to source dataset(images and labels)')
    parser.add_argument('--source_val_csv', type=str, help='path to source dataset(images and labels)')
    parser.add_argument('--verbose', type=bool, default=False, help='verbose debugging flag')
    
    args = parser.parse_args()

    root_dir = args.results_dir # Path to store results
    verbose = args.verbose # Debugging flag
    
    # Set our data loaders 
    source_ds_train, source_train_loader, source_ds_val, source_val_loader = load_dev_data(
        args.source_train_csv, args.source_val_csv, args.batch_size, verbose)
    
    if verbose: 
        source_train_labels = np.array([item['sex'] for item in source_ds_train])
        # Check the shape and data type of the target labels
        print("Source train labels shape:", source_train_labels.shape)
        print("Source train labels data type:", source_train_labels.dtype)

    model = SFCN()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    model = model.cuda()
    
    # Learning rate decay scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    run = wandb.init(
        project = "sex-classification",
        config = {
            "classifier": "SFCN",
            "learning_rate_initial": args.learning_rate,
            "batch": args.batch_size,
            "epochs": args.epochs,
        },
        name = "RUN-NAME", #set your run name here
        job_type = "train"
    )

    print("Start of training...", flush=True)
    
    train(
        source_train_loader,
        source_val_loader,
        model,
        optimizer,
        scheduler,
        args.epochs,
        root_dir,
        run,
    )
    
    print("End of training.", flush=True)
        