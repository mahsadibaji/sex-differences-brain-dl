import torch
import torch.nn as nn
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import wandb
import matplotlib.pyplot as plt

# Assuming these imports based on your initial code
from data_loader import *
from model import *

if __name__ == "__main__":

    # WandB Initialization
    run = wandb.init(
        project="sex-classification",
        name="RUN-NAME", #add your run name here
        job_type="eval",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="./results/", help='results directory')
    parser.add_argument('--source_test_csv', type=str, help='path to source dataset(images and labels)')
    parser.add_argument('--saved_model_path', type=str)
    parser.add_argument('--verbose', type=bool, default=False, help='verbose debugging flag')

    args = parser.parse_args()

    root_dir = args.results_dir
    verbose = args.verbose

    source_ds_test, source_test_loader = load_test_data(args.source_test_csv, verbose)

    model = SFCN()
    model.cuda()
    model.load_state_dict(torch.load(args.saved_model_path)) #load trained model weights
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    columns = ["id", "pred", "prob"]
    predictions = wandb.Table(columns=columns) #wandb table to record predictions results

    for step, batch in enumerate(source_test_loader):
        img, sex, sid = (batch["img"].cuda(), batch["sex"].cuda(), batch["sid"])
        sex = sex.squeeze()
        output = model(img).squeeze()

        pred_prob = output.detach().cpu().numpy()
        pred_label = (output > 0.5).float().detach().cpu().numpy()
        
        predictions.add_data(sid, pred_label, pred_prob)

        all_preds.append(pred_label)
        all_labels.append(sex.cpu().numpy())
        all_probs.append(pred_prob)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Log metrics to wandb
    wandb.log({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    wandb.log({
        'predictions': predictions
    })

    # Confusion Matrix Visualization
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f"{root_dir}/cm.png", dpi=500)
    plt.close()

    # Confidence Distribution Plots
    correct_probs = [all_probs[i] for i in range(len(all_labels)) if all_labels[i] == all_preds[i]]
    incorrect_probs = [all_probs[i] for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]

    plt.figure(figsize=(12, 6))
    plt.hist(correct_probs, bins=50, alpha=0.5, label='Correct Predictions', color='green')
    plt.hist(incorrect_probs, bins=50, alpha=0.5, label='Incorrect Predictions', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution of Predictions')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.savefig(f"{root_dir}/confidence_dist.png", dpi=500)

    # Finish the wandb run
    wandb.finish()
