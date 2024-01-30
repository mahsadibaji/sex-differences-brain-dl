import torch
import matplotlib.pylab as plt
import numpy as np
import torch.nn as nn
import numpy as np
import time

import wandb

def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, root_dir, run):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.train() # training mode - layer like dropout are active
    
    best_val_loss = 100000000

    train_time_start = time.time()
    
    loss_object = nn.BCELoss()

    for epoch in range(1,max_epochs +1):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        train_acc = 0
        val_acc = 0
    
        print("Epoch ", epoch, flush=True)
        print("Train:", end ="", flush=True)
        for step, batch in enumerate(train_loader):
            img, sex_label, sid = (batch["img"].cuda(), batch["sex"].cuda(), batch["sid"])

            optimizer.zero_grad()

            output = model(img).squeeze() # forward pass

            loss = loss_object(output, sex_label.float())

            loss.backward()

            train_loss += loss.item()

            pred_sex = (output > 0.5).float()

            train_acc += (pred_sex == sex_label).float().sum().item()

            optimizer.step()
            print("=", end = "", flush=True)

        train_loss = train_loss/(step+1)
        train_acc = train_acc/(len(train_loader.dataset))

        print(flush=True)
        print("Val:", end ="", flush=True)
        model.eval() # innference mode - layers like dropout get disabled
        with torch.no_grad():

                for step, batch in enumerate(val_loader):
                    val_img, sex_label, sid = (batch["img"].cuda(), batch["sex"].cuda(), batch["sid"])
                                        
                    output = model(val_img).squeeze()

                    loss = loss_object(output, sex_label.float())

                    val_loss += loss.item()

                    pred_sex = ( output > 0.5 ).float()
                    val_acc += (pred_sex == sex_label).float().sum().item()

                    print("=", end = "", flush=True)
                print()

                val_loss = val_loss/(step+1)
                val_acc = val_acc/(len(val_loader.dataset))

        if val_loss < best_val_loss:
            print("Saving model", flush=True)
            model_path = f"{root_dir}/sfcn_best.pth"
            torch.save(model.state_dict(), model_path)   

            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_val_accuracy"] = val_acc
            wandb.run.summary["best_train_loss"] = train_loss
            wandb.run.summary["best_train_accuracy"] = train_acc
            wandb.run.summary["best_model_epoch"] = epoch

        print(f"Training epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}, train accuracy: {train_acc}, val accuracy: {val_acc}, learning_rate: {optimizer.param_groups[0]['lr']}", flush=True)

        wandb.log({"epoch":epoch,
                   "train_loss": train_loss,
                   "val_loss": val_loss,
                   "train_accuracy": train_acc,
                   "val_accuracy": val_acc,
                   "learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step(val_loss)

    
    train_time_end = time.time()
    
    print("[INFO] total time taken to train the model: {:.2f}s".format(
    train_time_end - train_time_start), flush=True)
    
    return
