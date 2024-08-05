from dataset.getds import get_celeba
from model.model import get_model
from loss.loss import get_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import wandb
import json
import os
from metrics.metrics import evaluation
from alive_progress import alive_it
from datetime import datetime
import random
import numpy as np

def trainer(args):
    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")
    train_ld, valid_ld, test_ld, args = get_celeba(args)

    print(f"#TRAIN Batch: {len(train_ld)}")
    print(f"#VALID Batch: {len(valid_ld)}")
    print(f"#TEST Batch: {len(test_ld)}")

    run_name = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_dir = os.getcwd() + '/runs'

    if args.log:
        run = wandb.init(
            project='keypoint detection',
            entity='scalemind',
            config=args,
            name=run_name,
            force=True
        )
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    sv_dir = run_dir + f"/{run_name}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)
    
    best_model_path = sv_dir + f'/best.pt'
    last_model_path = sv_dir + f'/last.pt'

    model = get_model(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_train_params}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_ld) * args.epoch)

    old_valid_loss = 1e26
    loss_func = get_loss(args)
    for epoch in range(args.epoch):
        log_dict = {}
        
        model.train()
        total_loss = 0
        total_metrics = 0
        for img, landmarks  in alive_it(train_ld):
            img = img.to(device)
            landmarks = landmarks.to(device)

            pred = model(img)
            loss = loss_func(pred, landmarks)
            metric = evaluation(pred, landmarks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            total_metrics += metric.item()
        
        train_mean_loss = total_loss / len(train_ld)
        train_mean_metrics = total_metrics / len(train_ld)
        
        log_dict['train/loss'] = train_mean_loss
        log_dict['train/metrics'] = train_mean_metrics

        print(f"Epoch: {epoch} - Train Loss: {train_mean_loss} - Train Metric: {train_mean_metrics}")
        

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_metrics = 0
            for img, landmarks in alive_it(valid_ld):
                img = img.to(device)
                landmarks = landmarks.to(device)

                pred = model(img)
                loss = loss_func(pred, landmarks)
                metric = evaluation(pred, landmarks)

                total_loss += loss.item()
                total_metrics += metric.item()
        
            valid_mean_loss = total_loss / len(valid_ld)
            valid_mean_metrics = total_metrics / len(valid_ld)

        log_dict['valid/loss'] = valid_mean_loss
        log_dict['valid/metrics'] = valid_mean_metrics

        print(f"Epoch: {epoch} - Valid Loss: {valid_mean_loss} - Valid Metric: {valid_mean_metrics}")
        

        save_dict = {
            'args' : args,
            'model_state_dict': model.state_dict()
        }

        if valid_mean_loss < old_valid_loss:
            old_valid_loss = valid_mean_loss
            
            torch.save(save_dict, best_model_path)
        torch.save(save_dict, last_model_path)

        if args.log:
            run.log(log_dict)
            # wandb.log({"train loss": train_mean_loss , "train metrics": train_mean_metrics})
            # wandb.log({"valid loss": valid_mean_loss , "valid metrics": valid_mean_metrics})

    if args.log:
        run.log_model(path=best_model_path, name=f'{run_name}-best-model')
        run.log_model(path=last_model_path, name=f'{run_name}-last-model')