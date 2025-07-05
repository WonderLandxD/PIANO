import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
import torch.nn.functional as F
import numpy as np


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """
    Negative log-likelihood loss for survival analysis
    
    Args:
        hazards: Predicted hazards
        S: Survival probabilities
        Y: True survival times (ground truth bin)
        c: Censorship status (0 or 1)
        alpha: Weighting parameter for uncensored loss
        eps: Small value to prevent log(0)
    
    Returns:
        torch.Tensor: Computed loss
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


class NLLSurvLoss(object):
    """
    Negative Log-Likelihood Survival Loss for Cox-PH model
    
    Args:
        alpha (float): Weighting parameter for uncensored loss, default 0.15
    """
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


def train_wsi(model, train_loader, optimizer, scaler, device, epoch, use_amp=False, amp_dtype='float16'):
    """
    Train one epoch for WSI classification model
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (torch.optim): Optimizer
        scaler (torch.cuda.amp.GradScaler): Gradient scaler
        device (torch.device): Device to use
        epoch (int): Current epoch number
        use_amp (bool): Whether to use automatic mixed precision training
        amp_dtype (str): Data type for mixed precision training ('float16' or 'bfloat16')
    
    Returns:
        float: Average training loss
    """
    train_loader = tqdm(train_loader, ncols=100, colour='red', desc=f'Epoch {epoch}')
    total_loss = torch.zeros(1).to('cpu')
    
    model.train()
    
    dtype = torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Move batch data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with autocast(enabled=use_amp, dtype=dtype, device_type='cuda'):
            out_dicts = model(batch)
            loss = out_dicts['loss']

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss = (total_loss * i + loss.detach().cpu()) / (i + 1)
    
        train_loader.set_description(f'🎹 Train (\033[95m{model.training_mode}\033[0m) | Epoch {epoch} | \033[93mloss: {round(total_loss.item(), 3)}\033[0m')

    return total_loss.item()


def predict_wsi(model, test_loader, device, epoch):
    """
    Make predictions using trained WSI model
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use
        epoch (int): Current epoch number
    
    Returns:
        tuple: (Predictions, true labels, average loss)
    """
    labels = torch.tensor([], device='cpu')
    preds = torch.tensor([], device='cpu')
    test_loader = tqdm(test_loader, ncols=100, colour='blue', desc=f'🎯 Epoch {epoch} | \033[94mPredicting\033[0m')
    total_loss = torch.zeros(1).to('cpu')
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Move batch data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            out_dicts = model(batch)
            loss = out_dicts['loss']
            
            total_loss = (total_loss * i + loss.detach().cpu()) / (i + 1)
            labels = torch.cat([labels, batch['labels'].detach().cpu()], dim=0)
            preds = torch.cat([preds, out_dicts['logits'].detach().cpu()], dim=0)

    return preds.cpu(), labels.cpu(), total_loss.item()


def predict_wsi_surv(model, test_loader, device, epoch):
    """
    Make survival predictions using trained WSI survival model
    
    Args:
        model (nn.Module): Trained survival model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use
        epoch (int): Current epoch number
    
    Returns:
        tuple: (all_censorships, all_event_times, all_risk_scores, total_loss)
    """
    test_loader = tqdm(test_loader, ncols=100, colour='blue', desc=f'🎯 Epoch {epoch} | \033[94mSurvival Predicting\033[0m')
    total_loss = torch.zeros(1).to('cpu')
    
    all_risk_scores = np.zeros((len(test_loader)))
    all_censorships = np.zeros((len(test_loader)))
    all_event_times = np.zeros((len(test_loader)))
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Move batch data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            out_dicts = model(batch)
            loss = out_dicts['loss']
            
            total_loss = (total_loss * i + loss.detach().cpu()) / (i + 1)
            
            # Extract survival model outputs
            S = out_dicts['S']  # Survival probabilities
            
            # Calculate risk scores using the same method as training
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[i] = risk
            all_censorships[i] = batch['events'].item()
            all_event_times[i] = batch['survival_days'].item()

    return all_censorships, all_event_times, all_risk_scores, total_loss.item()


def wsi_create_ckpt(model, mil_name, epoch):
    """Create checkpoint dictionary for WSI model"""
    state_dict = model.state_dict()
    
    checkpoint = {
        'state_dict': state_dict,
        'epoch': epoch,
        'mil_name': mil_name
    }
    
    return checkpoint


def wsi_load_ckpt(ckpt_path, model):
    """Load checkpoint for WSI model"""
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['epoch']