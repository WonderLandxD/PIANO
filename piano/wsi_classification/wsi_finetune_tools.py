import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast
import torch.nn.functional as F
from .mil_baselines import ABMIL, SiMLP


class WSIClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, training_mode='abmil', dropout=0.25):
        """
        Args:
            feat_dim (int): Dimension of the feature
            num_classes (int): Number of output classes
            training_mode (str): Training mode, can be 'abmil', 'simlp'
        """
        super().__init__()
        self.training_mode = training_mode
        if training_mode == 'abmil':
            self.classifier = ABMIL(feat_dim, dropout=dropout, num_classes=num_classes)
        elif training_mode == 'simlp':
            self.classifier = SiMLP(feat_dim, dropout=dropout, num_classes=num_classes)

    def forward(self, x):
        logits = self.classifier(x)
        return logits
    

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
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.zeros(1).to('cpu')
    
    model.train()
    
    dtype = torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # Assuming WSIDataset returns a tuple (features, labels)
        feats = batch['feat'].to(device)
        label = batch['label'].to(device)

        with autocast(enabled=use_amp, dtype=dtype, device_type='cuda'):
            logits = model(feats)
            loss = criterion(logits, label)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss = (total_loss * i + loss.detach().cpu()) / (i + 1)
    
        train_loader.set_description(f'Train (mode: {model.training_mode}) | Epoch {epoch} | loss: {round(total_loss.item(), 3)}')
    
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
    test_loader = tqdm(test_loader, ncols=100, colour='blue', desc=f'Epoch {epoch} | Predicting')
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.zeros(1).to('cpu')
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Assuming WSIDataset returns a tuple (features, labels)
            feats = batch['feat'].to(device)  # batch[0] contains features
            label = batch['label'].to(device)     # batch[1] contains labels

            logits = model(feats)
            loss = criterion(logits, label)
            
            total_loss = (total_loss * i + loss.detach().cpu()) / (i + 1)
            labels = torch.cat([labels, label.detach().cpu()], dim=0)
            preds = torch.cat([preds, logits.detach().cpu()], dim=0)

    return preds.cpu(), labels.cpu(), total_loss.item()

def wsi_create_ckpt(model, training_mode, epoch):
    """Create checkpoint dictionary for WSI model"""
    if training_mode == 'abmil':
        state_dict = model.classifier.state_dict()
    elif training_mode == 'simlp':
        state_dict = model.classifier.state_dict()
    elif training_mode == 'mt_abmil':
        state_dict = model.classifier.state_dict()
    elif training_mode == 'mt_simlp':
        state_dict = model.classifier.state_dict()
    
    checkpoint = {
        'state_dict': state_dict,
        'epoch': epoch,
        'mode': training_mode
    }
    
    return checkpoint

def wsi_load_ckpt(ckpt_path, model):
    """Load checkpoint for WSI model"""
    checkpoint = torch.load(ckpt_path)
    model.classifier.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['epoch']