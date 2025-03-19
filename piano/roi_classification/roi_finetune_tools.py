import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast


class ROIClassifier(nn.Module):
    def __init__(self, backbone, num_classes, training_mode='linear_probe'):
        """
        Args:
            backbone (nn.Module): Backbone model
            num_classes (int): Number of output classes
            training_mode (str): Training mode, can be 'linear_probe', 'full_param' or 'mlp_probe'
        """
        super().__init__()
        self.backbone = backbone
        self.training_mode = training_mode
        
        if training_mode == 'linear_probe':
            # Freeze backbone parameters for linear probing
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.fc = nn.Linear(backbone.output_dim, num_classes)
        elif training_mode == 'mlp_probe':
            # Freeze backbone parameters for MLP probing
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.fc = nn.Sequential(
                nn.Linear(backbone.output_dim, 512),
                nn.GELU(),
                nn.Linear(512, num_classes)
            )
        else:  # full_param
            # Keep all parameters trainable
            self.fc = nn.Linear(backbone.output_dim, num_classes)

    def forward(self, x):
        features = self.backbone.encode_image(x)
        logits = self.fc(features)
        return logits


# ##############
# Training functions for ROI classification task
# ###############
def train_roi(model, train_loader, optimizer, scaler, device, epoch, use_amp=False, amp_dtype='float16'):
    """
    Train one epoch for ROI classification model
    
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

        imgs = batch['image'].to(device)
        label = batch['label'].to(device)

        with autocast(enabled=use_amp, dtype=dtype, device_type='cuda'):
            logits = model(imgs)
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

# ##############
# Prediction functions for ROI classification task
# ###############
def predict_roi(model, test_loader, device, epoch):
    """
    Make predictions using trained model
    
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
            imgs = batch['image'].to(device)
            label = batch['label'].to(device)

            logits = model(imgs)
            loss = criterion(logits, label)
            
            total_loss = (total_loss * i + loss.detach().cpu()) / (i + 1)
            labels = torch.cat([labels, label.detach().cpu()], dim=0)
            preds = torch.cat([preds, logits.detach().cpu()], dim=0)

    return preds.cpu(), labels.cpu(), total_loss.item()


def roi_create_ckpt(model, training_mode, epoch=None):
    """
    Save model checkpoint based on training mode
    
    Args:
        model (nn.Module): Model to save
        training_mode (str): Training mode ('linear_probe', 'mlp_probe', 'full_param')
        epoch (int): Current epoch number
    """

    checkpoint = {
        'training_mode': training_mode,
        'epoch': epoch
    }
    
    if training_mode in ['linear_probe', 'mlp_probe']:
        checkpoint['model'] = model.fc.state_dict()
    else:  # full_param
        checkpoint['model'] = model.state_dict()
    
    return checkpoint

def roi_load_ckpt(ckpt_path, model):
    """
    Load model checkpoint
    
    Args:
        ckpt_path (str): Path to checkpoint file
        model (nn.Module): Model to load checkpoint
    """
    checkpoint = torch.load(ckpt_path)
    if checkpoint['training_mode'] in ['linear_probe', 'mlp_probe']:
        model.fc.load_state_dict(checkpoint['model'])
    else:  # full_param
        model.load_state_dict(checkpoint['model'])
        
    print("\033[92m Checkpoint loaded successfully! \033[0m")
    return model, checkpoint['epoch']