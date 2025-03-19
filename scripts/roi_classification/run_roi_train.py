import argparse
import torch
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import torch.optim as optim

from piano import create_model
from piano.datasets.roi_datasets import ROIDataset
from piano.roi_classification.roi_finetune_tools import ROIClassifier, train_roi, predict_roi, roi_create_ckpt
from piano.utils.utils import planar_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    # Data related arguments
    parser.add_argument('--data_json', type=str, required=True, help='Path to data JSON file')
    parser.add_argument('--transform_config', type=str, default=None, help='Path to data augmentation config file')
    
    # Model related arguments
    parser.add_argument('--model_name', type=str, default='uni_v2', help='Pathology foundation model name')
    parser.add_argument('--training_mode', type=str, default='linear_probe', 
                      choices=['linear_probe', 'mlp_probe', 'full_param'], help='Training mode')
    parser.add_argument('--local_dir', type=bool, default=False, help='Using local directory of feature extractor')
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='Path to pretrained model')
    
    # Training related arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Mixed precision training arguments
    parser.add_argument('--amp_dtype', type=str, default='float32', 
                      choices=['float32', 'float16', 'bfloat16'], 
                      help='Data type for training')
    
    # Model saving related arguments
    parser.add_argument('--save_metric', type=str, default='accuracy',
                      choices=['accuracy', 'bal_accuracy', 'auc', 'f1', 'kappa'],
                      help='Metric to use for saving best model')
    parser.add_argument('--save_interval', type=int, default=5,
                      help='Save model checkpoint every N epochs')
    parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory to save intermediate checkpoints')
    
    # GPU related arguments
    parser.add_argument('--gpu_id', type=int, default=0, 
                      help='GPU device ID (default: 0). Set to -1 for CPU')
    
    return parser.parse_args()

def set_seed(seed):
    """Set all random seeds and deterministic flags"""
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def setup_experiment_folder(args):
    """Setup experiment folder and return paths"""
    # Get dataset name from json path
    dataset_name = os.path.splitext(os.path.basename(args.data_json))[0]
    # Create experiment folder
    exp_name = f"{dataset_name}_{args.model_name}_seed{args.seed}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(exp_dir, 'log.csv')
    if not os.path.exists(log_file):
        pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'accuracy', 
                            'bal_accuracy', 'auc', 'f1', 'kappa', 
                            'macro_specificity']).to_csv(log_file, index=False)
    
    return exp_dir, log_file

def update_training_plot(train_losses, val_losses, exp_dir):
    """Update and save training plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, 'loss_plot.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    print(f"Using device: {device}" + (f" (GPU {args.gpu_id})" if torch.cuda.is_available() and args.gpu_id >= 0 else ""))
    
    # Setup experiment folder
    exp_dir, log_file = setup_experiment_folder(args)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ROIDataset(args.data_json, mode='train', transform_config=args.transform_config)
    val_dataset = ROIDataset(args.data_json, mode='valid', transform_config=args.transform_config)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print(f"Creating {args.model_name} model...")
    backbone = create_model(args.model_name, checkpoint_path=args.pretrain_ckpt, local_dir=args.local_dir)
    model = ROIClassifier(backbone, num_classes=len(train_dataset.get_classes()), 
                         training_mode=args.training_mode).to(device)
    
    # Optimizer setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp_dtype != 'float32'
    scaler = GradScaler(enabled=use_amp)
    
    # Training loop
    best_metric = 0
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Training phase
        train_loss = train_roi(model, train_loader, optimizer, scaler, device, epoch, 
                             use_amp=use_amp, amp_dtype=args.amp_dtype)
        
        # Validation phase
        val_logits, val_labels, val_loss = predict_roi(model, val_loader, device, epoch)
        val_metrics = planar_metrics(val_logits, val_labels, len(train_dataset.get_classes()))
        
        # Update losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update plot
        update_training_plot(train_losses, val_losses, exp_dir)
        
        # Update log file
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **val_metrics
        }
        pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False)
        
        # Print metrics
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val {args.save_metric}: {val_metrics[args.save_metric]:.4f}")
        
        # Save best model based on selected metric
        if val_metrics[args.save_metric] > best_metric:
            best_metric = val_metrics[args.save_metric]
            checkpoint = roi_create_ckpt(model, args.training_mode, epoch)
            save_name = f"best_{args.save_metric}_{args.model_name}.pth"
            save_path = os.path.join(exp_dir, save_name)
            torch.save(checkpoint, save_path)
            print(f"Saved best model, validation {args.save_metric}: {best_metric:.4f}")
        
        # Save intermediate checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = roi_create_ckpt(model, args.training_mode, epoch)
            save_name = f"{args.model_name}_epoch{epoch+1}.pth"
            save_path = os.path.join(exp_dir, save_name)
            torch.save(checkpoint, save_path)
            print(f"Saved intermediate checkpoint: {save_name}")

if __name__ == '__main__':
    main()
