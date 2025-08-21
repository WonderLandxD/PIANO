import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import torch
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.optim as optim
import json
import scipy.stats as stats
from sklearn.metrics import confusion_matrix

from piano.model.patch_encoder import get_model_output_dim
from piano.datasets.wsi_datasets import WSIDatasetKFold

from piano.model.mil_factory import create_mil_model

from piano.utils.wsi_finetune_tools import train_wsi, predict_wsi, wsi_create_ckpt
from piano.utils.evaluation_metrics import planar_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    # Data related arguments
    parser.add_argument('--data_json', type=str, required=True, help='Path to data JSON file with k-fold structure')
    parser.add_argument('--few_shot', type=int, default=None, help='Number of samples per class for few-shot learning')
    
    # Model related arguments
    parser.add_argument('--pfm_name', type=str, default='conch_v1', help='Pathology foundation model name')
    parser.add_argument('--mil_name', type=str, default='abmil', help='Training mode')
    
    # Training related arguments
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--early_stopping', type=int, default=5, help='Number of epochs with no improvement to wait before early stopping')
    
    # K-fold related arguments
    parser.add_argument('--num_folds', type=int, default=None, help='Number of folds (if None, auto-detect from JSON)')
    parser.add_argument('--start_fold', type=int, default=0, help='Starting fold index')
    parser.add_argument('--end_fold', type=int, default=None, help='Ending fold index (if None, run all folds)')
    
    # Mixed precision training arguments
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', 
                      choices=['float32', 'float16', 'bfloat16'], 
                      help='Data type for training')
    
    # Model saving related arguments
    parser.add_argument('--save_metric', type=str, default='bal_accuracy',
                      choices=['accuracy', 'bal_accuracy', 'auc', 'f1', 'kappa', 'specificity'],
                      help='Metric to use for saving best model')
    parser.add_argument('--save_interval', type=int, default=50,
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

def get_num_folds_from_json(data_json_path):
    """Detect number of folds from JSON file"""
    with open(data_json_path, 'r') as f:
        data = json.load(f)
    
    # Find fold keys (e.g., fold_0, fold_1, etc.)
    fold_keys = [key for key in data.keys() if key.startswith('fold_')]
    return len(fold_keys)

def setup_experiment_folder(args, fold_idx=None):
    """Setup experiment folder and return paths"""
    # Get dataset name from json path
    dataset_name = os.path.splitext(os.path.basename(args.data_json))[0]
      
    # Create a clean and organized folder name
    main_exp_name = f"Kfold_{args.pfm_name}_{args.mil_name}"
    if args.few_shot is not None:
        main_exp_name += f"_fs{args.few_shot}"
    
    # Add timestamp to the experiment name
    main_exp_name += f"_seed{args.seed}"
    
    main_exp_dir = os.path.join(args.save_dir, dataset_name, main_exp_name)
    os.makedirs(main_exp_dir, exist_ok=True)
    
    if fold_idx is not None:
        # Create fold-specific subfolder
        fold_dir = os.path.join(main_exp_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Create log file for this fold
        log_file = os.path.join(fold_dir, 'log.tsv')
        if not os.path.exists(log_file):
            pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'accuracy', 
                                'bal_accuracy', 'auc', 'f1', 'kappa', 
                                'macro_specificity']).to_csv(log_file, index=False, sep='\t', float_format='%.6f')
        
        return fold_dir, log_file, main_exp_dir
    else:
        return main_exp_dir

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

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save a beautiful confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_predictions(val_logits, val_labels, val_dataset, exp_dir):
    """Save detailed predictions for validation set"""
    # Get predictions
    pred_probs = torch.softmax(val_logits, dim=1)
    pred_classes = torch.argmax(val_logits, dim=1)
    pred_confidence = torch.max(pred_probs, dim=1)[0]
    
    # Get class names
    class_names = val_dataset.get_classes()
    
    # Create predictions data
    predictions_data = []
    for i in range(len(val_labels)):
        slide_id = val_dataset.get_name(i)
        true_class = class_names[int(val_labels[i].item())]  # Convert to int
        pred_class = class_names[int(pred_classes[i].item())]  # Convert to int
        confidence = pred_confidence[i].item()
        correct = (val_labels[i] == pred_classes[i]).item()
        
        predictions_data.append({
            'slide_id': slide_id,
            'pred_class': pred_class,
            'true_class': true_class,
            'pred_confidence': confidence,
            'correct': correct
        })
    
    # Save to TSV
    predictions_df = pd.DataFrame(predictions_data)
    predictions_path = os.path.join(exp_dir, 'predictions.tsv')
    predictions_df.to_csv(predictions_path, index=False, sep='\t', float_format='%.6f')
    
    return predictions_df

def train_single_fold(args, fold_idx, device):
    """Train a single fold and return validation metrics"""
    print(f"\n=== Training Fold {fold_idx} ===")
    
    # Setup fold-specific experiment folder
    exp_dir, log_file, main_exp_dir = setup_experiment_folder(args, fold_idx)
    
    # Load datasets for this fold
    print(f"Loading datasets for fold {fold_idx}...")
    train_dataset = WSIDatasetKFold(args.data_json, fold_name=fold_idx, mode='train', pfm_name=args.pfm_name, few_shot=args.few_shot)
    val_dataset = WSIDatasetKFold(args.data_json, fold_name=fold_idx, mode='valid', pfm_name=args.pfm_name)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print(f"Creating \033[95m{args.mil_name}\033[0m slide-level finetune method under \033[91m{args.pfm_name}\033[0m...")
    feat_dim = get_model_output_dim(args.pfm_name)

    model = create_mil_model(model_name=args.mil_name, 
                             dim_in=feat_dim, 
                             num_classes=len(train_dataset.get_classes())
                             ).to(device)
    
    # Add training_mode attribute to model
    model.training_mode = args.mil_name
    
    # Optimizer setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp_dtype != 'float32'
    scaler = GradScaler(enabled=use_amp)
    
    # Training loop with early stopping
    best_metric = 0
    train_losses = []
    val_losses = []
    best_val_metrics = None
    epochs_without_improvement = 0
    
    print(f"Starting training for fold {fold_idx} with early stopping patience: {args.early_stopping}...")
    for epoch in range(args.num_epochs):
        # Training phase
        train_loss = train_wsi(model, train_loader, optimizer, scaler, device, epoch, 
                             use_amp=use_amp, amp_dtype=args.amp_dtype)
        
        # Validation phase
        val_logits, val_labels, val_loss = predict_wsi(model, val_loader, device, epoch)
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
        pd.DataFrame([log_data]).to_csv(log_file, mode='a', header=False, index=False, sep='\t', float_format='%.6f')
        
        # Print metrics
        print(f"\033[96mFold {fold_idx} Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val {args.save_metric}: {val_metrics[args.save_metric]:.4f}\033[0m")
        
        # Check for improvement and early stopping
        current_metric = val_metrics[args.save_metric]
        if current_metric > best_metric:
            best_metric = current_metric
            best_val_metrics = val_metrics.copy()
            best_val_logits = val_logits.clone()
            best_val_labels = val_labels.clone()
            epochs_without_improvement = 0
            
            # Save best model
            checkpoint = wsi_create_ckpt(model, args.mil_name, epoch)
            save_name = f"best_{args.save_metric}_{args.pfm_name}.pth"
            save_path = os.path.join(exp_dir, save_name)
            torch.save(checkpoint, save_path)
            # Display the message in dark green using ANSI escape codes
            print(f"\033[32mFold {fold_idx}: Saved best model, validation {args.save_metric}: {best_metric:.4f}\033[0m")
        else:
            epochs_without_improvement += 1
            print(f"Fold {fold_idx}: No improvement for {epochs_without_improvement} epochs")
            
            # Early stopping check
            if epochs_without_improvement >= args.early_stopping:
                print(f"\033[33mFold {fold_idx}: Early stopping triggered after {epoch+1} epochs (no improvement for {args.early_stopping} epochs)\033[0m")
                break
        
        # Save intermediate checkpoint every N epochs
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = wsi_create_ckpt(model, args.mil_name, epoch)
            save_name = f"{args.pfm_name}_epoch{epoch+1}.pth"
            save_path = os.path.join(exp_dir, save_name)
            torch.save(checkpoint, save_path)
            # Display the message in dark blue using ANSI escape codes
            print(f"\033[34mFold {fold_idx}: Saved intermediate checkpoint: {save_name}\033[0m")
    
    # After training, save confusion matrix and predictions for best model
    pred_classes = torch.argmax(best_val_logits, dim=1)
    class_names = train_dataset.get_classes()
    
    # Plot confusion matrix
    cm_path = os.path.join(exp_dir, 'confusion_matrix.png')
    plot_confusion_matrix(best_val_labels.numpy(), pred_classes.numpy(), class_names, cm_path)
    
    # Save detailed predictions
    save_predictions(best_val_logits, best_val_labels, val_dataset, exp_dir)
    
    print(f"Fold {fold_idx} completed. Best {args.save_metric}: {best_metric:.4f}")
    return best_val_metrics

def save_kfold_summary(args, all_fold_metrics, main_exp_dir):
    """Save k-fold cross-validation summary with detailed statistics"""
    # Create DataFrame from all fold metrics
    metrics_df = pd.DataFrame(all_fold_metrics)
    
    # Calculate detailed statistics for each metric
    summary_data = []
    metrics_to_analyze = ['accuracy', 'bal_accuracy', 'auc', 'f1', 'kappa', 'macro_specificity']
    
    for metric in metrics_to_analyze:
        values = metrics_df[metric].values
        n = len(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Sample standard deviation
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        degrees_freedom = n - 1
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_error = t_value * (std_val / np.sqrt(n))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        
        # Convert to percentages for display
        mean_pct = mean_val * 100
        std_pct = std_val * 100
        ci_lower_pct = ci_lower * 100
        ci_upper_pct = ci_upper * 100
        min_pct = np.min(values) * 100
        max_pct = np.max(values) * 100
        
        # Create formatted strings
        mean_std_str = f"{mean_pct:.2f}±{std_pct:.2f}"
        latex_str = f"${{{mean_pct:.2f}}}_{{{std_pct:.2f}}}$"
        latex_pm_str = f"${mean_pct:.2f}$±${std_pct:.2f}$"
        latex_scriptscriptstyle_str = f"{mean_pct:.2f}$\\scriptscriptstyle{{({std_pct:.2f})}}$"
        
        summary_data.append({
            'Metric': metric.upper().replace('_', ' '),
            'Mean (%)': f"{mean_pct:.2f}",
            'Std (%)': f"{std_pct:.2f}",
            'Min (%)': f"{min_pct:.2f}",
            'Max (%)': f"{max_pct:.2f}",
            '95% CI Lower (%)': f"{ci_lower_pct:.2f}",
            '95% CI Upper (%)': f"{ci_upper_pct:.2f}",
            'Mean ± Std': mean_std_str,
            'LaTeX Format': latex_str,
            'LaTeX PM Format': latex_pm_str,
            'LaTeX Scriptscriptstyle': latex_scriptscriptstyle_str,
            'N Folds': n
        })
    
    # Save individual fold results (keep original decimal format)
    fold_results_path = os.path.join(main_exp_dir, 'fold_results.tsv')
    metrics_df.to_csv(fold_results_path, index=False, sep='\t', float_format='%.6f')
    
    # Save summary statistics with better formatting
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(main_exp_dir, 'summary_results.tsv')
    summary_df.to_csv(summary_path, index=False, sep='\t')
    
    # Prepare summary table text
    summary_text = []
    summary_text.append("=" * 160)
    summary_text.append("K-FOLD CROSS-VALIDATION SUMMARY")
    summary_text.append("=" * 160)
    summary_text.append(f"{'Metric':<20} {'Mean(%)':<10} {'Std(%)':<10} {'95% CI(%)':<25} {'Min(%)':<10} {'Max(%)':<10} {'Mean±Std':<20} {'LaTeX':<25} {'LaTeX±':<20} {'LaTeX Script':<20}")
    summary_text.append("-" * 160)
    
    # Use original values for console display
    for metric in metrics_to_analyze:
        values = metrics_df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        degrees_freedom = len(values) - 1
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
        margin_error = t_value * (std_val / np.sqrt(len(values)))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        
        # Convert to percentages for display
        mean_pct = mean_val * 100
        std_pct = std_val * 100
        ci_lower_pct = ci_lower * 100
        ci_upper_pct = ci_upper * 100
        min_pct = np.min(values) * 100
        max_pct = np.max(values) * 100
        
        mean_std_str = f"{mean_pct:.2f}±{std_pct:.2f}"
        latex_str = f"${{{mean_pct:.2f}}}_{{{std_pct:.2f}}}$"
        latex_pm_str = f"${mean_pct:.2f}$±${std_pct:.2f}$"
        latex_scriptscriptstyle_str = f"{mean_pct:.2f}$\\scriptscriptstyle{{({std_pct:.2f})}}$"
        ci_str = f"[{ci_lower_pct:.2f}, {ci_upper_pct:.2f}]"
        
        summary_text.append(f"{metric.upper():<20} {mean_pct:<10.2f} {std_pct:<10.2f} {ci_str:<25} {min_pct:<10.2f} {max_pct:<10.2f} {mean_std_str:<20} {latex_str:<25} {latex_pm_str:<20} {latex_scriptscriptstyle_str:<20}")
    
    summary_text.append("=" * 160)
    summary_text.append(f"Results saved to: {main_exp_dir}")
    summary_text.append(f"- Individual fold results (decimal): fold_results.tsv")
    summary_text.append(f"- Summary statistics (formatted): summary_results.tsv")
    summary_text.append(f"- Summary table: summary_table.txt")
    
    # Save summary table to txt file
    summary_table_path = os.path.join(main_exp_dir, 'summary_table.txt')
    with open(summary_table_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_text))
    
    # Print summary to console
    print("\n" + '\n'.join(summary_text))
    
    return main_exp_dir

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set GPU device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    print(f"Using device: {device}" + (f" (GPU {args.gpu_id})" if torch.cuda.is_available() and args.gpu_id >= 0 else ""))
    
    # Auto-detect number of folds if not specified
    if args.num_folds is None:
        args.num_folds = get_num_folds_from_json(args.data_json)
        print(f"Auto-detected {args.num_folds} folds from JSON file")
    
    # Determine fold range
    start_fold = args.start_fold
    end_fold = args.end_fold if args.end_fold is not None else args.num_folds
    
    print(f"Running k-fold cross-validation with {args.num_folds} folds (folds {start_fold} to {end_fold-1})")
    
    # Get main experiment directory
    main_exp_dir = setup_experiment_folder(args)
    
    # Store results for all folds
    all_fold_metrics = []
    
    # Train each fold
    for fold_idx in range(start_fold, end_fold):
        fold_metrics = train_single_fold(args, fold_idx, device)
        fold_metrics['fold'] = fold_idx
        all_fold_metrics.append(fold_metrics)
    
    # Save k-fold summary
    summary_dir = save_kfold_summary(args, all_fold_metrics, main_exp_dir)
    print(f"\nK-fold summary saved to: {summary_dir}")

if __name__ == '__main__':
    main()
