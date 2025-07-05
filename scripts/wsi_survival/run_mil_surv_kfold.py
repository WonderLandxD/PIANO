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
from torch.amp import GradScaler
import torch.optim as optim
import json
import scipy.stats as stats
from lifelines import KaplanMeierFitter

from piano import get_model_output_dim
from piano.datasets.wsi_datasets import WSIDatasetSurvKFold

from piano.model.mil_factory import create_mil_model

from piano.utils.wsi_finetune_tools import train_wsi, predict_wsi_surv, wsi_create_ckpt
from piano.utils.evaluation_metrics import survival_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    # Data related arguments
    parser.add_argument('--data_json', type=str, required=True, help='Path to data JSON file with k-fold survival structure')
    parser.add_argument('--few_shot', type=int, default=None, help='Number of samples per class for few-shot learning')
    
    # Model related arguments
    parser.add_argument('--pfm_name', type=str, default='conch_v1', help='Pathology foundation model name')
    parser.add_argument('--mil_name', type=str, default='abmil', help='MIL model name')
    
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
    parser.add_argument('--save_metric', type=str, default='c_index',
                      choices=['c_index'],
                      help='Metric to use for saving best model')
    parser.add_argument('--save_interval', type=int, default=50,
                      help='Save model checkpoint every N epochs')
    parser.add_argument('--save_dir', type=str, default='/mnt/sdb/ljw/PIANO-Update/PIANO_THU/RESULTS/survival_prediction/mil_cross_val/',
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
    main_exp_name = f"SurvKfold_{args.pfm_name}_{args.mil_name}"
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
            pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'c_index']).to_csv(
                log_file, index=False, sep='\t', float_format='%.6f')
        
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

def plot_survival_predictions(risk_scores, event_times, censorships, save_path):
    """Plot and save Kaplan-Meier survival curves based on risk stratification"""
    # Convert event_times from days to months for plotting
    event_times_months = event_times / 30.44  # Convert days to months (average days per month)
    
    # Create event indicator (True for events, False for censored)
    events = (censorships == 1)
    
    # Stratify patients into high-risk and low-risk groups based on median risk score
    median_risk = np.median(risk_scores)
    high_risk_mask = risk_scores >= median_risk
    low_risk_mask = risk_scores < median_risk
    
    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Fit Kaplan-Meier curves for each risk group
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    # Fit high-risk group
    kmf_high.fit(
        durations=event_times_months[high_risk_mask], 
        event_observed=events[high_risk_mask], 
        label=f'High Risk (n={np.sum(high_risk_mask)})'
    )
    
    # Fit low-risk group  
    kmf_low.fit(
        durations=event_times_months[low_risk_mask], 
        event_observed=events[low_risk_mask], 
        label=f'Low Risk (n={np.sum(low_risk_mask)})'
    )
    
    # Plot survival curves with colors similar to the user's image
    kmf_high.plot_survival_function(ax=ax, color='red', linewidth=2.5)
    kmf_low.plot_survival_function(ax=ax, color='blue', linewidth=2.5)
    
    # Customize the plot to match the user's style
    ax.set_xlim(0, 100)
    ax.set_ylim(0.00, 1.00)
    ax.set_xlabel('Time (months)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Survival Probability', fontsize=16, fontweight='bold')
    ax.set_title('Prognosis evaluation', fontsize=20, fontweight='bold', pad=20)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Position legend in upper right
    ax.legend(loc='upper right', fontsize=12)
    
    # Make the plot area have rounded corners like the user's image
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print some statistics
    print(f"High-risk group: {np.sum(high_risk_mask)} patients, {np.sum(events[high_risk_mask])} events")
    print(f"Low-risk group: {np.sum(low_risk_mask)} patients, {np.sum(events[low_risk_mask])} events")

def save_survival_predictions(risk_scores, event_times, censorships, val_dataset, exp_dir):
    """Save detailed survival predictions for validation set"""
    
    # Create predictions data
    predictions_data = []
    for i in range(len(event_times)):
        slide_id = val_dataset.get_name(i)
        true_time = event_times[i]
        event_status = censorships[i]
        pred_risk = risk_scores[i]
        
        predictions_data.append({
            'slide_id': slide_id,
            'true_survival_time': true_time,
            'event_status': event_status,
            'predicted_risk_score': pred_risk,
            'is_event': event_status == 1
        })
    
    # Save to TSV
    predictions_df = pd.DataFrame(predictions_data)
    predictions_path = os.path.join(exp_dir, 'survival_predictions.tsv')
    predictions_df.to_csv(predictions_path, index=False, sep='\t', float_format='%.6f')
    
    return predictions_df

def train_single_fold(args, fold_idx, device):
    """Train a single fold and return validation metrics"""
    print(f"\n=== Training Survival Fold {fold_idx} ===")
    
    # Setup fold-specific experiment folder
    exp_dir, log_file, main_exp_dir = setup_experiment_folder(args, fold_idx)
    
    # Load datasets for this fold
    print(f"Loading survival datasets for fold {fold_idx}...")
    train_dataset = WSIDatasetSurvKFold(
        args.data_json, 
        fold_name=fold_idx, 
        mode='train', 
        pfm_name=args.pfm_name, 
        few_shot=args.few_shot
    )
    val_dataset = WSIDatasetSurvKFold(
        args.data_json, 
        fold_name=fold_idx, 
        mode='valid', 
        pfm_name=args.pfm_name
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Get survival info
    survival_info = train_dataset.get_survival_info()
    print(f"Survival info: {survival_info}")
    
    # Create model
    print(f"Creating \033[95m{args.mil_name}\033[0m survival model under \033[91m{args.pfm_name}\033[0m...")
    feat_dim = get_model_output_dim(args.pfm_name)
    
    # Use discretized survival classes
    num_classes = survival_info['num_classes']
    
    model = create_mil_model(
        mil_name=args.mil_name, 
        dim_in=feat_dim, 
        num_classes=num_classes,
        survival=True
    ).to(device)
    
    # Add training_mode attribute to model
    model.training_mode = f"{args.mil_name}_survival"
    
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
    
    print(f"Starting survival training for fold {fold_idx} with early stopping patience: {args.early_stopping}...")
    for epoch in range(args.num_epochs):
        # Training phase
        train_loss = train_wsi(model, train_loader, optimizer, scaler, device, epoch, 
                             use_amp=use_amp, amp_dtype=args.amp_dtype)
        
        # Validation phase
        all_censorships, all_event_times, all_risk_scores, val_loss = predict_wsi_surv(model, val_loader, device, epoch)
        val_metrics = survival_metrics(all_censorships, all_event_times, all_risk_scores)
        
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
            best_censorships = all_censorships.copy()
            best_event_times = all_event_times.copy()
            best_risk_scores = all_risk_scores.copy()
            epochs_without_improvement = 0
            
            # Save best model
            checkpoint = wsi_create_ckpt(model, model.training_mode, epoch)
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
            checkpoint = wsi_create_ckpt(model, model.training_mode, epoch)
            save_name = f"{args.pfm_name}_epoch{epoch+1}.pth"
            save_path = os.path.join(exp_dir, save_name)
            torch.save(checkpoint, save_path)
            # Display the message in dark blue using ANSI escape codes
            print(f"\033[34mFold {fold_idx}: Saved intermediate checkpoint: {save_name}\033[0m")
    
    # After training, save survival plots and predictions for best model
    plot_path = os.path.join(exp_dir, f'survival_predictions_fold{fold_idx}.png')
    plot_survival_predictions(best_risk_scores, best_event_times, best_censorships, plot_path)
    
    # Save detailed predictions
    save_survival_predictions(best_risk_scores, best_event_times, best_censorships, val_dataset, exp_dir)
    
    print(f"Fold {fold_idx} completed. Best {args.save_metric}: {best_metric:.4f}")
    return best_val_metrics

def save_kfold_summary(args, all_fold_metrics, main_exp_dir):
    """Save k-fold cross-validation summary with detailed statistics"""
    # Create DataFrame from all fold metrics
    metrics_df = pd.DataFrame(all_fold_metrics)
    
    # Calculate detailed statistics for each metric
    summary_data = []
    metrics_to_analyze = ['c_index']
    
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
    summary_text.append("SURVIVAL K-FOLD CROSS-VALIDATION SUMMARY")
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
    
    print(f"Running survival k-fold cross-validation with {args.num_folds} folds (folds {start_fold} to {end_fold-1})")
    
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
    print(f"\nSurvival k-fold summary saved to: {summary_dir}")

if __name__ == '__main__':
    main()
