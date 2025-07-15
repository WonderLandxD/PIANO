# KNN evaluation for slide-level features with K-Fold Cross Validation
import torch
import random
import numpy as np
from tqdm import tqdm
import os
import json
from torch.utils.data import DataLoader
# from uni.downstream.eval_patch_features.fewshot import eval_knn
from piano.utils.knn_evaluation_tools import eval_knn
from piano.datasets.oneslide_datasets import OneSlideDatasetKFold

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def get_device(device=None):
    """Get the appropriate device for computation"""
    device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device('cpu')

    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device

def extract_features_and_labels(dataset, batch_size=32, device='auto'):
    """Extract features and labels from dataset with GPU support"""
    device = get_device(device) if isinstance(device, str) else device
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True if device.type == 'cuda' else False)
    
    all_features = []
    all_labels = []
    
    print(f"Extracting features from {len(dataset)} samples using {device}...")
    
    with torch.no_grad():  # Disable gradient computation for faster inference
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", total=len(dataloader), ncols=100)):
            features = batch['features']  # [batch_size, 1, feat_dim] or [batch_size, feat_dim]
            labels = batch['labels']      # [batch_size]
            
            # Move to device for processing
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Squeeze the features if it's [batch_size, 1, feat_dim]
            if len(features.shape) == 3 and features.shape[1] == 1:
                features = features.squeeze(1)  # [batch_size, feat_dim]
            
            # Move back to CPU for storage
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            
            # if (batch_idx + 1) % 10 == 0:
            #     print(f"Processed {(batch_idx + 1) * batch_size} samples")
    
    # Concatenate all features and labels as tensors
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return features, labels

def get_available_folds(json_path):
    """Get available fold numbers from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        # Find fold keys
        fold_keys = [key for key in data_dict.keys() if key.startswith('fold_')]
        fold_numbers = [int(key.split('_')[1]) for key in fold_keys]
        return sorted(fold_numbers)
    
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return []

def run_kfold_evaluation(data_json, pfm_name, k=20, seed=42, device='auto'):
    """
    Run k-fold cross validation evaluation
    
    Args:
        data_json: path to JSON file with k-fold structure
        pfm_name: PFM model name
        k: number of neighbors for KNN
        seed: random seed
        device: device to use for computation ('auto', 'cuda', 'cpu', or torch.device)
    
    Returns:
        fold_results: results for each fold
        summary_stats: mean and std across folds
    """
    
    # Get device
    device = get_device(device) if isinstance(device, str) else device
    
    # Get available folds
    fold_numbers = get_available_folds(data_json)
    if not fold_numbers:
        raise ValueError("No valid folds found in JSON file")
    
    print(f"Found {len(fold_numbers)} folds: {fold_numbers}")
    
    # Store results for each fold
    fold_results = {}
    
    # Define metric keys
    knn_metric_keys = [f'knn{k}_acc', f'knn{k}_bacc', f'knn{k}_kappa', f'knn{k}_weighted_f1']
    proto_metric_keys = ['proto_acc', 'proto_bacc', 'proto_kappa', 'proto_weighted_f1']
    all_metric_keys = knn_metric_keys + proto_metric_keys
    
    # Initialize metric storage
    all_metrics = {key: [] for key in all_metric_keys}
    
    for fold_num in fold_numbers:
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_num}")
        print(f"{'='*60}")
        
        # Create datasets for this fold
        train_dataset = OneSlideDatasetKFold(data_json, fold_num, mode='train', pfm_name=pfm_name)
        valid_dataset = OneSlideDatasetKFold(data_json, fold_num, mode='valid', pfm_name=pfm_name)
        
        print(f"Fold {fold_num} - Train samples: {len(train_dataset)}")
        print(f"Fold {fold_num} - Valid samples: {len(valid_dataset)}")
        print(f"Fold {fold_num} - Classes: {train_dataset.get_classes()}")
        
        # Extract features and labels with GPU support
        train_feats, train_labels = extract_features_and_labels(train_dataset, device=device, batch_size=1)
        valid_feats, valid_labels = extract_features_and_labels(valid_dataset, device=device, batch_size=1)
        
        # Run evaluation
        print(f"\nRunning KNN evaluation for Fold {fold_num}...")
        knn_metrics, _, proto_metrics, _ = eval_knn(
            train_feats=train_feats,
            train_labels=train_labels,
            test_feats=valid_feats,
            test_labels=valid_labels,
            center_feats=True,
            normalize_feats=True,
            n_neighbors=k
        )
        
        # Store fold results
        fold_results[f'fold_{fold_num}'] = {
            'knn': knn_metrics,
            'proto': proto_metrics
        }
        
        # Collect metrics for summary statistics
        for key in knn_metric_keys:
            if key in knn_metrics:
                all_metrics[key].append(knn_metrics[key])
        
        for key in proto_metric_keys:
            if key in proto_metrics:
                all_metrics[key].append(proto_metrics[key])
        
        print(f"Fold {fold_num} KNN Results: {knn_metrics}")
        print(f"Fold {fold_num} Proto Results: {proto_metrics}")
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(all_metrics)
    
    return fold_results, summary_stats

def calculate_summary_statistics(all_metrics):
    """Calculate mean and standard deviation across folds"""
    summary_stats = {}
    
    for key, values in all_metrics.items():
        if values:  # Check if we have values
            values_array = np.array(values)
            summary_stats[key] = {
                'mean': np.mean(values_array),
                'std': np.std(values_array, ddof=1) if len(values_array) > 1 else 0.0,
                'values': values,
                'n_folds': len(values)
            }
    
    return summary_stats

def format_latex_results(mean, std):
    """Format results in three different LaTeX styles"""
    mean_str = f"{mean*100:.2f}"  # Convert to percentage
    std_str = f"{std*100:.2f}"
    
    formats = {
        'subscript': f"${{{mean_str}}}_{{{std_str}}}$",
        'pm': f"${mean_str}$±${std_str}$",
        'scriptsize': f"{mean_str}$\\scriptscriptstyle{{({std_str})}}$"
    }
    return formats

def save_kfold_results_tsv(fold_results, summary_stats, output_file="kfold_results.tsv"):
    """Save k-fold results to TSV file"""
    
    # Define metric display names
    metric_names = {
        'acc': 'Accuracy',
        'bacc': 'Balanced Accuracy', 
        'kappa': 'Cohen\'s Kappa',
        'weighted_f1': 'Weighted F1'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("Method\tMetric\tMean\tStd\tN_Folds\t")
        
        # Add fold columns
        fold_names = sorted(fold_results.keys())
        for fold_name in fold_names:
            f.write(f"{fold_name}\t")
        
        f.write("LaTeX_Subscript\tLaTeX_PM\tLaTeX_ScriptSize\n")
        
        # Write KNN results
        for key, stats in summary_stats.items():
            if key.startswith('knn'):
                metric = key.split('_', 1)[1] if '_' in key else key
                display_name = metric_names.get(metric, metric.replace('_', ' ').title())
                
                # Format LaTeX
                latex_formats = format_latex_results(stats['mean'], stats['std'])
                
                f.write(f"KNN\t{display_name}\t{stats['mean']:.4f}\t{stats['std']:.4f}\t{stats['n_folds']}\t")
                
                # Add fold values
                for fold_name in fold_names:
                    fold_value = fold_results[fold_name]['knn'].get(key, "N/A")
                    if isinstance(fold_value, (int, float)):
                        fold_value = f"{fold_value:.4f}"
                    f.write(f"{fold_value}\t")
                
                f.write(f"{latex_formats['subscript']}\t{latex_formats['pm']}\t{latex_formats['scriptsize']}\n")
        
        # Write Proto results
        for key, stats in summary_stats.items():
            if key.startswith('proto'):
                metric = key.replace('proto_', '')
                display_name = metric_names.get(metric, metric.replace('_', ' ').title())
                
                # Format LaTeX
                latex_formats = format_latex_results(stats['mean'], stats['std'])
                
                f.write(f"Proto\t{display_name}\t{stats['mean']:.4f}\t{stats['std']:.4f}\t{stats['n_folds']}\t")
                
                # Add fold values
                for fold_name in fold_names:
                    fold_value = fold_results[fold_name]['proto'].get(key, "N/A")
                    if isinstance(fold_value, (int, float)):
                        fold_value = f"{fold_value:.4f}"
                    f.write(f"{fold_value}\t")
                
                f.write(f"{latex_formats['subscript']}\t{latex_formats['pm']}\t{latex_formats['scriptsize']}\n")
    
    print(f"\nK-Fold results saved to: {output_file}")

def print_and_save_kfold_results(fold_results, summary_stats, output_file="summary_table.txt"):
    """Print formatted k-fold results and save to file"""
    
    # Define metric display names
    metric_names = {
        'acc': 'Accuracy',
        'bacc': 'Balanced Accuracy', 
        'kappa': 'Cohen\'s Kappa',
        'weighted_f1': 'Weighted F1'
    }
    
    # Prepare output content
    content_lines = []
    content_lines.append("="*80)
    content_lines.append("K-FOLD CROSS VALIDATION RESULTS")
    content_lines.append("="*80)
    content_lines.append(f"Number of Folds: {len(fold_results)}")
    content_lines.append("")
    
    # Print summary statistics
    content_lines.append("SUMMARY STATISTICS (Mean ± Std across folds)")
    content_lines.append("-" * 60)
    
    for method in ['knn', 'proto']:
        method_name = method.upper()
        content_lines.append(f"\n{method_name} Results:")
        content_lines.append("-" * 40)
        
        for key, stats in summary_stats.items():
            if (method == 'knn' and key.startswith('knn')) or (method == 'proto' and key.startswith('proto')):
                # Extract metric name from key
                if method == 'knn':
                    metric = key.split('_', 1)[1] if '_' in key else key
                else:
                    metric = key.replace('proto_', '')
                
                display_name = metric_names.get(metric, metric.replace('_', ' ').title())
                
                # Regular format
                regular_line = (f"{display_name:15}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                              f"(n={stats['n_folds']})")
                content_lines.append(regular_line)
                
                # LaTeX formats
                latex_formats = format_latex_results(stats['mean'], stats['std'])
                content_lines.append(f"  LaTeX Subscript : {latex_formats['subscript']}")
                content_lines.append(f"  LaTeX PM        : {latex_formats['pm']}")
                content_lines.append(f"  LaTeX ScriptSize: {latex_formats['scriptsize']}")
                content_lines.append("")
    
    # Print individual fold results
    content_lines.append("\n" + "="*80)
    content_lines.append("INDIVIDUAL FOLD RESULTS")
    content_lines.append("="*80)
    
    for fold_name in sorted(fold_results.keys()):
        content_lines.append(f"\n{fold_name.upper()}:")
        content_lines.append("-" * 30)
        
        fold_data = fold_results[fold_name]
        
        # KNN results for this fold
        content_lines.append("KNN:")
        for key, value in fold_data['knn'].items():
            if isinstance(value, (int, float)):
                content_lines.append(f"  {key}: {value:.4f}")
        
        # Proto results for this fold
        content_lines.append("Proto:")
        for key, value in fold_data['proto'].items():
            if isinstance(value, (int, float)):
                content_lines.append(f"  {key}: {value:.4f}")
    
    # Print to console
    for line in content_lines:
        print(line)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in content_lines:
            f.write(line + '\n')
    
    print(f"\nSummary table saved to: {output_file}")

def main(seed=42, k=20, data_json=None, pfm_name=None, output_dir=".", device='auto'):
    """Main evaluation function"""
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Get and print device info
    device = get_device(device)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("K-FOLD CROSS VALIDATION KNN EVALUATION")
    print("="*80)
    print(f"Dataset: {data_json}")
    print(f"PFM Name: {pfm_name}")
    print(f"K (neighbors): {k}")
    print(f"Random Seed: {seed}")
    print(f"Device: {device}")
    print(f"Output Directory: {output_dir}")
    
    # Run k-fold evaluation
    fold_results, summary_stats = run_kfold_evaluation(
        data_json=data_json,
        pfm_name=pfm_name,
        k=k,
        seed=seed,
        device=device
    )
    
    # Save results
    tsv_file = os.path.join(output_dir, "kfold_results.tsv")
    summary_file = os.path.join(output_dir, "summary_table.txt")
    
    print_and_save_kfold_results(fold_results, summary_stats, summary_file)
    save_kfold_results_tsv(fold_results, summary_stats, tsv_file)
    
    return fold_results, summary_stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='K-Fold KNN evaluation for slide-level features')
    parser.add_argument('--seed', type=int, default=50, help='Random seed for reproducibility')
    parser.add_argument('--data_json', type=str, default=None, help='Path to dataset JSON file with k-fold structure')
    parser.add_argument('--pfm_name', type=str, default=None, help='PFM model name')
    parser.add_argument('--k', type=int, default=20, help='Number of neighbors for KNN')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use for computation: auto, cuda, cpu, or specific GPU like cuda:0')
    parser.add_argument('--output_dir', type=str, 
                       default=None, 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory name based on dataset and PFM
    dataset_name = args.data_json.split('/')[-1].split('.json')[0]
    args.output_dir = os.path.join(args.output_dir, f'{dataset_name}_{args.pfm_name}')
    
    main(seed=args.seed, k=args.k, data_json=args.data_json, pfm_name=args.pfm_name, 
         output_dir=args.output_dir, device=args.device) 