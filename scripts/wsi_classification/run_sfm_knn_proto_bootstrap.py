# KNN evaluation for slide-level features
import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.utils import resample
# from uni.downstream.eval_patch_features.fewshot import eval_knn
from piano.utils.knn_evaluation_tools import eval_knn
from piano.datasets.oneslide_datasets import OneSlideDataset

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

def extract_features_and_labels(dataset, batch_size=32):
    """Extract features and labels from dataset"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_features = []
    all_labels = []
    
    print(f"Extracting features from {len(dataset)} samples...")
    
    for batch_idx, batch in enumerate(dataloader):
        features = batch['features']  # [batch_size, 1, feat_dim] or [batch_size, feat_dim]
        labels = batch['labels']      # [batch_size]
        
        # Squeeze the features if it's [batch_size, 1, feat_dim]
        if len(features.shape) == 3 and features.shape[1] == 1:
            features = features.squeeze(1)  # [batch_size, feat_dim]
        
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * batch_size} samples")
    
    # Concatenate all features and labels as tensors
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return features, labels

def run_bootstrap_evaluation(train_feats, train_labels, test_feats, test_labels, 
                           k=20, n_bootstrap=1000, confidence_level=0.95, random_state=None):
    """
    Run bootstrap evaluation using sklearn.utils.resample
    
    Args:
        train_feats: training features
        train_labels: training labels
        test_feats: test features  
        test_labels: test labels
        k: number of neighbors for KNN
        n_bootstrap: number of bootstrap iterations
        confidence_level: confidence level for intervals
        random_state: random state for reproducibility
    
    Returns:
        bootstrap_results: dict containing mean, std, and confidence intervals
    """
    print(f"\nRunning bootstrap evaluation with {n_bootstrap} iterations using sklearn.utils.resample...")
    
    # Convert tensors to numpy for sklearn compatibility
    test_feats_np = test_feats.numpy()
    test_labels_np = test_labels.numpy()
    
    # Store metrics for KNN
    knn_metrics_keys = [f'knn{k}_acc', f'knn{k}_bacc', f'knn{k}_kappa', f'knn{k}_weighted_f1']
    knn_metrics_values = {key: [] for key in knn_metrics_keys}
    
    # Store metrics for Proto
    proto_metrics_keys = ['proto_acc', 'proto_bacc', 'proto_kappa', 'proto_weighted_f1']
    proto_metrics_values = {key: [] for key in proto_metrics_keys}
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"Bootstrap iteration {i + 1}/{n_bootstrap}")
        
        # Generate bootstrap sample using sklearn.utils.resample
        boot_test_feats_np, boot_test_labels_np = resample(
            test_feats_np, test_labels_np,
            replace=True,
            n_samples=len(test_feats_np),
            random_state=random_state + i if random_state is not None else None
        )
        
        # Convert back to tensors
        boot_test_feats = torch.from_numpy(boot_test_feats_np)
        boot_test_labels = torch.from_numpy(boot_test_labels_np)
        
        # Run KNN evaluation on bootstrap sample
        knn_metrics, _, proto_metrics, _ = eval_knn(
            train_feats=train_feats,
            train_labels=train_labels,
            test_feats=boot_test_feats,
            test_labels=boot_test_labels,
            center_feats=True,
            normalize_feats=True,
            n_neighbors=k
        )
        
        # Store KNN metrics
        for key in knn_metrics_keys:
            if key in knn_metrics:
                knn_metrics_values[key].append(knn_metrics[key])
        
        # Store Proto metrics
        for key in proto_metrics_keys:
            if key in proto_metrics:
                proto_metrics_values[key].append(proto_metrics[key])
    
    # Calculate statistics using numpy
    def compute_stats(values):
        values_array = np.array(values)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return {
            'mean': np.mean(values_array),
            'std': np.std(values_array, ddof=1),  # Use sample standard deviation
            'ci_lower': np.percentile(values_array, lower_percentile),
            'ci_upper': np.percentile(values_array, upper_percentile)
        }
    
    bootstrap_results = {
        'knn': {},
        'proto': {}
    }
    
    # Compute statistics for KNN metrics
    for key in knn_metrics_keys:
        if knn_metrics_values[key]:  # Check if we have values
            bootstrap_results['knn'][key] = compute_stats(knn_metrics_values[key])
    
    # Compute statistics for Proto metrics
    for key in proto_metrics_keys:
        if proto_metrics_values[key]:  # Check if we have values
            bootstrap_results['proto'][key] = compute_stats(proto_metrics_values[key])
    
    return bootstrap_results

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

def save_bootstrap_results_tsv(bootstrap_results, standard_knn, standard_proto, 
                              output_file="bootstrap_results.tsv"):
    """Save bootstrap results to TSV file"""
    
    # Define metric display names
    metric_names = {
        'acc': 'Accuracy',
        'bacc': 'Balanced Accuracy', 
        'kappa': 'Cohen\'s Kappa',
        'weighted_f1': 'Weighted F1'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("Method\tMetric\tStandard_Value\tBootstrap_Mean\tBootstrap_Std\t"
               "CI_Lower\tCI_Upper\tLaTeX_Subscript\tLaTeX_PM\tLaTeX_ScriptSize\n")
        
        # Write KNN results
        for key, stats in bootstrap_results['knn'].items():
            metric = key.split('_', 1)[1] if '_' in key else key
            display_name = metric_names.get(metric, metric.replace('_', ' ').title())
            
            # Get standard value
            standard_value = standard_knn.get(key, "N/A")
            if isinstance(standard_value, (int, float)):
                standard_value = f"{standard_value:.4f}"
            
            # Format LaTeX
            latex_formats = format_latex_results(stats['mean'], stats['std'])
            
            f.write(f"KNN\t{display_name}\t{standard_value}\t"
                   f"{stats['mean']:.4f}\t{stats['std']:.4f}\t"
                   f"{stats['ci_lower']:.4f}\t{stats['ci_upper']:.4f}\t"
                   f"{latex_formats['subscript']}\t{latex_formats['pm']}\t{latex_formats['scriptsize']}\n")
        
        # Write Proto results
        for key, stats in bootstrap_results['proto'].items():
            metric = key.replace('proto_', '')
            display_name = metric_names.get(metric, metric.replace('_', ' ').title())
            
            # Get standard value
            standard_value = standard_proto.get(key, "N/A")
            if isinstance(standard_value, (int, float)):
                standard_value = f"{standard_value:.4f}"
            
            # Format LaTeX
            latex_formats = format_latex_results(stats['mean'], stats['std'])
            
            f.write(f"Proto\t{display_name}\t{standard_value}\t"
                   f"{stats['mean']:.4f}\t{stats['std']:.4f}\t"
                   f"{stats['ci_lower']:.4f}\t{stats['ci_upper']:.4f}\t"
                   f"{latex_formats['subscript']}\t{latex_formats['pm']}\t{latex_formats['scriptsize']}\n")
    
    print(f"\nBootstrap results saved to: {output_file}")

def print_and_save_bootstrap_results(results, confidence_level=0.95, 
                                   output_file="summary_table.txt"):
    """Print formatted bootstrap results and save to file"""
    
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
    content_lines.append("BOOTSTRAP EVALUATION RESULTS (using sklearn.utils.resample)")
    content_lines.append("="*80)
    content_lines.append(f"Confidence Level: {confidence_level:.1%}")
    content_lines.append("")
    
    for method in ['knn', 'proto']:
        method_name = method.upper()
        content_lines.append(f"{method_name} Results:")
        content_lines.append("-" * 50)
        
        for key, stats in results[method].items():
            # Extract metric name from key
            if method == 'knn':
                metric = key.split('_', 1)[1] if '_' in key else key
            else:
                metric = key.replace('proto_', '')
            
            display_name = metric_names.get(metric, metric.replace('_', ' ').title())
            
            # Regular format
            regular_line = (f"{display_name:15}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
            content_lines.append(regular_line)
            
            # LaTeX formats
            latex_formats = format_latex_results(stats['mean'], stats['std'])
            content_lines.append(f"  LaTeX Subscript : {latex_formats['subscript']}")
            content_lines.append(f"  LaTeX PM        : {latex_formats['pm']}")
            content_lines.append(f"  LaTeX ScriptSize: {latex_formats['scriptsize']}")
            content_lines.append("")
        
        content_lines.append("")
    
    # Print to console
    for line in content_lines:
        print(line)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in content_lines:
            f.write(line + '\n')
    
    print(f"\nSummary table saved to: {output_file}")

def main(seed=42, k=20, data_json=None, pfm_name=None, n_bootstrap=1000, 
         confidence_level=0.95, output_dir="."):
    """Main evaluation function"""
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = OneSlideDataset(data_json, mode='train', pfm_name=pfm_name)
    test_dataset = OneSlideDataset(data_json, mode='test', pfm_name=pfm_name)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.get_classes()}")
    
    # Extract features and labels
    train_feats, train_labels = extract_features_and_labels(train_dataset)
    test_feats, test_labels = extract_features_and_labels(test_dataset)
    
    # Run standard evaluation
    print("\nRunning standard KNN evaluation...")
    knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
        train_feats=train_feats,
        train_labels=train_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        center_feats=True,
        normalize_feats=True,
        n_neighbors=k
    )
    
    print(f"\nStandard KNN Results: {knn_eval_metrics}")
    print(f"Standard Proto Results: {proto_eval_metrics}")
    
    # Run bootstrap evaluation if requested
    if n_bootstrap > 0:
        bootstrap_results = run_bootstrap_evaluation(
            train_feats, train_labels, test_feats, test_labels,
            k=k, n_bootstrap=n_bootstrap, confidence_level=confidence_level,
            random_state=seed
        )
        
        # Print and save results
        tsv_file = os.path.join(output_dir, "bootstrap_results.tsv")
        summary_file = os.path.join(output_dir, "summary_table.txt")
        
        print_and_save_bootstrap_results(bootstrap_results, confidence_level, summary_file)
        save_bootstrap_results_tsv(bootstrap_results, knn_eval_metrics, proto_eval_metrics, tsv_file)
        
        return knn_eval_metrics, proto_eval_metrics, bootstrap_results
    
    return knn_eval_metrics, proto_eval_metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='KNN evaluation for slide-level features')
    parser.add_argument('--seed', type=int, default=50, help='Random seed for reproducibility')
    parser.add_argument('--data_json', type=str, 
                       default=None,
                       help='Path to dataset JSON file')
    parser.add_argument('--pfm_name', type=str, default=None, help='PFM model name')
    parser.add_argument('--k', type=int, default=20, help='Number of neighbors for KNN')
    parser.add_argument('--n_bootstrap', type=int, default=1000, help='Number of bootstrap iterations (0 to disable)')
    parser.add_argument('--confidence_level', type=float, default=0.95, help='Confidence level for bootstrap intervals')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    dataset_name = args.data_json.split('/')[-1].split('.json')[0]
    args.output_dir = os.path.join(args.output_dir, f'{dataset_name}_{args.pfm_name}')
    main(seed=args.seed, k=args.k, data_json=args.data_json, pfm_name=args.pfm_name,
         n_bootstrap=args.n_bootstrap, confidence_level=args.confidence_level, output_dir=args.output_dir)

