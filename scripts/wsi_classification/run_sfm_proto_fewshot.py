# KNN evaluation for slide-level features
import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader
# from uni.downstream.eval_patch_features.fewshot import eval_knn
from piano.utils.knn_evaluation_tools import eval_fewshot
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

def save_fewshot_results_tsv(fewshot_episodes, fewshot_dump, output_file="fewshot_results.tsv"):
    """Save few-shot results to TSV file"""
    
    # Define metric display names
    metric_names = {
        'acc': 'Accuracy',
        'bacc': 'Balanced Accuracy', 
        'kappa': 'Cohen\'s Kappa',
        'weighted_f1': 'Weighted F1'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("Metric\tMean\tStd\tLaTeX_Subscript\tLaTeX_PM\tLaTeX_ScriptSize\n")
        
        # Write results from fewshot_dump
        for key, value in fewshot_dump.items():
            if isinstance(value, dict) and 'mean' in value and 'std' in value:
                display_name = metric_names.get(key, key.replace('_', ' ').title())
                
                # Format LaTeX
                latex_formats = format_latex_results(value['mean'], value['std'])
                
                f.write(f"{display_name}\t{value['mean']:.4f}\t{value['std']:.4f}\t"
                       f"{latex_formats['subscript']}\t{latex_formats['pm']}\t{latex_formats['scriptsize']}\n")
    
    print(f"\nFew-shot results saved to: {output_file}")

def print_and_save_fewshot_results(fewshot_episodes, fewshot_dump, output_file="fewshot_summary.txt"):
    """Print formatted few-shot results and save to file"""
    
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
    content_lines.append("FEW-SHOT EVALUATION RESULTS")
    content_lines.append("="*80)
    content_lines.append("")
    
    # Print episode-level results
    content_lines.append("Episode-level Results:")
    content_lines.append("-" * 50)
    if isinstance(fewshot_episodes, dict):
        for key, value in fewshot_episodes.items():
            display_name = metric_names.get(key, key.replace('_', ' ').title())
            content_lines.append(f"{display_name}: {value}")
    else:
        content_lines.append(f"Episodes data: {fewshot_episodes}")
    content_lines.append("")
    
    # Print summary results
    content_lines.append("Summary Results:")
    content_lines.append("-" * 50)
    for key, value in fewshot_dump.items():
        if isinstance(value, dict) and 'mean' in value and 'std' in value:
            display_name = metric_names.get(key, key.replace('_', ' ').title())
            
            # Regular format
            regular_line = f"{display_name:15}: {value['mean']:.4f} ± {value['std']:.4f}"
            content_lines.append(regular_line)
            
            # LaTeX formats
            latex_formats = format_latex_results(value['mean'], value['std'])
            content_lines.append(f"  LaTeX Subscript : {latex_formats['subscript']}")
            content_lines.append(f"  LaTeX PM        : {latex_formats['pm']}")
            content_lines.append(f"  LaTeX ScriptSize: {latex_formats['scriptsize']}")
            content_lines.append("")
        else:
            display_name = key.replace('_', ' ').title()
            content_lines.append(f"{display_name}: {value}")
    
    # Print to console
    for line in content_lines:
        print(line)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in content_lines:
            f.write(line + '\n')
    
    print(f"\nSummary table saved to: {output_file}")

def main(seed=42, data_json=None, pfm_name=None, n_iter=100, n_way=None, n_shot=16, 
         n_query=None, center_feats=True, normalize_feats=True, average_feats=True, 
         output_dir="."):
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
    
    # Determine n_way (number of classes) if not provided
    if n_way is None:
        n_way = len(train_dataset.get_classes())
        print(f"Using n_way={n_way} (all classes)")
    
    # Determine n_query (number of test samples) if not provided
    if n_query is None:
        n_query = test_feats.shape[0]
        print(f"Using n_query={n_query} (all test samples)")
    
    # Run few-shot evaluation
    print(f"\nRunning few-shot evaluation with {n_iter} iterations...")
    print(f"Parameters: n_way={n_way}, n_shot={n_shot}, n_query={n_query}")
    
    fewshot_episodes, fewshot_dump = eval_fewshot(
        train_feats=train_feats,
        train_labels=train_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        n_iter=n_iter,  # draw n_iter few-shot episodes
        n_way=n_way,    # use all class examples
        n_shot=n_shot,  # n_shot examples per class
        n_query=n_query, # evaluate on all test samples
        center_feats=center_feats,
        normalize_feats=normalize_feats,
        average_feats=average_feats,
    )
    
    print(f"\nFew-shot Episodes: {fewshot_episodes}")
    print(f"Few-shot Summary: {fewshot_dump}")
    
    # Save results
    tsv_file = os.path.join(output_dir, "fewshot_results.tsv")
    summary_file = os.path.join(output_dir, "fewshot_summary.txt")
    
    print_and_save_fewshot_results(fewshot_episodes, fewshot_dump, summary_file)
    save_fewshot_results_tsv(fewshot_episodes, fewshot_dump, tsv_file)
    
    return fewshot_episodes, fewshot_dump

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Few-shot evaluation for slide-level features')
    parser.add_argument('--seed', type=int, default=50, help='Random seed for reproducibility')
    parser.add_argument('--data_json', type=str, 
                       default="/mnt/sdb/ljw/PIANO-Update/PIANO_THU/WSI_DATA/slide_feature_datasets/clinical_subtyping_gradining/bracs/bracs_3classes.json",
                       help='Path to dataset JSON file')
    parser.add_argument('--pfm_name', type=str, default=None, help='PFM model name')
    parser.add_argument('--n_iter', type=int, default=100, help='Number of few-shot episodes')
    parser.add_argument('--n_way', type=int, default=None, help='Number of classes (None for all classes)')
    parser.add_argument('--n_shot', type=int, default=16, help='Number of examples per class')
    parser.add_argument('--n_query', type=int, default=None, help='Number of query samples (None for all test samples)')
    parser.add_argument('--center_feats', action='store_true', default=True, help='Center features')
    parser.add_argument('--normalize_feats', action='store_true', default=True, help='Normalize features')
    parser.add_argument('--average_feats', action='store_true', default=True, help='Average features')
    parser.add_argument('--output_dir', type=str, default='/mnt/sdb/ljw/PIANO-Update/PIANO_THU/RESULTS/slidefeats_fewshot_classification/', help='Output directory for results')
    
    args = parser.parse_args()
    dataset_name = args.data_json.split('/')[-1].split('.json')[0]
    args.output_dir = os.path.join(args.output_dir, f'{dataset_name}_{args.pfm_name}')
    
    main(seed=args.seed, data_json=args.data_json, pfm_name=args.pfm_name,
         n_iter=args.n_iter, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query,
         center_feats=args.center_feats, normalize_feats=args.normalize_feats, 
         average_feats=args.average_feats, output_dir=args.output_dir)
