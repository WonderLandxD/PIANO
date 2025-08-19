import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import torch
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import torch.nn.functional as F

from piano.datasets.oneslide_datasets import OneSlideDatasetKFold

def parse_args():
    parser = argparse.ArgumentParser(description='Image-to-Image Retrieval using Pre-computed Slide Features')
    
    # Data related arguments
    parser.add_argument('--data_json', type=str, required=True, 
                       help='Path to data JSON file with k-fold structure')
    parser.add_argument('--fold_idx', type=int, default=0, 
                       help='Fold index to use for train/test split')
                       # K-fold related arguments
    parser.add_argument('--run_all_folds', action='store_true',
                       help='Run evaluation on all folds and compute summary statistics')
    
    # Feature related arguments
    parser.add_argument('--pfm_name', type=str, default=None, 
                       help='Name of the pre-computed slide-level features to use from dataset')
    
    # Retrieval related arguments
    parser.add_argument('--k_values', type=int, nargs='+', default=list(range(1, 21)), 
                       help='K values for Recall@K and mAP@K evaluation (default: 1-20)')
    parser.add_argument('--similarity_metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'dot_product'],
                       help='Similarity metric for retrieval')
    
    # Processing related arguments
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for feature loading')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output related arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save retrieval results')
    
    
    
    return parser.parse_args()

def set_seed(seed):
    """Set all random seeds and deterministic flags"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_num_folds_from_json(data_json_path):
    """Detect number of folds from JSON file"""
    with open(data_json_path, 'r') as f:
        data = json.load(f)
    
    # Find fold keys (e.g., fold_0, fold_1, etc.)
    fold_keys = [key for key in data.keys() if key.startswith('fold_')]
    return len(fold_keys)

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

def setup_output_folder(args, fold_idx=None):
    """Setup output folder and return path"""
    # Get dataset name from json path
    dataset_name = os.path.splitext(os.path.basename(args.data_json))[0]
    
    # Create experiment folder name
    if args.run_all_folds:
        exp_name = f"i2i_retrieval_kfold_{dataset_name}_{args.pfm_name}_seed{args.seed}"
    else:
        fold_idx = fold_idx if fold_idx is not None else args.fold_idx
        exp_name = f"i2i_retrieval_{dataset_name}_fold{fold_idx}_{args.pfm_name}_seed{args.seed}"
    
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir

def extract_slide_features(dataloader):
    """Extract slide-level features directly from dataset (features are already pre-computed)"""
    features_list = []
    labels_list = []
    slide_ids_list = []
    
    print("Loading pre-computed slide-level features...")
    with torch.no_grad():  # Disable gradient computation for faster processing
        for batch in tqdm(dataloader, desc="Loading features"):
            # Features are already slide-level (pre-computed)
            features = batch['features']  # [batch_size, 1, feat_dim] or [batch_size, feat_dim]
            labels = batch['labels']      # [batch_size]
            slide_ids = batch['slide_id'] # [batch_size]
            
            # Squeeze the features if it's [batch_size, 1, feat_dim]
            if len(features.shape) == 3 and features.shape[1] == 1:
                features = features.squeeze(1)  # [batch_size, feat_dim]
            
            # Store results
            features_list.append(features.cpu())
            labels_list.append(labels.cpu())
            slide_ids_list.extend(slide_ids)
    
    # Concatenate all features and labels
    all_features = torch.cat(features_list, dim=0)  # [N_total, feature_dim]
    all_labels = torch.cat(labels_list, dim=0)  # [N_total]
    
    return all_features, all_labels, slide_ids_list

def compute_similarity(query_features, gallery_features, metric='cosine'):
    """Compute similarity between query and gallery features"""
    if metric == 'cosine':
        # Normalize features for cosine similarity
        query_norm = F.normalize(query_features, p=2, dim=1)
        gallery_norm = F.normalize(gallery_features, p=2, dim=1)
        similarity = torch.mm(query_norm, gallery_norm.t())
    elif metric == 'euclidean':
        # Compute negative euclidean distance (higher is more similar)
        query_expanded = query_features.unsqueeze(1)  # [N_query, 1, feature_dim]
        gallery_expanded = gallery_features.unsqueeze(0)  # [1, N_gallery, feature_dim]
        distances = torch.norm(query_expanded - gallery_expanded, dim=2, p=2)
        similarity = -distances  # Negative distance for similarity
    elif metric == 'dot_product':
        similarity = torch.mm(query_features, gallery_features.t())
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    
    return similarity

def compute_recall_at_k(similarity_matrix, query_labels, gallery_labels, k_values):
    """Compute Recall@K for image-to-image retrieval"""
    n_queries = similarity_matrix.size(0)
    results = {}
    
    # Get top-k similar images for each query
    _, top_k_indices = torch.topk(similarity_matrix, k=max(k_values), dim=1, largest=True)
    
    for k in k_values:
        correct = 0
        for i in range(n_queries):
            query_label = query_labels[i]
            top_k_labels = gallery_labels[top_k_indices[i, :k]]
            
            # Check if any of the top-k retrieved images has the same label
            if torch.any(top_k_labels == query_label):
                correct += 1
        
        recall_k = correct / n_queries
        results[f'Recall@{k}'] = recall_k
        print(f"Recall@{k}: {recall_k:.4f} ({correct}/{n_queries})")
    
    return results

def compute_map_at_k(similarity_matrix, query_labels, gallery_labels, k_values):
    """Compute mAP@K (mean Average Precision at K) for image-to-image retrieval"""
    n_queries = similarity_matrix.size(0)
    results = {}
    
    # Get top-k similar images for each query
    _, top_k_indices = torch.topk(similarity_matrix, k=max(k_values), dim=1, largest=True)
    
    for k in k_values:
        ap_scores = []
        
        for i in range(n_queries):
            query_label = query_labels[i]
            top_k_labels = gallery_labels[top_k_indices[i, :k]]
            
            # Calculate Average Precision for this query
            relevant_items = (top_k_labels == query_label).float()
            
            if relevant_items.sum() == 0:
                # No relevant items found, AP = 0
                ap_scores.append(0.0)
            else:
                # Calculate precision at each relevant position
                precisions = []
                for j in range(k):
                    if relevant_items[j] == 1:  # If current item is relevant
                        # Precision@(j+1) = number of relevant items in top (j+1) / (j+1)
                        precision_at_j = relevant_items[:j+1].sum() / (j + 1)
                        precisions.append(precision_at_j.item())
                
                # Average Precision is the mean of precisions at relevant positions
                if len(precisions) > 0:
                    ap = sum(precisions) / len(precisions)
                else:
                    ap = 0.0
                ap_scores.append(ap)
        
        map_k = sum(ap_scores) / len(ap_scores)
        results[f'mAP@{k}'] = map_k
        print(f"mAP@{k}: {map_k:.4f}")
    
    return results

def save_retrieval_results(results, query_ids, gallery_ids, similarity_matrix, output_dir, k_values):
    """Save detailed retrieval results including Recall@K and mAP@K metrics"""
    # Save overall metrics (both Recall@K and mAP@K)
    results_df = pd.DataFrame([results])
    results_path = os.path.join(output_dir, 'retrieval_metrics.tsv')
    results_df.to_csv(results_path, index=False, sep='\t', float_format='%.6f')
    
    # Save detailed retrieval for each query
    _, top_k_indices = torch.topk(similarity_matrix, k=max(k_values), dim=1, largest=True)
    top_k_similarities = torch.gather(similarity_matrix, 1, top_k_indices)
    
    detailed_results = []
    for i, query_id in enumerate(query_ids):
        for rank in range(min(max(k_values), len(gallery_ids))):
            retrieved_idx = top_k_indices[i, rank].item()
            retrieved_id = gallery_ids[retrieved_idx]
            similarity_score = top_k_similarities[i, rank].item()
            
            detailed_results.append({
                'query_id': query_id,
                'retrieved_id': retrieved_id,
                'rank': rank + 1,
                'similarity': similarity_score
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(output_dir, 'detailed_retrieval.tsv')
    detailed_df.to_csv(detailed_path, index=False, sep='\t', float_format='%.6f')
    
    # Print summary of results
    print(f"\nResults saved to {output_dir}")
    print(f"- Overall metrics (Recall@K and mAP@K): retrieval_metrics.tsv")
    print(f"- Detailed retrieval: detailed_retrieval.tsv")
    
    # Print a summary of key metrics
    print(f"\n=== Summary ===")
    print(f"Recall@1: {results.get('Recall@1', 0):.4f}")
    print(f"Recall@5: {results.get('Recall@5', 0):.4f}")
    print(f"Recall@10: {results.get('Recall@10', 0):.4f}")
    print(f"Recall@20: {results.get('Recall@20', 0):.4f}")
    print(f"mAP@1: {results.get('mAP@1', 0):.4f}")
    print(f"mAP@5: {results.get('mAP@5', 0):.4f}")
    print(f"mAP@10: {results.get('mAP@10', 0):.4f}")
    print(f"mAP@20: {results.get('mAP@20', 0):.4f}")

def run_single_fold_evaluation(args, fold_idx, output_dir):
    """Run image-to-image retrieval evaluation for a single fold"""
    print(f"Running image-to-image retrieval on fold {fold_idx}")
    print(f"Using pre-computed features from: {args.pfm_name}")
    print(f"Similarity metric: {args.similarity_metric}")
    print(f"K values for evaluation: {args.k_values}")
    
    # Load datasets for the specified fold
    print(f"Loading datasets for fold {fold_idx}...")
    train_dataset = OneSlideDatasetKFold(args.data_json, fold_idx, mode='train', pfm_name=args.pfm_name)
    test_dataset = OneSlideDatasetKFold(args.data_json, fold_idx, mode='valid', pfm_name=args.pfm_name)
    
    print(f"Train set size: {len(train_dataset)} (gallery)")
    print(f"Test set size: {len(test_dataset)} (queries)")
    print(f"Classes: {train_dataset.get_classes()}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Extract features for gallery (train set)
    print("\n=== Loading Gallery Features (Train Set) ===")
    gallery_features, gallery_labels, gallery_ids = extract_slide_features(train_loader)
    
    # Extract features for queries (test set) 
    print("\n=== Loading Query Features (Test Set) ===")
    query_features, query_labels, query_ids = extract_slide_features(test_loader)
    print('\n')
    print(f"Gallery features shape: {gallery_features.shape}")
    print(f"Query features shape: {query_features.shape}")
    
    # Compute similarity matrix
    print(f"\n=== Computing Similarity Matrix ({args.similarity_metric}) ===")
    similarity_matrix = compute_similarity(query_features, gallery_features, args.similarity_metric)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Compute Recall@K metrics
    print("\n=== Computing Recall@K Metrics ===")
    recall_results = compute_recall_at_k(similarity_matrix, query_labels, gallery_labels, args.k_values)
    
    # Compute mAP@K metrics
    print("\n=== Computing mAP@K Metrics ===")
    map_results = compute_map_at_k(similarity_matrix, query_labels, gallery_labels, args.k_values)
    
    # Combine all results
    all_results = {**recall_results, **map_results}
    
    # Save single fold results if not running all folds
    if not args.run_all_folds:
        save_retrieval_results(all_results, query_ids, gallery_ids, similarity_matrix, output_dir, args.k_values)
        print(f"\nImage-to-image retrieval completed!")
        print(f"Results saved in: {output_dir}")
    
    return all_results

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
    
    # Get all available metrics from summary_stats and sort them
    all_metrics = list(summary_stats.keys())
    
    # Separate Recall and mAP metrics and sort by K value
    recall_metrics = sorted([m for m in all_metrics if m.startswith('Recall@')], 
                           key=lambda x: int(x.split('@')[1]))
    map_metrics = sorted([m for m in all_metrics if m.startswith('mAP@')], 
                        key=lambda x: int(x.split('@')[1]))
    
    # Combine in order: all Recall metrics first, then all mAP metrics
    metrics_to_save = recall_metrics + map_metrics
    
    # Create metric display names (same as metric names for retrieval)
    metric_names = {metric: metric for metric in metrics_to_save}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("Metric\tMean\tStd\tN_Folds\t")
        
        # Add fold columns
        fold_names = sorted(fold_results.keys())
        for fold_name in fold_names:
            f.write(f"{fold_name}\t")
        
        f.write("LaTeX_Subscript\tLaTeX_PM\tLaTeX_ScriptSize\n")
        
        # Write retrieval results
        for key in metrics_to_save:
            if key in summary_stats:
                stats = summary_stats[key]
                display_name = metric_names.get(key, key)
                
                # Format LaTeX
                latex_formats = format_latex_results(stats['mean'], stats['std'])
                
                f.write(f"{display_name}\t{stats['mean']:.4f}\t{stats['std']:.4f}\t{stats['n_folds']}\t")
                
                # Add fold values
                for fold_name in fold_names:
                    fold_value = fold_results[fold_name].get(key, "N/A")
                    if isinstance(fold_value, (int, float)):
                        fold_value = f"{fold_value:.4f}"
                    f.write(f"{fold_value}\t")
                
                f.write(f"{latex_formats['subscript']}\t{latex_formats['pm']}\t{latex_formats['scriptsize']}\n")
    
    print(f"\nK-Fold results saved to: {output_file}")

def print_and_save_kfold_results(fold_results, summary_stats, output_file="summary_table.txt"):
    """Print formatted k-fold results and save to file"""
    
    # Get all available metrics from summary_stats and sort them
    all_metrics = list(summary_stats.keys())
    
    # Separate Recall and mAP metrics and sort by K value
    recall_metrics = sorted([m for m in all_metrics if m.startswith('Recall@')], 
                           key=lambda x: int(x.split('@')[1]))
    map_metrics = sorted([m for m in all_metrics if m.startswith('mAP@')], 
                        key=lambda x: int(x.split('@')[1]))
    
    # Combine in order: all Recall metrics first, then all mAP metrics
    key_metrics = recall_metrics + map_metrics
    
    # Prepare output content
    content_lines = []
    content_lines.append("="*80)
    content_lines.append("K-FOLD CROSS VALIDATION RETRIEVAL RESULTS")
    content_lines.append("="*80)
    content_lines.append(f"Number of Folds: {len(fold_results)}")
    content_lines.append("")
    
    content_lines.append("Image-to-Image Retrieval Results:")
    content_lines.append("-" * 50)
    
    for key in key_metrics:
        if key in summary_stats:
            stats = summary_stats[key]
            
            # Regular format
            regular_line = (f"{key:15}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"(n={stats['n_folds']})")
            content_lines.append(regular_line)
            
            # LaTeX formats
            latex_formats = format_latex_results(stats['mean'], stats['std'])
            content_lines.append(f"  LaTeX Subscript : {latex_formats['subscript']}")
            content_lines.append(f"  LaTeX PM        : {latex_formats['pm']}")
            content_lines.append(f"  LaTeX ScriptSize: {latex_formats['scriptsize']}")
            content_lines.append("")
    
    # Print to console
    for line in content_lines:
        print(line)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in content_lines:
            f.write(line + '\n')
    
    print(f"\nSummary table saved to: {output_file}")

def run_kfold_evaluation(args):
    """Run k-fold cross validation evaluation"""
    
    # Get available folds
    fold_numbers = get_available_folds(args.data_json)
    if not fold_numbers:
        raise ValueError("No valid folds found in JSON file")
    
    print(f"Found {len(fold_numbers)} folds: {fold_numbers}")
    
    # Store results for each fold
    fold_results = {}
    
    # Define metric keys for retrieval evaluation
    retrieval_metric_keys = []
    for k in args.k_values:
        retrieval_metric_keys.extend([f'Recall@{k}', f'mAP@{k}'])
    
    # Initialize metric storage
    all_metrics = {key: [] for key in retrieval_metric_keys}
    
    for fold_num in fold_numbers:
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_num}")
        print(f"{'='*60}")
        
        # Run single fold evaluation
        fold_results_dict = run_single_fold_evaluation(args, fold_num, None)
        
        # Store fold results
        fold_results[f'fold_{fold_num}'] = fold_results_dict
        
        # Collect metrics for summary statistics
        for key in retrieval_metric_keys:
            if key in fold_results_dict:
                all_metrics[key].append(fold_results_dict[key])
        
        # Print fold summary
        print(f"Fold {fold_num} Results:")
        print(f"  Recall@1: {fold_results_dict.get('Recall@1', 0):.4f}")
        print(f"  Recall@5: {fold_results_dict.get('Recall@5', 0):.4f}")
        print(f"  Recall@10: {fold_results_dict.get('Recall@10', 0):.4f}")
        print(f"  mAP@1: {fold_results_dict.get('mAP@1', 0):.4f}")
        print(f"  mAP@5: {fold_results_dict.get('mAP@5', 0):.4f}")
        print(f"  mAP@10: {fold_results_dict.get('mAP@10', 0):.4f}")
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(all_metrics)
    
    return fold_results, summary_stats

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup output folder
    output_dir = setup_output_folder(args)
    print(f"Output directory: {output_dir}")
    
    if args.run_all_folds:
        print("="*80)
        print("K-FOLD CROSS VALIDATION IMAGE-TO-IMAGE RETRIEVAL")
        print("="*80)
        print(f"Dataset: {args.data_json}")
        print(f"PFM Name: {args.pfm_name}")
        print(f"Similarity Metric: {args.similarity_metric}")
        print(f"K values: {args.k_values}")
        print(f"Random Seed: {args.seed}")
        print(f"Output Directory: {output_dir}")
        
        # Run k-fold evaluation
        fold_results, summary_stats = run_kfold_evaluation(args)
        
        # Save results
        tsv_file = os.path.join(output_dir, "kfold_results.tsv")
        summary_file = os.path.join(output_dir, "summary_table.txt")
        
        print_and_save_kfold_results(fold_results, summary_stats, summary_file)
        save_kfold_results_tsv(fold_results, summary_stats, tsv_file)
        
        print(f"\nK-fold evaluation completed!")
        print(f"Results saved in: {output_dir}")
        
    else:
        # Run single fold evaluation
        run_single_fold_evaluation(args, args.fold_idx, output_dir)

if __name__ == '__main__':
    main()
