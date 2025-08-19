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

from piano.datasets.oneslide_datasets import OneSlideDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Image-to-Image Retrieval using Pre-computed Slide Features (Train-Val-Test)')
    
    # Data related arguments
    parser.add_argument('--data_json', type=str, required=True, 
                       help='Path to data JSON file with train/val/test structure')
    
    # Feature related arguments
    parser.add_argument('--pfm_name', type=str, default='chief', 
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
    parser.add_argument('--output_dir', type=str, default='/mnt/sdb/ljw/PIANO-Update/PIANO_Preview/results/retrieval_results',
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

def setup_output_folder(args):
    """Setup output folder and return path"""
    # Get dataset name from json path
    dataset_name = os.path.splitext(os.path.basename(args.data_json))[0]
    
    # Create experiment folder name
    exp_name = f"i2i_retrieval_tvt_{dataset_name}_{args.pfm_name}_seed{args.seed}"
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

def save_retrieval_results(results, query_ids, gallery_ids, similarity_matrix, output_dir, k_values, prefix="", save_individual_metrics=True):
    """Save detailed retrieval results including Recall@K and mAP@K metrics"""
    # Add prefix to filenames if provided
    metrics_filename = f'{prefix}_retrieval_metrics.tsv' if prefix else 'retrieval_metrics.tsv'
    detailed_filename = f'{prefix}_detailed_retrieval.tsv' if prefix else 'detailed_retrieval.tsv'
    
    # Save overall metrics (both Recall@K and mAP@K) in vertical format only if requested
    if save_individual_metrics:
        # Convert to vertical format: each metric is a row
        metrics_data = []
        for metric, value in results.items():
            metrics_data.append({'Metric': metric, 'Value': value})
        
        results_df = pd.DataFrame(metrics_data)
        results_path = os.path.join(output_dir, metrics_filename)
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
    detailed_path = os.path.join(output_dir, detailed_filename)
    detailed_df.to_csv(detailed_path, index=False, sep='\t', float_format='%.6f')
    
    # Print summary of results
    mode_name = prefix.upper() if prefix else "OVERALL"
    print(f"\n{mode_name} Results saved:")
    if save_individual_metrics:
        print(f"- Overall metrics: {metrics_filename}")
    print(f"- Detailed retrieval: {detailed_filename}")
    
    # Print a summary of key metrics
    print(f"\n=== {mode_name} Summary ===")
    print(f"Recall@1: {results.get('Recall@1', 0):.4f}")
    print(f"Recall@5: {results.get('Recall@5', 0):.4f}")
    print(f"Recall@10: {results.get('Recall@10', 0):.4f}")
    print(f"Recall@20: {results.get('Recall@20', 0):.4f}")
    print(f"mAP@1: {results.get('mAP@1', 0):.4f}")
    print(f"mAP@5: {results.get('mAP@5', 0):.4f}")
    print(f"mAP@10: {results.get('mAP@10', 0):.4f}")
    print(f"mAP@20: {results.get('mAP@20', 0):.4f}")

def run_retrieval_evaluation(args, query_mode, output_dir):
    """Run image-to-image retrieval evaluation with specified query mode (valid or test)"""
    print(f"\n{'='*60}")
    print(f"Running retrieval evaluation with {query_mode.upper()} as queries")
    print(f"{'='*60}")
    
    print(f"Using pre-computed features from: {args.pfm_name}")
    print(f"Similarity metric: {args.similarity_metric}")
    print(f"K values for evaluation: {args.k_values}")
    
    # Load datasets
    print(f"Loading datasets...")
    train_dataset = OneSlideDataset(args.data_json, mode='train', pfm_name=args.pfm_name)
    query_dataset = OneSlideDataset(args.data_json, mode=query_mode, pfm_name=args.pfm_name)
    
    print(f"Train set size: {len(train_dataset)} (gallery)")
    print(f"{query_mode.capitalize()} set size: {len(query_dataset)} (queries)")
    print(f"Classes: {train_dataset.get_classes()}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Extract features for gallery (train set)
    print(f"\n=== Loading Gallery Features (Train Set) ===")
    gallery_features, gallery_labels, gallery_ids = extract_slide_features(train_loader)
    
    # Extract features for queries
    print(f"\n=== Loading Query Features ({query_mode.capitalize()} Set) ===")
    query_features, query_labels, query_ids = extract_slide_features(query_loader)
    print('\n')
    print(f"Gallery features shape: {gallery_features.shape}")
    print(f"Query features shape: {query_features.shape}")
    
    # Compute similarity matrix
    print(f"\n=== Computing Similarity Matrix ({args.similarity_metric}) ===")
    similarity_matrix = compute_similarity(query_features, gallery_features, args.similarity_metric)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Compute Recall@K metrics
    print(f"\n=== Computing Recall@K Metrics for {query_mode.upper()} ===")
    recall_results = compute_recall_at_k(similarity_matrix, query_labels, gallery_labels, args.k_values)
    
    # Compute mAP@K metrics
    print(f"\n=== Computing mAP@K Metrics for {query_mode.upper()} ===")
    map_results = compute_map_at_k(similarity_matrix, query_labels, gallery_labels, args.k_values)
    
    # Combine all results
    all_results = {**recall_results, **map_results}
    
    # Save results with mode prefix (skip individual metrics files)
    save_retrieval_results(all_results, query_ids, gallery_ids, similarity_matrix, output_dir, args.k_values, query_mode, save_individual_metrics=False)
    
    return all_results

def save_combined_results(val_results, test_results, output_dir):
    """Save combined validation and test results"""
    # Combine results with prefixes
    combined_results = {}
    
    # Add validation results with 'val_' prefix
    for key, value in val_results.items():
        combined_results[f'val_{key}'] = value
    
    # Add test results with 'test_' prefix  
    for key, value in test_results.items():
        combined_results[f'test_{key}'] = value
    
    # Save combined results in vertical format
    # Convert to vertical format: each metric is a row with validation and test values
    combined_data = []
    
    # Get all unique metrics (without val_/test_ prefix)
    unique_metrics = set()
    for key in combined_results.keys():
        if key.startswith('val_'):
            unique_metrics.add(key[4:])  # Remove 'val_' prefix
        elif key.startswith('test_'):
            unique_metrics.add(key[5:])  # Remove 'test_' prefix
    
    # Sort metrics properly
    unique_metrics = list(unique_metrics)
    recall_metrics = sorted([m for m in unique_metrics if m.startswith('Recall@')], 
                           key=lambda x: int(x.split('@')[1]))
    map_metrics = sorted([m for m in unique_metrics if m.startswith('mAP@')], 
                        key=lambda x: int(x.split('@')[1]))
    sorted_metrics = recall_metrics + map_metrics
    
    for metric in sorted_metrics:
        val_key = f'val_{metric}'
        test_key = f'test_{metric}'
        val_value = combined_results.get(val_key, 'N/A')
        test_value = combined_results.get(test_key, 'N/A')
        difference = test_value - val_value if (val_value != 'N/A' and test_value != 'N/A') else 'N/A'
        
        # Calculate mean and standard deviation
        if val_value != 'N/A' and test_value != 'N/A':
            mean_value = (val_value + test_value) / 2
            # Sample standard deviation for n=2
            std_value = abs(test_value - val_value) / np.sqrt(2)
            mean_std_format = f"{mean_value:.4f} ± {std_value:.4f}"
            
            # LaTeX formats (convert to percentages like in the example: 95.87)
            mean_percent = mean_value * 100
            std_percent = std_value * 100
            
            latex_subscript = f"${{{mean_percent:.2f}}}_{{{std_percent:.2f}}}$"
            latex_pm = f"${mean_percent:.2f}$±${std_percent:.2f}$"
            latex_scriptsize = f"{mean_percent:.2f}$\\scriptscriptstyle{{({std_percent:.2f})}}$"
        else:
            mean_value = 'N/A'
            std_value = 'N/A'
            mean_std_format = 'N/A'
            latex_subscript = 'N/A'
            latex_pm = 'N/A'
            latex_scriptsize = 'N/A'
        
        combined_data.append({
            'Metric': metric,
            'Validation': val_value,
            'Test': test_value,
            'Difference': difference,
            'Mean': mean_value,
            'Std': std_value,
            'Mean±Std': mean_std_format,
            'LaTeX_Subscript': latex_subscript,
            'LaTeX_PM': latex_pm,
            'LaTeX_ScriptSize': latex_scriptsize
        })
    
    combined_df = pd.DataFrame(combined_data)
    combined_path = os.path.join(output_dir, 'combined_results.tsv')
    combined_df.to_csv(combined_path, index=False, sep='\t', float_format='%.6f')
    
    print(f"\nCombined results saved to: combined_results.tsv")
    
    # Print comparison summary
    print(f"\n=== VALIDATION vs TEST COMPARISON ===")
    
    # Get all available metrics and sort them
    available_metrics = list(val_results.keys())
    
    # Separate Recall and mAP metrics and sort by K value
    recall_metrics = sorted([m for m in available_metrics if m.startswith('Recall@')], 
                           key=lambda x: int(x.split('@')[1]))
    map_metrics = sorted([m for m in available_metrics if m.startswith('mAP@')], 
                        key=lambda x: int(x.split('@')[1]))
    
    # Combine in order: all Recall metrics first, then all mAP metrics
    key_metrics = recall_metrics + map_metrics
    
    print(f"{'Metric':<15} {'Validation':<12} {'Test':<12} {'Difference':<12}")
    print("-" * 55)
    
    for metric in key_metrics:
        val_score = val_results.get(metric, 0)
        test_score = test_results.get(metric, 0)
        diff = test_score - val_score
        print(f"{metric:<15} {val_score:<12.4f} {test_score:<12.4f} {diff:+12.4f}")
    
    return combined_results

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup output folder
    output_dir = setup_output_folder(args)
    print(f"Output directory: {output_dir}")
    
    print("="*80)
    print("TRAIN-VAL-TEST IMAGE-TO-IMAGE RETRIEVAL EVALUATION")
    print("="*80)
    print(f"Dataset: {args.data_json}")
    print(f"PFM Name: {args.pfm_name}")
    print(f"Similarity Metric: {args.similarity_metric}")
    print(f"K values: {args.k_values}")
    print(f"Random Seed: {args.seed}")
    print(f"Output Directory: {output_dir}")
    
    # Run retrieval evaluation with validation queries
    print(f"\n{'='*80}")
    print("PHASE 1: VALIDATION SET AS QUERIES")
    print(f"{'='*80}")
    val_results = run_retrieval_evaluation(args, 'valid', output_dir)
    
    # Run retrieval evaluation with test queries
    print(f"\n{'='*80}")
    print("PHASE 2: TEST SET AS QUERIES")
    print(f"{'='*80}")
    test_results = run_retrieval_evaluation(args, 'test', output_dir)
    
    # Save combined results and comparison
    combined_results = save_combined_results(val_results, test_results, output_dir)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved in: {output_dir}")
    print("Files generated:")
    print("- valid_detailed_retrieval.tsv (validation detailed)")
    print("- test_detailed_retrieval.tsv (test detailed)")
    print("- combined_results.tsv (validation and test comparison with statistics)")
    
    return combined_results

if __name__ == '__main__':
    main()
