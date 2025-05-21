import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from piano import create_model, get_model_output_dim
from piano.datasets.wsi_datasets import WSIDataset
from piano.wsi_classification.wsi_finetune_tools import WSIClassifier, predict_wsi, wsi_load_ckpt
from piano.utils.utils import planar_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    # Data related arguments
    parser.add_argument('--test_json', type=str, required=True, help='Path to test data JSON file')
    parser.add_argument('--few_shot', type=int, default=None, help='Number of samples per class for few-shot learning')
    
    # Model related arguments
    parser.add_argument('--model_name', type=str, default='uni_v2', help='Pathology foundation model name')
    parser.add_argument('--training_mode', type=str, default='abmil', 
                        choices=['abmil', 'simlp', 'mt_abmil', 'mt_simlp'], help='Training mode')
    parser.add_argument('--finetune_ckpt', type=str, required=True, help='Path to finetuned checkpoint')

    # GPU related arguments
    parser.add_argument('--gpu_id', type=int, default=0, 
                      help='GPU device ID (default: 0). Set to -1 for CPU')
    
    # Output related arguments
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save inference results')
    parser.add_argument('--plot_confusion', type=bool, default=True, help='Plot confusion matrix')
    
    return parser.parse_args()

def plot_confusion_matrix(confusion_mat, class_names, output_path):
    """Plot and save confusion matrix"""
    fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
    
    im = ax.imshow(confusion_mat, cmap='Blues')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    # Adding text annotations
    thresh = confusion_mat.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{confusion_mat[i, j]:.0f}',
                   ha="center", va="center", 
                   color="white" if confusion_mat[i, j] > thresh else "black")
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add class labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    print(f"Using device: {device}" + (f" (GPU {args.gpu_id})" if torch.cuda.is_available() and args.gpu_id >= 0 else ""))
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # If output directory is not specified, use the directory of the checkpoint
        args.output_dir = os.path.dirname(args.finetune_ckpt)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = WSIDataset(args.test_json, mode='test', pfm_name=args.model_name)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Get class names
    class_names = test_dataset.get_classes()
    num_classes = len(class_names)
    
    # Load model
    print(f"Loading {args.training_mode} slide-level finetune method under {args.model_name}...")
    feat_dim = get_model_output_dim(args.model_name)
    model = WSIClassifier(feat_dim, num_classes=num_classes, 
                          training_mode=args.training_mode).to(device)
    
    # Load checkpoint
    model, saved_epoch = wsi_load_ckpt(args.finetune_ckpt, model)
    model = model.to(device)
    
    # Run inference
    print(f"{args.model_name} inference using training mode: {args.training_mode} (from epoch: {saved_epoch})")
    test_logits, test_labels, test_loss = predict_wsi(model, test_loader, device, epoch=0)
    
    # Calculate metrics
    test_metrics = planar_metrics(test_logits, test_labels, num_classes)
    
    # Print results
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        if metric != 'confusion_mat':
            print(f"{metric}: {value:.4f}")
    
    # Plot confusion matrix if requested
    if args.plot_confusion and 'confusion_mat' in test_metrics:
        conf_mat_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(test_metrics['confusion_mat'], class_names, conf_mat_path)
        print(f"Confusion matrix saved to {conf_mat_path}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'test_results.csv')
    # Remove confusion matrix for CSV saving
    metrics_to_save = {k: v for k, v in test_metrics.items() if k != 'confusion_mat'}
    pd.DataFrame([metrics_to_save]).to_csv(results_file, index=False)
    print(f"\nTest results saved to {results_file}")
    
    # Save predictions for each WSI
    predictions = []
    for i, (logit, label) in enumerate(zip(test_logits, test_labels)):
        pred_class = torch.argmax(logit).item()
        # Convert true_class to integer
        true_class = int(label.item())
        prob = torch.softmax(logit, dim=0)[pred_class].item()
        
        # Get WSI name if available
        wsi_name = test_dataset.get_name(i) if hasattr(test_dataset, 'get_name') else f"WSI_{i}"
        
        predictions.append({
            'wsi_name': wsi_name,
            'pred_class': class_names[pred_class],
            'true_class': class_names[true_class],
            'pred_confidence': prob,
            'correct': pred_class == true_class
        })
    
    pred_file = os.path.join(args.output_dir, 'wsi_predictions.csv')
    pd.DataFrame(predictions).to_csv(pred_file, index=False)
    print(f"Per-WSI predictions saved to {pred_file}")

if __name__ == '__main__':
    main()
