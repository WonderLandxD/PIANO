import argparse
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader

from piano import create_model
from piano.datasets.roi_datasets import ROIDataset
from piano.roi_classification.roi_finetune_tools import ROIClassifier, predict_roi, roi_load_ckpt
from piano.utils.utils import planar_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    # Data related arguments
    parser.add_argument('--test_json', type=str, required=True, help='Path to test data JSON file')
    parser.add_argument('--transform_config', type=str, default=None, help='Path to data augmentation config file')
    
    # Model related arguments
    parser.add_argument('--model_name', type=str, default='uni_v2', help='Pathology foundation model name')
    parser.add_argument('--training_mode', type=str, default='linear_probe', 
                      choices=['linear_probe', 'mlp_probe', 'full_param'], help='Training mode')
    parser.add_argument('--local_dir', type=bool, default=False, help='Using local directory of feature extractor')
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--finetune_ckpt', type=str, required=True, help='Path to finetuned checkpoint')


    # GPU related arguments
    parser.add_argument('--gpu_id', type=int, default=0, 
                      help='GPU device ID (default: 0). Set to -1 for CPU')
    
    # Output related arguments
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save inference results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    print(f"Using device: {device}" + (f" (GPU {args.gpu_id})" if torch.cuda.is_available() and args.gpu_id >= 0 else ""))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ROIDataset(args.test_json, mode='test', transform_config=args.transform_config)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load model
    print(f"Loading {args.model_name} model...")
    backbone = create_model(args.model_name, checkpoint_path=args.pretrain_ckpt, local_dir=args.local_dir)
    model = ROIClassifier(backbone, num_classes=len(test_dataset.get_classes()), 
                         training_mode=args.training_mode).to(device)
    model, saved_epoch = roi_load_ckpt(args.finetune_ckpt, model)
    model = model.to(device)
    
    # Run inference
    print(f"{args.model_name} inference using training mode: {args.training_mode} from epoch: {saved_epoch}")
    test_logits, test_labels, test_loss = predict_roi(model, test_loader, device, epoch=0)
    
    # Calculate metrics
    test_metrics = planar_metrics(test_logits, test_labels, len(test_dataset.get_classes()))
    
    # Print results
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        if metric != 'confusion_mat':
            print(f"{metric.capitalize()}: {value:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'test_results.csv')
    pd.DataFrame([test_metrics]).to_csv(results_file, index=False)
    print(f"\nTest results saved to {results_file}")

if __name__ == '__main__':
    main()