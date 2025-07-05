# Create a list of patch feature path (.pt) for a given folder

# Usage: 
# python 5_run_create_feats_list.py \
#       --data_folder /path/to/data_folder \
#       --model_name model_name \
#       --dataset_name dataset_name \
#       --save_dir /path/to/save_dir


import os
import csv
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Create feature file list (csv format) from model directories.')
    parser.add_argument('--data_folder', type=str, required=True, help='Root directory path containing model feature folders')
    parser.add_argument('--model_name', type=str, required=True, help='Model name to search for in folder names and file names')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name for output file naming')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save CSV file')
    return parser.parse_args()

def find_model_feature_files(data_folder, model_name):
    """
    Find all .pth feature files in folders named with model_name,
    where the files also contain model_name in their filename
    """
    feature_files = []
    
    for root, dirs, files in os.walk(data_folder):
        # Check if current directory name contains model_name
        dir_name = os.path.basename(root)
        if model_name in dir_name:
            # Look for .pth files that also contain model_name in filename
            for file in files:
                if file.endswith('.pth') and model_name in file:
                    file_path = os.path.join(root, file)
                    feature_files.append(file_path)
    
    return sorted(feature_files)

def save_to_csv(file_list, save_path):
    """Save feature file list to CSV file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write file paths
        for file_path in file_list:
            writer.writerow([file_path])

def main():
    args = parse()
    
    # Find all feature files containing model_name
    feature_files = find_model_feature_files(args.data_folder, args.model_name)
    
    # Create save path
    csv_file_path = os.path.join(args.save_dir, f'{args.dataset_name}_{args.model_name}.csv')
    
    # Save to CSV
    save_to_csv(feature_files, csv_file_path)
    
    print(f"Found {len(feature_files)} feature files for model '{args.model_name}'")
    print(f"Saved feature file list to: {csv_file_path}")

if __name__ == '__main__':
    main() 