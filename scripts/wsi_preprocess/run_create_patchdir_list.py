import os
import csv
from datetime import date
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Create patch directory list (csv format) from thumbnail files.')
    parser.add_argument('--data_folder', type=str, required=True, help='Root directory path containing thumbnail files')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save CSV file')
    return parser.parse_args()

def find_thumbnail_dirs(data_folder):
    patch_dirs = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file == 'x20_thumbnail.jpg':
                file_path = os.path.join(root, file)
                dir_path = os.path.dirname(file_path)
                if os.path.basename(dir_path) == 'thumbnail':
                    # Get the parent directory of thumbnail folder
                    parent_dir = os.path.dirname(dir_path)
                    patch_dirs.append(parent_dir)
    
    return patch_dirs

def save_to_csv(dir_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for dir_path in dir_list:
            writer.writerow([dir_path])

def main():
    args = parse()
    
    # Find all patch directories containing thumbnail files
    patch_dirs = find_thumbnail_dirs(args.data_folder)
    
    # Create save path
    today = date.today()
    csv_file_path = os.path.join(args.save_dir, f'{args.dataset_name}_patch_dirs_{today}.csv')
    
    # Save to CSV
    save_to_csv(patch_dirs, csv_file_path)
    
    print(f"Found {len(patch_dirs)} patch directories")
    print(f"Saved directory list to: {csv_file_path}")

if __name__ == '__main__':
    main()
