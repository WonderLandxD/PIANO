import os
import csv
from datetime import date


import argparse

def parse():
    parser = argparse.ArgumentParser(description='Create WSI file list (csv format) for patch generation.')
    parser.add_argument('--data_folder', type=str, required=True, help='Root directory path containing WSI files')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save CSV file')
    parser.add_argument('--additional_file_types', type=str, nargs='+', default=['.svs', '.sdpc', '.tiff', '.tif', '.ndpi'], help='List of WSI file extensions to search for')
    return parser.parse_args()


def find_wsi_files(data_folder, file_types):
    wsi_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if any(file.endswith(ext) for ext in file_types):
                full_path = os.path.join(root, file)
                wsi_files.append(full_path)
    return wsi_files

def save_to_csv(file_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for filename in file_list:
            writer.writerow([filename])

def main():
    args = parse()
    
    # Find all WSI files
    wsi_files = find_wsi_files(args.data_folder, args.file_types)
    
    # Create save path
    today = date.today()
    csv_file_path = os.path.join(args.save_dir, f'{args.dataset_name}_wsi_{today}.csv')
    
    # Save to CSV
    save_to_csv(wsi_files, csv_file_path)
    
    print(f"Found {len(wsi_files)} WSI files")
    print(f"Saved file list to: {csv_file_path}")

if __name__ == '__main__':
    main()
