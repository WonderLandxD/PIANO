import argparse
import os
import csv
import logging
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from piano.slide_patch_tools import func_patching

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('patching.log'),
            logging.StreamHandler()
        ]
    )

def read_slide_list(csv_file_path):
    slide_list = []
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in tqdm(reader):
            slide_list.append(row[0])
    return slide_list

def generate_pair_list(slide_list, save_dir):
    pair_list = []
    for slide_path in tqdm(slide_list):
        slide_name = Path(slide_path).stem
        save_path = os.path.join(save_dir, slide_name)
        thumb_path = os.path.join(save_path, 'thumbnail/x20_thumbnail.jpg')
        if not os.path.exists(thumb_path):
            pair_list.append([slide_path, save_path])
    return pair_list

def distribute_processing_wsi2patch(pair_list, num_thread, args):
    sub_pair_list = np.array_split(pair_list, num_thread)

    with ThreadPoolExecutor(max_workers=num_thread) as executor:
        futures = {executor.submit(func_patching, args, sublist, i): i for i, sublist in enumerate(sub_pair_list)}

        for future in as_completed(futures):
            thread_index = futures[future]
            try:
                result = future.result()
                # logging.info(result)  
            except Exception as exc:
                logging.error(f'Thread {thread_index} generated an exception: {exc}', exc_info=True)

def validate_args(args):
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    if args.WSI_level != 1:
        raise ValueError("WSI_level must be set to 1")
    if args.n_thread <= 0:
        raise ValueError("Number of threads must be greater than 0")

def parse():
    parser = argparse.ArgumentParser(description='Multithreaded patch generation for hitopathology whole slide images using PIANO and opensdpc library.')
    parser.add_argument('--n_thread', type=int, default=16, help='Number of threads to use')
    # Hyperparameters for patch generation
    parser.add_argument('--patch_w', type=int, default=256, help='Width of patch')
    parser.add_argument('--patch_h', type=int, default=256, help='Height of patch')     
    parser.add_argument('--overlap_w', type=int, default=0, help='Overlap width of patch')
    parser.add_argument('--overlap_h', type=int, default=0, help='Overlap height of patch')             
    parser.add_argument('--WSI_level', type=float, default=1, help='WSI level for cropping patches')
    parser.add_argument('--blank_TH', type=float, default=0.7, help='Blank threshold for patch cutting')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for mask operations')
    parser.add_argument('--thumb_n', type=float, default=1, help='Thumbnail layer index')
    
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save patches')
    return parser.parse_args()

if __name__ == '__main__':
    # setup_logging()
    args = parse()
    validate_args(args)

    csv_file_path = Path(args.csv_path)
    csv_name = csv_file_path.stem
    args.save_dir = Path(args.save_dir) / csv_name

    slide_list = read_slide_list(csv_file_path)
    pair_list = generate_pair_list(slide_list, args.save_dir)

    logging.info(f'All data number: {len(slide_list)}, unprocessed data number: {len(pair_list)}')
    distribute_processing_wsi2patch(pair_list, args.n_thread, args)
