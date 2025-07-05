# Feature extraction from patches 

# Usage:
# python 4_run_create_wsi_features.py \
#     --csv_path /path/to/patch_dirs.csv \
#     --save_dir /path/to/save/features \
#     --model_name model_name \
#     --gpu_ids 0 1 2 3 \
#     --batch_size 1 \
#     --num_processes 4 \
#     --num_workers 1 \
#     --amp bf16 \
#     --image_loader pil \
#     --image_preprocess /path/to/transform_configs.yaml




import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.multiprocessing as mp

import argparse
import glob
import os
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from piano.preprocess.slide_feat_tools import func_feat_ext


def parse():
    parser = argparse.ArgumentParser(description='Multithreaded feature extraction for histopathology whole slide images using PIANO and opensdpc library.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for extracting features')
    parser.add_argument('--model_name', type=str, default=None, required=True, help='Pathology foundation model name of feature extractor')
    parser.add_argument('--local_dir', type=bool, default=False, help='Using local directory of feature extractor')
    parser.add_argument('--ckpt', type=str, default=None, required=False, help='Checkpoint path of feature extractor')
    parser.add_argument('--gpu_ids', type=int, nargs='+', required=True, help='List of GPU IDs to use')
    parser.add_argument('--num_processes', type=int, required=True, help='Number of processes to use for parallel processing')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save features')

    # Choose one of the following input methods:
    parser.add_argument('--csv_path', type=str, help='Path to CSV file containing slide directories')
    parser.add_argument('--patch_slide_dir', type=str, help='Directory to patches (cropped by slides)')

    parser.add_argument('--amp', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'], help='Mixed precision mode (fp32, fp16, bf16)')
    parser.add_argument('--image_loader', type=str, default='pil', choices=['pil', 'jpeg4py', 'opencv'], help='Image loading method (pil|jpeg4py|opencv)')
    parser.add_argument('--image_preprocess', type=str, default=f'{os.path.dirname(os.path.abspath(__file__))}/transform_configs/create_patch_feats_transforms.yaml', help='Path to the YAML file containing image preprocessing configurations')

    return parser.parse_args()



def generate_pair_list(slide_list, save_dir, model):
    pair_list = []
    for slide_path in slide_list:
        slide_name = os.path.basename(slide_path)
        thumb_path = os.path.join(slide_path, 'thumbnail/x20_thumbnail.jpg')
        save_path = os.path.join(save_dir, slide_name, model)
        feat_path = os.path.join(save_path, f'piano_{slide_name}_{model}.pth')
        if os.path.exists(thumb_path) and not os.path.exists(feat_path):
            pair_list.append([slide_path, feat_path])
    return pair_list


def distribute_processing_patch2feat(wsi_list, gpu_ids, num_processes, args):
    # Split the work based on number of processes
    sub_wsi_lists = np.array_split(wsi_list, num_processes)
    total_time = 0
    total_slides = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Assign specific GPU ID to each process in a round-robin manner
        futures = {
            executor.submit(
                func_feat_ext, 
                args, 
                sublist, 
                gpu_ids[i % len(gpu_ids)]  # Assign GPU ID in round-robin fashion
            ): i for i, sublist in enumerate(sub_wsi_lists)
        }
        
        for future in as_completed(futures):
            process_index = futures[future]
            try:
                process_time, num_slides = future.result()
                total_time += process_time
                total_slides += num_slides
            except Exception as exc:
                print(f'Process {process_index} on GPU {gpu_ids[process_index % len(gpu_ids)]} generated an exception: {exc}')
    
    if total_slides > 0:
        avg_time_per_slide = total_time / total_slides
        print(f'\nFeature Extraction Statistics:')
        print(f'Average processing time per WSI: {avg_time_per_slide:.2f} seconds')
        print(f'Total WSIs processed: {total_slides}')
        print(f'Total processing time: {total_time:.2f} seconds')

def validate_args(args):
    if not args.gpu_ids:
        raise ValueError("GPU IDs must be specified")
    if args.num_processes <= 0:
        raise ValueError("Number of processes must be greater than 0")
    if max(args.gpu_ids) >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU ID specified. Available GPUs: 0 to {torch.cuda.device_count()-1}")


if __name__ == '__main__':
    
    os.environ['OPENBLAS_NUM_THREADS'] = '1'  
    
    mp.set_start_method('spawn', force=True)

    args = parse()
    validate_args(args)

    # Read slide list from CSV if provided, otherwise use glob
    if args.csv_path and os.path.exists(args.csv_path):
        with open(args.csv_path, 'r', encoding='utf-8') as f:
            slide_list = [line.strip() for line in f.readlines()]
    else:
        slide_list = glob.glob(f'{args.patch_slide_dir}/*')

    if args.csv_path:
        save_dir = Path(args.save_dir) / args.csv_path.split('/')[-1].split('.csv')[0]
    elif args.patch_slide_dir:
        if args.patch_slide_dir.endswith('/'):
            save_dir = Path(args.save_dir) / args.patch_slide_dir.split('/')[-2]
        else:
            save_dir = Path(args.save_dir) / args.patch_slide_dir.split('/')[-1]
    else:
        RED = '\033[91m'
        END = '\033[0m'
        raise ValueError(f"{RED}Either csv_path or patch_slide_dir must be provided{END}")

    pair_list = generate_pair_list(slide_list, save_dir, args.model_name)

    logging.info(f'All data number: {len(slide_list)}, unprocessed data number: {len(pair_list)}')
    distribute_processing_patch2feat(pair_list, args.gpu_ids, args.num_processes, args)
