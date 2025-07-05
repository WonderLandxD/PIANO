import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.multiprocessing as mp

import argparse
import glob
import os
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from piano.preprocess.slide_onefeat_tools import func_onefeat_ext


def parse():
    parser = argparse.ArgumentParser(description='Multithreaded feature extraction for histopathology whole slide images using PIANO and opensdpc library.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for extracting features')
    parser.add_argument('--model_name', type=str, default=None, required=True, choices=['chief', 'prism', 'gigapath', 'titan'], help='Pathology foundation model name of feature extractor')
    parser.add_argument('--gpu_ids', type=int, nargs='+', required=True, help='List of GPU IDs to use')
    parser.add_argument('--num_processes', type=int, required=True, help='Number of processes to use for parallel processing')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save features')

    parser.add_argument('--csv_path', type=str, help='Path to CSV file containing slide directories')
    parser.add_argument('--amp', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'], help='Mixed precision mode (fp32, fp16, bf16)')

    return parser.parse_args()


def generate_pair_list(patchdir_feat_list, save_dir, model, args):
    pair_list = []
    for patchdir_feat_path in patchdir_feat_list:
        patchdir_feat_name = os.path.basename(patchdir_feat_path)
        slide_name = patchdir_feat_name.split('.pth')[0].split('piano_')[1].split(f'_{args.patch_model_name}')[0]
        save_path = os.path.join(save_dir, slide_name, model)
        feat_path = os.path.join(save_path, f'piano_{slide_name}_{model}.pth')
        if not os.path.exists(feat_path):
            pair_list.append([patchdir_feat_path, feat_path])
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
                func_onefeat_ext, 
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

    if args.model_name == 'chief':
        args.patch_model_name = 'ctranspath'
    elif args.model_name == 'prism':
        args.patch_model_name = 'virchow_v1'
    elif args.model_name == 'gigapath':
        args.patch_model_name = 'prov_gigapath'
    elif args.model_name == 'titan':
        args.patch_model_name = 'conch_v1_5'
        

    # Read slide list from CSV
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        patchdir_feat_list = [line.strip() for line in f.readlines()]

    save_dir = args.save_dir

    pair_list = generate_pair_list(patchdir_feat_list, save_dir, args.model_name, args)

    distribute_processing_patch2feat(pair_list, args.gpu_ids, args.num_processes, args)
