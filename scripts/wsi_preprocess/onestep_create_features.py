# Author: Jiawen Li (jw-li24@mails.tsinghua.edu.cn)


# example
# python onestep_patch_features_loop.py \
#     --output_dir /mnt/sdb/ljw/MoPE/histai_patch_features/HISTAI \
#     --csv_path /mnt/sdb/ljw/MoPE/split_csv_files/histai_part1_slide_001.csv \
#     --model_name uni_v1 \
#     --gpu_id 0 \
#     --batch_size 64 \
#     --patch_size 256 \
#     --overlap 0 \
#     --wsi_level 0 \
#     --blank_TH 0.2 \
#     --kernel_size 5 \
#     --save_patches False \
#     --patch_save_dir None


import argparse
import os
import glob
import torch
import time

from piano.preprocess.slide_feat_tools_onestep import extract_wsi_features
from piano.model.patch_encoder import create_patch_encoder


def parse_args():
    parser = argparse.ArgumentParser(description='Loop to create onestep features of tissue-contained patches at a time')

    # input parameters
    parser.add_argument('--output_dir', type=str, default='/mnt/sdb/ljw/MoPE/histai_patch_features/HISTAI', help='Output directory for features')
    parser.add_argument('--csv_path', type=str, default=None, help='Path to CSV file containing slide file paths')

    # model parameters
    parser.add_argument('--model_name', type=str, default='uni_v1', choices=['uni_v1', 'conch_v1', 'conch_v1_5', 'uni_v2', 'prov_gigapath', 'virchow_v1', 'virchow_v2'], help='Model name')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between patches')
    parser.add_argument('--wsi_level', type=int, default=0, help='WSI level')
    parser.add_argument('--blank_TH', type=float, default=0.2, help='Blank threshold')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size')

    # patch save parameters
    parser.add_argument('--save_patches', type=bool, default=False, help='Save patches')
    parser.add_argument('--patch_save_dir', type=str, default=None, help='Directory to save patches')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # validate args
    print('output_dir:', args.output_dir)
    print('csv_path:', args.csv_path)
    print('model_name:', args.model_name)
    print('gpu_id:', args.gpu_id)
    print('batch_size:', args.batch_size)
    print('patch_size:', args.patch_size)
    print('overlap:', args.overlap)
    print('wsi_level:', args.wsi_level)
    print('blank_TH:', args.blank_TH)
    print('kernel_size:', args.kernel_size)
    print('save_patches:', args.save_patches)
    print('patch_save_dir:', args.patch_save_dir)
    print('-'*100)


    #########################################################
    # STEP1: load slide list
    #########################################################

    with open(args.csv_path, 'r') as f:
        slide_list = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(slide_list)} slides from CSV: {args.csv_path}")

    #########################################################
    # STEP2: load model
    #########################################################

    model = create_patch_encoder(args.model_name)
    model.eval()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Use torch.compile to accelerate model
    if torch.cuda.is_available():
        print('Compiling model with torch.compile...')
        time_start = time.time()
        model = torch.compile(model, mode='max-autotune')
        print(f"Model compiled in {time.time() - time_start:.2f}s")
    else:
        print('No GPU available, model not compiled')

    #########################################################
    # STEP3: extract features
    #########################################################

    for i, slide_path in enumerate(slide_list):
        # for HISTAI dataset 
        slide_name = os.path.basename(slide_path).split('.')[0]
        case_name = slide_path.split('/')[-2]
        oncological_type = slide_path.split('/')[-3]
        output_path = os.path.join(args.output_dir, args.model_name, oncological_type, case_name, args.model_name + '_' + slide_name + '.pt')

        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skip: {i+1}/{len(slide_list)}, already exists: {output_path}")
            continue

        if 'x40' in slide_name:
            args.wsi_level = 1
        else:
            args.wsi_level = 0

        print(f"Progress: {i+1}/{len(slide_list)}, slide: {slide_path}")
        print('-'*100)
        time_start = time.time()
        extract_wsi_features(
            slide_path=slide_path, 
            feat_path=output_path, 
            model=model, 
            device=device, 
            batch_size=args.batch_size, 
            patch_size=args.patch_size, 
            overlap=args.overlap, 
            wsi_level=args.wsi_level, 
            blank_TH=args.blank_TH, 
            kernel_size=args.kernel_size, 
            save_patches=args.save_patches, 
            patch_save_dir=args.patch_save_dir)
        print(f"Slide {slide_path} completed, time: {time.time() - time_start:.2f}s")
        print('-'*100)
        print('\n\n\n')
