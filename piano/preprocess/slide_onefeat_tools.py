import os
import torch
import torch.utils
from tqdm import tqdm
from piano.model.slide_encoder import create_slide_encoder
import time

# ANSI color codes for progress bar
GREEN = '\033[92m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def load_patch_features(feat_file_path):
    """
    Load patch features from .pth file
    Args:
        feat_file_path: path to the .pth file containing patch features
    Returns:
        dict with 'feats' [N, C] and 'coords' [N, 2]
    """
    data = torch.load(feat_file_path, map_location='cpu')
    
    features = data['feats']  # [N, C]
    coords = data['coords']   # [N, 2]
    
    return {
        'feats': features.unsqueeze(0),  # [1, N, C]
        'coords': coords.unsqueeze(0)   # [1, N, 2]
    }

def func_onefeat_ext(args, pair_list, gpu_id):
    """
    Extract slide-level features from patch features
    Args:
        args: arguments containing model_name, batch_size, etc.
        pair_list: list of tuples (input_feat_path, output_feat_path)
        gpu_id: GPU device ID
    """
    device = torch.device(f"cuda:{gpu_id}")
    
    # Create slide-level model (not patch-level)
    model = create_slide_encoder(args.model_name)
    model.eval()
    model.to(device)

    total_process_time = 0
    processed_slides = 0

    amp_dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }[args.amp]

        # Create progress bar for this GPU
    pbar = tqdm(total=len(pair_list),
              desc=f'GPU {gpu_id}',
              position=gpu_id,
              ncols=100,
              colour='green',
              leave=False)

    for item, pair_path in enumerate(pair_list):
        try:
            start_time = time.time()
            input_feat_path = pair_path[0]  # Input patch feature file
            output_feat_path = pair_path[1]  # Output slide feature file
            slide_id = os.path.basename(input_feat_path).split('.pth')[0]

            # Check if input file exists
            if not os.path.exists(input_feat_path):
                tqdm.write(f"GPU {gpu_id}: Input file not found: {input_feat_path}")
                pbar.update(1)
                continue
            
            # Create output directory
            os.makedirs(os.path.dirname(output_feat_path), exist_ok=True)

            with torch.inference_mode():
                # Load patch features directly
                data = load_patch_features(input_feat_path)
                
                # Move to device
                input_feats = data['feats'].to(device, non_blocking=True)  # [1, N, C]
                input_coords = data['coords'].to(device, non_blocking=True)  # [1, N, 2]
                
                # Prepare input for the model
                model_input = {
                    'feats': input_feats,
                    'coords': input_coords
                }
                
                # Extract slide-level features
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    slide_feature = model(model_input)  # [1, C]
                
                # Save slide-level features
                torch.save({
                    'slide_feats': slide_feature.to(amp_dtype).cpu(),
                    'slide_id': slide_id
                }, output_feat_path)
                
                end_time = time.time()
                slide_time = end_time - start_time
                total_process_time += slide_time
                processed_slides += 1

                pbar.set_description(f'GPU {gpu_id} Slide {item+1}/{len(pair_list)} | Shape {slide_feature.shape}')
                
                # Print processing info using tqdm.write (won't interfere with progress bar)
                # tqdm.write(f"GPU {gpu_id}: {slide_id} | {input_feats.shape} -> {slide_feature.shape} | {slide_time:.2f}s")
                
                # Update progress bar
                pbar.update(1)
                
        except Exception as e:
            tqdm.write(f"GPU {gpu_id}: Error processing {slide_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            pbar.update(1)
    
    # Close progress bar
    pbar.close()

    return total_process_time, processed_slides