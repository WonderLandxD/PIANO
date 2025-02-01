import os
import re
import openslide
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import torch
import torch.utils
from piano.piano_timm import create_piano

try:
    from jpeg4py import JPEG, JPEGRuntimeError
    USE_JPEG4PY = True
except ImportError:
    from PIL import Image
    USE_JPEG4PY = False

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, slide_path, preprocess, image_loader='pil'):
        """Initialize dataset"""
        self.slide_path = slide_path
        self.preprocess = preprocess
        self.image_loader = image_loader
        
        self.patch_paths = sorted(glob.glob(os.path.join(self.slide_path, '*.jpg')))

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        if self.image_loader == 'jpeg4py':
            try:
                image = Image.fromarray(JPEG(patch_path).decode())
            except (JPEGRuntimeError, TypeError):
                image = Image.open(patch_path).convert('RGB')
        elif self.image_loader == 'opencv':
            import cv2
            image = cv2.cvtColor(cv2.imread(patch_path), cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        else:
            image = Image.open(patch_path).convert('RGB')
        return self.preprocess(image)

    def __len__(self):
        return len(self.patch_paths)

def func_feat_ext(args, pair_list, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    model, preprocess, _ = create_piano(args.model_name, args.ckpt, device=device)
    model.set_mode('eval')

    amp_dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }[args.amp]

    for item, pair_path in enumerate(pair_list):
        slide_path = pair_path[0]
        feat_path = pair_path[1]

        process_dataset = FeatureDataset(
            slide_path=slide_path,
            preprocess=preprocess,
            image_loader=args.image_loader
        )
        data_loader = torch.utils.data.DataLoader(
            process_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=False
        )

        os.makedirs(os.path.dirname(feat_path), exist_ok=True)

        with torch.inference_mode():
            features = []
            with tqdm(total=len(process_dataset), 
                     desc=f'GPU{gpu_id} Slide{item+1}/{len(pair_list)}',
                     position=gpu_id, 
                     ncols=90, 
                     leave=False) as pbar:
                
                slide_id = os.path.basename(slide_path)[:12]
                for batch_img in data_loader:
                    batch_img = batch_img.to(device, non_blocking=True)
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        patch_feat = model.encode_image(batch_img)
                    features.append(patch_feat.float())
                    pbar.update(len(batch_img))
                    pbar.set_postfix_str(f"ID:{slide_id}")

            feature_box = torch.cat(features, dim=0).cpu()
            torch.save(feature_box, feat_path)
    
