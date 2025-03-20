import os
import re
import openslide
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import torch
import torch.utils
from piano.piano import create_model
import torchvision
import yaml

try:
    from jpeg4py import JPEG, JPEGRuntimeError
    USE_JPEG4PY = True
except ImportError:
    from PIL import Image
    USE_JPEG4PY = False

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, slide_path, image_preprocess, image_loader='pil'):
        self.slide_path = slide_path
        self.preprocess = image_preprocess
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
        
        # "no{idx:06d}_{x_start:09d}x_{y_start:09d}y.jpg"
        filename = os.path.basename(patch_path)
        try:
            match = re.search(r'(\d+)x_(\d+)y\.jpg$', filename)
            if match:
                x_coord = int(match.group(1))
                y_coord = int(match.group(2))
            else:
                x_coord = 0
                y_coord = 0
        except:
            x_coord = 0
            y_coord = 0
            
        return self.preprocess(image), torch.tensor([x_coord, y_coord], dtype=torch.long)

    def __len__(self):
        return len(self.patch_paths)

def load_image_preprocess(cfg_path):
    if cfg_path is None:
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    transforms = []
    for transform_cfg in cfg['transforms']:
        transform_name = transform_cfg['name']
        transform_params = transform_cfg.get('params', {})
        
        if hasattr(torchvision.transforms, transform_name):
            transform_cls = getattr(torchvision.transforms, transform_name)
            transform = transform_cls(**transform_params)
            transforms.append(transform)
        else:
            raise ValueError(f"Invalid transform name: {transform_name}")
    
    return torchvision.transforms.Compose(transforms)

def func_feat_ext(args, pair_list, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    model = create_model(args.model_name, args.ckpt, local_dir=args.local_dir)
    model.eval()
    model.to(device)

    amp_dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }[args.amp]

    image_preprocess = load_image_preprocess(args.image_preprocess)

    for item, pair_path in enumerate(pair_list):
        slide_path = pair_path[0]
        feat_path = pair_path[1]

        process_dataset = FeatureDataset(
            slide_path=slide_path,
            image_preprocess=image_preprocess,
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
            drop_last=False,
            collate_fn=lambda batch: (torch.stack([item[0] for item in batch]), 
                                      torch.stack([item[1] for item in batch]))
        )

        os.makedirs(os.path.dirname(feat_path), exist_ok=True)

        with torch.inference_mode():
            features = []
            coordinates = []
            with tqdm(total=len(process_dataset), 
                     desc=f'GPU{gpu_id} Slide{item+1}/{len(pair_list)}',
                     position=gpu_id, 
                     ncols=90, 
                     leave=False) as pbar:
                
                slide_id = os.path.basename(slide_path)[:12]
                for batch_img, batch_coord in data_loader:
                    batch_img = batch_img.to(device, non_blocking=True)
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        patch_feat = model.encode_image(batch_img)
                    features.append(patch_feat.float())
                    coordinates.append(batch_coord)
                    pbar.update(len(batch_img))
                    pbar.set_postfix_str(f"ID:{slide_id}")

            feature_box = torch.cat(features, dim=0).cpu()
            coord_box = torch.cat(coordinates, dim=0).cpu()
            
            torch.save({
                'feats': feature_box,
                'coords': coord_box
            }, feat_path)