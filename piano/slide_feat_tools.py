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


class create_feat_datasests(torch.utils.data.Dataset):
    def __init__(self, slide_path, preprocess):
        self.slide_path = slide_path
        self.preprocess = preprocess
        self.patch_paths = glob.glob(os.path.join(self.slide_path, '*.jpg'))
        self.patch_paths = sorted(self.patch_paths, key=self.extract_number)

    def extract_number(self, file_path):
        match = re.search(r'no(\d+)', os.path.basename(file_path))
        if match:
            return int(match.group(1))  
        return 0  

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        image = Image.open(patch_path).convert('RGB')
        image_tensor = self.preprocess(image)
        return image_tensor

    def __len__(self):
        return len(self.patch_paths)

def func_feat_ext(args, pair_list, gpu_id):

    device = torch.device(f"cuda:{gpu_id}")
    model, preprocess, _ = create_piano(args.model_name, args.ckpt, device=device)
    model.set_mode('eval')

    progress_bar = tqdm(total=len(pair_list), desc=f'GPU {gpu_id}', position=gpu_id, ncols=75, leave=True)

    for item, pair_path in enumerate(pair_list):
        slide_path = pair_path[0]
        feat_path = pair_path[1]

        progress_bar.set_description(f'GPU {gpu_id} ({item+1}/{len(pair_list)})')
        progress_bar.refresh()

        process_dataset = create_feat_datasests(slide_path=slide_path, preprocess=preprocess)
        data_loader = torch.utils.data.DataLoader(process_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        total_samples = len(process_dataset)
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)

        with torch.inference_mode():
            features = []
            for batch_img in data_loader:
                batch_img = batch_img.to(device, non_blocking=True)
                patch_feat = model.encode_image(batch_img)
                features.append(patch_feat)

            # feature_box = np.concatenate(features, axis=0)
            feature_box = torch.cat(features, dim=0).cpu() 
            torch.save(feature_box, feat_path)

        progress_bar.update(1)


    progress_bar.close()
    
