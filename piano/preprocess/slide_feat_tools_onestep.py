# One-step feature extraction from the raw WSI, Shape from WSI -> [N, C]

import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data
import torchvision
import cv2
from tqdm import tqdm

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

try:
    import opensdpc
    OPENSDPC_AVAILABLE = True
except ImportError:
    OPENSDPC_AVAILABLE = False


def get_bg_mask(thumbnail, kernel_size=1):
    """Generate background mask using OTSU threshold"""
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    ret, threshold = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)
    
    if kernel_size > 1:
        close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(threshold), cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
        return (image_open / 255.0).astype(np.uint8)
    else:
        return (threshold / 255.0).astype(np.uint8)


def generate_patch_coordinates(img_x, img_y, x_size, y_size, x_overlap, y_overlap, bg_mask, blank_TH):
    """Generate valid patch coordinates based on background mask"""
    x_size_minus_overlap = x_size - x_overlap
    y_size_minus_overlap = y_size - y_overlap
    bg_mask_height, bg_mask_width = bg_mask.shape[0], bg_mask.shape[1]

    coordinates = []
    x_steps = int(np.floor((img_x - x_size) / x_size_minus_overlap + 1))
    y_steps = int(np.floor((img_y - y_size) / y_size_minus_overlap + 1))
    
    for i in range(x_steps):
        x_start_mask = int(np.floor(i * x_size_minus_overlap / img_x * bg_mask_width))
        x_end_mask = int(np.ceil((i * x_size_minus_overlap + x_size) / img_x * bg_mask_width))
        
        for j in range(y_steps):
            y_start_mask = int(np.floor(j * y_size_minus_overlap / img_y * bg_mask_height))
            y_end_mask = int(np.ceil((j * y_size_minus_overlap + y_size) / img_y * bg_mask_height))
            
            # Ensure indices are within bounds
            y_start_mask = max(0, min(y_start_mask, bg_mask_height - 1))
            y_end_mask = max(1, min(y_end_mask, bg_mask_height))
            x_start_mask = max(0, min(x_start_mask, bg_mask_width - 1))
            x_end_mask = max(1, min(x_end_mask, bg_mask_width))
            
            mask = bg_mask[y_start_mask:y_end_mask, x_start_mask:x_end_mask]
            
            if mask.size > 0 and np.sum(mask == 0) / mask.size < blank_TH:
                coordinates.append((i, j))
                
    return coordinates


class SimpleWSIPatchDataset(torch.utils.data.Dataset):
    """Simple dataset for WSI patches"""
    
    def __init__(self, slide_path, coordinates, wsi_level, patch_w, patch_h, overlap_w, overlap_h):
        self.slide_path = slide_path
        self.coordinates = coordinates
        self.wsi_level = wsi_level
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.overlap_w = overlap_w
        self.overlap_h = overlap_h
        self._slide = None
        self._level_downsample = None
        
        # Simple preprocessing
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _get_slide(self):
        """Open slide object"""
        if self._slide is None:
            if OPENSDPC_AVAILABLE:
                try:
                    self._slide = opensdpc.OpenSdpc(self.slide_path)
                except:
                    pass
            
            if self._slide is None and OPENSLIDE_AVAILABLE:
                self._slide = openslide.OpenSlide(self.slide_path)
                
            self._level_downsample = self._slide.level_downsamples[self.wsi_level]
        
        return self._slide

    def __getitem__(self, idx):
        slide = self._get_slide()
        i, j = self.coordinates[idx]
        
        # Calculate level 0 coordinates
        x_size_l0 = int(self.patch_w * self._level_downsample)
        y_size_l0 = int(self.patch_h * self._level_downsample)
        x_overlap_l0 = int(self.overlap_w * self._level_downsample)
        y_overlap_l0 = int(self.overlap_h * self._level_downsample)
        
        x_start = int(i * (x_size_l0 - x_overlap_l0))
        y_start = int(j * (y_size_l0 - y_overlap_l0))
        
        # Read patch
        image = slide.read_region((x_start, y_start), self.wsi_level, 
                                 (self.patch_w, self.patch_h)).convert('RGB')
        
        # Pad if necessary
        if image.size != (self.patch_w, self.patch_h):
            padded_image = Image.new('RGB', (self.patch_w, self.patch_h), (255, 255, 255))
            padded_image.paste(image, (0, 0))
            image = padded_image
        
        processed_image = self.preprocess(image)
        coordinates = torch.tensor([x_start, y_start], dtype=torch.long)
        
        return processed_image, coordinates

    def __len__(self):
        return len(self.coordinates)


def extract_wsi_features(slide_path, feat_path, model, device, batch_size=64, 
                        patch_size=256, overlap=0, wsi_level=0, blank_TH=0.1, 
                        kernel_size=5, save_patches=False, patch_save_dir=None):
    """
    Extract features from a single WSI
    
    Args:
        slide_path: Path to WSI file
        feat_path: Path to save features
        model: Feature extraction model
        device: Computing device
        batch_size: Batch size for feature extraction
        patch_size: Size of patches
        overlap: Overlap between patches
        wsi_level: WSI pyramid level to use
        blank_TH: Threshold for blank region filtering
        kernel_size: Kernel size for morphological operations
        save_patches: Whether to save patch images
        patch_save_dir: Directory to save patch images
    
    Returns:
        Number of patches processed
    """
    
    
    # Create patch save directory if needed
    if save_patches and patch_save_dir is not None:
        os.makedirs(patch_save_dir, exist_ok=True)
        slide_name = os.path.basename(slide_path).split('.')[0]
        print(f"Will save patches to: {patch_save_dir}")
    
    # Open slide
    slide = None
    if OPENSDPC_AVAILABLE:
        try:
            slide = opensdpc.OpenSdpc(slide_path)
        except:
            pass
    
    if slide is None and OPENSLIDE_AVAILABLE:
        slide = openslide.OpenSlide(slide_path)
    
    if slide is None:
        raise ValueError(f"Cannot open slide {slide_path}")
    
    # Get slide properties
    thumbnail_level = slide.level_count - 1
    thumbnail = np.array(slide.read_region((0, 0), thumbnail_level, 
                                          slide.level_dimensions[thumbnail_level]).convert('RGB'))
    img_x, img_y = slide.level_dimensions[0]
    level_downsample = slide.level_downsamples[wsi_level]
    
    # Generate background mask
    black_pixel = np.where((thumbnail[:, :, 0] < 50) & 
                          (thumbnail[:, :, 1] < 50) & 
                          (thumbnail[:, :, 2] < 50))
    thumbnail[black_pixel] = [255, 255, 255]
    thumbnail_bgr = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
    bg_mask = get_bg_mask(thumbnail_bgr, kernel_size=kernel_size)

    # Calculate patch parameters
    x_size_l0 = int(patch_size * level_downsample)
    y_size_l0 = int(patch_size * level_downsample)
    x_overlap_l0 = int(overlap * level_downsample)
    y_overlap_l0 = int(overlap * level_downsample)
    
    # Generate coordinates
    coordinates = generate_patch_coordinates(img_x, img_y, x_size_l0, y_size_l0, 
                                           x_overlap_l0, y_overlap_l0, bg_mask, blank_TH)
    
    slide.close()
    
    if not coordinates:
        print(f"No valid patches found for {slide_path}")
        return 0

    # Create dataset and dataloader
    dataset = SimpleWSIPatchDataset(
        slide_path=slide_path,
        coordinates=coordinates,
        wsi_level=wsi_level,
        patch_w=patch_size,
        patch_h=patch_size,
        overlap_w=overlap,
        overlap_h=overlap
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Extract features
    features = []
    coordinates_list = []
    patch_count = 0
    
    print(f"Processing {len(coordinates)} patches...")
    with torch.inference_mode():
        for batch_img, batch_coord in tqdm(dataloader, total=len(dataloader), desc='Extracting features', ncols=100):
            batch_img = batch_img.to(device)
            
            # Save patches if requested
            if save_patches and patch_save_dir is not None:
                generate_patches(batch_img, batch_coord, slide_name, patch_save_dir, patch_count)
            
            # Extract features
            patch_feat = model.encode_image(batch_img)
            
            features.append(patch_feat.cpu())
            coordinates_list.append(batch_coord)
    
    # Combine results and save
    if features:
        feature_box = torch.cat(features, dim=0)
        coord_box = torch.cat(coordinates_list, dim=0)
        
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        torch.save({
            'feats': feature_box.detach().cpu(),
            'coords': coord_box
        }, feat_path)
        
        print(f"Saved features to {feat_path} | Shape: {feature_box.shape}")
        
        if save_patches and patch_save_dir is not None:
            print(f"Saved {patch_count} patch images to {patch_save_dir}")
    
    return len(coordinates) 


def generate_patches(batch_img, batch_coord, slide_name, patch_save_dir, patch_count):
    for i, img_tensor in enumerate(batch_img):
        # Convert tensor back to PIL image
        # Denormalize first
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img_tensor.cpu() * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to PIL and save
        img_pil = torchvision.transforms.ToPILImage()(img_denorm)
        coord_x, coord_y = batch_coord[i]
        patch_filename = f"no{patch_count + i:06d}_{coord_x:09d}x_{coord_y:09d}y.jpg"
        img_pil.save(os.path.join(patch_save_dir, patch_filename))
    
    patch_count += len(batch_img)