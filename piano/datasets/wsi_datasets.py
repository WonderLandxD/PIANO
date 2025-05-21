import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import yaml
import os


# ############## 
# Dataset for WSI classification task 
# ###############
class WSIDataset(Dataset):
    """
    Dataset class for WSI classification using JSON file.
    The JSON file should contain 'train' and 'valid' splits.
    """
    def __init__(self, 
                 json_path, 
                 mode='train',
                 pfm_name=None,
                 few_shot=None
                 ):
        """
        Initialize WSI dataset
        
        Args:
            json_path (str): Path to JSON file containing both training and validation data
            mode (str): 'train' or 'valid' to specify which split to use
            pfm_name (str): Name of the PFM model
            few_shot (int, optional): Number of samples per class for few-shot learning
        """
        self.json_path = json_path
        print(f"\033[33m Loading Dataset ({mode} split) from {self.json_path} \033[0m")
        self.mode = mode
        self.pfm_name = pfm_name
        self.few_shot = few_shot
        
        # Load data from JSON
        self.load_data_from_json()
        
        # Create label mapping
        unique_labels = sorted(set(item['label'] for item in self.data))
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.classes = list(self.label_map.keys())
        
        # Apply few-shot sampling if specified and in training mode
        if few_shot is not None and mode == 'train':
            self.data = self.sample_few_shot_data()
            print(f"Few-shot learning with {few_shot} samples per class")
        
        print(f"Loaded {len(self.data)} samples for {mode} mode")
        print(f"Found {len(self.classes)} classes: {self.classes}")
    
    def load_data_from_json(self):
        """Load data from JSON file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            if self.mode not in data_dict:
                raise KeyError(f"Mode '{self.mode}' not found in JSON file")
            
            self.data = data_dict[self.mode]
            
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            self.data = []
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        item = self.data[idx]
        feat_path = item['feat_path'].replace("<PFM_NAME>", self.pfm_name)
        label = item['label']
        
        # Load and process image
        try:
            feat = torch.load(feat_path, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f"Error loading image {feat_path}: {e}")
            # Return a black feature as placeholder
            feat = torch.zeros(1, 1024)
        
        # Convert label to index
        label_idx = self.label_map[label]
        
        return {
            'feat': feat['feats'],
            'coords': feat['coords'],
            'label': torch.tensor(label_idx, dtype=torch.long),
            'path': feat_path
        }
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalanced datasets"""
        class_counts = {cls: 0 for cls in self.label_map}
        for item in self.data:
            class_counts[item['label']] += 1
        
        # Calculate weights (1/frequency)
        weights = [1.0 / max(class_counts[cls], 1) for cls in self.classes]
        # Normalize weights
        total = sum(weights)
        weights = [w / total * len(weights) for w in weights]
        
        return torch.tensor(weights, dtype=torch.float)
    
    def get_classes(self):
        """Return list of classes"""
        return self.classes
    
    def get_label_map(self):
        """Return label mapping"""
        return self.label_map
    
    def get_name(self, idx):
        """
        Extract WSI name from the feature path.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            str: WSI name extracted from path
        """
        if idx >= len(self.data):
            return f"Unknown_{idx}"
            
        item = self.data[idx]
        feat_path = item['feat_path'].replace("<PFM_NAME>", self.pfm_name)
        
        # Extract WSI name from path
        # Format: .../piano_WSI_NAME_pfm.pth
        try:
            # Get the base filename
            base_name = os.path.basename(feat_path)
            # Remove the extension
            base_name = os.path.splitext(base_name)[0]
            # Extract WSI name - assuming format 'piano_WSINAME_pfmname'
            if base_name.startswith('piano_'):
                parts = base_name.split('_')
                if len(parts) >= 3:
                    # Join all parts except the first (piano) and the last (pfm name)
                    wsi_name = '_'.join(parts[1:-1])
                    return wsi_name
            
            # If we couldn't parse it properly, return the full base name
            return base_name
        except:
            return f"WSI_{idx}"
    
    def sample_few_shot_data(self):
        """Sample few-shot data from the dataset"""
        import random
        few_shot_data = []
        # Group data by label
        label_data = {}
        for item in self.data:
            label = item['label']
            if label not in label_data:
                label_data[label] = []
            label_data[label].append(item)
        
        # Sample few-shot data for each class
        for label in self.classes:
            if label in label_data:
                samples = label_data[label]
                # If we have fewer samples than few_shot, use all samples
                n_samples = min(self.few_shot, len(samples))
                # Random sampling without replacement
                sampled = random.sample(samples, n_samples)
                few_shot_data.extend(sampled)
        
        return few_shot_data 