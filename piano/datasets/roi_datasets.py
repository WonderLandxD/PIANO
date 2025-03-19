import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import yaml


# ############## 
# Dataset for ROI classification task 
# ###############
class ROIDataset(Dataset):
    """
    Dataset class for ROI classification using JSON file.
    The JSON file should contain 'train' and 'valid' splits.
    """
    def __init__(self, 
                 json_path, 
                 transform=None, 
                 mode='train',
                 transform_config=None):
        """
        Initialize ROI dataset
        
        Args:
            json_path (str): Path to JSON file containing both training and validation data
            transform (callable, optional): Transform to be applied to images
            mode (str): 'train' or 'valid' to specify which split to use
            transform_config (str, optional): Path to YAML config file for transforms
        """
        self.json_path = json_path
        print(f"\033[33m Loading Dataset ({mode} split) from {self.json_path} \033[0m")
        self.mode = mode
        
        # Load transform from config if provided
        if transform_config:
            self.transform = self.load_transform_from_config(transform_config)
        else:
            # Default transform if none provided
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]) if transform is None else transform
        
        # Load data from JSON
        self.load_data_from_json()
        
        # Create label mapping
        unique_labels = sorted(set(item['label'] for item in self.data))
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.classes = list(self.label_map.keys())
        
        print(f"Loaded {len(self.data)} samples for {mode} mode")
        print(f"Found {len(self.classes)} classes: {self.classes}")
    
    def load_transform_from_config(self, config_path):
        """Load transform from YAML config file"""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        transform_list = []
        for transform_cfg in cfg['transforms']:
            transform_name = transform_cfg['name']
            transform_params = transform_cfg.get('params', {})
            
            if hasattr(transforms, transform_name):
                transform_cls = getattr(transforms, transform_name)
                transform = transform_cls(**transform_params)
                transform_list.append(transform)
            else:
                raise ValueError(f"Invalid transform name: {transform_name}")
        
        return transforms.Compose(transform_list)
    
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
        img_path = item['image_path']
        label = item['label']
        
        # Load and process image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as placeholder
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Convert label to index
        label_idx = self.label_map[label]
        
        return {
            'image': img,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'path': img_path
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