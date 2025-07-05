import torch
import torch.nn as nn
from .madeleine.factory import create_model_from_pretrained


class MADELEINEModel(nn.Module):
    def __init__(self, local_dir: str = None):
        super().__init__()
        
        # Create the MADELEINE model using the factory function
        self.slide_model = create_model_from_pretrained(local_dir=local_dir)
    
    def forward(self, input):
        # Get the features from input
        feats = input['feats'] 
        
        # Use the encode_he method as specified by the user
        output = self.slide_model.encode_he(feats)
        return output 