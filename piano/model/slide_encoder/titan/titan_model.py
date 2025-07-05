import torch
import torch.nn as nn
from transformers import AutoModel


class TITANModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.slide_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    
    def forward(self, input):
        patch_features = input['feats']
        coordinates = input['coords']
        patch_size_lv0 = 1024
        output = self.slide_model.encode_slide_from_patch_features(patch_features, coordinates, patch_size_lv0)
        return output 