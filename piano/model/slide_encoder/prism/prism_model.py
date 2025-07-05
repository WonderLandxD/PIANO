import torch
import torch.nn as nn
from transformers import AutoModel


class PRISMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.slide_model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
    
    def forward(self, input):
        patch_features = input['feats']
        result = self.slide_model.slide_representations(patch_features)
        output = result['image_embedding']
        return output 