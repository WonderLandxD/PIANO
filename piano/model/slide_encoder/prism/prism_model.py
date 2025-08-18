import torch
import torch.nn as nn
from transformers import AutoModel


class PRISMModel(nn.Module):
    def __init__(self):
        super().__init__()
        local_dir = '/mnt/sdb/ljw/hf_cache/models--paige-ai--Prism/snapshots/adeb3376909b4fb3fba5a5243a1df05d42d57fdd'
        # self.slide_model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
        self.slide_model = AutoModel.from_pretrained(local_dir, trust_remote_code=True)
    
    def forward(self, input):
        patch_features = input['feats']
        result = self.slide_model.slide_representations(patch_features)
        output = result['image_embedding']
        return output 