import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import os
class GigaPathModel(nn.Module):
    def __init__(self, local_dir: str = None):
        super().__init__()

        from .GigaPath.slide_encoder import create_model

        if local_dir is None:
            slide_path = hf_hub_download(repo_id="prov-gigapath/prov-gigapath", filename="slide_encoder.pth")
        else:
            slide_path = os.path.join(local_dir, "slide_encoder.pth")
        
        self.slide_model = create_model(slide_path, "gigapath_slide_enc12l768d", 1536)
    
    def forward(self, input):
        patch_features = input['feats']
        coordinates = input['coords']
        output = self.slide_model(patch_features, coordinates)[0]
        return output 