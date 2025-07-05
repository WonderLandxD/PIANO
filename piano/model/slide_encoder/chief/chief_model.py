import torch
import torch.nn as nn
import os
from huggingface_hub import hf_hub_download

class CHIEFModel(nn.Module):
    def __init__(self, local_dir: str = None):
        super().__init__()

        from .CHIEF.CHIEF import CHIEF

        if local_dir is None:
            word_embedding_path = hf_hub_download(repo_id="JWonderLand/CHIEF_unofficial", filename="Text_emdding.pth")
            td = torch.load(hf_hub_download(repo_id="JWonderLand/CHIEF_unofficial", filename="CHIEF_pretraining.pth"))
        else:
            word_embedding_path = os.path.join(local_dir, 'Text_emdding.pth')
            td = torch.load(os.path.join(local_dir, 'CHIEF_pretraining.pth'))
            
        self.slide_model = CHIEF(size_arg='small', word_embedding_path=word_embedding_path, dropout=True, n_classes=2)
        self.slide_model.load_state_dict(td, strict=True)
    
    def forward(self, input):
        patch_features = input['feats']
        anatomical=13
        result = self.slide_model(patch_features, anatomical)
        output = result['WSI_feature'] 
        return output 