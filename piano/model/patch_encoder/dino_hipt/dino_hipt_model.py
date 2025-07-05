import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from ..base_model import BaseModel, register_model


@register_model('dino_hipt')
class HIPTDinoModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            vit256_path = checkpoint_path.get('vit256_path', None)
            vit4k_path = checkpoint_path.get('vit4k_path', None)
        else:
            from ..model_registry import get_model_hf_path
            repo_id = get_model_hf_path('dino_hipt')
            vit256_path = hf_hub_download(repo_id=repo_id, filename="vit256_small_dino.pth")
            vit4k_path = hf_hub_download(repo_id=repo_id, filename="vit4k_xs_dino.pth")
        
        from .HIPT_4K.hipt_4k import HIPT_4K
        
        self.backbone = HIPT_4K(
            model256_path=vit256_path, 
            model4k_path=vit4k_path, 
            device256=torch.device('cpu'), 
            device4k=torch.device('cpu')
        )
        

        self.image_preprocess = None
        self.output_dim = 384
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone.model256(x)
        return output
    
    def get_img_token(self, x):

        with torch.set_grad_enabled(self.backbone.training):

            features = self.backbone.model256.forward_features(x) if hasattr(self.backbone.model256, 'forward_features') else self.backbone.model256(x)
            
            if isinstance(features, torch.Tensor) and len(features.shape) >= 3:
                class_token = features[:, 0]  
                patch_tokens = features[:, 1:]  
                return {"patch_tokens": patch_tokens, "class_token": class_token}
            else:
                return {"patch_tokens": features, "class_token": features}
