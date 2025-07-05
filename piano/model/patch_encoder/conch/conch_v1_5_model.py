import torch
import torch.nn as nn
from transformers import AutoModel

from ..base_model import BaseModel, register_model


@register_model('conch_v1_5')
class CONCHV1_5Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            pass
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('conch_v1_5')
        
        self.titan = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)
        conch, eval_transform = self.titan.return_conch()
        self.backbone = conch
        self.image_preprocess = eval_transform

        self.output_dim = 768
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output 