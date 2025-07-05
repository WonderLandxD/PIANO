import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from ..base_model import BaseModel, register_model


@register_model('beph')
class BEPHModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        from mmengine.config import Config
        from .beph_load import init_model
        if local_dir == True and checkpoint_path is not None:
            cfg = Config.fromfile(checkpoint_path)
            checkpoint_path = checkpoint_path.get('checkpoint_path', None)
        else:
            from ..model_registry import get_model_hf_path
            repo_id = get_model_hf_path('beph')
            cfg = Config.fromfile('/mnt/sdb/ljw/PIANO-Custom/piano/beph_backbone.py')
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="BEPH_backbone.pth")
        self.backbone = init_model(cfg, checkpoint=checkpoint_path, device='cpu')
        print(f'\033[32mSuccessfully loaded BEPH from {checkpoint_path}\033[0m')
        self.output_dim = 768
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone.extract_feat(x)[0]
        return output

    def get_img_token(self, x):
        raise NotImplementedError("BEPH model does not support image token extraction") 