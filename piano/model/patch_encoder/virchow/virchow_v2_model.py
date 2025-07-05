import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from ..base_model import BaseModel, register_model


@register_model('virchow_v2')
class Virchow2Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for Virchow2 model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('virchow_v2')
            from timm.layers import SwiGLUPacked
            self.backbone = timm.create_model(checkpoint_path, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            preprocess = create_transform(**resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone))

        self.image_preprocess = preprocess
        self.output_dim = 2560
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)  # size: 1 x 261 x 1280

            class_token = output[:, 0]    # size: 1 x 1280
            patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding
    
    def get_img_token(self, x):
        output = self.backbone.forward_features(x)
        patch_token = output[:, 5:]
        class_token = output[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token} 