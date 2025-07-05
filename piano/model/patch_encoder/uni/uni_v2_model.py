import torch
import torch.nn as nn
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from ..base_model import BaseModel, register_model


@register_model('uni_v2')
class UNI2Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        from timm.layers import SwiGLUPacked
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        if local_dir == True and checkpoint_path is not None:
            backbone = timm.create_model('vit_giant_patch14_224', pretrained=False, **timm_kwargs)
            backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
            self.backbone = backbone
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('uni_v2')
            self.backbone = timm.create_model(checkpoint_path, pretrained=True, **timm_kwargs)
            preprocess = create_transform(**resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone))
        
        self.preprocess = preprocess
        self.image_preprocess = self.preprocess
        self.output_dim = 1536
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output
    
    def get_img_token(self, x):
        output = self.backbone.forward_features(x)
        patch_token = output[:, 9:]
        class_token = output[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token} 