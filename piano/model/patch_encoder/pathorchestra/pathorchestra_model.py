import torch
import torch.nn as nn
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from ..base_model import BaseModel, register_model


@register_model('pathorchestra')
class PathOrchestraModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            
            backbone = timm.create_model("vit_large_patch16_224", pretrained=False, img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
            backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
            self.backbone = backbone
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
            self.image_preprocess = preprocess
        else:
            from ..model_registry import get_model_hf_path
            self.backbone = timm.create_model("hf-hub:yf-research/PathOrchestra_V1.0.0.0", pretrained=True, init_values=1e-5, dynamic_img_size=True)
            preprocess = create_transform(**resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone))
            self.image_preprocess = preprocess
        self.output_dim = 1024
        
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output
    
    def get_img_token(self, x):
        output = self.backbone.forward_features(x)
        patch_token = output[:, 1:]
        class_token = output[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token} 

