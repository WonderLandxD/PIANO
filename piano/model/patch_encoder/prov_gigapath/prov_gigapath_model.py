import torch
import torch.nn as nn
from torchvision import transforms
import timm

from ..base_model import BaseModel, register_model


@register_model('prov_gigapath')
class ProvGigaPathModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            model_args = {'model_name': "vit_giant_patch14_dinov2", 'img_size': 224, 'in_chans': 3, 'patch_size': 16, 'embed_dim': 1536, 'depth': 40, 'num_heads': 24, 'init_values': 1e-05, 'mlp_ratio': 5.33334, 'num_classes': 0}
            backbone = timm.create_model(**model_args, pretrained=False)
            backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
            self.backbone = backbone
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('prov_gigapath')
            self.backbone = timm.create_model(checkpoint_path, pretrained=True)
        
        preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.image_preprocess = preprocess
        self.output_dim = 1536
        
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output
    
    def get_img_token(self, x):
        output = self.backbone.forward_features(x)
        patch_token = output[:, 1:]
        class_token = output[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token} 