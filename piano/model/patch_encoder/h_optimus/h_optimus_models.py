import torch
import torch.nn as nn
from torchvision import transforms
import timm

from ..base_model import BaseModel, register_model


@register_model('h_optimus_0')
class HOptimus0Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for H-optimus 0 model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('h_optimus_0')
            
        self.backbone = timm.create_model(checkpoint_path, 
                                       pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517)
                ),
            ])
        self.image_preprocess = self.preprocess
        self.output_dim = 1536

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output

    def get_img_token(self, x):
        output = self.backbone.forward_features(x)
        patch_token = output[:, 5:]
        class_token = output[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token}
    

@register_model('h_optimus_1')
class HOptimus1Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for H-optimus 1 model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('h_optimus_1')
            
        self.backbone = timm.create_model(checkpoint_path, 
                                       pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517)
                ),
            ])
        self.image_preprocess = self.preprocess
        self.output_dim = 1536

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output

    def get_img_token(self, x):
        output = self.backbone.forward_features(x)
        patch_token = output[:, 5:]
        class_token = output[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token} 