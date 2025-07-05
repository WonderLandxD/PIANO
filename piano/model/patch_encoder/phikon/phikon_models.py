import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, ViTModel

from ..base_model import BaseModel, register_model


@register_model('phikon_v1')
class PhikonV1Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for phikon_v1 model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('phikon_v1')

        self.preprocess = AutoImageProcessor.from_pretrained("owkin/phikon")
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

        def phikonprocessor(image):
            output = self.preprocess(image, return_tensor="pt")
            return torch.from_numpy(output['pixel_values'][0])
        
        self.image_preprocess = phikonprocessor
        self.output_dim = 768


    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            outputs = self.backbone(x)
            outputs = outputs.last_hidden_state[:, 0, :]
        return outputs

    def get_img_token(self, x):
        output = self.backbone(x)
        patch_token = output.last_hidden_state[:, 1:]
        class_token = output.last_hidden_state[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token}
    


@register_model('phikon_v2')
class PhikonV2Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for phikon_v2 model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('phikon_v2')

        self.backbone = AutoModel.from_pretrained(checkpoint_path)
        self.preprocess = AutoImageProcessor.from_pretrained(checkpoint_path)

        def phikonprocessor(image):
            output = self.preprocess(image, return_tensor="pt")
            return torch.from_numpy(output['pixel_values'][0])
        
        self.image_preprocess = phikonprocessor
        self.output_dim = 1024


    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            outputs = self.backbone(x)
            outputs = outputs.last_hidden_state[:, 0, :]
        return outputs

    def get_img_token(self, x):
        output = self.backbone(x)
        patch_token = output.last_hidden_state[:, 1:]
        class_token = output.last_hidden_state[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token} 