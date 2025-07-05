import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel, register_model


@register_model('plip')
class PLIPModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            pass
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('plip')
        from transformers import CLIPProcessor, CLIPModel
        self.backbone = CLIPModel.from_pretrained(checkpoint_path)
        self.processor = CLIPProcessor.from_pretrained(checkpoint_path)

        self.image_preprocess = self._image_preprocess
        self.text_preprocess = self._text_preprocess
        self.output_dim = 512
    
    def _image_preprocess(self, image):
        inputs = self.processor(images=image, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)
    
    def _text_preprocess(self, text):
        inputs = self.processor(text=text, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        return inputs['input_ids'].squeeze(0)
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone.get_image_features(pixel_values=x)
            output = F.normalize(output, dim=-1)
        return output
    
    def encode_text(self, text):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone.get_text_features(input_ids=text)
            output = F.normalize(output, dim=-1)
        return output
    
    def get_img_token(self, x):
        """
        Output patch tokens from the image encoder
        """
        with torch.set_grad_enabled(self.backbone.training):
            # get the output of the image encoder
            visual_outputs = self.backbone.vision_model(pixel_values=x)
            # extract patch tokens and class token
            patch_tokens = visual_outputs.last_hidden_state[:, 1:]
            class_token = visual_outputs.last_hidden_state[:, 0]
        return {"patch_tokens": patch_tokens, "class_token": class_token}


@register_model("openai_clip_p16")
class OpenAICLIPModel(PLIPModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            pass
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('openai_clip_p16')
        
        super().__init__(checkpoint_path)
        self.output_dim = 512 