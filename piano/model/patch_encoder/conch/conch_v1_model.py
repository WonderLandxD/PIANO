import torch
import torch.nn as nn

from ..base_model import BaseModel, register_model


@register_model('conch_v1')
class CONCHModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            pass
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('conch_v1')
            
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        self.backbone, self.preprocess = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=checkpoint_path)
        self.tokenizer = get_tokenizer()

        self.image_preprocess = self.preprocess
        self.text_preprocess = self._text_preprocess
        self.output_dim = 512
    
    def _text_preprocess(self, text):
        from conch.open_clip_custom import tokenize
        inputs = tokenize(texts=text, tokenizer=self.tokenizer) # [1, 128]
        return inputs.squeeze(0)

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone.encode_image(x) # already normalized
        return output
    
    def encode_text(self, text):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone.encode_text(text) # already normalized
        return output
    
    def get_img_token(self, x):
        raise NotImplementedError("Conch model does not support image token extraction") 