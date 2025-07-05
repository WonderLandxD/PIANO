import importlib.util
import os
import torch
import torch.nn as nn
from torchvision import transforms

from ..base_model import BaseModel, register_model


@register_model('musk')
class MuskModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for Musk model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('musk')
            
        from musk import utils, modeling
        from timm.models import create_model
        model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate(checkpoint_path, model, 'model|module', '')
        self.backbone = model
        self.preprocess = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.image_preprocess = self.preprocess
        self.output_dim = 1024

        musk_spec = importlib.util.find_spec("musk")
        if musk_spec is None:
            raise ImportError("musk package is not installed.")
        
        musk_path = os.path.dirname(musk_spec.submodule_search_locations[0])
        tokenizer_path = os.path.join(musk_path, "musk", "models", "tokenizer.spm")
        from transformers import XLMRobertaTokenizer
        self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
    
    def _text_preprocess(self, text):
        from musk import utils  

        if isinstance(text_list, str):  
            text_list = [text_list]

        text_ids = []
        for txt in text_list:
            txt_ids, _ = utils.xlm_tokenizer(txt, self.tokenizer, max_len=100)
            text_ids.append(torch.tensor(txt_ids).unsqueeze(0)) 

        text_ids = torch.cat(text_ids, dim=0) 
        return text_ids
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(
                image=x,
                with_head=False,
                out_norm=True,
                ms_aug=True,
                return_global=True  
                )[0]
        return output
    
    def encode_text(self, text):
        with torch.set_grad_enabled(self.backbone.training):
            padding_mask = (text == self.tokenizer.pad_token_id).long()
            text_embeddings = self.backbone(
                text_description=text,
                padding_mask=padding_mask,
                with_head=False, 
                out_norm=True,
                ms_aug=False,
                return_global=True 
                )[1]
        
        return text_embeddings

    def get_img_token(self, x):
        output = self.backbone(image=x, with_head=False, out_norm=False, ms_aug=False, return_global=False)[0]
        class_token = output[:, 0, :]
        patch_token = output[:, 1:, :]
        return {"patch_tokens": patch_token, "class_token": class_token} 