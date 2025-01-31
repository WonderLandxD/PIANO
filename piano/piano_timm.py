import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

import importlib.util
import os


import torch
import torch.nn.functional as F
from torchvision import transforms

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from transformers import CLIPProcessor, CLIPModel
from transformers import XLMRobertaTokenizer


MODEL_HF_PATHS = {
    "plip": "vinid/plip",  
    "openai_clip_p16": "openai/clip-vit-base-patch16",  
    "conch_v1": "local_dir",
    "uni_v1": "hf-hub:MahmoodLab/uni",
    "uni_v2": "hf-hub:MahmoodLab/UNI2-h",
    "prov_gigapath_tile": "hf_hub:prov-gigapath/prov-gigapath",
    "virchow_v1": "hf-hub:paige-ai/Virchow",
    "virchow_v2": "hf-hub:paige-ai/Virchow2",
    "musk": "hf_hub:xiangjx/musk",
    "h_optimus_0": "hf-hub:bioptimus/H-optimus-0",
}


def get_model_hf_path(model_name):
    """
    return the huggingface path for the model
    """
    if model_name not in MODEL_HF_PATHS:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_HF_PATHS.keys())}")
    return MODEL_HF_PATHS[model_name]



MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


class BaseModel:
    def __init__(self):
        self.image_preprocess = None
        self.text_preprocess = self._default_text_preprocess 

    def _default_text_preprocess(self, text):
        raise NotImplementedError("This model does not support text preprocessing.")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")

    def encode_image(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def encode_text(self, text):
        return NotImplementedError("This model does not support text encoding.")
    
    def set_mode(self, mode):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ValueError(f"Invalid mode {mode}. Use 'train' or 'eval'.")


@register_model('plip')
class PLIPModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(checkpoint_path)  # "vinid/plip"
        self.processor = CLIPProcessor.from_pretrained(checkpoint_path)
        self.device = device
        self.model = self.model.to(self.device)

        self.backbone = self.model
        self.image_preprocess = self._image_preprocess
        self.text_preprocess = self._text_preprocess
    
    def _image_preprocess(self, image):
        inputs = self.processor(images=image, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)
    
    def _text_preprocess(self, text):
        inputs = self.processor(text=text, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        return inputs['input_ids'].squeeze(0)
    
    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model.get_image_features(pixel_values=x)
            output = F.normalize(output, dim=-1)
        return output
    
    def encode_text(self, text):
        with torch.set_grad_enabled(self.model.training):
            output = self.model.get_text_features(input_ids=text)
            output = F.normalize(output, dim=-1)
        return output


@register_model("openai_clip_p16")
class OpenAICLIPModel(PLIPModel):
    def __init__(self, checkpoint_path, device="cpu"):
        # "openai/clip-vit-base-patch32"
        super().__init__(checkpoint_path, device)


@register_model('conch_v1')
class CONCHModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        self.model, self.preprocess = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=checkpoint_path)
        self.tokenizer = get_tokenizer()
        self.device = device
        self.model = self.model.to(self.device)

        self.backbone = self.model
        self.image_preprocess = self.preprocess
        self.text_preprocess = self._text_preprocess
    
    def _text_preprocess(self, text):
        from conch.open_clip_custom import tokenize
        inputs = tokenize(texts=text, tokenizer=self.tokenizer) # [1, 128]
        return inputs.squeeze(0)

    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model.encode_image(x) # already normalized

        return output
    
    def encode_text(self, text):
        with torch.set_grad_enabled(self.model.training):
            output = self.model.encode_text(text) # already normalized
        
        return output


@register_model('uni_v1')
class UNIModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        self.model = timm.create_model(checkpoint_path, 
                                       pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.device = device
        self.model = self.model.to(self.device)
        self.preprocess = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

        self.backbone = self.model
        self.image_preprocess = self.preprocess
        
    
    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)
        return output
    

@register_model('uni_v2')
class UNI2Model(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
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
        self.model = timm.create_model(checkpoint_path, pretrained=True, **timm_kwargs)
        self.preprocess = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

        self.device = device
        self.model = self.model.to(self.device)
        self.backbone = self.model
        self.image_preprocess = self.preprocess
    
    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)
        return output


@register_model('prov_gigapath_tile')
class ProvGigaPathModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()

        self.model = timm.create_model(checkpoint_path, pretrained=True)
        self.device = device
        self.model = self.model.to(self.device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.backbone = self.model
        self.image_preprocess = self.preprocess
        
    
    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)
        return output
    

@register_model('virchow_v1')
class VirchowModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        # need to specify MLP layer and activation function for proper init
        self.model = timm.create_model(checkpoint_path, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        self.device = device
        self.model = self.model.to(self.device)

        self.preprocess = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        self.backbone = self.model
        self.image_preprocess = self.preprocess

    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)  # size: 1 x 257 x 1280

            class_token = output[:, 0]    # size: 1 x 1280
            patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1) # size: 1 x 2560
        return embedding


@register_model('virchow_v2')
class Virchow2Model(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__(checkpoint_path, device)
    
    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)  # size: 1 x 261 x 1280

            class_token = output[:, 0]    # size: 1 x 1280
            patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding


@register_model('musk')
class MuskModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        from musk import utils
        from timm.models import create_model
        model = create_model("musk_large_patch16_384")
        utils.load_model_and_may_interpolate(checkpoint_path, model, 'model|module', '')
        self.model = model.to(device)
        self.preprocess = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.backbone = self.model
        self.image_preprocess = self.preprocess

        musk_spec = importlib.util.find_spec("musk")
        if musk_spec is None:
            raise ImportError("musk package is not installed.")
        
        musk_path = os.path.dirname(musk_spec.submodule_search_locations[0])
        tokenizer_path = os.path.join(musk_path, "musk", "models", "tokenizer.spm")
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
        with torch.set_grad_enabled(self.model.training):
            output = self.model(
                image=x,
                with_head=False,
                out_norm=True,
                ms_aug=True,
                return_global=True  
                )[0]
        return output
    
    def encode_text(self, text):
        with torch.set_grad_enabled(self.model.training):
            padding_mask = (text == self.tokenizer.pad_token_id).long()
            text_embeddings = self.model(
                text_description=text,
                padding_mask=padding_mask,
                with_head=False, 
                out_norm=True,
                ms_aug=False,
                return_global=True 
                )[1]
        
        return text_embeddings


@register_model('h_optimus_0')
class HOptimusModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        self.model = timm.create_model(checkpoint_path, 
                                       pretrained=True, init_values=1e-5, dynamic_img_size=False)
        self.device = device
        self.model = self.model.to(self.device)
        self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517)
                ),
            ])
        self.backbone = self.model
        self.image_preprocess = self.preprocess

    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)
        return output



def create_piano(model_name, checkpoint_path, device='cpu'):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]

    model = model_class(checkpoint_path, device)

    image_preprocess = model.image_preprocess
    text_preprocess = model.text_preprocess

    return model, image_preprocess, text_preprocess