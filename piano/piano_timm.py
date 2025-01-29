import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn.functional as F
from torchvision import transforms

import timm
from transformers import CLIPProcessor, CLIPModel


MODEL_HF_PATHS = {
    "plip": "vinid/plip",  
    "openai_clip_p16": "openai/clip-vit-base-patch16",  
    "conch_v1_0": "local_dir",
    "uni-1": "local_dir",
    "uni-2": "local_dir",
    "prov_gigapath_patch": "local_dir",
}


def get_model_hf_path(model_name):
    """
    根据模型类型返回对应的 Hugging Face 地址。
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
        # 指定模型名称为 "openai/clip-vit-base-patch32"
        super().__init__(checkpoint_path, device)


@register_model('conch_v1_0')
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
        inputs = tokenize(texts=text, tokenizer=self.tokenizer)
        return inputs.squeeze(0)

    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model.encode_image(x) # already normalized

        return output
    
    def encode_text(self, text):
        with torch.set_grad_enabled(self.model.training):
            output = self.model.encode_text(text) # already normalized
        
        return output


@register_model('uni-1')
class UNIModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        model_args = {
            'model_name': 'vit_large_patch16_224',
            'img_size': 224, 
            'patch_size': 16, 
            'init_values': 1e-5, 
            'num_classes': 0, 
            'dynamic_img_size': True
        }
        self.model = timm.create_model(**model_args)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.device = device
        self.model = self.model.to(self.device)
        self.preprocess = transforms.Compose([
                            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ])

        self.backbone = self.model
        self.image_preprocess = self.preprocess
        
    
    def forward(self, x):
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)
        return output
    

@register_model('uni-2')
class UNI2Model(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
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
        self.model = timm.create_model(
            pretrained=False, **timm_kwargs
        )
        self.model.load_state_dict(torch.load(os.path.join(checkpoint_path), map_location="cpu"), strict=True)
        self.device = device
        self.model = self.model.to(self.device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224),
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


@register_model('prov_gigapath_patch')
class ProvGigaPathModel(BaseModel):
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        model_args = {
            'model_name': "vit_giant_patch14_dinov2",
            'img_size': 224,
            'in_chans': 3,
            'patch_size': 16,
            'embed_dim': 1536,
            'depth': 40,
            'num_heads': 24,
            'init_values': 1e-05,
            'mlp_ratio': 5.33334,
            'num_classes': 0
        }
        self.model = timm.create_model(**model_args)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.device = device
        self.model = self.model.to(self.device)
        self.preprocess = transforms.Compose([
                                            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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