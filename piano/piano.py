import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

import importlib.util
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform





MODEL_HF_PATHS = {
    "plip": "vinid/plip",  
    "openai_clip_p16": "openai/clip-vit-base-patch16",  
    "conch_v1": "hf_hub:MahmoodLab/conch",
    "uni_v1": "hf-hub:MahmoodLab/uni",
    "uni_v2": "hf-hub:MahmoodLab/UNI2-h",
    "prov_gigapath": "hf_hub:prov-gigapath/prov-gigapath",
    "virchow_v1": "hf-hub:paige-ai/Virchow",
    "virchow_v2": "hf-hub:paige-ai/Virchow2",
    "musk": "hf_hub:xiangjx/musk",
    "h_optimus_0": "hf-hub:bioptimus/H-optimus-0",
    "ctranspath": "YOUR_LOCAL_PATH",
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


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_preprocess = self._default_image_preprocess
        self.text_preprocess = self._default_text_preprocess

    def _default_image_preprocess(self, image):
        raise NotImplementedError("This model does not support image preprocessing.")

    def _default_text_preprocess(self, text): 
        raise NotImplementedError("This model does not support text preprocessing.")
    
    def forward(self, x):
        return self.backbone(x)

    def encode_image(self, x):
        return self(x)
    
    def encode_text(self, text):
        if not hasattr(self, '_text_preprocess'):
            raise NotImplementedError("This model does not support text encoding.")
        return self(text)


@register_model('plip')
class PLIPModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            pass
        else:
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
            checkpoint_path = get_model_hf_path('openai_clip_p16')
        
        super().__init__(checkpoint_path)
        self.output_dim = 512


@register_model('conch_v1')
class CONCHModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            pass
        else:
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


@register_model('uni_v1')
class UNIModel(BaseModel):
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
            checkpoint_path = get_model_hf_path('uni_v1')
            self.backbone = timm.create_model(checkpoint_path, pretrained=True, init_values=1e-5, dynamic_img_size=True)
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

    

@register_model('virchow_v1')
class VirchowModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for Virchow model")
        else:
            checkpoint_path = get_model_hf_path('virchow_v1')
            from timm.layers import SwiGLUPacked
            self.backbone = timm.create_model(checkpoint_path, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            preprocess = create_transform(**resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone))
            
        self.image_preprocess = preprocess
        self.output_dim = 2560

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)  # size: 1 x 257 x 1280

            class_token = output[:, 0]    # size: 1 x 1280
            patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1) # size: 1 x 2560
        return embedding
    
    def get_img_token(self, x):
        output = self.backbone.forward_features(x)
        patch_token = output[:, 5:]
        class_token = output[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token}


@register_model('virchow_v2')
class Virchow2Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for Virchow2 model")
        else:
            checkpoint_path = get_model_hf_path('virchow_v2')
            from timm.layers import SwiGLUPacked
            self.backbone = timm.create_model(checkpoint_path, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            preprocess = create_transform(**resolve_data_config(self.backbone.pretrained_cfg, model=self.backbone))

        self.image_preprocess = preprocess
        self.output_dim = 2560
        
        super().__init__(checkpoint_path)
    
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)  # size: 1 x 261 x 1280

            class_token = output[:, 0]    # size: 1 x 1280
            patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

            # concatenate class token and average pool of patch tokens
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding
    
    def get_img_token(self, x):
        output = self.backbone.forward_features(x)
        patch_token = output[:, 5:]
        class_token = output[:, 0]
        return {"patch_tokens": patch_token, "class_token": class_token}


@register_model('musk')
class MuskModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for Musk model")
        else:
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

@register_model('h_optimus_0')
class HOptimusModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for H-optimus model")
        else:
            checkpoint_path = get_model_hf_path('h_optimus_0')
            
        self.backbone = timm.create_model(checkpoint_path, 
                                       pretrained=True, init_values=1e-5, dynamic_img_size=False)
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


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 2:
            return tuple(x)
        else:
            raise ValueError(f"Expected length 2, got {len(x)}")
    return (x, x)

class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


@register_model('ctranspath')
class CTransPathModel(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            backbone = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
            backbone.head = nn.Identity()
            backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu")['model'], strict=True)
            self.backbone = backbone
        else:
            raise NotImplementedError("Huggingface directory not supported for CTransPath model")
        
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.image_preprocess = preprocess
        self.output_dim = 768
            
        
    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output

    def get_img_token(self, x):
        raise NotImplementedError("CTransPath model does not support image token extraction")


def create_model(model_name, checkpoint_path=None, local_dir=False):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(checkpoint_path=checkpoint_path, local_dir=local_dir)

    return model


if __name__ == "__main__":
    model = create_model("musk").cuda()
    image = torch.randn(1, 3, 384, 384).cuda()
    output = model.get_img_token(image)
    print(output["patch_tokens"].shape)
    print(output["class_token"].shape)
    # print(model(image).shape)
    text = ["a histopathology image of breast cancer"]
    # print(model.image_preprocess)
    # preprocess text
    # text_tensor = model.text_preprocess(text).unsqueeze(0).cuda()
    # print(model.encode_text(text_tensor).shape)
    # print(model.backbone)