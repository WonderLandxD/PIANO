import torch.nn as nn


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