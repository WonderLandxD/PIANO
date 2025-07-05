import torch
import torch.nn as nn
import open_clip

from ..base_model import BaseModel, register_model


@register_model('quiltnet_b_32')
class QuiltNetB32Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for QuiltNet-B-32 model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('quiltnet_b_32')

        # 加载完整模型和预处理
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:wisdomik/QuiltNet-B-32'
        )
        self.tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')

        # 设置视觉主干结构
        self.backbone_visual = self.model.visual       
        self.backbone = self.backbone_visual           
        self.model = self.model                        

        self.image_preprocess = self.preprocess_train
        self.text_preprocess = self._text_preprocess
        self.output_dim = 512

    def _text_preprocess(self, text):
        """
        处理文本为tokenized输入
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            torch.Tensor: 文本token IDs
        """
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text)
        return tokens

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)  # 本质就是 self.backbone_visual(x)
        return output  # 通常是 class token proj 后的 embedding

    def encode_text(self, text_tokens):
        """
        编码文本tokens为文本特征
        
        Args:
            text_tokens: 已tokenized的文本输入
            
        Returns:
            torch.Tensor: 文本特征
        """
        with torch.set_grad_enabled(self.model.training):
            text_features = self.model.encode_text(text_tokens)
        return text_features

    def get_img_token(self, x):
        raise NotImplementedError("QuiltNet model does not support image token extraction")
    

@register_model('quiltnet_b_16')
class QuiltNetB16Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for QuiltNet-B-16 model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('quiltnet_b_16')

        # 加载完整模型和预处理
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:wisdomik/QuiltNet-B-16'
        )
        self.tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-16')

        # 设置视觉主干结构
        self.backbone_visual = self.model.visual       
        self.backbone = self.backbone_visual           
        self.model = self.model                        

        self.image_preprocess = self.preprocess_train
        self.text_preprocess = self._text_preprocess
        self.output_dim = 512

    def _text_preprocess(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text)
        return tokens

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output

    def encode_text(self, text_tokens):
        with torch.set_grad_enabled(self.model.training):
            text_features = self.model.encode_text(text_tokens)
        return text_features

    def get_img_token(self, x):
        raise NotImplementedError("QuiltNet model does not support image token extraction")
    

@register_model('quiltnet_b_16_pmb')
class QuiltNetB16_PMB_Model(BaseModel):
    def __init__(self, checkpoint_path=None, local_dir=False):
        super().__init__()
        if local_dir == True and checkpoint_path is not None:
            raise NotImplementedError("Local directory not supported for QuiltNet-B-16-PMB model")
        else:
            from ..model_registry import get_model_hf_path
            checkpoint_path = get_model_hf_path('quiltnet_b_16_pmb')

        # 加载完整模型和预处理
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:wisdomik/QuiltNet-B-16-PMB'
        )
        self.tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-16-PMB')

        # 设置视觉主干结构
        self.backbone_visual = self.model.visual       
        self.backbone = self.backbone_visual           
        self.model = self.model                        

        self.image_preprocess = self.preprocess_train
        self.text_preprocess = self._text_preprocess
        self.output_dim = 512

    def _text_preprocess(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text)
        return tokens

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            output = self.backbone(x)
        return output

    def encode_text(self, text_tokens):
        with torch.set_grad_enabled(self.model.training):
            text_features = self.model.encode_text(text_tokens)
        return text_features

    def get_img_token(self, x):
        raise NotImplementedError("QuiltNet model does not support image token extraction") 