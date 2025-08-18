import torch
import torch.nn as nn
import sys

from .cobra_utils.mamba2 import Mamba2Enc
from .cobra_utils.abmil import BatchedABMIL
import torch.nn.functional as F
from einops import rearrange
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from huggingface_hub import hf_hub_download
import os


class Embed(nn.Module):
    def __init__(self, dim, embed_dim=1024,dropout=0.25):
        super(Embed, self).__init__()

        self.head = nn.Sequential(
             nn.LayerNorm(dim),
             nn.Linear(dim, embed_dim),
             nn.Dropout(dropout) if dropout else nn.Identity(),
             nn.SiLU(),
             nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.head(x) 

class Cobra(nn.Module):
    """
    Cobra model for processing and aggregating embeddings with attention.
    This model utilizes separate embedding layers for different input dimensions, followed by a
    normalization layer and a mamba-based encoder (Mamba2Enc). It then applies multi-head
    attention using BatchedABMIL modules to compute attention maps and aggregate the input features.
    Parameters:
        embed_dim (int, optional):
            Dimensionality of the embedding vectors. Default is 768.
        input_dims (list of int, optional):
            A list of input feature dimensions. Each feature dimension corresponds to a key in the
            embedding module dictionary. Default is [384, 512, 1024, 1280, 1536].
        num_heads (int, optional):
            Number of attention heads. Each head processes a slice of the embedded features.
            Default is 8.
        layers (int, optional):
            Number of layers in the Mamba2Enc encoder. Default is 2.
        dropout (float, optional):
            Dropout rate used throughout the model to prevent overfitting. Default is 0.25.
        att_dim (int, optional):
            The hidden dimensionality for the attention branch (BatchedABMIL) per attention head.
            Default is 96.
        d_state (int, optional):
            Dimensionality of the internal state in the Mamba2Enc encoder. Default is 128.
    Methods:
        forward(x, multi_fm_mode=False, fm_idx=None, get_attention=False):
            Forward pass through the Cobra network.
            Args:
                x (Tensor or list of Tensors):
                    Input tensor if single feature map; list of tensors if multi_fm_mode is True.
                    Each tensor should have a shape corresponding to the respective key in the embedding module.
                multi_fm_mode (bool, optional):
                    Flag to indicate that multiple feature maps (different modalities) are provided.
                    Default is False.
                fm_idx (int, optional):
                    In multi_fm_mode, if provided, selects a specific feature map index for feature aggregation.
                    Default is None.
                get_attention (bool, optional):
                    If True, the method returns the computed attention matrix rather than the aggregated features.
                    Default is False.
            Returns:
                Tensor:
                    If get_attention is True, returns the attention matrix computed from the input.
                    Otherwise, returns the aggregated feature representation obtained after applying the attention mechanism.
                    For multi_fm_mode with a provided fm_idx, returns the features corresponding to the selected modality.
            Raises:
                AssertionError:
                    If the dimensions of the embedded features do not match the expected sizes during
                    concatenation or aggregation, assertions will be raised to signal the discrepancy.
    Example:
        >>> model = Cobra()
        >>> # Processing random input
        >>> x = torch.randn(1, 100, 768)  # batch size: 1, sequence length: 100, feature dimension: 768
        >>> features = model(x)
    """

    def __init__(self,embed_dim=768,input_dims=[384,512,1024,1280,1536], num_heads=8,layers=2,dropout=0.25,att_dim=96,d_state=128):
        super().__init__()
        
        self.embed = nn.ModuleDict({str(d):Embed(d,embed_dim) for d in input_dims})
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.mamba_enc = Mamba2Enc(embed_dim,embed_dim,n_classes=embed_dim,layer=layers,dropout=dropout,d_state=d_state)
   
        self.num_heads = num_heads
        self.attn = nn.ModuleList([BatchedABMIL(input_dim=int(embed_dim/num_heads),hidden_dim=att_dim,
                                        dropout=dropout,n_classes=1) for _ in range(self.num_heads)]) 
        
    def forward(self, x, multi_fm_mode=False, fm_idx=None, get_attention=False):
        if multi_fm_mode:
            fm_embs = torch.concat([self.embed[str(xi.shape[-1])](xi) for xi in x],dim=0)
            assert fm_embs.shape[-1]==self.embed_dim, fm_embs.shape
            assert len(fm_embs.shape)==3, fm_embs.shape
            assert fm_embs.shape[0]==len(x), fm_embs.shape
            logits = torch.mean(fm_embs,dim=0) 
        else:
            logits = self.embed[str(x.shape[-1])](x)

        h = self.norm(self.mamba_enc(logits))
        
        if self.num_heads > 1:
            h_ = rearrange(h, 'b t (e c) -> b t e c',c=self.num_heads)

            attention = []
            for i, attn_net in enumerate(self.attn):
                _, processed_attention = attn_net(h_[:, :, :, i], return_raw_attention = True)
                attention.append(processed_attention)
                
            A = torch.stack(attention, dim=-1)

            A = rearrange(A, 'b t e c -> b t (e c)',c=self.num_heads).mean(-1).unsqueeze(-1)
            A = torch.transpose(A,2,1)
            A = F.softmax(A, dim=-1) 
        else: 
            A = self.attn[0](h)
        
        if get_attention:
            return A
        
        if multi_fm_mode:
            if fm_idx:
                feats = torch.bmm(A,x[fm_idx]).squeeze(0).squeeze(0)
            else:
                feats=[]
                for i,xi in enumerate(x):
                    feats.append(torch.bmm(A,xi).squeeze(0).squeeze(0))
                    assert len(feats[i].shape)==1 and feats[i].shape[0]==xi.shape[-1], feats[i].shape
        else:
            feats = torch.bmm(A,x).squeeze(1)
        
        return feats


def get_cobraII(download_weights=False, checkpoint_path="weights/cobraII.pth.tar"):
    """
    Load the COBRAII model.

    Parameters:
    - download_weights (bool): If True, download the model weights from Hugging Face Hub.
    - checkpoint_path (str): Path to the model checkpoint file.

    Returns:
    - Cobra: The loaded COBRAII model.
    
    Raises:
    - FileNotFoundError: If the checkpoint file is not found and download_weights is False.
    """
    if download_weights:
        download_path = hf_hub_download("KatherLab/COBRA", filename="cobraII.pth.tar")
        checkpoint_path = download_path
    else:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found and download_weights is False")
    state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = Cobra(layers=1, input_dims=[512,1024,1280,1536],
                  num_heads=4, dropout=0.2, att_dim=256, d_state=128)
    if "state_dict" in list(state_dict.keys()):
        chkpt = state_dict["state_dict"]
        cobra_weights = {k.split("momentum_enc.")[-1]:v for k,v in chkpt.items() if "momentum_enc" in k and "momentum_enc.proj" not in k}
        if len(list(cobra_weights.keys())) == 0:
            # from stamp finetuning
            print("Loading STAMP model..")
            cobra_weights = {k.split("cobra.")[-1]:v for k,v in chkpt.items() if "cobra" in k}
    else:
        cobra_weights = state_dict
    model.load_state_dict(cobra_weights, strict=True)
    print("\033[92mCOBRAII model loaded successfully\033[0m")
    return model


class COBRA2_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = get_cobraII(download_weights=True, checkpoint_path=None)
        
    def forward(self, input):
        patch_features = input['feats']

        output = self.model(patch_features)
        return output

if __name__ == '__main__':
    model = COBRA2_Model().cuda()
    input_feat = torch.randn([1, 3000, 1024]).cuda()
    input_feat = {'feats': input_feat}
    output = model(input_feat)
    print(output.shape)