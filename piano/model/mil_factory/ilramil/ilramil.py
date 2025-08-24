import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from piano.utils.wsi_finetune_tools import NLLSurvLoss

"""
Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification
Jinxi Xiang et al. ICLR 2023
"""


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K):
        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, _ = self.multihead_attn(Q, K, V)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat


class NLP(nn.Module):
    """
    Non-Local Pooling
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        ret = self.mha(global_embedding, X)
        return ret


class ILRAMIL(nn.Module):
    def __init__(self, dim_in=1024, dim_hidden=None, num_classes=2, num_layers=2,
                 num_heads=8, topk=2, ln=False, dropout=0.25, survival=False):
        super().__init__()
        
        # Network configuration parameters
        self.dim_in = dim_in
        if dim_hidden is None:
            self.dim_hidden = dim_in // 2
        else:
            self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.topk = topk  # This is the rank of latent matrix; tune this parameter for your applications!
        self.ln = ln
        self.survival = survival
        
        # Stack multiple GAB blocks
        gab_blocks = []
        for idx in range(num_layers):
            block = GAB(
                dim_in=self.dim_in if idx == 0 else self.dim_hidden,
                dim_out=self.dim_hidden,
                num_heads=num_heads,
                num_inds=topk,
                ln=ln
            )
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # Non-local pooling for classification
        self.pooling = NLP(dim=self.dim_hidden, num_heads=num_heads, ln=ln)

        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=self.dim_hidden, out_features=num_classes)
        )


        if self.survival:
            self.loss_fn = NLLSurvLoss()  
        else:
            self.loss_fn = nn.CrossEntropyLoss() 

    def forward(self, input_dict, return_loss=True):
        # Extract features from input dict
        x = input_dict['features']  # N x dim_in -> 1 x N x dim_in
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if needed
        
        label = input_dict.get('labels', None)
        
        # Pass through GAB blocks
        for block in self.gab_blocks:
            x = block(x)

        # Non-local pooling to get global features
        feat = self.pooling(x)  # 1 x 1 x dim_hidden
        
        # Classification
        logits = self.classifier(feat).squeeze(1)  # 1 x num_classes
        
        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': feat,
            'features': feat
        }
        
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            
            output_dict.update({
                'Y_hat': Y_hat,
                'hazards': hazards,
                'S': S
            })
        
        # Calculate loss
        loss = None
        if return_loss and label is not None:
            if self.survival and 'events' in input_dict:
                loss = self.loss_fn(hazards=output_dict['hazards'], 
                                   S=output_dict['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:
                loss = self.loss_fn(logits, label)
        
        output_dict['loss'] = loss
        return output_dict