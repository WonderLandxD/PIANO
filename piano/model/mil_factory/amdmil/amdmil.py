import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from piano.utils.wsi_finetune_tools import NLLSurvLoss


class AMD_Layer(nn.Module):
    def __init__(
        self,
        dim,
        agent_num=256,
        heads = 8,
    ):
        super().__init__()
        self.dim_head = dim//heads
        self.agent_num = agent_num
        self.denoise = nn.Linear(self.dim_head,self.dim_head)
        self.mask = nn.Linear(self.dim_head,self.dim_head)
        self.get_thresh = nn.Linear(dim,1)
        self.heads = heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.agent = nn.Parameter(torch.randn(heads, agent_num, self.dim_head))

    def forward(self, x , return_WSI_attn = False):
        forward_return = {}
        b, _, _, h = *x.shape, self.heads
        # obtain the qkv matrix
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        agent = self.agent.unsqueeze(0).expand(b,-1,-1,-1)
        # Perform agent calculations
        q = torch.matmul(q,agent.transpose(-1,-2))
        k = torch.matmul(agent,k.transpose(-1,-2))
        softmax = nn.Softmax(dim=-1)
        q *= self.scale
        q = softmax(q)
        k = softmax(k)
        kv = torch.matmul(k,v) 
        kv_c = kv.reshape(b,self.agent_num,-1)
        thresh = self.get_thresh(kv_c).squeeze().mean()
        thresh = F.sigmoid(thresh)
        # Perform mask and denoise operations
        denoise = self.denoise(kv)
        denoise = torch.sigmoid(denoise)
        mask = self.mask(kv)
        mask = torch.sigmoid(mask)
        mask = torch.where(mask > thresh, torch.ones_like(mask), torch.zeros_like(mask))
        kv = kv * mask + denoise
        # Obtain weighted features
        kv = softmax(kv)
        out = torch.matmul(q,kv)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        forward_return['amd_out'] = out
        if return_WSI_attn:
            WSI_attn = torch.matmul(q,k)
            forward_return['WSI_attn'] = WSI_attn 
        return forward_return


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class AMDLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, agent_num=512, tem=0, pool=False, thresh=None, thresh_tem='classical', kaiming_init=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = AMD_Layer(
            dim = dim,
            agent_num=agent_num,
            heads = 8,          
        )

    def forward(self, x, return_WSI_attn = False):
        forward_return = self.attn(self.norm(x), return_WSI_attn)
        x = x + forward_return['amd_out']
        new_forward_return = {}
        new_forward_return['amd_out'] = x
        if return_WSI_attn:
            new_forward_return['WSI_attn'] = forward_return['WSI_attn']

        return new_forward_return


'''
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
'''
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class AMD_MIL(nn.Module):
    def __init__(self, dim_in, embed_dim=512, dropout=0.25, act=nn.ReLU(), 
                 agent_num=256, num_classes=1000, survival=False):
        super(AMD_MIL, self).__init__()
        
        self.survival = survival
        
        # Position encoding layer
        self.pos_layer = PPEG(dim=embed_dim) # PPEG from TransMIL

        # Feature projection
        self._fc1 = [nn.Linear(dim_in, embed_dim)]
        self._fc1 += [act]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*self._fc1)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        
        # AMD layers
        self.amdlayer1 = AMDLayer(dim=embed_dim, agent_num=agent_num)
        self.amdlayer2 = AMDLayer(dim=embed_dim, agent_num=agent_num)
        self.norm = nn.LayerNorm(embed_dim)         
        
        # Final classification layer
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(embed_dim, num_classes)
        
        # Loss function selection
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            
        self.apply(initialize_weights)

    def forward(self, input_dict, return_loss=True):
        # Extract features from input dictionary
        x = input_dict['features']  # input_dict contains features, coords, and labels (optional)
        
        B = x.shape[0]
        N = x.shape[1]
        
        # Feature projection
        h = self._fc1(x) 
        
        # Fit for PPEG - pad to square shape
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) 
        B = h.shape[0]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        
        # AMD layers
        h = self.amdlayer1(h)['amd_out']
        h = self.pos_layer(h, _H, _W)
        
        # Get attention weights if needed
        return_attn = 'return_attn' in input_dict and input_dict['return_attn']
        amd_return = self.amdlayer2(h, return_attn)
        h = amd_return['amd_out']
        
        # Extract CLS token and get logits
        h = self.norm(h)[:,:1] 
        h = h.squeeze(1)
        logits = self.fc(h)

        # Initialize output dictionary
        output_dict = {
            'logits': logits,
        }
        
        # Add attention weights if computed
        if return_attn and 'WSI_attn' in amd_return:
            # Extract attention for original patches (excluding padding)
            raw_attn = amd_return['WSI_attn'][:,:,0,1:N+1].mean(1).transpose(0,1)
            output_dict['raw_attn'] = raw_attn
        
        # Survival analysis calculations
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            
            output_dict.update({
                'Y_hat': Y_hat,
                'hazards': hazards,
                'S': S
            })

        # Loss calculation
        if return_loss and 'labels' in input_dict:
            if self.survival and 'events' in input_dict:
                # Use survival loss: loss_fn(hazards=hazards, S=S, Y=label, c=event)
                loss = self.loss_fn(hazards=output_dict['hazards'], 
                                   S=output_dict['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:
                # Use standard classification loss
                loss = self.loss_fn(logits, input_dict['labels'])
        else:
            loss = None

        output_dict['loss'] = loss
        return output_dict


if __name__ == "__main__":
    # Test with new interface
    features = torch.randn(1, 1000, 1024)  # [B, N, C]
    input_dict = {
        'features': features,
        'labels': torch.randint(0, 10, (1,)),  # classification labels
        'return_attn': True
    }
    
    model = AMD_MIL(
        dim_in=1024, 
        embed_dim=512,
        num_classes=10,
        agent_num=256
    )
    
    # Calculate total and learnable parameters
    total_params = sum(p.numel() for p in model.parameters())
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    output = model(input_dict)
    print(f"Output keys: {output.keys()}")
    print(f"Logits shape: {output['logits'].shape}")
    if 'raw_attn' in output:
        print(f"Attention shape: {output['raw_attn'].shape}")
    print(f"Loss: {output['loss']}")
    print(f"Total parameters: {total_params/1e6:.2f} M")
    print(f"Learnable parameters: {learnable_params/1e6:.2f} M")