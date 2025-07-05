import torch
from torch import nn
from einops import repeat
from .translayer import TransLayer1
import torch.nn.functional as F
from piano.utils.wsi_finetune_tools import NLLSurvLoss


class Attention(nn.Module):
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D, bias=bias)]

        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K, bias=bias)]
        self.attention = nn.Sequential(*self.attention)

    def forward(self, x, no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)
        
        if no_norm:
            return x, A_ori
        else:
            return x, A


class AttentionGated(nn.Module):
    def __init__(self, input_dim=512, bias=False, dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention_a = [nn.Linear(self.L, self.D, bias=bias), nn.ReLU()]
        self.attention_b = [nn.Linear(self.L, self.D, bias=bias), nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

    def forward(self, x, no_norm=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        else:
            return x, A


class DAttention(nn.Module):
    def __init__(self, input_dim=512, act='relu', gated=False, bias=False, dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim, bias, dropout)
        else:
            self.attention = Attention(input_dim, act, bias, dropout)

    def forward(self, x, return_attn=False, no_norm=False):
        return self.attention(x, no_norm)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class RRTEncoder(nn.Module):
    def __init__(self, mlp_dim=512, attn='rrt', region_num=8, drop_out=0.1, n_layers=1, 
                 n_heads=8, pool='cls_token', da_act='tanh', **kwargs):
        super(RRTEncoder, self).__init__()
        
        self.final_dim = mlp_dim
        self.pool = pool
        
        if pool == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        elif pool == 'attn':
            self.pool_fn = DAttention(self.final_dim, da_act)

        self.norm = nn.LayerNorm(self.final_dim)

        self.layer1 = TransLayer1(
            dim=mlp_dim, 
            head=n_heads, 
            drop_out=drop_out, 
            attn=attn, 
            n_region=region_num,
            **kwargs
        )

        if n_layers >= 2:
            self.layers = []
            for i in range(n_layers - 1):
                self.layers += [TransLayer1(
                    dim=mlp_dim, 
                    head=n_heads, 
                    drop_out=drop_out, 
                    attn=attn, 
                    n_region=region_num,
                    **kwargs
                )]
            self.layers = nn.Sequential(*self.layers)
        else:
            self.layers = nn.Identity()

    def forward(self, x, no_pool=False, return_attn=False, no_norm=False):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            shape_len = 4
        batch, num_patches, C = x.shape 
        patch_idx = 0
        
        # cls_token
        if self.pool == 'cls_token':
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
            x = torch.cat((cls_tokens, x), dim=1)
            patch_idx = 1

        # translayer1
        x = self.layer1(x)
        
        # additional layers
        for layer in self.layers.children():
            x = layer(x)

        x = self.norm(x)

        if no_pool:
            if shape_len == 2:
                x = x.squeeze(0)
            elif shape_len == 4:
                x = x.transpose(1, 2)
                x = x.reshape(batch, C, int(num_patches**0.5), int(num_patches**0.5))
            return x
            
        if self.pool == 'cls_token':
            logits = x[:, 0, :]
        elif self.pool == 'avg':
            logits = x.mean(dim=1)
        elif self.pool == 'attn':
            if return_attn:
                logits, a = self.pool_fn(x, return_attn=True, no_norm=no_norm)
            else:
                logits = self.pool_fn(x)
        else:
            logits = x

        if shape_len == 2:
            logits = logits.squeeze(0)
        elif shape_len == 4:
            logits = logits.transpose(1, 2)
            logits = logits.reshape(batch, C, int(num_patches**0.5), int(num_patches**0.5))

        if return_attn:
            return logits, a
        else:
            return logits


class RRT_MIL(nn.Module):
    def __init__(self, dim_in, dim_hidden=None, dropout=0.25, num_classes=1000, survival=False):
        super().__init__()
        if dim_hidden is None:
            self.dim_hidden = dim_in // 2
        else:
            self.dim_hidden = dim_hidden
        
        self.survival = survival
        
        self.attn_1 = nn.Sequential(
            nn.Linear(dim_in, self.dim_hidden),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.attn_2 = nn.Sequential(
            nn.Linear(dim_in, self.dim_hidden),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attn_3 = nn.Linear(self.dim_hidden, 1)

        self.fc = nn.Linear(dim_in, num_classes)

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.rrt_encoder = RRTEncoder(
            mlp_dim=dim_in,
            attn='rrt',
            region_num=8,
            n_layers=2,
            n_heads=8,
        )

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']
        x = self.rrt_encoder(x)
        attn_1 = self.attn_1(x)
        attn_2 = self.attn_2(x)
        attn = attn_1.mul(attn_2)
        attn = self.attn_3(attn)
        A = torch.transpose(attn, -1, -2)
        A = torch.softmax(A, dim=-1)
        output = torch.matmul(A, x).squeeze(1)
        logits = self.fc(output)

        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': attn,
        }
        
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



# class RRT_MIL(nn.Module):
#     def __init__(self, dim_in, mlp_dim=512, act='relu', num_classes=1000, dropout=0.25,
#                  attn='rrt', pool='cls_token', region_num=8, n_layers=2, n_heads=8, 
#                  da_act='relu', trans_dropout=0.1, survival=False, **kwargs):
#         super(RRT_MIL, self).__init__()

#         self.survival = survival
        
#         # Feature projection layer
#         self.patch_to_emb = [nn.Linear(dim_in, mlp_dim)]
#         if act.lower() == 'relu':
#             self.patch_to_emb += [nn.ReLU()]
#         elif act.lower() == 'gelu':
#             self.patch_to_emb += [nn.GELU()]

#         self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
#         self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

#         # RRT Encoder
#         self.encoder = RRTEncoder(
#             mlp_dim=mlp_dim,
#             attn=attn,
#             region_num=region_num,
#             n_layers=n_layers,
#             n_heads=n_heads,
#             pool=pool,
#             da_act=da_act,
#             drop_out=trans_dropout,
#             **kwargs
#         )

#         # Final classification layer
#         self.fc = nn.Linear(self.encoder.final_dim, num_classes)
        
#         # Loss function selection
#         if self.survival:
#             self.loss_fn = NLLSurvLoss()
#         else:
#             self.loss_fn = nn.CrossEntropyLoss()
        
#         self.apply(initialize_weights)

#     def forward(self, input_dict, return_loss=True):
#         # Extract features from input dictionary
#         x = input_dict['features']  # input_dict contains features, coords, and labels (optional)
        
#         # Feature projection
#         x = self.patch_to_emb(x)  # n*mlp_dim
#         x = self.dp(x)

#         # Get attention weights if needed
#         if 'return_attn' in input_dict and input_dict['return_attn']:
#             x, attn_weights = self.encoder(x, return_attn=True, no_norm=True)
#         else:
#             x = self.encoder(x, return_attn=False)
#             attn_weights = None
        
#         # Final prediction
#         logits = self.fc(x)

#         # Initialize output dictionary
#         output_dict = {
#             'logits': logits,
#         }
        
#         # Add attention weights if computed
#         if attn_weights is not None:
#             output_dict['raw_attn'] = attn_weights
        
#         # Survival analysis calculations
#         if self.survival:
#             Y_hat = torch.topk(logits, 1, dim=1)[1]
#             hazards = torch.sigmoid(logits)
#             S = torch.cumprod(1 - hazards, dim=1)
            
#             output_dict.update({
#                 'Y_hat': Y_hat,
#                 'hazards': hazards,
#                 'S': S
#             })

#         # Loss calculation
#         if return_loss and 'labels' in input_dict:
#             if self.survival and 'events' in input_dict:
#                 # Use survival loss: loss_fn(hazards=hazards, S=S, Y=label, c=event)
#                 loss = self.loss_fn(hazards=output_dict['hazards'], 
#                                    S=output_dict['S'], 
#                                    Y=input_dict['labels'], 
#                                    c=input_dict['events'])
#             else:
#                 # Use standard classification loss
#                 loss = self.loss_fn(logits, input_dict['labels'])
#         else:
#             loss = None

#         output_dict['loss'] = loss
#         return output_dict


if __name__ == "__main__":
    # Test with new interface
    features = torch.randn(1, 1000, 1024)  # [B, N, C]
    input_dict = {
        'features': features,
        'labels': torch.randint(0, 10, (1,)),  # classification labels
        'return_attn': False
    }

    
    model = RRT_MIL(
        dim_in=1024, 
        dim_hidden=512,
        dropout=0.25,
        num_classes=10,
        survival=False
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