import torch
import torch.nn.functional as F
from torch import nn
from thop import profile
import numpy as np
from piano.utils.wsi_finetune_tools import NLLSurvLoss


def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError
    

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\
    

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)
    

class RetNetRelPos(nn.Module):
    def __init__(self, embed_dim, retention_heads, hidden_dim):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // retention_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(retention_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = hidden_dim
        
    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            
            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos
    

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 512, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # chunk * N * n_classes
        return A, x


class Retention(nn.Module):
    def __init__(self, embed_dim, num_heads, gate_fn='swish'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads 

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.ret_rel_pos = RetNetRelPos(embed_dim=embed_dim, retention_heads=num_heads, hidden_dim=self.embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scaling = self.embed_dim ** -0.5

        self.group_norm = RMSNorm(self.embed_dim, eps=1e-6, elementwise_affine=False)
    
    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output
    
    def forward(self, x):

        bsz, tgt_len, _ = x.size()

        (sin, cos), inner_mask = self.ret_rel_pos(slen=tgt_len, activate_recurrent=False, chunkwise_recurrent=False)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        output = self.parallel_forward(qr, kr, v, inner_mask)  # three methods, now just the parallel method

        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output   # [chunksize, 256, C]

        return output



class RetMIL(nn.Module):
    def __init__(self, dim_in, num_heads, window_size=256, stride=256, num_classes=2, survival=False):
        super().__init__()
        self.embed_dim = dim_in
        self.num_heads = num_heads
        self.window_size = window_size
        self.stride = stride
        self.survival = survival

        self.local_retention = Retention(embed_dim=self.embed_dim, num_heads=self.num_heads, gate_fn='swish')

        self.local_attn_pool = Attn_Net_Gated(L=self.embed_dim, D=self.embed_dim // 2, dropout=0.25, n_classes=1)

        self.global_retention = Retention(embed_dim=self.embed_dim, num_heads=self.num_heads, gate_fn='swish')

        self.global_attn_pool = Attn_Net_Gated(L=self.embed_dim, D=self.embed_dim // 2, dropout=0.25, n_classes=1)

        self.classifier = nn.Linear(self.embed_dim, num_classes, bias=False)

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']   # input_dict contain features, coords, and labels (optional)
        B, N, C = x.shape

        # N_ = int(np.ceil(N / self.windows_size) * self.windows_size)
        # if N_ == N:
        #     N_ += self.window_size
        
        # _N = int(np.floor(N / self.window_size) * self.window_size)
        # if _N == N:
        #     _N -= N
        
        # c_x, r_x = x.split((_N, N-_N), dim=1)
        # a_x = 

        # split N // stride, remain                                 (img_x - x_size) / (x_size - x_overlap) + 1


        _N = (N // 512) * 512             
        c_x, r_x = x.split((_N, N - _N), dim=1)  # chunk_x, remain_x; r_x: [1, 200, C]
        if N-_N != 0:
            full_repeats = 512 // (N-_N) 
            partial_repeat = 512 % (N-_N)

            full_repeat_r_x = r_x.repeat(1, full_repeats, 1)
            partial_r_x = r_x[:, :partial_repeat, :]
            added_x = torch.cat([full_repeat_r_x, partial_r_x], dim=1) 
            d_x = torch.cat([c_x, added_x], dim=1)
        else:
            d_x = x
        # print(d_x.shape)
        ###############
        # full_repeats = N // 256

        # r_x_add = r_x[:, :256-(N-N_), :]

        # d_x = torch.cat((x, r_x_add), dim=1)
        d_x = d_x.reshape(B, d_x.shape[1] // 512 , 512, C)  # [B, 2560, C] -> [B, 10, 256, C]    chunk_sequence, stride

        local_x = self.local_retention(d_x.squeeze(0))  # [10, 256, C]

        local_A, local_x = self.local_attn_pool(local_x) # [10, C]
        local_A = torch.transpose(local_A, 2, 1)
        local_A = F.softmax(local_A, dim=2)
        local_fe_sq = torch.matmul(local_A, local_x)

        global_x = self.global_retention(local_fe_sq.transpose(1, 0))   # [1, 10, C]

        global_A, global_x = self.global_attn_pool(global_x) # [1, C]
        global_A = torch.transpose(global_A, 2, 1)
        global_A = F.softmax(global_A, dim=2)
        global_fe = torch.matmul(global_A, global_x)

        logits = self.classifier(global_fe.squeeze(0))

        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': global_A,
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
    

if __name__ == '__main__':
    input_data = torch.randn([1, 2560, 384]).cuda()
    input_dict = {'features': input_data}
    model = RetMIL(embed_dim=384, num_heads=8, window_size=256, stride=256, num_classes=2).cuda()
    output = model(input_dict)
    print(output['logits'].shape)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total_m_params = n_parameters / 1e6  # 转换为以"M"为单位

    # print(f"full parameter: {total_m_params:.2f} M")

    # # input_data = torch.randn([1, 1000, 384]).cuda()
    # flops, params = profile(model, inputs=(input_data,))
    # print(f"FLOPs: {flops / 1e6} M FLOPs")
   

# [B, N, C]
    
# 1, 1000, 384 -> 1, 1, 384

# [1, 25600, 384] -> [1, 100, 256, 384] -> [100, 256, 384] -> [100, 256, 384] -> [100, 1, 384] -> [1, 100, 384] -> [1, 100, 384] -> [1, 1, 384]
# [25601] 99 * 256 REMAIN 1 3 -> 256 
    
# [1, 25600, 384] -> [1, 25600, 384] - > [1, 1, 384]
    