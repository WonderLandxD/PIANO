import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from piano.utils.wsi_finetune_tools import NLLSurvLoss

warnings.filterwarnings("ignore")


class M4(nn.Module):
    def __init__(self, dim_in, experts_out, towers_out, towers_hidden, tasks=1, num_expert=4, survival=False, num_classes=2):
        super(M4, self).__init__()
        self.feat_input = dim_in
        self.tasks = tasks
        self.experts_out = experts_out
        self.towers_hidden = towers_hidden
        self.towers_out = towers_out
        self.num_expert = num_expert
        self.survival = survival
        
        # layer1
        self.experts = nn.ModuleList([MP_AMIL(self.feat_input, is_expert=True) for i in range(self.num_expert)])
        self.gate1_fc1 = nn.Sequential(nn.Linear(self.feat_input, 512), nn.ReLU())
        self.gate1_fc2 = nn.ModuleList([nn.Sequential(nn.Linear(128, self.num_expert), nn.Softmax(dim=-1)) for i in range(self.tasks)])

        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.experts_out, num_classes)

        # tower
        # self.towers = nn.ModuleList(
        #     [Tower(self.experts_out, self.towers_out, self.towers_hidden) for i in range(self.tasks)])

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_dict, return_loss=True):
        # Extract features from input_dict
        x = input_dict['features']
        
        # layer1
        x = x.squeeze(0)
        expert_output = [self.experts[i](x_path=x) for i in range(self.num_expert)]
        expert_output = torch.stack(expert_output).squeeze(1)  
        expert_output_split = torch.split(expert_output, expert_output.shape[-1] // 4, dim=-1)#split expert output into 4 parts along the channel dimension
        gate_out = [self.gate1_fc2[i](self.gate1_fc1(x.mean(dim=0)).view(4, -1)) for i in range(self.tasks)]
        tower_in = []
        split_size = expert_output.shape[-1] // 4
        for i in range(self.tasks):#Aggregate experts through the gate
            tower_in_i = torch.zeros(expert_output.shape[-1], device=x.device)
            for j in range(4):
                start = j * split_size
                end = (j + 1) * split_size
                tower_in_i[start:end] = torch.matmul(gate_out[i][j], expert_output[:, start:end])
            tower_in.append(tower_in_i)

        # task specific tower
        final_feats = torch.stack(tower_in)  # (self.tasks,512)
        logits = self.classifier(final_feats)
        
        
        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': None,  # M4 doesn't have explicit attention weights like ABMIL
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

class MP_AMIL(nn.Module):
    def __init__(self, feat_input=2048, size_arg="small", n_classes=4, is_expert=True):
        super(MP_AMIL, self).__init__()
        self.is_expert = is_expert
        self.size_dict = {"small": [feat_input, 512, 128], "big": [feat_input, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.1)]
        self.fc = nn.Sequential(*fc)
        self.cpe = ConvPosEnc(dim=size[1], k=3)
        self.norm1 = nn.LayerNorm(size[1])
        self.hierarchical = Aggregator(dim=size[1], seg=4)
        self.attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=0.25, n_classes=1)

        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(size[1], n_classes)
        initialize_weights(self)

    def forward(self, **kwargs):
        h = kwargs['x_path'].unsqueeze(0)  # [1,n,2048]
        h = self.fc(h)  # [1,n,512]
        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [1,n, 512]
        B, _, C = h.shape  # [1,n,512]
        h = self.cpe(h, [_H, _W])# Position Encoding by Convolutional Network
        cur = self.norm1(h)  
        cur = self.hierarchical(cur, [_H, _W])  # hierarchical embedding

        h = h + cur
        A, h = self.attention_net(h.squeeze(0))
        A = torch.transpose(A, 1, 0)

        # if 'attention_only' in kwargs.keys():
        #     if kwargs['attention_only']:
        #         return A

        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)

        if self.is_expert:
            return M
        else:
            h = self.classifier(M)
            h = self.softmax(h)
            return h


class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
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
        A = self.attention_c(A)  # N x n_classes
        return A, x


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Aggregator(nn.Module):
    def __init__(self, dim, seg=4):
        super().__init__()
        self.dim = dim
        self.seg = seg  

        seg_dim = self.dim // self.seg

        # self.norm0 = nn.SyncBatchNorm(seg_dim)
        self.act0 = nn.Hardswish()

        self.agg1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        # self.norm1 = nn.SyncBatchNorm(seg_dim)
        self.act1 = nn.Hardswish()

        self.agg2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        # self.norm2 = nn.SyncBatchNorm(seg_dim)
        self.act2 = nn.Hardswish()

        self.agg3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)
        # self.norm3 = nn.SyncBatchNorm(seg_dim)
        self.act3 = nn.Hardswish()

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        seg_dim = self.dim // self.seg

        x = x.split([seg_dim] * self.seg, dim=1)  # split Channels to self.seg parts

        x0 = self.act0(x[0])  # only activation
        x1 = self.act1(self.agg1(x[1]))
        x2 = self.act2(self.agg2(x[2]))
        x3 = self.act3(self.agg3(x[3]))

        x = torch.cat([x0, x1, x2, x3], dim=1)

        x = x.flatten(2).transpose(1, 2)

        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x


class ConvPosEnc(nn.Module):  # Position Encoding by Convolutional Network
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        # Depthwise convolution.
        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        return x


if __name__ == "__main__":
    model = M4(feat_input=2048, experts_out=512, towers_out=2, towers_hidden=128, tasks=1, num_expert=4)
    x = torch.randn(1, 100, 2048)
    input_dict = {'features': x}
    output = model(input_dict)
    print("Output logits shape:", output['logits'].shape)
    print("Loss:", output['loss'])