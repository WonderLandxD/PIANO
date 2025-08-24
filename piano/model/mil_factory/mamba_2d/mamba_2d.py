import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .mamba_simple import MambaConfig as SimpleMambaConfig
from .mamba_simple import Mamba as SimpleMamba
from piano.utils.wsi_finetune_tools import NLLSurvLoss

def split_tensor(data, batch_size):
    num_chk = int(np.ceil(data.shape[0] / batch_size))
    return torch.chunk(data, num_chk, dim=0)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MambaMIL_2D(nn.Module):
    def __init__(self, dim_in=1024, drop_out=0.25, num_classes=2, survival=False, pos_emb_type=None):   
        super(MambaMIL_2D, self).__init__()
        
        # self.args = args
        self._fc1 = [nn.Linear(dim_in, 128)]
        self._fc1 += [nn.GELU()]
        if drop_out > 0:
            self._fc1 += [nn.Dropout(drop_out)]

        self._fc1 = nn.Sequential(*self._fc1)
        
        self.norm = nn.LayerNorm(128)
        
        self.layers = nn.ModuleList()
        config = SimpleMambaConfig(
            d_model = 128,
            n_layers = 1,
            d_state = 16,
            inner_layernorms = False,
            pscan = True,
            use_cuda = False,
            mamba_2d = True,
            mamba_2d_max_w = 100000,
            mamba_2d_max_h = 100000,
            mamba_2d_pad_token = 'trainable',
            mamba_2d_patch_size = 512
        )
        self.layers = SimpleMamba(config)
        self.config = config

        self.n_classes = num_classes
        self.survival = survival

        self.attention = nn.Sequential(
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(128, self.n_classes)

        self.pos_emb_type = pos_emb_type
        if pos_emb_type == 'linear':
            self.pos_embs = nn.Linear(2, 128)
            self.norm_pe = nn.LayerNorm(128)
            self.pos_emb_dropout = nn.Dropout(0.25)
        else:
            self.pos_embs = None

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.apply(initialize_weights)

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']   # input_dict contain features, coords, and labels (optional)
        coords = input_dict.get('coords', None)  # coords might be optional
        coords = coords.squeeze(0).to(x.dtype)  # [num_patch, 2] or [1, num_patch, 2]
        
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)   # (1, num_patch, feature_dim)
        h = x # .float()  # [1, num_patch, feature_dim]

        h = self._fc1(h)  # [1, num_patch, mamba_dim];   project from feature_dim -> mamba_dim

        # Add Pos_emb
        if self.pos_emb_type == 'linear':
            pos_embs = self.pos_embs(coords)
            h = h + pos_embs.unsqueeze(0)
            h = self.pos_emb_dropout(h)

        h = self.layers(h, coords, self.pos_embs)

        h = self.norm(h)   # LayerNorm
        A = self.attention(h) # [1, W, H, 1]

        if len(A.shape) == 3:
            A = torch.transpose(A, 1, 2)
        else:  
            A = A.permute(0,3,1,2)
            A = A.view(1,1,-1)
            h = h.view(1,-1,self.config.d_model)

        A = F.softmax(A, dim=-1)  # [1, 1, num_patch]  # A: attention weights of patches
        h = torch.bmm(A, h) # [1, 1, 512] , weighted combination to obtain slide feature
        h = h.squeeze(0)  # [1, 512], 512 is the slide dim

        logits = self.classifier(h)  # [1, n_classes]

        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': A,
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
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers  = self.layers.to(device)
        
        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)



    
if __name__ == '__main__':

    model = MambaMIL_2D(dim_in=1024, drop_out=0.25, num_classes=2, survival=False, pos_emb_type=None).cuda()
    print(model)

    input_data = torch.randn([1, 2560, 1024]).cuda()
    coords = torch.randn([1, 2560, 2]).cuda()
    input_dict = {'features': input_data, 'coords': coords}
    output = model(input_dict)
    print(output['logits'].shape)