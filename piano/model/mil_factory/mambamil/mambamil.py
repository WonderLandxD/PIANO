"""
MambaMIL
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from .mamba_local.mamba_ssm import SRMamba
from .mamba_local.mamba_ssm import BiMamba
from .mamba_local.mamba_ssm import Mamba
import torch.nn.functional as F
from piano.utils.wsi_finetune_tools import NLLSurvLoss


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(nn.Module):
    def __init__(self, dim_in, num_classes, dropout, act='gelu', survival=False, layer=2, rate=10, type="SRMamba"):
        super(MambaMIL, self).__init__()
        self._fc1 = [nn.Linear(dim_in, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.survival = survival

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.num_classes = num_classes
        self.classifier = nn.Linear(512, self.num_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = type

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.apply(initialize_weights)

    def forward(self, input_dict, return_loss=True):
        # Extract features from input dict
        if isinstance(input_dict, dict):
            if 'features' in input_dict:
                x = input_dict['features']
            elif 'feature' in input_dict:
                x = input_dict['feature']
            else:
                raise KeyError("Input dict must contain 'features' or 'feature' key")
            label = input_dict.get('labels', None)
        else:
            # Backward compatibility
            x = input_dict
            label = None
        
        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # N x dim_in -> 1 x N x dim_in
        
        h = x.float()  # [B, n, dim_in]
        
        h = self._fc1(h)  # [B, n, 512]

        if self.type == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate)
                h = h + h_
        elif self.type == "Mamba" or self.type == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_

        h = self.norm(h)
        A = self.attention(h) # [B, n, 1]
        A = torch.transpose(A, 1, 2)  # [B, 1, n]
        A = F.softmax(A, dim=-1) # [B, 1, n]
        h = torch.bmm(A, h) # [B, 1, 512]
        h = h.squeeze(1)  # [B, 512]

        logits = self.classifier(h)  # [B, num_classes]
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)  # Ensure batch dimension
        
        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': A.squeeze(1),  # [B, n] - attention weights as raw attention
            'features': h
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
        if return_loss and label is not None:
            if self.survival and isinstance(input_dict, dict) and 'events' in input_dict:
                # Use survival loss: loss_fn(hazards=hazards, S=S, Y=label, c=event)
                loss = self.loss_fn(hazards=output_dict['hazards'], 
                                   S=output_dict['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:
                # Use standard classification loss
                loss = self.loss_fn(logits, label)
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


if __name__ == "__main__":
    model = MambaMIL(dim_in=1024, num_classes=2, dropout=0.25, act='gelu', survival=False, layer=2, rate=10, type="SRMamba").cuda()
    input_dict = {
        'features': torch.randn(1, 1024, 1024).cuda(),
        'labels': torch.randint(0, 2, (1,)).cuda()
    }
    output_dict = model(input_dict)
    print(output_dict)