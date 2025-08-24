import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from piano.utils.wsi_finetune_tools import NLLSurvLoss


class ABMIL(nn.Module):
    def __init__(self, dim_in, dim_hidden=None, dropout=0.25, num_classes=1000, survival=False):
        super().__init__()
        if dim_hidden is None:
            self.dim_hidden = dim_in // 2
        else:
            self.dim_hidden = dim_hidden
        self.survival = survival
        
        self.attn_module = nn.Sequential(
            nn.Linear(dim_in, self.dim_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_hidden, 1)
        )
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(dim_in, num_classes)

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']   # input_dict contain features, coords, and labels (optional)
        attn = self.attn_module(x)  # [B, N, 1]
        A = torch.transpose(attn, -1, -2) # [B, 1, N]
        A = torch.softmax(A, dim=-1) # [B, 1, N]
        output = torch.matmul(A, x).squeeze(1) # [B, C]
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
    

class GatedABMIL(nn.Module):
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

        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(dim_in, num_classes)

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']
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
