import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from piano.model.mil_factory.layers.layers import create_mlp
from piano.model.mil_factory.layers.layers import GlobalAttention, GlobalGatedAttention
from piano.utils.wsi_finetune_tools import NLLSurvLoss


class ABMIL(nn.Module):
    def __init__(self, 
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 attn_dim: int = 384,
                 num_classes: int = 2, 
                 survival: bool = False):
        super().__init__()
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        self.global_attn = GlobalAttention(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )
    
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']   # input_dict contain features, coords, and labels (optional)
        h = self.patch_embed(x)  # Apply patch embedding MLP
        attn = self.global_attn(x)  # Apply global attention
        A = torch.transpose(attn, -2, -1)  # Transpose attention matrix
        A = F.softmax(A,dim = -1)
        h = torch.bmm(A,h).squeeze(dim=1)      # Initialize output dictionary
        logits = self.classifier(h)  # Classify the aggregated features
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
    def __init__(self, 
                 in_dim: int = 1024,
                 embed_dim: int = 512,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 attn_dim: int = 384,
                 num_classes: int = 2, 
                 survival: bool = False):
        super().__init__()
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] *
                     (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        self.global_attn = GlobalGatedAttention(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )
    
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']   # input_dict contain features, coords, and labels (optional)
        h = self.patch_embed(x)  # Apply patch embedding MLP
        attn = self.global_attn(x)  # Apply global attention
        A = torch.transpose(attn, -2, -1)  # Transpose attention matrix
        A = F.softmax(A,dim = -1)
        h = torch.bmm(A,h).squeeze(dim=1)      # Initialize output dictionary
        logits = self.classifier(h)  # Classify the aggregated features
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
    

