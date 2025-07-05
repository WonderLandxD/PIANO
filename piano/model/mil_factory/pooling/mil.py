import torch
import torch.nn as nn
from piano.utils.wsi_finetune_tools import NLLSurvLoss


class MeanPool(nn.Module):
    def __init__(self, dim_in, num_classes=1000, survival=False):
        super().__init__()
        self.dim_in = dim_in
        self.num_classes = num_classes
        self.survival = survival
        

        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']
        x = torch.mean(x, dim=-2, keepdim=False)
        logits = self.fc(x)
        
        # Initialize output dictionary
        output_dict = {
            'logits': logits,
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
        
        # Loss calculation
        if return_loss and 'labels' in input_dict:
            if self.survival and 'events' in input_dict:

                loss = self.loss_fn(hazards=output_dict['hazards'], 
                                   S=output_dict['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:

                loss = self.loss_fn(logits, input_dict['labels'])
        else:
            loss = None
        
        output_dict['loss'] = loss
        return output_dict
        

class MaxPool(nn.Module):
    def __init__(self, dim_in, num_classes=1000, survival=False):
        super().__init__()
        self.dim_in = dim_in
        self.num_classes = num_classes
        self.survival = survival
        

        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features']
        x, _ = torch.max(x, dim=-2, keepdim=True)
        logits = self.fc(x)
        
        # Initialize output dictionary
        output_dict = {
            'logits': logits,
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

        # Loss calculation
        if return_loss and 'labels' in input_dict:
            if self.survival and 'events' in input_dict:

                loss = self.loss_fn(hazards=output_dict['hazards'], 
                                   S=output_dict['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:

                loss = self.loss_fn(logits, input_dict['labels'])
        else:
            loss = None

        output_dict['loss'] = loss
        return output_dict