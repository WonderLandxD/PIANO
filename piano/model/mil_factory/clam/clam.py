import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from piano.utils.wsi_finetune_tools import NLLSurvLoss



"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
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

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, dim_in=1024, dim_hidden=512, dropout=0.25, num_classes=2, 
                 k_sample=8, instance_loss_fn=None, subtyping=False, survival=False):
        super().__init__()

        self.dim_hidden_1 = dim_in
        if dim_hidden is None:
            self.dim_hidden_2 = dim_in // 2
        else:
            self.dim_hidden_2 = dim_hidden
        self.k_sample = k_sample
        self.num_classes = num_classes
        self.subtyping = subtyping
        self.survival = survival
        
        self.attention_net = Attn_Net_Gated(
            L=self.dim_hidden_1,
            D=self.dim_hidden_2,
            dropout=dropout,
            n_classes=1
        )
        
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.dim_hidden_1, num_classes)
        
        # Handle the case when num_classes=0 (no classification head) for instance classifiers
        if num_classes == 0:
            instance_classifiers = [nn.Identity()]
        else:
            instance_classifiers = [nn.Linear(self.dim_hidden_1, 2) for _ in range(num_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            
        if instance_loss_fn is None:
            self.instance_loss_fn = nn.CrossEntropyLoss()
        else:
            self.instance_loss_fn = instance_loss_fn
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()
    
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, input_dict, return_loss=True):
        h = input_dict['features'].squeeze(0)
        label = input_dict.get('labels', None)

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)

        results_dict = {}
        if return_loss and label is not None:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze()
            
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
            
            results_dict.update({
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds)
            })

        M = torch.mm(A, h)
        logits = self.classifier(M)
        
        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': A_raw,
            'features': M
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
        loss = None
        if return_loss and label is not None:
            if self.survival and 'events' in input_dict:
                # Use survival loss: loss_fn(hazards=hazards, S=S, Y=label, c=event)
                loss = self.loss_fn(hazards=output_dict['hazards'], 
                                   S=output_dict['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:
                # Use standard classification loss
                loss = self.loss_fn(logits, label)
            if 'instance_loss' in results_dict:
                loss += results_dict['instance_loss']

        output_dict['loss'] = loss
        output_dict.update(results_dict)
        
        return output_dict
    

class CLAM_MB(CLAM_SB):
    def __init__(self, dim_in=1024, dim_hidden=512, dropout=0.25, num_classes=2, 
                 k_sample=8, instance_loss_fn=None, subtyping=False, survival=False):
        super().__init__(dim_in, dim_hidden, dropout, num_classes, k_sample, instance_loss_fn, subtyping, survival)
        
        self.dim_hidden_1 = dim_in
        if dim_hidden is None:
            self.dim_hidden_2 = dim_in // 2
        else:
            self.dim_hidden_2 = dim_hidden
        self.k_sample = k_sample
        self.num_classes = num_classes
        self.subtyping = subtyping
        self.survival = survival

        
        self.attention_net = Attn_Net_Gated(
            L=self.dim_hidden_1,
            D=self.dim_hidden_2,
            dropout=dropout,
            n_classes=num_classes
        )
        
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            bag_classifiers = [nn.Identity()]
            instance_classifiers = [nn.Identity()]
        else:
            bag_classifiers = [nn.Linear(self.dim_hidden_1, 1) for _ in range(num_classes)]
            instance_classifiers = [nn.Linear(self.dim_hidden_1, 2) for _ in range(num_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            
        if instance_loss_fn is None:
            self.instance_loss_fn = nn.CrossEntropyLoss()
        else:
            self.instance_loss_fn = instance_loss_fn

    def forward(self, input_dict, return_loss=True):
        h = input_dict['features'].squeeze(0)
        label = input_dict.get('labels', None)

        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)

        results_dict = {}
        if return_loss and label is not None:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.num_classes).squeeze()
            
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
            
            results_dict.update({
                'instance_loss': total_inst_loss,
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_preds)
            })

        M = torch.mm(A, h)
        
        logits = torch.empty(1, self.num_classes).float().to(M.device)
        for c in range(self.num_classes):
            logits[0, c] = self.classifiers[c](M[c])
        
        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': A_raw,
            'features': M
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
        loss = None
        if return_loss and label is not None:
            if self.survival and 'events' in input_dict:
                # Use survival loss: loss_fn(hazards=hazards, S=S, Y=label, c=event)
                loss = self.loss_fn(hazards=output_dict['hazards'], 
                                   S=output_dict['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:
                # Use standard classification loss
                loss = self.loss_fn(logits, label)
            if 'instance_loss' in results_dict:
                loss += results_dict['instance_loss']

        output_dict['loss'] = loss
        output_dict.update(results_dict)
        
        return output_dict