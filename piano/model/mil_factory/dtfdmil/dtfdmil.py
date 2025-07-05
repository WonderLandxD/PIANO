import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from piano.utils.wsi_finetune_tools import NLLSurvLoss


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        if self.numRes > 0:
            x = self.resBlocks(x)
        return x


class Attention(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):  ## x: N x L
        forward_return = {}
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x)  ## K x L
        pred = self.classifier(afeat)  ## K x num_cls
        forward_return['logits'] = pred
        if return_WSI_feature:
            forward_return['WSI_feature'] = afeat
        if return_WSI_attn:
            forward_return['WSI_attn'] = AA.transpose(0, 1)
        return forward_return


def get_cam_1d(classifier, features):
    """Calculate class activation mapping"""
    features = features.squeeze(0)  # N x F
    fc_weights = classifier.fc.weight  # C x F
    cam = torch.matmul(fc_weights, features.t())  # C x N
    return cam


class DTFDMIL(nn.Module):
    def __init__(self, dim_in=1024, dim_hidden=512, num_classes=2, 
                 num_groups=4, numLayer_Res=0, classifier_dropout=0.25, 
                 attCls_dropout=0.25, distill='MaxMinS', survival=False):
        super().__init__()
        
        # Validate distill parameter
        valid_distill_options = ['MaxMinS', 'MaxS', 'AFS']
        if distill not in valid_distill_options:
            raise ValueError(f"Invalid distill parameter: '{distill}'. "
                           f"Only supports the following options: {valid_distill_options}")
        
        # Network configuration parameters
        self.dim_in = dim_in
        if dim_hidden is None:
            self.dim_hidden = dim_in // 2
        else:
            self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.distill = distill
        self.survival = survival
        
        # Build network components
        self.dimReduction = DimReduction(dim_in, self.dim_hidden, numLayer_Res=numLayer_Res)
        self.attention = Attention(self.dim_hidden)
        self.classifier = Classifier_1fc(self.dim_hidden, num_classes, classifier_dropout)
        self.attCls = Attention_with_Classifier(
            L=self.dim_hidden, 
            num_cls=num_classes, 
            droprate=attCls_dropout
        )
        
        if self.survival:
            self.loss_fn = NLLSurvLoss()  
        else:
            self.loss_fn = nn.CrossEntropyLoss()  

    def forward(self, input_dict, return_loss=True):
        features = input_dict['features'].squeeze(0)  # N x dim_in
        label = input_dict.get('labels', None)
        
        # Calculate number of instances per group
        total_instances = features.size(0)
        instance_per_group = max(1, total_instances // self.num_groups)
        
        # Group features
        if total_instances < self.num_groups:
            # If instances are less than groups, repeat features
            features = features.repeat(self.num_groups // total_instances + 1, 1)
            features = features[:self.num_groups * instance_per_group]
        
        # Split into groups
        inputs_pseudo_bags = torch.chunk(features, self.num_groups, dim=0)
        
        slide_sub_preds = []
        slide_sub_labels = []
        slide_pseudo_feat = []
        
        # First tier: process each group
        for subFeat_tensor in inputs_pseudo_bags:
            if label is not None:
                slide_sub_labels.append(label)
            
            # Dimension reduction
            tmidFeat = self.dimReduction(subFeat_tensor)  # group_size x dim_hidden
            
            # Attention mechanism
            tAA = self.attention(tmidFeat).squeeze(0)  # group_size
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # group_size x dim_hidden
            tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x dim_hidden
            
            # Classification prediction
            tPredict = self.classifier(tattFeat_tensor)  # 1 x num_classes
            slide_sub_preds.append(tPredict)
            
            # Get patch-level predictions for instance selection
            patch_pred_logits = get_cam_1d(self.classifier, tattFeats.unsqueeze(0)).squeeze(0)
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  # group_size x num_classes
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)
            
            # Select top and bottom instances
            if patch_pred_softmax.size(0) > 0:
                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                
                # Ensure indices don't exceed range
                max_instances = min(instance_per_group, sort_idx.size(0))
                topk_idx_max = sort_idx[:max_instances].long()
                topk_idx_min = sort_idx[-max_instances:].long()
                
                # Select features based on distillation strategy
                if self.distill == 'MaxMinS':
                    topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                    selected_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                elif self.distill == 'MaxS':
                    selected_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                elif self.distill == 'AFS':
                    selected_feat = tattFeat_tensor
                else:
                    selected_feat = tattFeat_tensor
                
                slide_pseudo_feat.append(selected_feat)
            else:
                # If no features, use aggregated features
                slide_pseudo_feat.append(tattFeat_tensor)
        
        # Merge results
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  # total_selected x dim_hidden
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  # num_groups x num_classes
        
        # Second tier: global classification
        global_output = self.attCls(slide_pseudo_feat)
        global_logits = global_output['logits']  # 1 x num_classes
        
        # Initialize output dictionary
        output_dict = {
            'logits': global_logits,
            'sub_logits': slide_sub_preds,
            'raw_attn': slide_pseudo_feat,
            'features': slide_pseudo_feat
        }
        
        if self.survival:
            Y_hat = torch.topk(global_logits, 1, dim=1)[1]
            hazards = torch.sigmoid(global_logits)
            S = torch.cumprod(1 - hazards, dim=1)
            
            output_dict.update({
                'Y_hat': Y_hat,
                'hazards': hazards,
                'S': S
            })
        
        # Calculate loss
        loss = None
        if return_loss and label is not None:
            if self.survival and 'events' in input_dict:
                loss_second = self.loss_fn(hazards=output_dict['hazards'], 
                                          S=output_dict['S'], 
                                          Y=input_dict['labels'], 
                                          c=input_dict['events'])
                
                # First tier loss 
                slide_sub_labels = torch.cat(slide_sub_labels, dim=0) if slide_sub_labels else label.repeat(slide_sub_preds.size(0))
                loss_first = nn.CrossEntropyLoss()(slide_sub_preds, slide_sub_labels)
                
                # Total loss
                loss = loss_first + loss_second
            else:
                # First tier loss
                slide_sub_labels = torch.cat(slide_sub_labels, dim=0) if slide_sub_labels else label.repeat(slide_sub_preds.size(0))
                loss_first = self.loss_fn(slide_sub_preds, slide_sub_labels)
                
                # Second tier loss
                loss_second = self.loss_fn(global_logits, label)
                
                # Total loss
                loss = loss_first + loss_second
        
        output_dict['loss'] = loss
        
        if 'WSI_attn' in global_output:
            output_dict['WSI_attn'] = global_output['WSI_attn']
        
        return output_dict
