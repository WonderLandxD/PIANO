import torch
import torch.nn as nn
import torch.nn.functional as F
from piano.utils.wsi_finetune_tools import NLLSurvLoss


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

# class IClassifier(nn.Module):
#     def __init__(self, feature_extractor, feature_size, output_class):
#         super().__init__()
        
#         self.feature_extractor = feature_extractor      
#         self.fc = nn.Linear(feature_size, output_class)
        
#     def forward(self, x):
#         device = x.device
#         feats = self.feature_extractor(x) # N x K
#         c = self.fc(feats.view(feats.shape[0], -1)) # N x C
#         return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, nonlinear=True): # K, L, N
        super().__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = feats
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    

class DSMIL(nn.Module):
    def __init__(self, dim_in, num_classes=1000, nonlinear=True, survival=False):
        super().__init__()
        self.num_classes = num_classes
        self.nonlinear = nonlinear
        self.survival = survival

        # 根据任务类型自动选择损失函数
        if self.survival:
            self.loss_fn = NLLSurvLoss()  # 生存分析损失函数
        else:
            self.loss_fn = nn.CrossEntropyLoss()  # 分类损失函数

        self.i_classifier = FCLayer(dim_in, num_classes)
        self.b_classifier = BClassifier(dim_in, num_classes, nonlinear=nonlinear)

    def forward(self, input_dict, return_loss=True):
        x = input_dict['features'].squeeze(0)
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        # Initialize output dictionary
        output_dict = {
            'logits': prediction_bag,
            'raw_attn': A,
        }
        
        # 生存分析计算
        if self.survival:
            Y_hat = torch.topk(prediction_bag, 1, dim=1)[1]
            hazards = torch.sigmoid(prediction_bag)
            S = torch.cumprod(1 - hazards, dim=1)
            
            output_dict.update({
                'Y_hat': Y_hat,
                'hazards': hazards,
                'S': S
            })

        # Loss calculation
        if return_loss and 'labels' in input_dict:
            if self.survival and 'events' in input_dict:
                # 使用生存分析损失: loss_fn(hazards=hazards, S=S, Y=label, c=event)
                loss = self.loss_fn(hazards=output_dict['hazards'], 
                                   S=output_dict['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:
                # 使用标准分类损失
                loss = self.loss_fn(prediction_bag, input_dict['labels'])
        else:
            loss = None

        output_dict['loss'] = loss
        return output_dict

