import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from piano.utils.wsi_finetune_tools import NLLSurvLoss

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout= d
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class CrossLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, d=0.3):
        super().__init__()
        self.dim = dim
        self.num_heads = 8
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(d)

    def forward(self, q, k, v):
        batch_size, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape
        
        # Reshape for multi-head attention
        # [batch, seq_len, dim] -> [batch, num_heads, seq_len, head_dim]
        Q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        # [batch, num_heads, seq_len_q, seq_len_k]
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # [batch, num_heads, seq_len_q, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # Average attention weights across heads for output compatibility
        attention = attn_weights.mean(dim=1)  # [batch, seq_len_q, seq_len_k]
        
        # Reshape output back to original format
        # [batch, num_heads, seq_len_q, head_dim] -> [batch, seq_len_q, dim]
        output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.dim
        )
        
        return output, attention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout1d(drop)
        self.act2 = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        #x = self.act2(x)
        return x


class optimizer_triple(nn.Module):
    def __init__(self, in_feature,out_feature,drop=0.):
        super().__init__()
        self.infeature = in_feature
        self.outfeature = out_feature
        self.drop_rate = drop
        #print(self.infeature)
        self.fc1 = nn.Linear(self.infeature,self.outfeature)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout1d(self.drop_rate)

        self.fc2 = nn.Linear(self.outfeature,self.outfeature)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout1d(self.drop_rate)

    def forward(self, x, mode):
        if mode == 'global':
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            x = self.act2(x)

        else:
            x = self.fc1(x)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.act2(x)
            x = self.drop2(x)
        
        return x

class DGRMIL(nn.Module):
    def __init__(self, dim_in, num_classes=2, L=512, D=128, n_lesion = 11, attn_mode="gated", dropout_node=0.0,dropout_patch=0.0,initialize=False, survival=False):
        super().__init__()
        self.L = L
        self.D = D
        self.n_lesion = n_lesion
        self.attn_mode = attn_mode
        self.initialize = initialize
        self.survival = survival
        # global lesion representation learning 
        self.m = 0.4
 
        self.lesionRrepresentation = nn.Parameter(torch.randn(1,self.n_lesion, dim_in))
        self.normalcenter = nn.Parameter(torch.randn(1, self.L),requires_grad=False)
        self.postivecenter = nn.Parameter(torch.randn(1, self.L),requires_grad=False)
        # encoder instances -> 

        self.token = nn.Parameter(torch.randn(1, 1, L))


        self.triple_optimizer = optimizer_triple(in_feature=dim_in,out_feature=self.L,drop=dropout_patch)

        self.encoder_instances = nn.Sequential(
            TransLayer(dim=self.L,d=dropout_node),
            nn.LayerNorm(self.L),
        )
        
        # encoder global lesion representation -> 
        self.encoder_globalLesion = nn.Sequential(
            TransLayer(dim=self.L,d=dropout_node),
            nn.LayerNorm(self.L),
        )
              
        
        self.crossffn = nn.Sequential(
            nn.Linear(self.L,self.L),
            nn.LayerNorm(self.L),
        )
 
        self.crossattention =  CrossLayer(dim=self.L,d=dropout_node)
        self.fc = nn.Sequential(
            nn.Linear(self.L,num_classes)
        )

        # 根据任务类型自动选择损失函数
        if self.survival:
            self.loss_fn = NLLSurvLoss()  # 生存分析损失函数
        else:
            self.loss_fn = nn.CrossEntropyLoss()  # 分类损失函数


    def forward(self, input_dict, bag_mode='normal', return_WSI_attn = False, return_WSI_feature = False, return_loss=True):
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
        
        forward_return = {}
        x = self.triple_optimizer(x,mode='instances')

        H = self.encoder_instances(x)

        lesion_enhacing = self.triple_optimizer(self.lesionRrepresentation,mode='global')

        #x = self.triple_optimizer(x)
        #H = self.encoder_instances(x)
        #lesion_enhacing = self.triple_optimizer(self.lesionRrepresentation)

        lesion_token =  torch.cat((self.token,lesion_enhacing), dim=1)

        lesion = self.encoder_globalLesion(lesion_token)
                                
        

        out,A = self.crossattention(lesion,H,H) # 1 x n x L -> 1 x 5 x n 
        out = self.crossffn(out)
        out = out[:,0,:]
        if return_WSI_attn:
            WSI_attn = A[:,0,:].transpose(0,1)
            print(WSI_attn.shape)
            forward_return['WSI_attn'] = WSI_attn
        if return_WSI_feature:
            forward_return['WSI_feature'] = out
        
        # Classification
        logits = self.fc(out)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)  # Ensure batch dimension
        forward_return['logits'] = logits
        
        # survival analysis
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            
            forward_return.update({
                'Y_hat': Y_hat,
                'hazards': hazards,
                'S': S
            })
        
        # Calculate loss
        loss = None
        if return_loss and label is not None:
            if self.survival and 'events' in input_dict:
                loss = self.loss_fn(hazards=forward_return['hazards'], 
                                   S=forward_return['S'], 
                                   Y=input_dict['labels'], 
                                   c=input_dict['events'])
            else:
                loss = self.loss_fn(logits, label)
        forward_return['loss'] = loss
        
        # print(cls.shape)
        if self.training:
            with torch.no_grad(): 
                if bag_mode == 'normal': 
                    x = x.squeeze(0)  
                    negative_instances = torch.mean(x,dim=0,keepdim=True)
                    self._momentum_update_nc(negative_instances)

                else:
                    x = x.squeeze(0)  
                    postive_instances = torch.mean(x,dim=0,keepdim=True)
                    self._momentum_update_p(postive_instances)
            forward_return['A'] = A
            forward_return['H'] = H
            forward_return['postivecenter'] = self.postivecenter
            forward_return['normalcenter'] = self.normalcenter
            forward_return['lesion_enhacing'] = lesion_enhacing
            return forward_return
        else: 
            return forward_return

    
    @torch.no_grad()
    def _momentum_update_p(self,postive):
        self.postivecenter.data = self.postivecenter.data * self.m + postive.data * (1. - self.m)
    
    @torch.no_grad()
    def _momentum_update_nc(self,negative):
        
        self.normalcenter.data = self.normalcenter.data * self.m + negative.data * (1. - self.m)
            

@torch.no_grad()
def concat_all_gather(tensor):
    
    tensor_gather = [torch.ones_like(tensor) 
                     for _ in range(torch.distributions.get_world.size())]
    
    torch.distributions.all_gather(tensor_gather,tensor,async_op = False)
    
    output = torch.cat(tensor_gather,dim=0)
    
    return output     