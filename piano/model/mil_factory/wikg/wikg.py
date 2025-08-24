import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from piano.utils.wsi_finetune_tools import NLLSurvLoss

class WiKG(nn.Module):
    def __init__(self, dim_in=384, dim_hidden=None, num_classes=2, topk=6, agg_type='bi-interaction', dropout=0.3, pool='attn', survival=False):
        super().__init__()
        
        # Network configuration parameters
        self.dim_in = dim_in
        if dim_hidden is None:
            self.dim_hidden = dim_in // 2
        else:
            self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.topk = topk
        self.agg_type = agg_type
        self.pool = pool
        self.survival = survival
        
        # Validate aggregation type
        valid_agg_types = ['gcn', 'sage', 'bi-interaction']
        if agg_type not in valid_agg_types:
            raise ValueError(f"Invalid agg_type: '{agg_type}'. "
                           f"Only supports the following options: {valid_agg_types}")
        
        # Validate pooling type
        valid_pool_types = ['mean', 'max', 'attn']
        if pool not in valid_pool_types:
            raise ValueError(f"Invalid pool type: '{pool}'. "
                           f"Only supports the following options: {valid_pool_types}")

        # Feature transformation
        self._fc1 = nn.Sequential(nn.Linear(dim_in, self.dim_hidden), nn.LeakyReLU())
        
        # Head and tail transformation for knowledge graph
        self.W_head = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.W_tail = nn.Linear(self.dim_hidden, self.dim_hidden)

        # Attention scaling
        self.scale = self.dim_hidden ** -0.5

        # Aggregation layers based on type
        if self.agg_type == 'gcn':
            self.linear = nn.Linear(self.dim_hidden, self.dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(self.dim_hidden * 2, self.dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.dim_hidden, self.dim_hidden)
            self.linear2 = nn.Linear(self.dim_hidden, self.dim_hidden)
        
        # Activation and dropout
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        # Normalization and classification
        self.norm = nn.LayerNorm(self.dim_hidden)
        # Handle the case when num_classes=0 (no classification head)
        if num_classes == 0:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Linear(self.dim_hidden, num_classes)

        # Readout/pooling layer
        if pool == "mean":
            self.readout = global_mean_pool 
        elif pool == "max":
            self.readout = global_max_pool 
        elif pool == "attn":
            att_net = nn.Sequential(
                nn.Linear(self.dim_hidden, self.dim_hidden // 2), 
                nn.LeakyReLU(), 
                nn.Linear(self.dim_hidden // 2, 1)
            )     
            self.readout = GlobalAttention(att_net)
        
        # Automatically select loss function based on survival or classification
        if self.survival:
            self.loss_fn = NLLSurvLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

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
        
        # Feature transformation
        x = self._fc1(x)    # [B, N, C]

        # Global context integration
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        # Head and tail embeddings for knowledge graph construction
        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # Construct neighbor relationships using attention
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # Prepare indices for advanced indexing
        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)

        # Get neighbor embeddings
        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # [B, N, topk, dim_hidden]

        # Apply softmax to get neighbor probabilities
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))

        # Gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        # Aggregation based on type
        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        # Apply dropout
        h = self.message_dropout(embedding)
        # Global readout/pooling
        h = self.readout(h.squeeze(0), batch=None)
        h = h.squeeze(1)

        h = self.norm(h)
        
        # Classification
        logits = self.classifier(h)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)  # Ensure batch dimension
        
        # Initialize output dictionary
        output_dict = {
            'logits': logits,
            'raw_attn': topk_prob,  # Attention weights as raw attention
            'features': h.unsqueeze(0) if len(h.shape) == 1 else h
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
            if self.survival and 'events' in input_dict:
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