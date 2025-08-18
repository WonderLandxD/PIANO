import torch
import torch.nn as nn
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib
import os
import sys

@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress both stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def split_clusters(idx, n):
    m = idx.shape[0]
    if m == 0:
        # Handle empty cluster: return n empty tensors
        return [torch.tensor([], dtype=idx.dtype, device=idx.device) for _ in range(n)]
    
    # Randomly shuffle the indices
    shuffled_idx = idx[torch.randperm(m, device=idx.device)]
    
    # Calculate split sizes
    base_size = m // n
    remainder = m % n
    
    # Split into n parts
    splits = []
    start = 0
    for i in range(n):
        # First 'remainder' splits get one extra element
        size = base_size + (1 if i < remainder else 0)
        if size > 0:
            splits.append(shuffled_idx[start:start + size])
        else:
            splits.append(torch.tensor([], dtype=idx.dtype, device=idx.device))
        start += size
    
    return splits

def build_kmeans(feats, k, n, distance='euclidean'):
    device = feats.device
    
    # Suppress kmeans output
    with suppress_output():
        labels, centroids = kmeans(X=feats, num_clusters=k, distance=distance, device=device)
    
    # Find indices for each cluster and split them
    slices_per_cluster = []
    for c in range(k):
        cluster_indices = (labels == c).nonzero(as_tuple=False).squeeze(1)
        cluster_splits = split_clusters(cluster_indices, n)
        slices_per_cluster.append(cluster_splits)

    bag_idx = []
    for b in range(n):
        parts = [slices_per_cluster[c][b] for c in range(k)]
        bag_idx.append(torch.cat(parts, dim=0))

    bag_feats_list = [feats[bag_idx[b]] for b in range(n)]

    return bag_feats_list


class Attention(nn.Module):
    def __init__(self, L, D, K=1, dropout=0.25):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.module = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x):
        # 确保输入和模块参数类型匹配
        module_dtype = next(self.module.parameters()).dtype
        x = x.to(dtype=module_dtype)
        return self.module(x), x


class ProTEA(nn.Module):
    def __init__(self, input_dim, num_clusters, num_experts, llm_decoder_ckpt_path, loss_type='clip'):
        super().__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.num_experts = num_experts
        self.loss_type = loss_type
        self.temp = nn.Parameter(torch.tensor(0.07))

        self.Main_Attn = Attention(L=self.input_dim, D=self.input_dim, K=1)

        self.Attn_Nets = nn.ModuleList([
            Attention(L=self.input_dim, D=self.input_dim, K=1) for _ in range(self.num_experts)
        ]) 

        self.llm_decoder = AutoModelForCausalLM.from_pretrained(llm_decoder_ckpt_path)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_decoder_ckpt_path)
        
        # Two projection layers to match image and text embedding dimension
        self.text_projection = nn.Linear(self.llm_decoder.config.hidden_size, self.input_dim)
        self.image_projection = nn.Linear(self.input_dim, self.input_dim)

    
    def forward(self, patch_feats, input_ids, train_mode=True):
        """
        INPUT:
        patch_feats: list of [N, C], list sequence length is B
        input_ids: [B, seq_len]
        """

        # WSI part
        image_embeds = []

        for i, img_feats in enumerate(patch_feats):
            # 获取模型参数的数据类型
            model_dtype = next(self.parameters()).dtype
            
            # 确保输入数据类型匹配
            img_feats = img_feats.to(dtype=model_dtype)
            h = F.normalize(img_feats, dim=1)

            A_global, h_global = self.Main_Attn(h)
            A_global = torch.transpose(A_global, 1, 0)
            A_global_raw = A_global
            A_global = F.softmax(A_global, dim=1)
            global_feats = torch.matmul(A_global, h_global)

            bag_feats_list = build_kmeans(h, k=self.num_clusters, n=self.num_experts)

            for j, bag_feat in enumerate(bag_feats_list):
                A, bag_feat = self.Attn_Nets[j](bag_feat)
                A = torch.transpose(A, 1, 0)
                A_raw = A
                A = F.softmax(A, dim=1)
                # expert_feat = torch.matmul(A, bag_feat)
                if j == 0:
                    expert_feats = torch.matmul(A, bag_feat)
                else:
                    expert_feats = torch.cat([expert_feats, torch.matmul(A, bag_feat)], dim=0)

            v_global = F.normalize(global_feats, dim=-1) # [1, C]
            v_bags = F.normalize(expert_feats, dim=-1) # [N, C]  # NOTE: 后续需要这里的前k个拿出来再与其他四个文本做clip对比学习，this version just uses vanilla one-to-one clip loss

            sim = (v_bags @ v_global.T)
            alpha = torch.softmax(sim, dim=0) # [N, 1]
            v_mix = torch.matmul(alpha.transpose(0, 1), v_bags) # [1, C]

            v_final = 0.5 * v_global + 0.5 * v_mix

            image_embeds.append(v_final) 

        image_embeds = torch.cat(image_embeds, dim=0) # [B, C]

        if input_ids is not None:
            # Text part
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            text_input_embeds = self.llm_decoder.get_input_embeddings()(input_ids)

            text_output = self.llm_decoder(inputs_embeds=text_input_embeds, attention_mask=attention_mask, output_hidden_states=True)
            text_hidden_states = text_output.hidden_states[-1]  # [B, seq_len, hidden_dim], hidden_dim = 896
            
            # Mean pooling with attention mask to get [B, hidden_dim]
            text_embeds_pooled = (text_hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
            text_embeds = self.text_projection(text_embeds_pooled)  # [B, input_dim]
        
        # Project image and text embeddings to match image embedding dimension
        image_embeds = self.image_projection(image_embeds)  # [B, input_dim]
        
        if train_mode:  
            # Calculate CLIP Loss
            contrastive_loss = calculate_loss(image_embeds, text_embeds, loss_type=self.loss_type, temperature=self.temp)
        else:
            contrastive_loss = None

        output_dict = {
            'image_embeds': image_embeds, 
            'text_embeds': text_embeds if input_ids is not None else None, 
            'loss': contrastive_loss
        }

        return output_dict

            

def calculate_loss(image_embeds, text_embeds, loss_type, temperature):

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    if loss_type == 'clip':
        logits = torch.matmul(image_embeds, text_embeds.T) / temperature  # [B, B]
        labels = torch.arange(len(image_embeds), device=image_embeds.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
    elif loss_type == 'siglip':
        logits = torch.matmul(image_embeds, text_embeds.T) / temperature  # [B, B]
        
        batch_size = len(image_embeds)
        labels = torch.eye(batch_size, device=image_embeds.device) * 2 - 1  # [B, B]
        
        loss = -F.logsigmoid(labels * logits).mean()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss



            



            

    
            