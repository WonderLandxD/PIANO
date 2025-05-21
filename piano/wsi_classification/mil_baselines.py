import torch
import torch.nn as nn


#### weakly supervised learning
class ABMIL(nn.Module):
    def __init__(self, dim_in, dropout=0.25, num_classes=1000):
        super().__init__()
        dim_hidden = 512
        self.attn_module = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, 1)
        )
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):  # x: [B, N, C]
        attn = self.attn_module(x)  # [B, N, 1]
        A = torch.transpose(attn, -1, -2) # [B, 1, N]
        A = torch.softmax(A, dim=-1) # [B, 1, N]
        output = torch.matmul(A, x).squeeze(1) # [B, C]
        logits = self.fc(output)
        return logits
    
    
####  unsupervised learning
class SiMLP(nn.Module):
    def __init__(self, dim_in, dropout=0.25, num_classes=1000):
        super().__init__()
        self.dim_hidden = 512
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, self.dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim_hidden, num_classes)
        )
    
    def forward(self, x): # x: [B, N, C]
        feat = x.mean(dim=1) # [B, C]
        logits = self.mlp(feat)
        return logits 