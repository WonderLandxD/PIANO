""" adapted from: https://github.com/mahmoodlab/MADELEINE/blob/main/core/models/abmil.py
    Guillaume Jaume, Anurag Jayant Vaidya, Andrew Zhang,
    Andrew H Song, Richard J. Chen, Sharifa Sahai, Dandan
    Mo, Emilio Madrigal, Long Phi Le, and Mahmood Faisal.
    Multistain pretraining for slide representation learning in
    pathology. In European Conference on Computer Vision.
    Springer, 2024.
"""
import torch 
from torch import nn
import torch.nn.functional as F

class BatchedABMIL(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=256, dropout=False, n_classes=1, n_heads = 1, activation='softmax'):
        super(BatchedABMIL, self).__init__()

        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention_a = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        ])

        self.attention_b = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        ])
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, return_raw_attention=False):
        assert len(x.shape)==3, x.shape
        a = self.attention_a(x) 
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A) 
        if self.activation == 'softmax':
            activated_A = F.softmax(A, dim=1)
        elif self.activation == 'leaky_relu':
            activated_A = F.leaky_relu(A)
        elif self.activation == 'relu':
            activated_A = F.relu(A)
        elif self.activation == 'sigmoid': 
            activated_A = torch.sigmoid(A)
        else:
            raise NotImplementedError('Activation not implemented.')

        if return_raw_attention:
            return activated_A, A

        return activated_A