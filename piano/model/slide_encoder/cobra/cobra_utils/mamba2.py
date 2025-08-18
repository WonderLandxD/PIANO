"""
Adapted from: https://github.com/isyangshu/MambaMIL/blob/main/models/MambaMIL.py
Shu Yang, Yihui Wang, and Hao Chen. MambaMIL: En-
hancing Long Sequence Modeling with Sequence Reorder-
ing in Computational Pathology. In proceedings of Medi-
cal Image Computing and Computer Assisted Intervention –
MICCAI 2024. Springer Nature Switzerland, 2024
"""

import torch
import torch.nn as nn
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from mamba_ssm import Mamba2


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Mamba2Enc(nn.Module):
    def __init__(
        self,
        in_dim,
        dim,
        n_classes,
        dropout=0.25,
        act="gelu",
        layer=2,
        rate=10,
        d_state=64,
    ):
        super(Mamba2Enc, self).__init__()
        self._fc1 = [nn.Linear(in_dim, dim)]
        if act.lower() == "relu":
            self._fc1 += [nn.ReLU()]
        elif act.lower() == "gelu":
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList()

        for _ in range(layer):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(dim),
                    Mamba2(
                        d_model=dim,
                        d_state=d_state,
                        d_conv=4,
                        expand=2,
                    ),
                )
            )

        self.n_classes = n_classes
        self.classifier = nn.Linear(dim, self.n_classes)
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x  # .float()

        h = self._fc1(h)

        for layer in self.layers:
            h_ = h
            h = layer[0](h)
            h = layer[1](h)
            h = h + h_

        logits = self.classifier(h)
        return logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)