import torch
import torch.nn as nn
import numpy as np

class Zerovel(nn.Module):
    def __init__(self, seq_len=25, dev='cuda'):
        super(Zerovel, self).__init__()
        self.seq_len = seq_len
        self.dev = dev

    def forward(self, x):
        b, d, l, k = x.shape
        x = x
        p = x[:, :, -1, :].unsqueeze(-2)
        x = p.repeat([1 for _ in range(b - 2)] + [self.seq_len, 1]).squeeze()
        return x.permute(0, 2, 1, 3)
