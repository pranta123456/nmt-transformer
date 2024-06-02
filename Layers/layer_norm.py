import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, unbiased=False, keepdims=True)

        out = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * out + self.beta 