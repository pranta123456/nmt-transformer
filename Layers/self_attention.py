import torch
import torch.nn as nn

import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, h, d_model):
        super(MultiHeadSelfAttention, self).__init__()

        self.h = h 
        self.d_model = d_model 
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_T = nn.Linear(self.d_model, self.d_model, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.score = None

    def forward(self, q, k, v, mask=None):
        # Generating Q, K, V for the input
        Q, K, V = self.W_Q(q), self.W_K(k), self.W_V(v)
        # Split the values for each head
        Q, K, V = self.split_head(Q), self.split_head(K), self.split_head(V)

        output, self.score = self.scaled_dot_product_attention(Q, K, V, mask=mask)
        # output --> (N, h, seq_len, dim)
        output = self.concat_multihead(output)
        output = self.W_T(output)

        return output 
    
    def split_head(self, x):
        batch_size, seq_len, dim = x.size()
        head_dim = dim // self.h 

        output = x.view(batch_size, seq_len, self.h, head_dim).transpose(1, 2)

        return output
    
    def concat_multihead(self, x):
        batch_size, n_head, seq_len, d_v = x.size()
        dim = d_v * n_head 
        output = x.transpose(1,2).contiguous().view(batch_size, seq_len, dim)

        return output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        _, _, _, d_k = K.size()
        K_t = K.transpose(2,3)

        score = (Q @ K_t) / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        
        score = self.softmax(score)

        output = score @ V 

        return output, score