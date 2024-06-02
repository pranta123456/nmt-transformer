import torch.nn as nn

from Layers.self_attention import MultiHeadSelfAttention
from Layers.layer_norm import LayerNorm 
from Layers.feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, h, drop_prob):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadSelfAttention(h, d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.layer_norm1 = LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_hidden, p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x, src_mask):
        _x = x 
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.dropout1(x)
        x = self.layer_norm1(x + _x)

        _x = x 
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.layer_norm2(x + _x)

        return x 