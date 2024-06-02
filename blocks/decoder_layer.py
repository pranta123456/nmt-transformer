import torch.nn as nn

from Layers.self_attention import MultiHeadSelfAttention
from Layers.layer_norm import LayerNorm 
from Layers.feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, h, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(h, d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.layer_norm1 = LayerNorm(d_model)

        self.enc_dec_attention = MultiHeadSelfAttention(h, d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.layer_norm2 = LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_hidden, p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.layer_norm3 = LayerNorm(d_model)

    def forward(self, dec, enc, src_mask, tgt_mask):
        _x = dec 
        x = self.self_attention(q=dec, k=dec, v=dec, mask=tgt_mask)

        x = self.dropout1(x)
        x = self.layer_norm1(x + _x)

        if enc is not None:
            _x = x 
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            x = self.dropout2(x)
            x = self.layer_norm2(x + _x)
        
        _x = x 
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.layer_norm3(x + _x)

        return x 