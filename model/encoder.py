import torch.nn as nn

from blocks.encoder_layer import EncoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, d_hidden, h, n_layers, drop_prob, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(enc_voc_size, d_model, max_len, drop_prob=drop_prob, device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model, 
                                        d_hidden, 
                                        h, 
                                        drop_prob)
                                        for _ in range(n_layers)])
    
    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x