import torch.nn as nn

from blocks.decoder_layer import DecoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, d_hidden, h, n_layers, drop_prob, device):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob=drop_prob, device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model,
            d_hidden,
            h,
            drop_prob)
            for _ in range(n_layers)])
        
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        tgt = self.emb(tgt)

        for layer in self.layers:
            tgt = layer(tgt, enc_src, src_mask, tgt_mask)
        
        output = self.linear(tgt)

        return output