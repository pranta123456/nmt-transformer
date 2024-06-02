import torch
import torch.nn as nn 

from model.encoder import Encoder 
from model.decoder import Decoder 

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, 
            enc_voc_size, dec_voc_size, d_model, d_hidden, 
            n_layers, h, max_len, drop_prob, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(enc_voc_size, 
                                max_len, 
                                d_model, 
                                d_hidden, 
                                h, 
                                n_layers, 
                                drop_prob,
                                device=device)
        
        self.decoder = Decoder(dec_voc_size, 
                                max_len, 
                                d_model, 
                                d_hidden, 
                                h, 
                                n_layers, 
                                drop_prob,
                                device=device)

        self.encoder.apply(self.initialize_weights)
        self.decoder.apply(self.initialize_weights)
        
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_rep = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_rep, trg_mask, src_mask)

        return output
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform(m.weight.data)