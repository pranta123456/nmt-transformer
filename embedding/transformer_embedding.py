import torch.nn as nn

from embedding.positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model, device)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.token_emb(x)
        pos_enc = self.pos_encoding(x)

        return self.dropout(tok_emb + pos_enc)