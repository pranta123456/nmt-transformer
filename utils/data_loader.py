import math
import io 

import torch 
import torch.nn as nn 
import torchtext 
from collections import Counter
from torchtext.vocab import vocab


class Datautils:
    def __init__(self, de_tokenizer, en_tokenizer):
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer

    def build_vocab(self, filepath, tokenizer, min_freq):
        counter = Counter()
        with io.open(filepath, encoding='utf8') as f:
            for line in f:
                counter.update(tokenizer(line))
        
        result_vocab = vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=min_freq)
        result_vocab.set_default_index(result_vocab['<unk>'])
        return result_vocab, counter

    def process_data(self, filepaths, de_vocab, en_vocab):
        de_iter = iter(io.open(filepaths[0], encoding='utf8'))
        en_iter = iter(io.open(filepaths[1], encoding='utf8'))

        data = []
        for (raw_de, raw_en) in zip(de_iter, en_iter):
            de_tensor = torch.tensor([de_vocab[token] for token in self.de_tokenizer(raw_de)], dtype=torch.long)
            en_tensor = torch.tensor([en_vocab[token] for token in self.en_tokenizer(raw_en)], dtype=torch.long)
            data.append((de_tensor, en_tensor))

        return data


class LabelSmoothing(nn.Module):
    def __init__(self, dec_voc_size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.size = dec_voc_size
        self.padding_idx = padding_idx
        self.eps = smoothing
        self.conf = 1 - smoothing
        self.criterion = nn.CrossEntropyLoss() #nn.KLDivLoss(reduction='sum')
        self.true_dist = None
            
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.eps/(self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.conf)
        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
              true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def itos_mapper(vocab, counter):
    itos = {vocab[k] : k for k in counter}
    itos[vocab['<unk>']] = '<unk>'
    itos[vocab['<pad>']] = '<pad>'
    itos[vocab['<sos>']] = '<sos>'
    itos[vocab['<eos>']] = '<eos>'
    return itos


def idx_to_word(x, itos):
    words = []
    for i in x:
        word = itos[i.item()]
        if '<' not in word:
            words.append(i)
    return " ".join(words)