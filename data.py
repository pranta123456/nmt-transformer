from torchtext.utils import download_from_url, extract_archive
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import *
from utils.data_loader import Datautils, itos_mapper
from utils.tokenizer import Tokenizer


def generate_batch(batched_data):
    de_batch, en_batch = [], []
    for (de_item, en_item) in batched_data:
        de_batch.append(torch.cat([torch.tensor(de_vocab['<sos>']).reshape(1), de_item, torch.tensor(de_vocab['<eos>']).reshape(1)], dim=0))
        en_batch.append(torch.cat([torch.tensor(en_vocab['<sos>']).reshape(1), en_item, torch.tensor(en_vocab['<eos>']).reshape(1)], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=de_vocab['<pad>'])
    en_batch = pad_sequence(en_batch, padding_value=en_vocab['<pad>'])

    return de_batch, en_batch

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

tokenizer = Tokenizer()
de_tokenizer = tokenizer.tokenize_de
en_tokenizer = tokenizer.tokenize_en
loader = Datautils(de_tokenizer, en_tokenizer)

de_vocab, de_cnt = loader.build_vocab(train_filepaths[0], de_tokenizer, min_freq=2)
en_vocab, en_cnt = loader.build_vocab(train_filepaths[1], en_tokenizer, min_freq=2)

train_data = loader.process_data(train_filepaths, de_vocab, en_vocab)
val_data = loader.process_data(val_filepaths, de_vocab, en_vocab)
test_data = loader.process_data(test_filepaths, de_vocab, en_vocab)

train_iter = DataLoader(train_data, batch_size=100, shuffle=True, collate_fn=generate_batch, drop_last=False)
val_iter = DataLoader(val_data, batch_size=6, shuffle=True, collate_fn=generate_batch, drop_last=False)
test_iter = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=generate_batch, drop_last=False)

src_itos = itos_mapper(de_vocab, de_cnt)
trg_itos = itos_mapper(en_vocab, en_cnt)

src_pad_idx = de_vocab['<pad>']
trg_pad_idx = en_vocab['<pad>']
trg_sos_idx = en_vocab['<sos>']

enc_voc_size = len(de_vocab)
dec_voc_size = len(en_vocab)