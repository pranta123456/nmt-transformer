# nmt-transformer

This repository contains the implementation of the ground breaking model architecture Transformer from the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) and its pretraining for NMT from Germen to English on Multi30k Dataset.
 
![transformer](https://github.com/pranta123456/nmt-transformer/assets/59831039/3c38bbd7-2aef-4846-ba29-5ae121467bd6)

# Pretraining

To train the model on Multi30k Dataset you can run the following command with specific arguments.
```
python3 train.py --nodes 1 --gpus 8 --epochs 1000 --train_batch_size 128 --val_batch_size 128 --log_file_name training_logs.txt
```

# Translation

To translate from German to English, you can run the following command with specific arguments.
```
python3 nmt.py --model_weights_path <path> --sampling_strategy beam_search --beam_size 2 --generated_seq_length 10 --gpu 0 --input_text '<German text>'
```
