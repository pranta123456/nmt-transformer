import math
import time
import argparse
import os

import torch.nn as nn
from torch.optim import Adam 
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

from data import *
from config import *
from model.transformer import Transformer 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def initialize_weights(m):
#     if hasattr(m, 'weight') and m.weight.dim() > 1:
#         nn.init.kaiming_uniform(m.weight.data)


def rate(step, d_model, factor, warmup_steps):
    if step == 0:
        step = 1
    return factor * (
          d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
      )


class TrainLogger(object):
    def __init__(self, file, total_epoch):
        self.file = os.path.join('./logs', file)
        self.curr_epoch = 0
        self.total_epoch = total_epoch
        with open(self.file, 'w') as f:
            pass

    def add_entry(self, train_loss, val_loss):
        self.curr_epoch += 1
        s = 'Epoch: [{}/{}] train_loss: {} val_loss: {}\n'.format(self.curr_epoch, self.total_epoch, train_loss, val_loss)
        with open(self.file, 'a') as f:
            f.write(s)


# model = Transformer(src_pad_idx=src_pad_idx,
#                     trg_pad_idx=trg_pad_idx,
#                     trg_sos_idx=trg_sos_idx,
#                     enc_voc_size=enc_voc_size,
#                     dec_voc_size=dec_voc_size,
#                     d_model=d_model,
#                     d_hidden=d_hidden,
#                     n_layers=n_layers,
#                     h=n_heads,
#                     max_len=max_len,
#                     drop_prob=drop_prob,
#                     device=device)

# model.apply(initialize_weights)


# optimizer = Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps, betas=(0.9, 0.98))

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
#                                                         verbose=True,
#                                                         factor=factor,
#                                                         patience=patience)

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
#                                               lr_lambda=lambda step : rate(step, d_model=512, factor=1.0, warmup_steps=4000))

# criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx, label_smoothing=0.1)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        src = batch[0].T.to(device)
        trg = batch[1].T.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
    lr = optimizer.param_groups[0]["lr"]
        # print('step :', round((i / len(iterator)) *  100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator), lr


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for batch in iterator:
            src = batch[0].T.to(device)
            trg = batch[1].T.to(device)
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_reshape = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_reshape)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def run(gpu, args):
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args.world_size,                              
        rank=rank                                               
    )
    
    model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    d_model=d_model,
                    d_hidden=d_hidden,
                    n_layers=n_layers,
                    h=n_heads,
                    max_len=max_len,
                    drop_prob=drop_prob,
                    device=torch.device(f"cuda:{gpu}"))
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    optimizer = Adam(params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                              lr_lambda=lambda step : rate(step, d_model=512, factor=1.0, warmup_steps=warmup))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                                     verbose=True,
    #                                                     factor=factor,
    #                                                     patience=patience)
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Define Sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_data,
    	num_replicas=args.world_size,
    	rank=rank
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
    	val_data,
    	num_replicas=args.world_size,
    	rank=rank
    )

    # Define Loader
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                               batch_size = args.train_batch_size,
                                               shuffle = False,
                                               num_workers = 0,
                                               pin_memory = True,
                                               sampler = train_sampler,
                                               collate_fn = generate_batch)
    
    val_loader = torch.utils.data.DataLoader(dataset = val_data,
                                               batch_size = args.val_batch_size,
                                               shuffle = False,
                                               num_workers = 0,
                                               pin_memory = True,
                                               sampler = val_sampler,
                                               collate_fn = generate_batch)
    
    train_losses, test_losses = [], []
    best_loss = inf
    logger = TrainLogger(args.log_file_name, args.epochs)
    for step in range(args.epochs):
        # start_time = time.time()
        train_loss, lr = train(model, train_loader, optimizer, criterion, clip)
        valid_loss = evaluate(model, val_loader, criterion)
        # end_time = time.time()

        # if step > warmup:
        #     scheduler.step(valid_loss)
        # train_loss_tensor = torch.tensor([train_loss], device=gpu)
        # val_loss_tensor = torch.tensor([valid_loss], device=gpu)

        # dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        # dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)

        # avg_train_loss = train_loss_tensor.item() / args.world_size
        # avg_val_loss = val_loss_tensor.item() / args.world_size
        # train_losses.append(train_loss)
        # test_losses.append(valid_loss)
        # bleus.append(bleu)
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     torch.save(model.state_dict(), './saved_model/best_model.pt')

        # f = open('./results/train_loss.txt', 'w')
        # f.write(str(train_losses))
        # f.close()

        # f = open('/content/drive/MyDrive/transformer/result/bleu.txt', 'w')
        # f.write(str(bleus))
        # f.close()

        # f = open('./results/test_loss.txt', 'w')
        # f.write(str(test_losses))
        # f.close()
        
        # save_last_5_checkpoints(step)

        # f = open(f'./logs/{args.log_file_name}.txt', 'w')
        # f.write(f"Epoch: [{step+1}/{args.epochs}] train_loss: {train_loss:.3f} val_loss: {valid_loss:.3f}")
        # f.write("\n")
        # f.close()
        logger.add_entry(train_loss, valid_loss)

        # print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | lr: {lr}')
        # print(f'\tVal Loss: {valid_loss:.3f} ')
        # print(f'\tBLEU Score: {bleu:.3f}')
    if gpu == 0:
        torch.save(model.state_dict(), './saved_model/final_model_v1.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type = int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--train_batch_size', default=100, type=int)
    parser.add_argument('--val_batch_size', default=100, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--log_file_name', default='', type=str)

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(run, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()