import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np
import networkx as nx
import argparse
import warnings

from model.transformer import Transformer
from data import *
from config import *

def idx_to_words(x, itos):
    words = []
    for i in x:
        word = itos[i.item()]
        if '<' not in word:
            words.append(word)
    return " ".join(words)


def modified_state_dict(model_path):
    state_dict = torch.load(model_path)
    clean_state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
    return clean_state_dict


def greedy_decoding(src, start_symbol, model, device):
    src_mask = model.make_src_mask(src)
    enc_rep = model.encoder(src, src_mask)
    
    ys = torch.zeros(1, 1).fill_(en_vocab[start_symbol]).type_as(src.data).to(device)
    max_len = src.size()[1] + 20 - 2
    for i in range(max_len - 1):
        trg_mask = model.make_trg_mask(ys)
        output = model.decoder(ys, enc_rep, trg_mask, src_mask)
        # if i == 0:
        # output_idx = torch.randint(len(output[0]), (1,)).item()
        # else:
        output_idx = output[0].max(dim=1)[1][-1]
        if output_idx == en_vocab['<eos>']:
            break
        else:
            ys = torch.cat([ys, torch.zeros(1, 1).type_as(src.data).fill_(output_idx)], dim=1)
        
    return ys


def get_log_prob(logits, token_id):
    # Compute the softmax of the logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities)
    
    # Get the log probability of the token
    token_log_probability = log_probabilities[token_id].item()
    return token_log_probability


def greedy_sampling(logits, beams):
    return torch.topk(logits, beams).indices


def top_k_sampling(logits, temperature, top_k, beams):
    assert top_k >= 1
    assert beams <= top_k

    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    new_logits = torch.clone(logits)
    new_logits[indices_to_remove] = float('-inf')

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(new_logits / temperature, dim=-1)

    # Sample n tokens from the resulting distribution
    next_tokens = torch.multinomial(probabilities, beams)

    return next_tokens


def nucleus_sampling(logits, temperature, p, beams):
    assert p > 0
    assert p <= 1

    # Sort the probabilities in descending order and compute cumulative probabilities
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = torch.nn.functional.softmax(sorted_logits / temperature, dim=-1)
    cumulative_probabilities = torch.cumsum(probabilities, dim=-1)

    # Create a mask for probabilities that are in the top-p
    mask = cumulative_probabilities < p

    # If there's not n index where cumulative_probabilities < p, we use the top n tokens instead
    if mask.sum() > beams:
        top_p_index_to_keep = torch.where(mask)[0][-1].detach().cpu().tolist()
    else:
        top_p_index_to_keep = beams

    # Only keep top-p indices
    indices_to_remove = sorted_indices[top_p_index_to_keep:]
    sorted_logits[indices_to_remove] = float('-inf')

    # Sample n tokens from the resulting distribution
    probabilities = torch.nn.functional.softmax(sorted_logits / temperature, dim=-1)
    next_tokens = torch.multinomial(probabilities, beams)

    return next_tokens

    
def beam_search(input_ids, node, bar, length, beams, sampling, enc_rep, src_mask, model, graph, temperature=0.1):
    if length == 0:
        return None
    
    trg_mask = model.make_trg_mask(input_ids)
    outputs = model.decoder(input_ids, enc_rep, trg_mask, src_mask)
    predictions = outputs

    # Get the predicted next sub-word (here we use top-k search)
    logits = predictions[0, -1, :]

    if sampling == 'greedy':
        top_token_ids = greedy_sampling(logits, beams)
    elif sampling == 'top_k':
        top_token_ids = top_k_sampling(logits, temperature, 20, beams)
    elif sampling == 'nucleus':
        top_token_ids = nucleus_sampling(logits, temperature, 0.5, beams)

    for j, token_id in enumerate(top_token_ids):
        bar.update(1)

        # Compute the score of the predicted token
        token_score = get_log_prob(logits, token_id)
        cumulative_score = graph.nodes[node]['cumscore'] + token_score

        # Add the predicted token to the list of input ids
        new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)

        # Add node and edge to graph
        token = idx_to_words([token_id], trg_itos)#tokenizer.decode(token_id, skip_special_tokens=True)
        current_node = list(graph.successors(node))[j]
        graph.nodes[current_node]['tokenscore'] = np.exp(token_score) * 100
        graph.nodes[current_node]['cumscore'] = cumulative_score
        graph.nodes[current_node]['sequencescore'] = 1/(len(new_input_ids.squeeze())) * cumulative_score
        graph.nodes[current_node]['token'] = token + f"_{length}_{j}"

        # Recursive call
        beam_search(new_input_ids, current_node, bar, length-1, beams, sampling, enc_rep, src_mask, model, graph, 1)


def get_best_sequence(G):
    # Create a list of leaf nodes
    leaf_nodes = [node for node in G.nodes() if G.out_degree(node)==0]

    # Get the leaf node with the highest cumscore
    max_score_node = None
    max_score = float('-inf')
    for node in leaf_nodes:
        if G.nodes[node]['sequencescore'] > max_score:
            max_score = G.nodes[node]['sequencescore']
            max_score_node = node

    # Retrieve the sequence of nodes from this leaf node to the root node in a list
    path = nx.shortest_path(G, source=0, target=max_score_node)

    # Return the string of token attributes of this sequence
    sequence = " ".join([G.nodes[node]['token'].split('_')[0] for node in path])
    
    return sequence, max_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights_path', default='', type=str)
    parser.add_argument('--sampling_strategy', default='greedy', type=str)
    parser.add_argument('--generated_seq_length', default=10, type=int)
    parser.add_argument('--beam_size', default=2, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--input_text', default='', type=str)

    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Model Loading ...")
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
                    device=device).to(device)

    new_state_dict = modified_state_dict(args.model_weights_path)
    model.load_state_dict(new_state_dict)
    print("Model Loaded Successfully")

    src_tensor = torch.tensor([de_vocab[token] for token in de_tokenizer(args.input_text)], dtype=torch.long)
    src = torch.cat([torch.tensor(de_vocab['<sos>']).reshape(1), src_tensor, torch.tensor(de_vocab['<eos>']).reshape(1)], dim=0).unsqueeze(0).to(device)

    if args.sampling_strategy == 'greedy':
        token_ids = greedy_decoding(src, '<sos>', model, device)
        generated_sentence = idx_to_words(token_ids.squeeze(), trg_itos)
    else:
        src_mask = model.make_src_mask(src)
        enc_rep = model.encoder(src, src_mask)

        ys = torch.zeros(1, 1).fill_(en_vocab['<sos>']).type_as(src.data).to(device)

        # Create a balanced tree with height 'length' and branching factor 'k'
        graph = nx.balanced_tree(args.beam_size, args.generated_seq_length, create_using=nx.DiGraph())
        bar = tqdm(total=len(graph.nodes))

        # Add 'tokenscore', 'cumscore', and 'token' attributes to each node
        for node in graph.nodes:
            graph.nodes[node]['tokenscore'] = 100
            graph.nodes[node]['cumscore'] = 0
            graph.nodes[node]['sequencescore'] = 0
            graph.nodes[node]['token'] = '<sos>'

        # Start generating text
        beam_search(ys, 0, bar, args.generated_seq_length, args.beam_size, 'greedy', enc_rep, src_mask, model, graph, 1)
        sentence, max_score = get_best_sequence(graph)
        generated_sentence = sentence[5:]

    print(f"Translated Text: {generated_sentence}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
