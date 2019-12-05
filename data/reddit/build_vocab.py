"""Builds vocabulary file from data."""

import argparse
import collections
import json
import os
import pickle
import re


def build_counter(train_data, initial_counter=None):
    train_tokens = []
    for u in train_data:
        for c in train_data[u]['x']:
            train_tokens.extend([s for s in c])

    all_tokens = []
    for i in train_tokens:
        all_tokens.extend(i)    
    train_tokens = []

    if initial_counter is None:
        counter = collections.Counter()
    else:
        counter = initial_counter

    counter.update(all_tokens)
    all_tokens = []

    return counter


def build_vocab(counter, vocab_size=10000):
    pad_symbol, unk_symbol = 0, 1
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = count_pairs[:(vocab_size - 2)] # -2 to account for the unknown and pad symbols

    words, _ = list(zip(*count_pairs))

    vocab = {}
    vocab['<PAD>'] = pad_symbol
    vocab['<UNK>'] = unk_symbol

    for i, w in enumerate(words):
        if w != '<PAD>':
            vocab[w] = i + 1

    return {'vocab': vocab, 'size': vocab_size, 'unk_symbol': unk_symbol, 'pad_symbol': pad_symbol}


def load_leaf_data(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        to_ret = data['user_data']
        data = None
    return to_ret


def save_vocab(vocab, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    pickle.dump(vocab, open(os.path.join(target_dir, 'reddit_vocab.pck'), 'wb'))


def main():
    args = parse_args()

    json_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    json_files.sort()

    counter = None
    train_data = {}
    for f in json_files:
        print('loading {}'.format(f))
        train_data = load_leaf_data(os.path.join(args.data_dir, f))
        print('counting {}'.format(f))
        counter = build_counter(train_data, initial_counter=counter)
        print()
        train_data = {}

    if counter is not None:
        vocab = build_vocab(counter, vocab_size=args.vocab_size)
        save_vocab(vocab, args.target_dir)
    else:
        print('No files to process.')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', 
        help='dir with training file;',
        type=str,
        required=True)
    parser.add_argument('--vocab-size', 
        help='size of the vocabulary;',
        type=int,
        default=10000,
        required=False)
    parser.add_argument('--target-dir', 
        help='dir with training file;',
        type=str,
        default='./',
        required=False)

    return parser.parse_args()


if __name__ == '__main__':
    main()
