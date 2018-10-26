"""Utils for language models."""

import json
import numpy as np
import re


# ------------------------
# utils for shakespeare dataset

ALL_LETTERS = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):
    """Returns one-hot vector with given size and value 1 at given index."""
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    """Returns one-hot representation of given letter."""
    index = max(0,ALL_LETTERS.find(letter)) # treating ' ' as unknown character
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(max(0, ALL_LETTERS.find(c))) # added max to account for -1
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    """Split given line/phrase into list of words

    Args:
        line: string representing phrase to be split
    
    Return:
        list of strings, with each string representing a word
    """
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    """Returns index of given word based on given lookup dictionary

    returns the length of the lookup dictionary if word not found

    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    """
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, indd, max_words=25):
    """Converts given phrase into list of word indices
    
    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer 
    representing unknown index to returned list until the list's length is 
    max_words

    Args:
        line: string representing phrase/sequence of words
        indd: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list

    Return:
        indl: list of word indices, one index for each word in phrase
    """
    line_list = split_line(line) # split phrase in words
    indl = []
    for word in line_list:
        cind = _word_to_index(word, indd)
        indl.append(cind)
        if (len(indl) == max_words):
            break
    for i in range(max_words - len(indl)):
        indl.append(len(indd))
    return indl


def bag_of_words(line, vocab):
    """Returns bag of words representation of given phrase using given vocab.

    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values

    Return:
        integer list
    """
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def get_word_emb_arr(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab


def val_to_vec(size, val):
    """Converts target into one-hot.

    Args:
        size: Size of vector.
        val: Integer in range [0, size].
    Returns:
         vec: one-hot vector with a 1 in the val element.
    """
    assert 0 <= val < size
    vec = [0 for _ in range(size)]
    vec[int(val)] = 1
    return vec

