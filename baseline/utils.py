import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import nltk
import string
nltk.download('punkt')
from collections import defaultdict
import json
from os.path import exists
import torch

def plot_img(img):
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    default_img = np.clip(np.array(img).transpose(1,2,0)*std + mean, 0, 1)
    plt.axis('off')
    plt.imshow(default_img)
    plt.show()

def tokenize(text):
    return [token for token in word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation)))]

def get_all_captions(dataset):
    # if exists('captions.json'):
    #     with open('captions.json', 'r') as f:
    #         return json.load(f)
    all_captions = []
    d = len(dataset)
    for i, (_, captions) in enumerate(dataset):
        print(f'{i/d*100:.2f}%\r', end='')
        all_captions.extend(captions)
    print()
    return all_captions
        
def tokenize_captions(captions):
    return [tokenize(cap) for cap in captions]

def build_dict(tokens, special_tokens, tok2idx=None, idx2tok=None):
    """
        tokens: a list of lists of tokens
        special_tokens: some special tokens
    """
    if tok2idx is None or idx2tok is None:
        tok2idx = defaultdict(lambda: 0)
        idx2tok = []
    
    for t in chain(special_tokens, *tokens):
        if t not in tok2idx:
            tok2idx[t] = len(idx2tok)
            idx2tok.append(t)
    
    return tok2idx, idx2tok

def build_dict_from(path):
    with open(path, 'r') as f:
        idx2tok = json.load(f)
    tok2idx = defaultdict(lambda: 0, [(t, i) for i, t in enumerate(idx2tok)])
    return tok2idx, idx2tok

def captions2idxs(captions, words2idxs, max_length=20):
    cs = [tokenize(c) for c in captions]
    length = min(max_length, max(len(c) for c in cs)) + 2
    return torch.tensor([to_pad_idxs(c, words2idxs, length) for c in cs], dtype=torch.long)

def to_pad_idxs(tokens, words2idxs, seq_length):
    ts = ['<START>'] + tokens
    length = min(len(ts) + 1, seq_length)
    return words2idxs(ts[:seq_length - 1] + ['<END>'] + ['<PAD>'] * (seq_length - length))
