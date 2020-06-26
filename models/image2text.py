import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CocoCaptions

from encoders import EmbeddingGridInception

import pandas as pd
import os
import sys
import random

from torch.utils.data import Dataset, DataLoader
from transformers.optimization import AdamW

from PIL import Image
import requests
from io import BytesIO
from IPython.display import display
from multiprocessing.pool import ThreadPool

from IPython.display import display
from collections import namedtuple
from time import perf_counter
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from torch.distributions.categorical import Categorical
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch.nn.functional as F

class Image2TextDescriptor(nn.Module):
    def __init__(self,
                 device,
                 SPECIAL_TOKENS=('img', 'desc', 'pad'),
                 pretrained_model_text='gpt2',
                 max_description_len=40):
        super().__init__()
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SpecialIds = namedtuple('speacil_ids', self.SPECIAL_TOKENS)
        self.max_length = max_description_len
        self.device = device

        self.tokenizer, self.special_ids = self._build_tokenizer(pretrained_model_text)
        self.gpt2 = self._build_gpt2(len(self.tokenizer), pretrained_model_text)

        # cnn = self._build_cnn(pretrained_model_image)
        # self.embedding = EmbeddingFinalLayer(cnn, self.gpt2.config.hidden_size)#.to(device)
        self.embedding = EmbeddingGridInception(self.gpt2.config.hidden_size)
        self.to(device)

    def train(self, mode=True):
        self.embedding.train(mode)
        self.gpt2.train(mode)
        for param in self.gpt2.parameters():
            param.requires_grad_(mode)

    def target_transform(self, target):
        target = random.choice(target)
        text_tok_ids = self.tokenizer.encode(target, max_length=self.max_length)
        text_tok_ids.extend([self.tokenizer.eos_token_id])
        return text_tok_ids

    def save(self, path='weights/model_full_trained'):
        state = {
            'gpt_dict': self.gpt2.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'tokenizer': self.tokenizer,
        }
        torch.save(state, path)
        # print('!!! model is saved !!!')

    def load(self, path='weights/model_full_trained'):
        state = torch.load(path)
        self.gpt2.load_state_dict(state['gpt_dict'])
        self.embedding.load_state_dict(state['embedding_dict'])
        self.tokenizer = state['tokenizer']
        # print('!!! model weights are loaded !!!')

    def forward(self, img_data, text_tok_ids, attn_mask=None):
        context_embeds = self.get_context_embeds(img_data)
        text_embeds = self.gpt2.transformer.wte(text_tok_ids)
        full_embeds = torch.cat((context_embeds, text_embeds), dim=1)
        tok_type_ids = torch.tensor([[self.special_ids.img] * (context_embeds.size(1) - 1) +
                                     [self.special_ids.desc] * (text_embeds.size(1) + 1)], dtype=torch.long)
        tok_type_ids = tok_type_ids.expand(context_embeds.size(0), -1).to(self.device)

        if attn_mask is not None:
            mask = torch.ones_like(tok_type_ids, dtype=torch.float).to(self.device)
            mask[:, context_embeds.size(1):] = attn_mask
            attn_mask = mask

        labels_ids = self.get_labels(
            full_embeds.shape[:-1],
            context_embeds.size(1),
            text_tok_ids
        ).to(self.device)

        out = self.gpt2(inputs_embeds=full_embeds,
                        labels=labels_ids,
                        token_type_ids=tok_type_ids,
                        attention_mask=attn_mask)
        return out

    def get_context_embeds(self, img_data):
        img_tok_emb, desc_tok_emb = self.gpt2.transformer.wte(
            torch.tensor([self.special_ids.img, self.special_ids.desc]).to(self.device)
        )

        img_embeds = self.embedding(img_data)
        context_embeds = torch.cat((img_tok_emb.expand(img_embeds.size(0), 1, -1),
                                    img_embeds,
                                    desc_tok_emb.expand(img_embeds.size(0), 1, -1)), dim=1)

        return context_embeds

    def get_labels(self, shape, context_len, tok_ids):
        labels_ids = torch.empty(shape, dtype=torch.long).fill_(
            -100)  # -100 is mask value for labels in hugginface models
        mask = (tok_ids == self.special_ids.pad)
        labels_ids[:, context_len:] = tok_ids.masked_fill(mask, -100)

        return labels_ids

    def _build_tokenizer(self, pretrained_model):
        special_dct = {t: f"<{t.upper()}>" for t in self.SPECIAL_TOKENS}
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        tokenizer.add_special_tokens({'additional_special_tokens': list(special_dct.values())})
        special_ids = self.SpecialIds(**{k: tokenizer.encode(v)[0] for k, v in special_dct.items()})

        return tokenizer, special_ids

    def _build_gpt2(self, vocab_size, pretrained_model):
        gpt2 = AutoModelWithLMHead.from_pretrained(pretrained_model)
        gpt2.resize_token_embeddings(vocab_size)

        return gpt2

    def _build_cnn(self, pretrained_model):
        if pretrained_model == 'resnet18':
            return models.resnet18(pretrained=True)
        else:
            raise ValueError(f'{pretrained_model} is not supported yet :(')