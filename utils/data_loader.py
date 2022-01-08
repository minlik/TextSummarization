import os
import sys

from utils import config

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from utils.config import *


# 生成词典
class Vocab:
    def __init__(self, sentences, min_freq=1, reserved_tokens=None):
        self.idx2token = list()
        self.token2idx = {}
        token_freqs = defaultdict(int)
        self.UNK_TOKEN = '<UNK>'
        for sentence in sentences:
            for token in sentence.split(' '):
                token_freqs[token] += 1
        unique_tokens = reserved_tokens if reserved_tokens else []
        unique_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq]
        if self.UNK_TOKEN not in unique_tokens:
            unique_tokens = [self.UNK_TOKEN] + unique_tokens
        for token in unique_tokens:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        self.unk = self.token2idx[self.UNK_TOKEN]

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, token):
        return self.token2idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx2token[idx] for idx in ids]

    # 将 source token 转化为 ids, 其中 unk_token 加入到 oovs
    def convert_text_to_ids(self, text_tokens):
        ids = []
        oovs = []
        unk_id = self.unk
        for token in text_tokens:
            i = self[token]
            if i == unk_id:
                if token not in oovs:
                    oovs.append(token)
                oov_idx = oovs.index(token)
                ids.append(oov_idx + len(self))
            else:
                ids.append(i)
        return ids, oovs

    # 将 title token 转化为 ids，考虑 source token 中出现的 oovs
    def convert_title_to_ids(self, title_tokens, oovs):
        ids = []
        unk_id = self.unk
        for token in title_tokens:
            i = self[token]
            if i == unk_id:
                if token in oovs:
                    token_idx = oovs.index(token) + len(self)
                    ids.append(token_idx)
                else:
                    ids.append(unk_id)
            else:
                ids.append(i)
        return ids


class MyDataset(Dataset):
    def __init__(self, vocab, text, title=None):
        self.is_train = True if title is not None else False
        self.vocab = vocab
        self.text = text
        self.title = title

    def __getitem__(self, i):
        # 得到原文中的 token_id，以及 oovs
        text_ids, oovs = self.vocab.convert_text_to_ids(self.text[i].split())
        if not self.is_train:
            return {'text_ids': text_ids,
                    'oovs': oovs,
                    'len_oovs': len(oovs)}
        else:
            # title 的首尾分别加入 BOS_TOKEN 和 EOS_TOKEN
            title_ids = [self.vocab[BOS_TOKEN]] + self.vocab.convert_title_to_ids(self.title[i].split(), oovs) + [self.vocab[EOS_TOKEN]]
            return {'text_ids': text_ids,
                    'oovs': oovs,
                    'len_oovs': len(oovs),
                    'title_ids': title_ids}

    def __len__(self):
        return len(self.text)


def load_data(path):
    # 数据的加载
    with open(path, 'r') as f:
        lines = f.readlines()
    xs, ys = [], []
    for line in lines:
        x, y = line.split('\t')
        xs.append(x.strip())
        ys.append(y.strip())
    return xs, ys


def replace_oovs(in_tensor, vocab):
    # 将文本张量中所有OOV单词的id, 全部替换成 UNK_TOKEN 对应的 id，以便模型可以直接处理
    oov_token = torch.full(in_tensor.shape, vocab.unk, dtype=torch.long).to(config.device)
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor


