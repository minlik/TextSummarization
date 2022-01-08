import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import random
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import config
from utils.data_loader import replace_oovs, load_data, Vocab


class Encoder(nn.Module):
    def __init__(self, vocab):
        super(Encoder, self).__init__()
        self.vocab_size = len(vocab)
        # Embedding 层设定 padding_idx 值
        self.embedding = nn.Embedding(self.vocab_size, config.emb_size, padding_idx=vocab[config.PAD_TOKEN])
        self.gru = nn.GRU(config.emb_size, config.hidden_size, num_layers=config.num_layers, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, enc_input, text_lengths):
        # enc_input: [batch_size, seq_len], 经过 padding 处理过的输入token_id
        # text_lengths: [batch_size], padding 之前，输入 tokens 的长度
        embedded = self.dropout(self.embedding(enc_input)) # [batch_size, seq_len, emb_size]
        # 输入 GRU 前，将 padded_sequence 打包，去掉 padding token，加快训练速度
        embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        # GRU 的输出 分为两部分，output 对应每个 token 的最后一层隐状态，hidden 对应最后一个字符的所有层的隐状态
        # output: [batch_size, seq_len, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        output, hidden = self.gru(embedded)
        # GRU 训练完成后，再恢复 padding token 状态
        output, _ = pad_packed_sequence(output, batch_first=True)
        # 输出再经过一个 linear 层，增加模型复杂度
        output = self.relu(self.linear(output))
        return output, hidden[-1].detach()


class Decoder(nn.Module):
    def __init__(self, vocab, attention):
        super(Decoder, self).__init__()
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, config.emb_size, padding_idx=vocab[config.PAD_TOKEN])
        self.attention = attention
        self.gru = nn.GRU(config.emb_size + config.hidden_size, config.hidden_size, batch_first=True)
        self.linear = nn.Linear(config.emb_size + 2 * config.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        # 设置 PGN 网络架构的参数，用于计算 p_gen
        if config.pointer:
            self.w_gen = nn.Linear(config.hidden_size * 2 + config.emb_size, 1)

    def forward(self, dec_input, prev_hidden, enc_output, text_lengths, coverage_vector):
        # 与 Encoder 不同，Decoder 的计算是分步进行的，每次输入一个时间步的 dec_input，同时输出这个时间步的 dec_output
        # dec_input = [batch_size]
        # prev_hidden = [batch_size, hidden_size]
        # enc_output = [batch_size, src_len, hidden_size]
        dec_input = dec_input.unsqueeze(1) # [batch_size, 1]
        embedded = self.embedding(dec_input) # [batch_size, 1, dec_len]
        # 加入 coverage 机制后，attention 的计算公式参考 https://zhuanlan.zhihu.com/p/453600830
        attention_weights, coverage_vector = self.attention(embedded, enc_output, text_lengths, coverage_vector)
        attention_weights = attention_weights.unsqueeze(1) # [batch_size, 1, enc_len]
        # 根据 attention weights，计算 context vector
        c = torch.bmm(attention_weights, enc_output) # [batch_size, 1, hidden_size]
        # 将经过 embedding 处理过的 decoder 输入，和上下文向量一起送入到 GRU 网络中
        gru_input = torch.cat([embedded, c], dim=2)
        # dec_output: [batch_size, 1, hidden_size]
        # dec_hidden: [1, batch_size, hidden_size]
        # prev_hidden 是上个时间步的隐状态，作为 decoder 的参数传入进来
        dec_output, dec_hidden = self.gru(gru_input, prev_hidden.unsqueeze(0))
        # 将输出映射到 vocab_size 维度，以便计算每个 vocab 的生成概率
        dec_output = self.linear(torch.cat((dec_output.squeeze(1), c.squeeze(1), embedded.squeeze(1)), dim=1)) # [batch_size, vocab_size]
        dec_hidden = dec_hidden.squeeze(0)
        p_gen = None
        # 计算 p_gen
        if config.pointer:
            x_gen = torch.cat([dec_hidden, c.squeeze(1), embedded.squeeze(1)], dim=1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))
        return dec_output, dec_hidden, attention_weights.squeeze(1), p_gen, coverage_vector


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.linear = nn.Linear(config.hidden_size * 2 + config.emb_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_input, enc_output, text_lengths, coverage_vector):
        # enc_output = [batch_size, seq_len, hidden_size]
        # dec_input = [batch_size, hidden_size]
        # text_lengths = [batch_size]
        # coverage_vector = [batch_size, seq_len]
        seq_len = enc_output.shape[1]
        hidden_size = enc_output.shape[-1]
        s = dec_input.repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]
        coverage_vector_copy = coverage_vector.unsqueeze(2).repeat(1, 1, hidden_size) # [batch_size, seq_len, hidden_size]
        # enc_output, s, coverage_vector_copy 维度统一，用于计算 attention
        x = torch.tanh(self.linear(torch.cat([enc_output, s, coverage_vector_copy], dim=2)))
        attention = self.v(x).squeeze(-1) # [batch_size, seq_len]
        max_len = enc_output.shape[1]
        # mask = [batch_size, seq_len]，遮蔽掉 Decoder 当前时间步之后的单词
        mask = torch.arange(max_len).expand(text_lengths.shape[0], max_len) >= text_lengths.unsqueeze(1)
        attention.masked_fill_(mask.to(config.device), float('-inf'))
        attention_weights = self.softmax(attention)
        # 更新 coverage_vector
        coverage_vector += attention_weights
        return attention_weights, coverage_vector # [batch, seq_len], [batch_size, seq_len]


# seq2seq 模型架构
class Seq2seq(nn.Module):
    def __init__(self, vocab):
        super(Seq2seq, self).__init__()
        attention = Attention()
        self.encoder = Encoder(vocab)
        self.decoder = Decoder(vocab, attention)

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        # src = [batch_size, src_len]
        # tgt = [batch_size, tgt_len]
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.vocab_size
        enc_output, prev_hidden = self.encoder(src, src_lengths)
        dec_input = tgt[:, 0]
        dec_outputs = torch.zeros(batch_size, tgt_len, vocab_size)
        for t in range(tgt_len - 1):
            dec_output, prev_hidden, _, _ = self.decoder(dec_input, prev_hidden, enc_output, src_lengths)
            dec_outputs[:, t, :] = dec_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = tgt[:, t] if teacher_force else top1
        return dec_outputs


# PGN 模型架构
class PGN(nn.Module):
    def __init__(self, vocab):
        super(PGN, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.device = config.device

        attention = Attention()
        self.encoder = Encoder(vocab)
        self.decoder = Decoder(vocab, attention)

    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        # 应用 PGN 公式，计算最终单词的概率分布。由于PGN网络会copy原文中的单词，因此需要考虑原文中 OOV 单词的影响
        if not config.pointer:
            return p_vocab
        batch_size = x.shape[0]
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        p_vocab_weighted = p_gen * p_vocab
        attention_weighted = (1 - p_gen) * attention_weights
        # 加入 max_oov 维度，将原文中的 OOV 单词考虑进来
        extension = torch.zeros((batch_size, max_oov), dtype=torch.float).to(self.device)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=-1)
        # p_gen * p_vocab + (1 - p_gen) * attention_weights, 将 attention weights 中的每个位置 idx 映射成该位置的 token_id
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)
        # 输出最终的 vocab distribution [batch_size, vocab_size + len(oov)]
        return final_distribution

    def forward(self, src, tgt, src_lengths, len_oovs, teacher_forcing_ratio=0.5):
        # src = [batch_size, src_len]，Encoder 原文输入
        # tgt = [batch_size, tgt_len]，Decoder 摘要输入
        # src_lengths = [batch_size]， Encoder 原文长度
        # len_oovs = [batch_size, max_oovs]， Encoder 原文中 oov 的长度

        # 将 oov 替换成 <UNK>， 以便 Encoder 可以处理
        src_copy = replace_oovs(src, self.vocab)
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.vocab_size
        # encoder 过程
        enc_output, prev_hidden = self.encoder(src_copy, src_lengths)
        # decoder 的第一个输入
        dec_input = tgt[:, 0]
        dec_outputs = torch.zeros(batch_size, tgt_len, vocab_size + max(len_oovs))
        coverage_vector = torch.zeros_like(src, dtype=torch.float32).to(config.device)
        # 依次处理每一个 decoder 时间步的输入
        for t in range(tgt_len - 1):
            # 将 oov 替换成 <UNK>， 以便 Dncoder 可以处理
            dec_input = replace_oovs(dec_input, self.vocab)
            dec_output, prev_hidden, attention_weights, p_gen, coverage_vector = self.decoder(dec_input, prev_hidden, enc_output, src_lengths, coverage_vector)
            final_distribution = self.get_final_distribution(src, p_gen, dec_output, attention_weights, max(len_oovs))
            # 随机使用 teacher forcing 训练，增加模型稳定性
            teacher_force = random.random() < teacher_forcing_ratio
            # 将这个时间步得到的每个单词的概率，赋值给 dec_outputs
            dec_outputs[:, t, :] = final_distribution
            top1 = final_distribution.argmax(1)
            dec_input = tgt[:, t] if teacher_force else top1
        return dec_outputs, attention_weights, coverage_vector


if __name__ == '__main__':
    train_text, train_title = load_data(config.train_save_path)
    train_text = train_text[:1000]
    train_title = train_title[:1000]
    vocab = Vocab(train_text + train_title, reserved_tokens=config.reserved_tokens)
    model = PGN(vocab)
    print(model)
