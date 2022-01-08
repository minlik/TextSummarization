import os

import torch
import random
from utils import config
from src.model import Seq2seq, PGN
from utils.data_loader import *

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 模型预测过程
def predict(model, vocab, text, max_len=20):
    # 预测的长度大于 max_len 或者遇到 EOS_TOKEN 时停止
    model.eval()
    dec_words = []
    with torch.no_grad():
        # 处理输入
        src, oovs = vocab.convert_text_to_ids(text)
        src_lengths = torch.tensor([len(src)])
        src = torch.tensor(src).reshape(1, -1)
        src_copy = replace_oovs(src, vocab)
        enc_output, prev_hidden = model.encoder(src_copy, src_lengths)
        # Decoder 的第一个输入为 EOS_TOKEN
        dec_input = torch.tensor([vocab[config.BOS_TOKEN]]).to(device)
        # 依次处理每个时间步的 decoder 过程
        for t in range(max_len):
            dec_output, prev_hidden, attention_weights, p_gen = model.decoder(dec_input, prev_hidden, enc_output, src_lengths)
            final_distribution = model.get_final_distribution(src,p_gen, dec_output, attention_weights, len(oovs))
            dec_output = final_distribution.argmax(-1)
            token_id = dec_output.item()
            # 对 token_id 进行解码，转换成单词
            # 遇到 EOS_TOKEN 时停止
            if dec_output.item() == vocab[config.EOS_TOKEN]:
                dec_words.append(config.EOS_TOKEN)
                break
            # token_id 在 vocab 里面，直接输出
            elif token_id < len(vocab):
                dec_words.append(vocab.idx2token[token_id])
            # token_id 在 oovs 里面，输入 oovs 对应的该单词。oovs 来源于原文输入。
            elif token_id < len(vocab) + len(oovs):
                dec_words.append(oovs[token_id - len(vocab)])
            # 其他情况，输入 UNK_TOKEN
            else:
                dec_words.append(vocab.UNK_TOKEN)
            # 将 decoder output 作为下一个时刻的 decoder input，并将其中的 oovs 替换成 UNK_TOKEN
            dec_input = replace_oovs(dec_output, vocab)
    return dec_words


if __name__ == '__main__':
    train_text, train_title = load_data(config.train_save_path)
    if config.train_sample > 0:
        train_text = train_text[:config.train_sample]
        train_title = train_title[:config.train_sample]
    vocab = Vocab(train_text + train_title, reserved_tokens=config.reserved_tokens)

    # 加载训练好的模型
    model = PGN(vocab)
    model.load_state_dict((torch.load(config.model_load_path)))

    # 随机打印预测的结果
    for i in range(10):
        idx = random.randint(0, config.train_sample)
        text = train_text[idx].split()
        title = train_title[idx].split()
        print('>', ''.join(text))
        print('=', ''.join(title))
        output_words = predict(model, vocab, text)
        print('<', ''.join(output_words))
        print('')
