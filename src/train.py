import os
import sys

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt
import torch
from torch import nn
import datetime

from model import Seq2seq, PGN
from src.evaluate import evaluate
from utils.data_loader import Vocab, MyDataset, load_data
from utils import config

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


def print_bar():
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + now_time)


def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs):
    # 模型训练过程
    model = model.to(config.device)
    model.train()
    print_bar()
    print('Start Training...')
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        total_loss = 0
        for i, (text, text_len, title_in, title_out, oovs, len_oovs) in enumerate(train_dataloader):
            text = text.to(config.device)
            title_in = title_in.to(config.device)
            title_out = title_out.to(config.device)
            optimizer.zero_grad()
            title_pred, attention_weights, coverage_vector = model(text, title_in, text_len, len_oovs)
            # 计算 cross entropy loss
            ce_loss = loss_fn(title_pred.transpose(1, 2).to(config.device), title_out)
            if config.coverage:
                # 计算 coverage loss
                c_t = torch.min(attention_weights, coverage_vector)
                cov_loss = torch.mean(torch.sum(c_t, dim=1))
                # 计算整体 loss
                loss = ce_loss + config.cov_lambda * cov_loss
            total_loss += loss.item()
            loss.backward()
            # 梯度截断
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        # 每个 epoch 结束，验证模型的精度
        avg_val_loss = evaluate(model, val_dataloader, loss_fn)
        print_bar()
        print(f'epoch: {epoch+1}/{epochs}, training loss: {avg_train_loss:.4f}, validation loss: {avg_val_loss:.4f}')
        # if epoch == 0 or avg_val_loss < min_val_loss:
        if (epoch + 1) % 20 == 0:
            model_path = root_path + '/src/saved_model/' + 'model_' + str(epoch+1) + '.pt'
            torch.save(model.state_dict(), model_path)
            print(f'The model has been saved for epoch {epoch + 1}')
        # min_val_loss = avg_val_loss

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
    return train_loss, val_loss


def collate_fn(batch):
    # 将 dataset 中的数据进行整理，得到 dataloader 需要的格式
    # 1. text 和 title 加入 padding 处理，统一每个 batch 中的句子长度
    # 2. 统计原文中的 oov 单词
    is_train = 'title_ids' in batch[0]
    text = [torch.tensor(example['text_ids']) for example in batch]
    text_len = torch.tensor([len(example['text_ids']) for example in batch])
    padded_text = pad_sequence(text, batch_first=True, padding_value=vocab[config.PAD_TOKEN])
    oovs = [example['oovs'] for example in batch]
    len_oovs = [example['len_oovs'] for example in batch]
    if is_train:
        title_in = [torch.tensor(example['title_ids'][:-1]) for example in batch]
        title_out = [torch.tensor(example['title_ids'][1:]) for example in batch]
        padded_title_in = pad_sequence(title_in, batch_first=True, padding_value=vocab[config.PAD_TOKEN])
        padded_title_out = pad_sequence(title_out, batch_first=True, padding_value=vocab[config.PAD_TOKEN])
        return padded_text, text_len, padded_title_in, padded_title_out, oovs, len_oovs
    return padded_text, text_len, oovs, len_oovs


if __name__ == '__main__':
    train_text, train_title = load_data(config.train_save_path)
    if config.train_sample > 0:
        train_text = train_text[:config.train_sample]
        train_title = train_title[:config.train_sample]
    vocab = Vocab(train_text + train_title, reserved_tokens=config.reserved_tokens)
    train_dataset = MyDataset(vocab, train_text, train_title)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)

    val_text, val_title = load_data(config.val_save_path)
    if config.val_sample > 0:
        val_text = val_text[:config.val_sample]
        val_title = val_title[:config.val_sample]
    val_dataset = MyDataset(vocab, val_text, val_title)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)

    model = PGN(vocab)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab[config.PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loss, val_loss = train(model, train_dataloader, val_dataloader, loss_fn, optimizer, config.epochs)
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend()
    plt.title('Training and validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
