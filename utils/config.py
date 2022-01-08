import os
import torch

# 将路径加入到环境变量中
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 设置模型参数
emb_size = 128
hidden_size = 256
num_layers = 2
dropout = 0.5
# 设置 PGN + coverage
pointer = True
coverage = True
cov_lambda = 1 # 计算总体loss时，设置coverage loss的权重。

# 设置训练参数
batch_size = 16
epochs = 10
lr = 1e-3
max_grad_norm = 2 # 梯度最大截断值，避免出现梯度爆炸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_sample = 100 # 训练集采样，调试时使用。-1时使用全部训练集。
val_sample = 100 # 验证集采样，训练时使用。-1时为全部验证集。

# 设置文件路径
content_path = os.path.join(root_path, 'data', 'train_text.txt')
title_path = os.path.join(root_path, 'data', 'train_label.txt')
train_save_path = os.path.join(root_path, 'data', 'sina-article-train.txt')
val_save_path = os.path.join(root_path, 'data', 'sina-article-test.txt')
model_load_path = os.path.join(root_path, 'src', 'saved_model', 'model.pt')

# 定义词典中预留的 token
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
reserved_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
