import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
import torch
from utils import config

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


# 模型验证过程
def evaluate(model, val_dataloader, loss_fn):
    val_loss = []
    model.eval()
    with torch.no_grad():
        val_loss = []
        for i, (text, text_len, title_in, title_out, oovs, len_oovs) in enumerate(val_dataloader):
            text = text.to(config.device)
            title_in = title_in.to(config.device)
            title_out = title_out.to(config.device)
            title_pred, _, _ = model(text, title_in, text_len, len_oovs)
            loss = loss_fn(title_pred.transpose(1, 2).to(config.device), title_out)
            val_loss.append(loss.item())
    return np.mean(val_loss)
