### 一、介绍
本项目实现了生成式自动文本摘要的模型训练及预测，项目用到的主要架构如下：
1. seq2seq：利用 Encoder-Decoder 架构，结合 Attention，对文本摘要生成任务进行建模和训练。Encoder 和 Decoder 中，分别使用双层和单层 GRU 结构来抽取文本特征。
2. PGN：使用PGN(Pointer Generator Network)，可以直接 copy 原文中的重要单词和短语，缓解文本摘要生成过程中可能出现 OOV 问题。
3. Coverage 机制：在模型中，加入 Coverage Loss，对过往时刻已经生成的单词进行惩罚，缓解文本摘要生成过程中可能出现的重复问题。

关于本项目中的各项细节可以参考以下文章：

[文本摘要（一）：任务介绍](https://zhuanlan.zhihu.com/p/451808468)

[文本摘要（二）：TextRank](https://zhuanlan.zhihu.com/p/452359234)

[文本摘要（三）：数据处理](https://zhuanlan.zhihu.com/p/452359234)

[文本摘要（四）：seq2seq 介绍及实现](https://zhuanlan.zhihu.com/p/452475603)

[文本摘要（五）：seq2seq 训练及预测](https://zhuanlan.zhihu.com/p/452703432)

[文本摘要（六）：生成任务中的采样方法](https://zhuanlan.zhihu.com/p/453286395)

[文本摘要（七）：PGN 模型架构](https://zhuanlan.zhihu.com/p/453600830)

### 二、 框架
```
├── data
│   ├── sina-article-test.txt
│   ├── sina-article-train.txt
│   ├── train_label.txt
│   └── train_text.txt
├── src
│   ├── saved_model
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
└── utils
    ├── config.py
    ├── data_loader.py
    ├── multi_proc_utils.py
    ├── params_utils.py
    └── preprocess.py 
```

### 三、使用
1. 数据预处理，生成分词后的 'data/sina-article-train.txt' 及 'data/sina-article-test.txt' 文件.
```bash
python utils/preprocess.py
```
2. 模型训练，训练好的模型储存在 'src/saved_model' 文件夹中。
```bash
python src/train.py
```
3. 模型预测，修改 config 中的模型加载路径 model_load_path
```bash
python src/predict.py
```

### 四、TODO
- [ ] ROUGE 指标评测
- [ ] 参数调整
- [ ] 模型部署