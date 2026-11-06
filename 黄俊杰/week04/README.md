# week04 作业：Transformer 语言模型

- 任务：字符级语言模型，输入 64 个字预测下一个字，语料 `corpus_general.txt`（6,021,645 字）
- 模型：Embedding 128 + 位置编码 + 2 层 Transformer Block，参数量 1.30 M
- 训练：Adam，lr 1e-3，batch 128，seq_len 64，5 轮
- 结果：验证集 PPL 71.25 → 41.97，权重 `best_model.pt`，曲线 `train-curve.png`
- 续写（temperature 0.8）：

```
人工智能 -> 人工智能在哪
今天天气 -> 今天天气净烟。
我想问一下 -> 我想问一下是什么意思？
```

- 运行：`corpus_general.txt` 放脚本同目录，执行 `python language_model.py`
