# week04 作业：Transformer 语言模型

- 任务：字符级语言模型，输入 64 个字预测下一个字，语料用课件的 `corpus_general.txt`（6,021,645 字）
- 模型：Embedding 128 + 位置编码 + 2 层 Block（4 头自注意力、FFN 128→256→128、Pre-LN）+ 输出层，参数量 1.30 M
- 训练：Adam，lr = 1e-3，batch = 128，seq_len = 64，5 轮
- 结果：验证集 PPL 71.25 → 41.97，权重 `best_model.pt`，曲线 `train-curve.png`

## 效果

用 `best_model.pt` 续写三句（temperature = 0.8，30 个字，只取到第一个换行）：

```
人工智能 -> 人工智能在哪
今天天气 -> 今天天气净烟。
我想问一下 -> 我想问一下是什么意思？
```

## 流程

1. 读取语料，按出现频率建 4000 字的字表
2. 语料按行打乱后切 95% 训练 / 5% 验证
3. 切样本：连续 64 个字为输入，往后错一位为目标
4. 训练 5 轮，每轮在验证集上算 PPL，存 PPL 最低的那轮权重
5. 训练结束画曲线，并用最优权重续写三句

## 运行

1. 把课件的 `corpus_general.txt` 放到脚本同目录
2. `python language_model.py`
3. 终端打印每轮 loss / PPL，目录下生成 `train-curve.png`、`best_model.pt`

## 文件

- `language_model.py`：训练 + 生成脚本
- `best_model.pt`：验证集 PPL 最低的权重（含字表）
- `train-curve.png`：PPL / loss 曲线
