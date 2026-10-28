# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

"""
基于 pytorch 框架编写模型训练（softmax 多分类版）

规律：x 是一个 5 维向量，它最大的那个数出现在第几维，这个样本就属于第几类。
      例：x = [0.10, 0.90, 0.30, 0.20, 0.50]
          最大值 0.90 在第 2 维 -> 第 2 类
      内部标签用 0~4 表示（0 对应第 1 维 / 第 1 类），打印时显示成 1~5。

与 TorchDemo.py（二分类）的区别：
  1. 输出层从 1 维改成 5 维：1 维只能表示"是/否"，5 类要 5 个输出分数；
  2. sigmoid 换成 softmax：把 5 个分数压成 5 个和为 1 的概率；
  3. 均方差损失换成交叉熵 nn.CrossEntropyLoss
     —— 它内部已经带了 log_softmax，所以模型 forward 直接输出 logits 就行，
        千万不要自己再套一次 softmax 再喂给它；
  4. 标签必须是 LongTensor，形状是 (batch_size,) 而不是 (batch_size, 1)；
  5. 判断结果用 argmax（取概率最大的那一类），而不是和 0.5 比大小。

关于精度：一个线性层理论上就能学会这个规律（权重矩阵学成"对角大、其余小"的
对比矩阵，argmax(Wx) 就等于 argmax(x)），但交叉熵在接近 100% 时梯度会变得很小，
所以准确率通常停在 98%~99%，属于正常现象，不是代码写错了。
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_softmax.bin")
INPUT_SIZE = 5      # 输入向量维度
NUM_CLASSES = 5     # 类别数：5 维 -> 5 类


def normalize(x):
    """把 [0, 1] 的随机数平移到 [-1, 1]。
    只是缩放，不改变"哪一维最大"这个规律（argmax 不变），
    但让 5 个维度之间的差异更明显，训练收敛快很多（约 0.88 -> 0.99）。
    训练、测试、预测必须用同一个变换，所以统一放在这里调用。"""
    return (x - 0.5) * 2


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)   # 线性层，输出 5 个 logits
        self.loss = nn.functional.cross_entropy            # 交叉熵损失（自带 log_softmax）

    # 当输入真实标签，返回loss值；无真实标签，返回预测的 logits
    def forward(self, x, y=None):
        logits = self.linear(x)          # (batch_size, 5) -> (batch_size, 5)
        if y is not None:
            return self.loss(logits, y)  # 预测值和真实值计算损失，y 是 LongTensor
        else:
            return logits                # 输出 logits，要概率时再套 softmax


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值在第几维，就属于第几类（0~4）
def build_sample():
    x = np.random.random(INPUT_SIZE)
    y = int(np.argmax(x))  # 最大值所在的维度下标 = 类别
    return x, y


# 随机生成一批样本
# 5 个类别的样本数量天然是均匀的（每个维度成为最大值的概率都是 1/5）
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 输入是 float，标签必须是 long（交叉熵要求）
    return torch.FloatTensor(normalize(np.array(X))), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)
    counts = [0] * NUM_CLASSES
    for t in y.tolist():
        counts[t] += 1
    print("本次预测集中各类样本个数：" + "，".join(
        "第%d类:%d" % (i + 1, c) for i, c in enumerate(counts)))
    with torch.no_grad():
        logits = model(x)                        # model.forward(x)，拿到 logits
        y_pred = torch.argmax(logits, dim=1)     # 分数最高的那一类 = 预测类别
        correct = int((y_pred == y).sum())
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    # 配置参数
    epoch_num = 30        # 训练轮数
    batch_size = 20       # 每次训练样本个数
    train_sample = 2000   # 每轮训练总共训练的样本总数
    learning_rate = 0.05  # 学习率

    # 固定随机种子，方便复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 建立模型
    model = TorchModel(INPUT_SIZE, NUM_CLASSES)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 每轮打乱一次，保证每个 batch 里 5 类样本混在一起
        index = torch.randperm(train_sample)
        for batch_index in range(train_sample // batch_size):
            idx = index[batch_index * batch_size: (batch_index + 1) * batch_size]
            x = train_x[idx]     # 取出一个batch数据作为输入
            y = train_y[idx]
            loss = model(x, y)   # 计算loss  model.forward(x,y)
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            optim.zero_grad()    # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)    # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print("模型已保存到：%s" % MODEL_PATH)

    # 看看模型学到了什么：理想情况下对角线上的权重明显大于其它位置
    print("学到的权重矩阵（行=类别，列=输入的第几维）：")
    print(np.round(model.linear.weight.detach().numpy(), 2))
    print("偏置：%s" % np.round(model.linear.bias.detach().numpy(), 2))

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")    # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")   # 画loss曲线
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.show()


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel(INPUT_SIZE, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model(torch.FloatTensor(normalize(np.array(input_vec))))  # 模型预测
        probs = torch.softmax(logits, dim=1)          # logits -> 5 类概率
    for vec, p in zip(input_vec, probs.tolist()):
        cls = int(np.argmax(p))
        print("输入：%s  (最大值在第%d维)" % ([round(v, 4) for v in vec], int(np.argmax(vec)) + 1))
        print("  预测：第%d类（标签%d），概率：%.4f" % (cls + 1, cls, p[cls]))
        print("  各类概率：%s" % [round(v, 4) for v in p])


if __name__ == "__main__":
    train_mode = False  # 改成 False 则跳过训练，只加载已有模型做预测
    if train_mode:
        main()
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]]


    test_vec2 = [[0.1, 0.3, 0.2, 0.222222, 0.333333],
                [0.1, 0.11, 0.1111, 0.1113, 0.1114],
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]]
    predict(MODEL_PATH, test_vec2)
