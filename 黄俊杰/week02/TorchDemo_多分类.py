# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，它最大数字在第几维，就属于第几类（共5类，类别下标0~4）

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层：5维输入 -> 5个类别的打分
        self.loss = nn.functional.cross_entropy  # 多分类损失：交叉熵（内部已包含 softmax）

    # 当输入真实标签，返回loss值；无真实标签，返回预测值（5个类别的打分）
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, 5) -> (batch_size, 5)，每行是5个类别的打分
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果，这里不再套 softmax，softmax 已在 loss 内部


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值在第几维（下标0~4），就属于第几类
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # argmax 返回最大值所在的下标，这就是类别标签
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # Y 用 LongTensor：CrossEntropyLoss 要求标签是 long 类型的类别下标（不是 one-hot）
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中各类别样本数：%s" % torch.bincount(y, minlength=5).tolist())
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)，得到 (100, 5) 的打分
        pred = torch.argmax(y_pred, dim=1)  # 打分最高的那一维就是预测类别
    correct = int((pred == y).sum().item())  # 预测类别和真实标签一致就算对
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率

    # 固定随机种子：保证每次运行结果一致（数据、初始权重都一样）
    # 不固定的话，每轮 acc 会在 0.93~1.00 之间跳，最后的预测结果也会变
    torch.manual_seed(42)
    np.random.seed(42)

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        # 每轮打乱一次样本顺序，避免模型记住数据的排列顺序
        index = torch.randperm(train_sample)
        train_x, train_y = train_x[index], train_y[index]
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.xlabel("epoch")  # 横轴：训练轮数
    plt.legend()
    plt.savefig("train-curve.png", dpi=200)  # 保存训练曲线图片，随作业一起提交
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model(torch.FloatTensor(input_vec))  # 模型预测，得到 (N, 5) 的打分
        prob = torch.softmax(logits, dim=1)  # 需要概率时，在模型外面单独做一次 softmax
    for vec, logit, p in zip(input_vec, logits, prob):
        pred = int(torch.argmax(logit))  # 打分最高的那一维就是预测类别
        true = int(np.argmax(vec))  # 真实类别：这个向量里最大值的下标
        print("输入：%s, 真实类别：%d, 预测类别：%d, 概率值：%f"
              % (vec, true, pred, float(p[pred])))


if __name__ == "__main__":
    main()

    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]

    predict("model.bin", test_vec)
