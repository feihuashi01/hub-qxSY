# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大数字在第几维就属于第几类

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size=5):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)  # 隐藏层
        self.linear2 = nn.Linear(64, 32)  # 隐藏层
        self.linear3 = nn.Linear(32, output_size)  # 输出层
        self.activation = nn.ReLU()  # 激活函数
        self.loss = nn.CrossEntropyLoss() # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x1 = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 64)
        x1 = self.activation(x1)  # 激活函数
        x2 = self.linear2(x1)  # (batch_size, 64) -> (batch_size, 32)
        x2 = self.activation(x2)  # 激活函数
        x3 = self.linear3(x2)  # (batch_size, 32) -> (batch_size, output_size)
        y_pred = x3  # 直接返回 logits
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，返回“哪个位置上的数最大”，这个位置就是标签
# 例如 [0.54, 0.71, 0.60, 0.54, 0.42]，返回标签1，因为第1个数最大
def build_sample(n_features=5):
    x = np.random.rand(n_features)
    y = int(np.argmax(x))
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)



# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        y_p = torch.softmax(y_pred, dim=1)  # 归一化为概率值
        y_p = torch.argmax(y_p, dim=1)  # 取最大值
        for i in range(test_sample_num):
            if y_p[i] == y[i]:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 800  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
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
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:32]  train_y[0:32] train_x[32:64]  train_y[32:64]
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
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    result = torch.softmax(result, dim=1)  # 归一化为概率值
    result = torch.argmax(result, dim=1)  # 取最大值
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, int(res)))  # 打印结果

if __name__ == "__main__":
    # main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model.bin", test_vec)
