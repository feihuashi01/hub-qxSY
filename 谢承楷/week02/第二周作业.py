# coding:utf8
# 解决 OpenMP 库冲突问题（Windows下多线程MKL报错专用）
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
任务：5维向量多分类任务
输入：5维随机向量x
规律：找出向量中数值最大元素的下标，作为标签（0,1,2,3,4）
模型学习目标：输入5维向量，预测最大值所在的位置

=========== 曾经踩过的坑总结 ===========
坑1：在模型__init__里直接调用torch.softmax(pred, dim=-1)，实例化时报错
原因：torch.softmax是计算函数，不是层对象，不能在初始化时直接执行；forward前向传播里才可以调用。

坑2：forward推理分支直接return torch.argmax(y_pred)
问题：输入是batch批量样本时，argmax会压缩维度，输出shape从[batch,5]变成[batch]，后续循环遍历会异常。
解决：forward统一返回概率分布，argmax放到forward外部（evaluate/predict函数）计算。

坑3：使用nn.functional.cross_entropy时，标签传了one-hot向量
知识点：CrossEntropyLoss = log_softmax + NLLLoss，内部自带softmax。
要求：输入必须是未经过softmax的logits，标签必须是类别索引(LongTensor，数字0~4)，不能是one-hot。

MSE 回归 / 二分类：标签要和输出同维度，append([y])
CrossEntropy 多分类：标签是类别序号一维数组，直接append(y)

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 单层线性层，输入5维，输出5维logits（原始得分，不做softmax）
        self.linear = nn.Linear(input_size, 5)
        # 交叉熵损失函数，内部自带softmax
        self.loss = nn.functional.cross_entropy

    # 前向传播函数
    # 参数 x：输入特征 [batch_size, input_size]
    # 参数 y：真实标签，类别索引，LongTensor；y=None代表推理预测模式
    # 返回值：训练模式返回loss；推理模式返回softmax概率分布
    def forward(self, x, y=None):
        # 线性层计算，得到logits，shape: [batch_size,5]
        y_pred = self.linear(x)
        if y is not None:
            # 训练阶段：传入logits和类别索引，计算交叉熵损失
            return self.loss(y_pred, y)
        else:
            # 推理阶段：对logits做softmax，转为概率分布
            prob = torch.softmax(y_pred, dim=-1)
            return prob


# 生成单个样本
# 输入：无，随机生成5维向量
# 返回：x：5维向量；y：最大值对应的下标（类别索引0~4）
def build_sample():
    x = np.random.random(5)
    max_idx = np.argmax(x)
    return x, max_idx


# 批量生成数据集
# total_sample_num：样本总数
# 返回：特征张量X，标签张量Y（Y必须是LongTensor，适配交叉熵）
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试评估函数，计算模型在测试集上的准确率
# 坑：原版本每次调用都重新随机生成测试样本，造成acc曲线抖动
# 优化方案：提前固定测试集，不再每次生成
def evaluate(model, test_x, test_y):
    model.eval()  # 模型切换为评估模式，关闭dropout/bn等训练专用层
    correct, wrong = 0, 0
    with torch.no_grad():  # 推理阶段关闭梯度计算，节省显存加速
        y_prob = model(test_x)  # 模型前向传播，输出概率分布
        y_pred = torch.argmax(y_prob, dim=-1)  # 在外部求最大概率下标，不放在forward内部
        for y_p, y_t in zip(y_pred, test_y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    acc = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    # 配置超参数
    epoch_num = 20  # 训练总轮数
    batch_size = 20  # 每个批次样本数量
    train_sample = 5000  # 每一轮训练总样本数量
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率，步长大会造成参数震荡

    # 构建模型
    model = TorchModel(input_size)
    # 选择优化器Adam，更新模型参数
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []  # 保存每一轮的[准确率，平均loss]，用于画图

    # ========== 数据集准备 ==========
    # 训练集，每轮训练使用
    train_x, train_y = build_dataset(train_sample)
    # 固定测试集！只生成一次，全程复用，解决acc震荡问题
    test_x, test_y = build_dataset(1000)

    # 训练主循环
    for epoch in range(epoch_num):
        model.train()  # 切换为训练模式
        watch_loss = []  # 保存当前epoch所有batch的loss，用于求平均

        # 按批次循环，遍历全部训练样本
        for batch_index in range(train_sample // batch_size):
            # 切片取出一个batch数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            # 前向传播计算loss
            loss = model(x, y)
            # 反向传播，计算梯度
            loss.backward()
            # 利用梯度更新权重参数
            optim.step()
            # 梯度清零，防止梯度累积（顺序不能乱！）
            optim.zero_grad()
            watch_loss.append(loss.item())

        # 打印本轮训练信息
        mean_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, mean_loss))
        # 在固定测试集上评估准确率
        acc = evaluate(model, test_x, test_y)
        log.append([acc, mean_loss])

    # 保存训练完成后的模型权重
    torch.save(model.state_dict(), "model_distribute.bin")

    # 绘制loss和acc曲线
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss") # 损失曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    # 加载保存好的权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = torch.softmax(model.forward(torch.FloatTensor(input_vec)), dim=-1).round(decimals=4)
    # 取概率最大的类别下标
    pred_idx = torch.argmax(result, dim=-1)
    # 遍历打印每条样本结果
    for vec, res, idx in zip(input_vec, result, pred_idx):
        print(f"输入：{vec}, 预测类别：{int(idx)}, 概率分布：{res.numpy()}")


if __name__ == "__main__":
    # main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model_distribute.bin", test_vec)
