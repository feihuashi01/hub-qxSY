#第二周作业（郑志强）


"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个任意5维向量，
它最大数字在第几维就属于第几类

"""

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import random 
import json 
import matplotlib.pyplot as plt


#定义函数
class TorchModel(nn.Module):
    def __init__(self, input_size):  # 
        super(TorchModel, self).__init__()
        self.hidden = nn.Linear(input_size,16)  #增加一个隐藏层，提升非线性拟合能力 (5维 -> 16维 -> 5维)
        self.linear = nn.Linear(16, 5) 
        self.activation = nn.ReLU()   #增加激活函数
        self.loss = nn.CrossEntropyLoss()

# 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.hidden(x)   #经过隐藏层
        x = self.activation(x)  #激活
        x = self.linear(x)  # (batch_size, 5) -> (batch_size, 5)5个类别得分
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果

#生成样本：寻找5维向量中最大值的索引作为类别
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)     #获取最大值的索引
    return x , y

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
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))  #交叉熵要求标签为整型


# 测试代码：评估模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 500
    x, y = build_dataset(test_sample_num)
    
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，输出形状为 (100, 5)
        
        # 用 argmax 沿着维度1获取预测得分最高的索引
        pred_classes = torch.argmax(y_pred, dim=1) 
        
        # 统计预测正确的个数
        correct = (pred_classes == y).sum().item()
        
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 100  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            # 取出一个batch数据
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            
            loss = model(x, y)  # 计算loss
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
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    
    # 修改点7：使用 softmax 将得分转为概率，用 argmax 获取类别
    probs = torch.softmax(result, dim=1)
    pred_classes = torch.argmax(result, dim=1)
    
    for i, (vec, p, c) in enumerate(zip(input_vec, probs, pred_classes)):
        # 打印输入、预测类别、对应概率
        print("输入：%s, 预测类别：%d, 对应概率：%.4f" % (vec, c.item(), p[c].item()))
        # 如果想看所有类别的概率，可以取消下面这行的注释
        # print("各类别概率：", p.tolist())


if __name__ == "__main__":
    #main()
    
    # 2. 运行预测
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843], # 预测类别：4，对应概率(0.8180)
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],  # 预测类别：3，对应概率(0.5229)
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392], # 预测类别：0，对应概率(0.9970)
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]  # 预测类别：2，对应概率(0.9054)
   
    predict("model.bin", test_vec)
        
