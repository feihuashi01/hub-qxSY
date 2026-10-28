import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchDemo(nn.Module):
    def __init__(self,input_size):
        super(TorchDemo, self).__init__()
        self.linear = nn.Linear(input_size,input_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=-1)  # 输出预测结果

def build_sample():
    x = np.random.random(5)
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    return x,y

def build_dataset(total_num):
    X = []
    Y = []
    for i in range(total_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)                        # (N,5) 概率
        pred_cls = y_pred.argmax(dim=1)          # 预测类别
        true_cls = y.argmax(dim=1)               # 真实类别
        correct = (pred_cls == true_cls).sum().item()
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc

def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01

    model = TorchDemo(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])


    torch.save(model.state_dict(), "model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchDemo(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        cls = int(res.argmax())
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, cls, res[cls].item()))



if __name__ == "__main__":
    main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model.bin", test_vec)