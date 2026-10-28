import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import os


# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 获取当前脚本所在目录
cur_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(cur_dir, "multi_model.bin")
img_path = os.path.join(cur_dir, "train_curve.png")


# 1.构建数据集
def build_dataset(sample_num):
    x = []
    y = []
    for i in range(sample_num):
        vec = [random.random() for _ in range(5)]
        label = np.argmax(vec)
        x.append(vec)
        y.append(label)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# 2.构建模型
class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super().__init__()
        self.linear = nn.Linear(input_size, class_num)

    def forward(self, x, y=None):
        pred = self.linear(x)
        if y is not None:
            loss = nn.functional.cross_entropy(pred, y)
            return loss
        else:
            return pred


# 3.评估函数
def evaluate(model, x_test, y_test):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        pred_logits = model(x_test)
        pred_label = torch.argmax(pred_logits, dim=1)
        correct = torch.sum(pred_label == y_test).item()
        total = len(y_test)
    acc = correct / total
    return acc


# 4. 主函数
def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    test_sample_num = 100
    input_size = 5
    class_num = 5
    learning_rate = 0.01

    model = TorchModel(input_size, class_num)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    x_train, y_train = build_dataset(train_sample)
    x_test, y_test = build_dataset(test_sample_num)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            start = batch_index * batch_size
            end = start + batch_size
            x = x_train[start:end]
            y = y_train[start:end]

            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        mean_loss = np.mean(watch_loss)
        acc = evaluate(model, x_test, y_test)
        log.append((acc, mean_loss))
        print("第%d轮平均loss:%f, 准确率:%f" % (epoch + 1, mean_loss, acc))

    torch.save(model.state_dict(), model_path)
    # 绘图保存
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.title("loss and accuracy curve")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(img_path, dpi=300)
    print(f"曲线图已经保存到 {img_path}")
    plt.close()


# 5.预测函数
def predict():
    if not os.path.exists(model_path):
        print("模型文件不存在，请先运行main()训练！")
        return
    model = TorchModel(5, 5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    x, y = build_dataset(10)
    with torch.no_grad():
        pred_logits = model(x)
        pred_label = torch.argmax(pred_logits, dim=1)
        print("预测标签:", pred_label.numpy())
        print("真实标签:", y.numpy())


if __name__ == "__main__":
    main()
    predict()

