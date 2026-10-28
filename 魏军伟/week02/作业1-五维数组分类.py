import math
import random
import sys

# 第二周作业：五维向量「最大数字在第几维」就属于第几类（5 分类）

random.seed(2026)

# ---------------------------------------------------------------- 1. 造数据
# 生成 n 个五维随机向量，标签 = 最大值所在的维度序号（0~4）
def make_data(n):
    data = []
    for _ in range(n):
        x = [random.gauss(0.0, 1.0) for _ in range(5)]
        y_true = 0
        for j in range(1, 5):
            if x[j] > x[y_true]:
                y_true = j
        data.append((x, y_true))
    return data

# print(make_data(5))
print("生成数据...")
TRAIN = make_data(12000)    # 训练集
TEST = make_data(1000)      # 测试集（独立生成，用于检验泛化）

CLASS = 5                   # 类别数 = 维度数


# ---------------------------------------------------------------- 2. 模型
# 第 c 类的得分（logits）：score_c = W[c][0]*x0 + W[c][1]*x1 + ... + W[c][4]*x4 + B[c]
# 再对 5 个得分做 softmax，得到属于每一类的概率 p_c
# 预测类别 = argmax(p) = argmax(score)

# 权重随机初始化
W = [[random.gauss(0.0, 0.1) for _ in range(5)] for _ in range(CLASS)]
B = [0.0 for _ in range(CLASS)]


def softmax(scores):
    """softmax：先减去最大值再做 exp，防止 e^score 溢出（数值稳定写法）"""
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    total = sum(e)
    return [v / total for v in e]


def func(x):
    """前向传播：输入五维向量，输出 5 个类别的概率"""
    scores = []
    for c in range(CLASS):
        s = B[c]
        for j in range(5):
            s += W[c][j] * x[j]
        scores.append(s)
    return softmax(scores)


def loss(p, y_true):
    """交叉熵损失：只看真实类别对应的那个概率，概率越接近 1，损失越接近 0"""
    return -math.log(p[y_true] + 1e-12)


def predict(x):
    """预测类别：概率最大的那一类"""
    p = func(x)
    best = 0
    for c in range(1, CLASS):
        if p[c] > p[best]:
            best = c
    return best


# 学习率设置（比原代码的 0.1 大一些，配合衰减使用）
lr = 0.3

# batch size（原代码是 1，这里调大一点让梯度更稳、训练更快）
batch_size = 32

# ---------------------------------------------------------------- 3. 训练过程
for epoch in range(120):
    epoch_loss = 0.0
    counter = 0

    # 梯度累加器：W 是 5x5 矩阵，B 是长度 5 的向量
    grad_W = [[0.0 for _ in range(5)] for _ in range(CLASS)]
    grad_B = [0.0 for _ in range(CLASS)]

    # 每个 epoch 前打乱数据（随机梯度下降的标准做法）
    random.shuffle(TRAIN)

    for x, y_true in TRAIN:
        # 前向：算出 5 类的概率
        p = func(x)
        epoch_loss += loss(p, y_true)
        counter += 1

        # 梯度计算（推导见文件末尾）：
        #   dL/dscore_c = p_c - 1{c == y_true}
        #   dscore_c/dW[c][j] = x_j        dscore_c/dB[c] = 1
        for c in range(CLASS):
            g = p[c] - (1.0 if c == y_true else 0.0)   # 这就是 dL/dscore_c
            for j in range(5):
                grad_W[c][j] += g * x[j]               # 链式法则乘上 x_j
            grad_B[c] += g                             # 链式法则乘上 1

        if counter == batch_size:
            # 权重更新（SGD：沿梯度反方向走一小步）
            for c in range(CLASS):
                for j in range(5):
                    W[c][j] = W[c][j] - lr * grad_W[c][j] / batch_size
                B[c] = B[c] - lr * grad_B[c] / batch_size
            counter = 0
            # 梯度清零
            grad_W = [[0.0 for _ in range(5)] for _ in range(CLASS)]
            grad_B = [0.0 for _ in range(CLASS)]

    epoch_loss = epoch_loss / len(TRAIN)

    print("第%d轮训练 loss:%f" % (epoch, epoch_loss))

    lr = lr * 0.99               # 学习率每轮衰减 1%，越到后面步子迈得越小，便于收敛

    if epoch_loss < 0.01:      # 损失足够小就提前停止
        print("loss 已足够小，提前结束训练")
        break

# ---------------------------------------------------------------- 4. 训练结果
print("\n训练后的权重矩阵 W（行=类别 c，列=输入维度 j）：")
for c in range(CLASS):
    print("  第%d类: " % c + "  ".join("%7.3f" % W[c][j] for j in range(5)))
print("偏置 B: " + "  ".join("%7.3f" % b for b in B))
print("--------------------------------------------------")

print("\n几个具体例子（输入 -> 模型预测 / 真实类别 / 各类概率）：")
demo = [
    [0.20, -1.00, 3.50, 0.10, 2.00],
    [-2.00, 0.50, 0.40, 1.90, -3.30],
    [1.000, 0.999, 0.10, -5.00, 0.30],   # 两个维度几乎并列，最难的情况
    [0.05, -0.02, 0.01, 0.03, 0.04],     # 五个数都很接近，最最难的情况
]
for x in demo:
    p = func(x)
    y_pred = predict(x)
    y_true = 0
    for j in range(1, 5):
        if x[j] > x[y_true]:
            y_true = j
    print("  输入 %s" % x)
    print("    预测第%d类 / 真实第%d类  %s" %
          (y_pred, y_true, "正确" if y_pred == y_true else "错误"))
    print("    各类概率: " + "  ".join("%d:%.4f" % (c, p[c]) for c in range(CLASS)))

