"""
字符级语言模型：RNN / LSTM / Transformer 三种结构，含 PPL 与「下一字」预测。

用法:
    # 训练（默认 Transformer；加 --model lstm / --model rnn 可切回循环结构）
    python language_model.py --epochs 10
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20

    # 训练完立刻演示：输入前几个字 → 下一个字概率最大的字
    python language_model.py --model transformer --epochs 10 --prompt "今天天气" --topk 5

    # 用已训好的权重做预测 / 交互
    python language_model.py --load best_model_transformer.pt --prompt "今天天气"
    python language_model.py --load best_model_transformer.pt --interactive

三个结构的核心差别:
    RNN/LSTM 从左到右逐字读，位置 t 的隐状态天然只包含第 1..t 个字的信息，
    「不能看未来」是结构自带的性质，不需要额外处理。

    Transformer 的自注意力一次看到整条序列，位置 t 能直接关注 t+1、t+2……
    如果训练时不遮挡右侧，模型就会「偷看」下一秒的答案：训练 loss 掉得飞快，
    但推理时没有未来可看，生成结果立刻崩掉（信息泄漏）。
    所以必须加因果掩码（causal mask），把每个位置能看到的位置限制在 j <= i。
    具体实现见 CausalSelfAttention。
"""

import argparse
import glob
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    Transformer 每轮要过完整条语料，默认用不重叠切块（--stride 0 自动取 seq_len）。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型：RNN / LSTM ───────────────────────────

class RecurrentLM(nn.Module):
    """
    RNN / LSTM 语言模型。

    循环结构本身就是「因果」的：t 时刻的隐状态只由 0..t 的输入累积而来，
    所以这里不需要任何 mask，输出 logits[:, t] 天然只依赖 x[:, :t+1]。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits


# ─────────────────────────── 模型：Transformer ───────────────────────────

class CausalSelfAttention(nn.Module):
    """
    多头自注意力 + 因果掩码（causal mask）。

    为什么必须加 mask:
        自注意力的公式是 softmax(QKᵀ/√d)·V，QKᵀ 会算出「每个位置对每个位置」的分数，
        包括右侧的未来位置。语言模型的任务是用前 t 个字预测第 t+1 个字，
        若训练时位置 t 能看到 t+1..T，模型就直接抄答案了：
            - 训练/验证 loss 异常低（因为答案就在输入里）
            - 推理时只有左侧上文，没有答案可抄，输出全是垃圾
        这就是 teacher forcing 下的信息泄漏，也是 Transformer 做语言模型
        与做 BERT（双向、可以看右边）最本质的区别。

    做法:
        构造下三角布尔矩阵 mask[i][j] = (j <= i)：
            位置 0 只能看 0
            位置 1 只能看 0,1
            ...
            mask = [[1,0,0,0],
                    [1,1,0,0],
                    [1,1,1,0],
                    [1,1,1,1]]
        在 softmax 之前，把 mask 为 0 的位置的分数填成 -inf，
        softmax 之后这些位置的权重就是 0，等于「完全看不见」。
        注意 mask 必须加在 softmax 之前，加在之后只是把权重清零但没有重新归一化，
        剩下的权重之和不再是 1，就错了。
    """

    def __init__(self, d_model, n_head, block_size, dropout):
        super().__init__()
        assert d_model % n_head == 0, f"d_model({d_model}) 必须能被 n_head({n_head}) 整除"
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # 因果掩码只需建一次，所有 batch/head 共用，所以形状是 (1, 1, block, block)。
        # persistent=False: 不写进 state_dict，否则换 block_size 加载旧权重会 size mismatch。
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        if T > self.causal_mask.size(-1):
            raise ValueError(f"序列长度 {T} 超过 block_size {self.causal_mask.size(-1)}")

        # 一次线性层算出 Q/K/V，再切成三份：(B, T, C) -> (B, H, T, head_dim)
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 缩放点积注意力，分数形状 (B, H, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ── 关键一步：遮住右侧 ──
        # causal_mask[:, :, :T, :T] 取左上角 T×T；为 False（上三角）的位置填 -inf。
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)      # 未来位置的权重变成 0
        att = self.attn_drop(att)

        out = att @ v                      # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    """Transformer 里的前馈网络：先升维 4 倍再降回来，给每个位置做非线性变换。"""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """
    一个 Transformer 块（Pre-LN 结构）:
        x = x + Attn(LN(x))      ← 注意力和 MLP 都是残差分支
        x = x + MLP(LN(x))
    Pre-LN（先 LayerNorm 再进子层）比原始论文的 Post-LN 更容易训稳，
    小模型上也基本不需要复杂的学习率调参。
    """

    def __init__(self, d_model, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    字符级 Transformer 语言模型（GPT 式，decoder-only）。

    结构: 字嵌入 + 位置嵌入 → N × TransformerBlock → LayerNorm → 线性头 → (B,T,V)
    每个位置的输出 logits[:, t] 经过因果掩码后只依赖 x[:, :t+1]，
    所以最后一个位置 logits[:, -1] 就是「下一个字」的未归一化概率。
    """

    def __init__(self, vocab_size, block_size, d_model, n_head, num_layers, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # 可学习的位置嵌入：模型需要知道「谁在前谁在后」，否则自注意力是置换不变的。
        # 只有 block_size 个位置，所以推理时上文长度不能超过 block_size（超了要截断）。
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, block_size, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        # 权重共享：输入字嵌入和输出分类头用同一套参数（char LM 里很常见，
        # 既省参数又让「输入/输出同一个字」的表示一致）。
        self.head.weight = self.tok_emb.weight
        # 残差分支的投影层按 1/√(2N) 缩小初始化，层数多了也不会让残差数值爆炸。
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"输入长度 {T} 超过 block_size {self.block_size}")

        pos = torch.arange(T, device=x.device)          # 0,1,...,T-1
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h)                             # (B, T, vocab_size) logits


def build_model(cfg, vocab_size, dropout=None):
    """
    按配置字典构建模型。训练脚本和 pinyin_ime.py 共用，避免两处模型定义漂移。

    cfg 需要包含 model 字段；transformer 还需要 hidden_dim(=d_model)、
    n_head、block_size、num_layers；rnn/lstm 用 embed_dim/hidden_dim。
    """
    model_type = cfg.get("model", "lstm")
    drop = cfg.get("dropout", 0.1) if dropout is None else dropout

    if model_type == "transformer":
        return TransformerLM(
            vocab_size=vocab_size,
            block_size=int(cfg.get("block_size") or cfg.get("seq_len") or 64),
            d_model=int(cfg["hidden_dim"]),
            n_head=int(cfg.get("n_head", 8)),
            num_layers=int(cfg["num_layers"]),
            dropout=drop,
        )

    return RecurrentLM(
        vocab_size=vocab_size,
        embed_dim=int(cfg["embed_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        model_type=model_type,
        dropout=drop,
    )


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def make_lr_lambda(warmup_steps, total_steps, min_ratio=0.1):
    """
    warmup + 余弦退火的学习率曲线。
    Transformer 刚初始化时注意力分数很乱，直接上大学习率容易发散，
    所以前 warmup_steps 步从 0 线性升到 base_lr，之后余弦降到 base_lr * min_ratio。
    """
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return fn


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, grad_clip=1.0):
    """
    跑一轮。optimizer 为 None 表示评估模式（不建计算图、不更新参数）。
    返回 (平均交叉熵 loss, PPL)；PPL = exp(loss)，可理解为「模型在每一步平均在多少个字里犹豫」。
    """
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_tokens = 0

    # 评估时关掉自动求导，省显存也更快
    with torch.enable_grad() if train else torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 梯度裁剪：语言模型偶尔会遇到极端 batch，裁一下更稳
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, math.exp(avg_loss)


# ─────────────────────────── 推理：下一个字 ───────────────────────────

def context_limit(model):
    """
    上文最多喂多少字。
    Transformer 的绝对位置表只有 block_size 个位置，超长必须截断到最近 block_size 字；
    RNN 理论上不限长，这里给个 512 的上限只为避免推理太慢。
    """
    return getattr(model, "block_size", 512)


def encode_prefix(text, char2idx, limit):
    """把上文转成 id 序列，丢掉词表外的字，并只保留最近 limit 个字。返回 (ids, 丢弃字数)。"""
    known = [c for c in text if c in char2idx]
    skipped = len(text) - len(known)
    return [char2idx[c] for c in known][-limit:], skipped


@torch.no_grad()
def next_char_probs(model, text, char2idx, device):
    """
    输入前几个字，返回词表上的下一个字概率分布。
    只取模型最后一个位置的输出：因为在因果掩码下，位置 t 的输出只由前 t+1 个字决定。
    """
    limit = context_limit(model)
    ids, skipped = encode_prefix(text, char2idx, limit)
    if not ids:
        return None, skipped, 0

    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)[0, -1]                 # (vocab_size,)
    probs = F.softmax(logits.float(), dim=-1)
    return probs, skipped, len(ids)


@torch.no_grad()
def generate(model, text, char2idx, idx2char, device, gen_len):
    """
    贪心续写：每一步选概率最大的字，拼到上文后面，再预测下一个。
    这里没有 BOS/EOS 特殊符号，换行符只是普通字符，所以不把它当停止符。
    每步都把整段上文重新过一遍模型（O(T²)），只为演示；
    真实系统会缓存 KV，做增量解码。
    """
    limit = context_limit(model)
    ids, _ = encode_prefix(text, char2idx, limit)
    if not ids:
        return ""

    out = list(ids)
    for _ in range(gen_len):
        x = torch.tensor([out[-limit:]], dtype=torch.long, device=device)
        out.append(int(model(x)[0, -1].argmax()))
    return "".join(idx2char[i] for i in out)


def show_prediction(model, text, char2idx, idx2char, device, topk=5, gen_len=0):
    """打印「上文 → 下一个字」的 Top-k 概率，可选再贪心续写一段。"""
    probs, skipped, used = next_char_probs(model, text, char2idx, device)
    if probs is None:
        print(f"上文「{text}」里没有词表中的字，无法预测。")
        return

    if skipped:
        print(f"（提示：上文有 {skipped} 个字不在词表中，已忽略）")
    limit = context_limit(model)
    if used == limit and len([c for c in text if c in char2idx]) > limit:
        print(f"（提示：上文超过 {limit} 字，只用了最近 {limit} 个字）")

    k = min(topk, probs.numel())
    top_p, top_i = torch.topk(probs, k)

    print(f"上文「{text}」 → 下一个字概率 Top{k}:")
    for rank, (p, i) in enumerate(zip(top_p.tolist(), top_i.tolist()), 1):
        ch = idx2char[i].replace("\n", "\\n")
        print(f"  {rank}. 「{ch}」  {p:.2%}")

    best = idx2char[int(top_i[0])].replace("\n", "\\n")
    print(f"概率最大的字: 「{best}」")

    if gen_len > 0:
        print(f"贪心续写 {gen_len} 字: {generate(model, text, char2idx, idx2char, device, gen_len)}")


def interactive(model, char2idx, idx2char, device, topk, gen_len):
    """交互模式：每行输入一段上文，输出下一个字的 Top-k 和续写。q 退出。"""
    print("=" * 56)
    print("  字符级语言模型 · 下一字预测")
    print("  输入一段话回车 → 看下一个字的概率分布；q 退出")
    print("=" * 56)
    while True:
        text = input("\n上文> ").strip()
        if text in ("q", "quit", "exit"):
            print("退出。")
            break
        if not text:
            continue
        show_prediction(model, text, char2idx, idx2char, device, topk, gen_len)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="字符级语言模型：RNN / LSTM / Transformer")
    parser.add_argument("--model",      default="transformer", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64, help="训练窗口长度；transformer 里也当 block_size")
    parser.add_argument("--stride",     type=int,   default=0,
                        help="切块间隔：0=自动(不重叠，等于 seq_len)，1=密集重叠(样本数×seq_len)")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128, help="仅 rnn/lstm 使用")
    parser.add_argument("--hidden_dim", type=int,   default=256, help="rnn/lstm 的隐层维度；transformer 的 d_model")
    parser.add_argument("--num_layers", type=int,   default=None, help="默认 rnn/lstm=2，transformer=4")
    parser.add_argument("--n_head",     type=int,   default=8, help="transformer 注意力头数，需整除 hidden_dim")
    parser.add_argument("--dropout",    type=float, default=None, help="默认 rnn/lstm=0.3，transformer=0.1")
    parser.add_argument("--lr",         type=float, default=None, help="默认 rnn/lstm=1e-3，transformer=5e-4")
    parser.add_argument("--weight_decay", type=float, default=None, help="默认 rnn/lstm=0，transformer=0.01")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="学习率 warmup 占总步数的比例")
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default=None,
                        help="checkpoint 路径；默认 best_model_<model>.pt，避免覆盖别的结构")
    parser.add_argument("--load",       default=None, help="载入已有 checkpoint，跳过训练做预测")
    parser.add_argument("--prompt",     default=None, help="输入前几个字，输出下一个字概率最大的字")
    parser.add_argument("--topk",       type=int,   default=5)
    parser.add_argument("--gen_len",    type=int,   default=20, help="--prompt 时额外贪心续写的字数，0=不续写")
    parser.add_argument("--interactive", action="store_true", help="进入交互式下一字预测")
    args = parser.parse_args()

    # 三种结构合适的默认超参不同，这里按模型类型补默认值（命令行显式传入则尊重）
    if args.model == "transformer":
        args.num_layers   = 4    if args.num_layers   is None else args.num_layers
        args.dropout      = 0.1  if args.dropout      is None else args.dropout
        args.lr           = 5e-4 if args.lr           is None else args.lr
        args.weight_decay = 0.01 if args.weight_decay is None else args.weight_decay
    else:
        args.num_layers   = 2    if args.num_layers   is None else args.num_layers
        args.dropout      = 0.3  if args.dropout      is None else args.dropout
        args.lr           = 1e-3 if args.lr           is None else args.lr
        args.weight_decay = 0.0  if args.weight_decay is None else args.weight_decay

    if args.stride <= 0:
        args.stride = args.seq_len
    if args.save is None:
        args.save = f"best_model_{args.model}.pt"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 推理模式：直接加载权重 ──
    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]
        cfg = ckpt.get("model_cfg") or ckpt["args"]
        model = build_model(cfg, len(char2idx), dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"载入 {args.load}（{cfg.get('model', 'lstm')}，词表 {len(char2idx)} 字）")
        if args.prompt is not None:
            show_prediction(model, args.prompt, char2idx, idx2char, device, args.topk, args.gen_len)
        else:
            interactive(model, char2idx, idx2char, device, args.topk, args.gen_len)
        return

    print(f"device: {device}  model: {args.model.upper()}")
    print(f"seq_len: {args.seq_len}  stride: {args.stride}  batch: {args.batch_size}")

    # ── 数据准备 ──
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── 模型 ──
    model_cfg = {
        "model":      args.model,
        "embed_dim":  args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout":    args.dropout,
        "n_head":     args.n_head,
        "block_size": args.seq_len,   # transformer 的位置表与掩码长度
    }
    model = build_model(model_cfg, vocab_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}  (层数 {args.num_layers}，dropout {args.dropout})")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps))

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'LR':>9}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>9}  {'Val PPL':>9}  {'Sec':>5}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scheduler, args.grad_clip)
        va_loss, va_ppl = run_epoch(model, val_loader, criterion, device)

        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "model_cfg": model_cfg,      # 新格式：推理时据此复原模型结构
                "args": vars(args),          # 保留旧字段，兼容老脚本
            }, args.save)
            marker = "  *"
        else:
            marker = ""

        print(f"{epoch:>6}  {optimizer.param_groups[0]['lr']:>9.2e}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>9.4f}  {va_ppl:>9.2f}  {time.time() - t0:>5.1f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")
    print(f'试试: python language_model.py --load {args.save} --prompt "今天天气" --topk 5')

    # ── 训练完直接演示预测 ──
    if args.prompt is not None or args.interactive:
        ckpt = torch.load(args.save, map_location=device)
        model = build_model(ckpt["model_cfg"], vocab_size, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        if args.prompt is not None:
            show_prediction(model, args.prompt, char2idx, idx2char, device, args.topk, args.gen_len)
        else:
            interactive(model, char2idx, idx2char, device, args.topk, args.gen_len)


if __name__ == "__main__":
    main()
