"""
字符级语言模型（Embedding + Transformer 版）
============================================

和 language_model.py（Embedding + LSTM）做同一件事：输入连续 seq_len 个字符，
在每个位置预测“下一个字符”，用交叉熵 / PPL 评估和早停保存。区别只在建模方式：

    LSTM 版：       Embedding → LSTM（隐状态按时间顺序传递历史）→ Linear(V)
    Transformer 版：Embedding + 位置编码 → N × 因果自注意力(Decoder-only) → LayerNorm → Linear(V)

三个关键点：
1. 因果掩码（causal mask）：位置 t 只能看到 ≤ t 的 token，否则就是“偷看答案”，
   训练 loss 会假性下降，但生成时完全不会写。
2. 训练技巧：pre-LN、GELU、权重共享（输出层 = 词向量矩阵）、warmup + cosine 学习率、
   梯度裁剪。Transformer 比 LSTM 更依赖这些技巧，去掉后容易不收敛或发散。
3. 生成：训练完（或 --load 加载权重后）按 temperature / top_k 采样续写。

用法:
    # 训练（CPU 建议先用 --max_chars 缩小语料，见文末“训练建议”）
    python transformer_language_model.py --corpus corpus_general.txt --epochs 10

    # 只加载权重续写，不训练
    python transformer_language_model.py --load best_transformer.pt --prompt 今天天气 --n_gen 200

训练建议:
    语料 1700 万字符、stride=seq_len 时一轮约 27 万个 batch，CPU 上非常慢。
    先跑通用:  --max_chars 2000000 --seq_len 64 --d_model 256 --num_layers 4 \
                --batch_size 64 --epochs 5 --stride 64
    想要更好的效果再逐步加语料 / 加层数，PPL 和 LSTM 版对比时请保持同样的数据切分。
"""

import argparse
import glob
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────── 数据（与 language_model.py 完全一致） ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """按出现频率取前 max_vocab 个字（None = 全部字符）。"""
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """x = 连续 seq_len 个字符，y = 往后错一位（每个位置都要预测下一个字）。"""

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class TransformerLM(nn.Module):
    """Decoder-only Transformer 语言模型：每个位置输出词表上的下一个字符分布。"""

    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_len=64, tie_weights=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.block_size = max_len
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)   # 字向量：Embedding
        self.pos = nn.Embedding(max_len, d_model)        # 可学习的位置向量
        self.drop = nn.Dropout(dropout)

        # 一层 = 多头因果自注意力 + 前馈网络，pre-LN 更稳
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 输出投影：把每个位置的表征映射回词表。tie_weights 时与字向量共享参数，
        # 小模型上这个做法很划算（少 vocab × d_model 个参数，效果通常还更好）。
        self.head = nn.Linear(d_model, vocab_size, bias=not tie_weights)
        if tie_weights:
            self.head.weight = self.embed.weight

        # 因果掩码：上三角（含右侧）为 -inf，位置 t 只能看 0..t
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1),
            persistent=False,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        b, t = x.shape
        assert t <= self.block_size, f"序列长度 {t} 超过 block_size {self.block_size}"
        pos = torch.arange(t, device=x.device).unsqueeze(0)      # (1, T)
        h = self.drop(self.embed(x) + self.pos(pos))             # (B, T, D)
        mask = self.causal_mask[:t, :t]                          # (T, T)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.head(self.norm(h))                           # (B, T, V)


# ─────────────────────────── 生成 ───────────────────────────

@torch.no_grad()
def generate(model, char2idx, idx2char, prompt="", n_chars=200,
             temperature=0.8, top_k=20, device="cpu"):
    """给定开头，自回归地逐字续写（每次只取最后一个位置的预测）。"""
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:                       # 提示词全是生僻字时，用句号起头
        ids = [char2idx.get("。", 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = [idx2char[i] for i in ids]

    for _ in range(n_chars):
        cond = idx[:, -model.block_size:]               # 上下文超长就滑窗
        logits = model(cond)[:, -1, :]                  # 只关心最后一个位置的预测
        if temperature <= 0:                            # greedy
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        out.append(idx2char[next_id.item()])
    return "".join(out)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None,
              scheduler=None, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    total_loss, total_tokens = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 50.0))     # 训练初期 loss 很大时防止 exp 溢出
    return avg_loss, ppl


def make_scheduler(optimizer, total_steps, warmup_steps, min_ratio=0.1):
    """线性 warmup + cosine 衰减到 lr * min_ratio。"""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(cfg, vocab_size):
    return TransformerLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["ff_dim"],
        dropout=cfg["dropout"],
        max_len=cfg["seq_len"],
        tie_weights=cfg["tie_weights"],
    )


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="字符级语言模型：Embedding + Transformer，预测下一个字符")
    parser.add_argument("--corpus",      default="*.txt", help="语料 glob，如 corpus_general.txt")
    parser.add_argument("--max_chars",   type=int,   default=0, help="只取语料前 N 个字符，0=全部")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--seq_len",     type=int,   default=64, help="上下文长度（同时是最大生成长度）")
    parser.add_argument("--stride",      type=int,   default=0, help="切块间隔，0=seq_len（不重叠）")
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--d_model",     type=int,   default=256)
    parser.add_argument("--nhead",       type=int,   default=8)
    parser.add_argument("--num_layers",  type=int,   default=4)
    parser.add_argument("--ff_dim",      type=int,   default=1024, help="前馈层隐藏维度")
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--tie_weights", default=True, action=argparse.BooleanOptionalAction,
                        help="输出层与字向量共享权重（--no-tie_weights 关闭）")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup",      type=int,   default=200, help="warmup 步数")
    parser.add_argument("--grad_clip",   type=float, default=1.0)
    parser.add_argument("--val_ratio",   type=float, default=0.05)
    parser.add_argument("--max_vocab",   type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=0)
    parser.add_argument("--save",        default="best_transformer.pt")
    parser.add_argument("--load",        default="", help="加载 .pt 后只做生成，不训练")
    parser.add_argument("--prompt",      default="今天天气", help="生成用的开头（也用于每轮采样）")
    parser.add_argument("--n_gen",       type=int,   default=200, help="生成多少字")
    parser.add_argument("--sample_chars", type=int,  default=60, help="训练中每轮采样多少字，0=不采样")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer")

    # ── 只加载权重做生成 ──
    if args.load:
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        char2idx, idx2char = ckpt["char2idx"], ckpt["idx2char"]
        model = build_model(cfg, len(char2idx)).to(device)
        model.load_state_dict(ckpt["model_state"])
        text = generate(model, char2idx, idx2char, args.prompt, args.n_gen,
                        args.temperature, args.top_k, device)
        print(f"\n提示: {args.prompt}\n续写: {text}")
        return

    # ── 数据准备 ──
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认 --corpus 路径正确。")
    if args.max_chars:
        text = text[:args.max_chars]
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 按行切分训练 / 验证（与 LSTM 版一致，PPL 才能直接对比）
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    stride = args.stride or args.seq_len
    train_ds = CharDataset(train_text, char2idx, args.seq_len, stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, stride)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("语料太短，切不出一个完整样本，请减小 --seq_len 或增大 --max_chars。")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, num_workers=args.num_workers)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("切出的 batch 数为 0：请减小 --batch_size 或增大 --max_chars。")
    print(f"训练样本: {len(train_ds):,}  验证样本: {len(val_ds):,}  "
          f"每轮 batch: {len(train_loader):,}")

    # ── 模型 ──
    cfg = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "seq_len": args.seq_len,
        "tie_weights": args.tie_weights,
    }
    model = build_model(cfg, vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()

    # 只对二维及以上参数做 weight decay（LayerNorm / bias 不加）
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, min(args.warmup, total_steps // 10))
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps)

    best_val_ppl = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scheduler, args.grad_clip)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, device)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

        if args.sample_chars > 0:
            sample = generate(model, char2idx, idx2char, args.prompt,
                              args.sample_chars, args.temperature, args.top_k, device)
            print(f"        采样: {sample}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # 用最佳权重再采样一段
    ckpt = torch.load(args.save, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    text = generate(model, char2idx, idx2char, args.prompt, args.n_gen,
                    args.temperature, args.top_k, device)
    print(f"\n提示: {args.prompt}\n续写: {text}")


if __name__ == "__main__":
    main()
