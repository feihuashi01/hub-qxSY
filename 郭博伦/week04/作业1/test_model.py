"""
test_model.py — 最小测试：前向预测 + 续写
"""

import sys
import glob
import random
import torch
import torch.nn.functional as F

from transformer_lm import TransformerLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    c = ck["args"]
    model = TransformerLM(
        len(ck["char2idx"]), c["d_model"], c["heads"], c["layers"],
        c["seq_len"], 0.0,
        use_pos=not c.get("no_pos", False),
        causal=not c.get("no_mask", False),
    ).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck["char2idx"], ck["idx2char"], c["seq_len"]


def get_prompts(cli_prompts):
    """优先用命令行给的；否则从 corpus_*.txt 里随机摘几段开头"""
    if cli_prompts:
        return [p.strip() for p in cli_prompts.split(",") if p.strip()]
    for c in sorted(glob.glob("corpus_*.txt")):
        with open(c, encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if len(l.strip()) >= 12]
        if lines:
            random.seed(42)
            return [l[:8] for l in random.sample(lines, min(3, len(lines)))]
    return ["今天天气", "人工智能", "我们一起去"]


@torch.no_grad()
def predict_next(model, char2idx, idx2char, text, k=5):
    ids = [char2idx[ch] for ch in text if ch in char2idx]
    if not ids:
        return []
    logits = model(torch.tensor([ids], device=DEVICE))[0, -1]
    probs = F.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    return [(idx2char[int(i)], top.values[j].item()) for j, i in enumerate(top.indices)]


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, n_chars=30, temperature=0.8, seq_len=64):
    ids = [char2idx[ch] for ch in prompt if ch in char2idx]
    out = list(ids)
    for _ in range(n_chars):
        logits = model(torch.tensor([out[-seq_len:]], device=DEVICE))[0, -1]
        if temperature <= 0.01:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        out.append(nxt)
    return "".join(idx2char[i] for i in out[len(ids):])


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else sorted(glob.glob("tf_*.pt"))[0]
    cli_prompts = None
    if "--prompts" in sys.argv:
        cli_prompts = sys.argv[sys.argv.index("--prompts") + 1]

    model, char2idx, idx2char, seq_len = load(ckpt)
    prompts = get_prompts(cli_prompts)
    print(f"ckpt: {ckpt}   device: {DEVICE}\n")

    # ── 1. 前向：预测下一个字 ──
    print("[预测下一个字]")
    for prompt in prompts:
        top = predict_next(model, char2idx, idx2char, prompt)
        cand = "  ".join(f"{ch}({p:.3f})" for ch, p in top)
        print(f"  {prompt!r} → {cand}")
    print()

    # ── 2. 续写 ──
    print("[续写]")
    p = prompts[0]
    for temp in [0.0, 0.8, 1.2]:
        gen = generate(model, char2idx, idx2char, p, 30, temp, seq_len)
        print(f"  T={temp:<4} {p} | {gen}")


if __name__ == "__main__":
    main()