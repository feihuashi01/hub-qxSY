import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import math
import os
import random

#==========================超参数==============================
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
n_head = 16
n_embed = 512
seq_len = 16
n_layer = 2
LR = 3e-4
base_file = os.path.abspath(__file__)
base_path = os.path.dirname(base_file)
max_epoch = 2

class LargeLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head = 8, n_layer = 2, seq_len = 16, dropout= 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed, padding_idx=0)
        self.pos_embedding = nn.Embedding(seq_len, n_embed, padding_idx=0)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_embed,
            nhead=n_head,
            dim_feedforward=4 * n_embed,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embed)
        self.linear = nn.Linear(n_embed, vocab_size)
        self.dropout = nn.Dropout(p= dropout)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, T = x.shape
        token = self.token_embedding(x)
        pos_idx = torch.arange(0, T, device= device)
        pos = self.pos_embedding(pos_idx)
        input = token + pos

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        memory = torch.zeros_like(input)
        input = self.decoder(input, memory = memory, tgt_mask= tgt_mask)
        input = self.ln(input)
        return self.linear(input)

class CharDataSet(Dataset):
    def __init__(self, text, char2idx, seq_len, stride=1):
        self.text = text
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.char2idx = char2idx
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)
        

    def __getitem__(self, index):
        start = int(index*self.stride)
        x = self.data[start: start + self.seq_len].to(non_blocking=True, device=device)
        y = self.data[start + 1: start + self.seq_len + 1].to(non_blocking=True, device=device)
        return x, y

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)


# text = """
# 女朋友和我打冷战，怎么办啊
# 谁有范冰冰的苹果这部电影
# 表哥的儿子叫我什么
# """
# vocab = {}
# with open(os.path.join(base_path, "./vocab.txt"), mode="r", encoding="utf-8") as file:
#     for idx, line in enumerate(file):
#         token = line.rstrip("\n") # 去掉换行符！重点，不然token带\n
#         vocab[token] = idx
# dataset = CharDataSet(text, vocab, 3,  1)
# for x, y in DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True):
#     print("x: ,", x)
#     print("x shape is: ,", x.shape)
#     print("x reshape: ", x.reshape(-1))
#     break

#训练
lines = []
with open(os.path.join(base_path, "./corpus_general.txt"), mode="r", encoding="utf-8") as file:
    for line in file.readlines():
        lines.append(line)
text = "\n".join(lines)

vocab = {}
idx2vocab = {}
with open(os.path.join(base_path, "./vocab.txt"), mode="r", encoding="utf-8") as file:
    for idx, line in enumerate(file):
        token = line.rstrip("\n") # 去掉换行符！重点，不然token带\n
        vocab[token] = idx
        idx2vocab[idx] = token
print("total vocab: ", len(vocab))

split_idx = int(0.4 * len(lines))
training_data = lines[: split_idx]
split_idx = int(0.9 * len(lines))
val_data = lines[split_idx:]
print("training data: ", len(training_data))

training_text = "\n".join(training_data)
val_text = "\n".join(val_data)

training_dataset = CharDataSet(training_text, vocab, seq_len=seq_len)
val_dataset = CharDataSet(val_text, vocab, seq_len=seq_len)

def train():
    model = LargeLanguageModel(
        vocab_size=len(vocab),
        n_embed=n_embed,
        n_layer=n_layer
        ).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    training_loader = DataLoader(training_dataset, batch_size=64, shuffle=False, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True, pin_memory=True)
    print("training loader size: ", len(training_loader))
    print("val loader size: ", len(val_loader))
    for epoch in range(1, max_epoch):
        step = 0
        # 训练
        model.train()
        # 不要用普通 float，用 tensor 放在 device
        train_loss_sum = torch.tensor(0.0, device=device)
        for x, y in training_loader:
            logits = model(x)
            logits_flat = logits.reshape(-1, logits.size(-1))
            y_flat = y.reshape(-1)
            loss = model.loss_fn(logits_flat, y_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss
            if step % 10 == 0:
                print(f"epoch : {epoch}， step: {step}, loss: {loss.item()}")
            step +=1
        # 验证
        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        for x, y in val_loader:
            with torch.no_grad():
                logits = model(x)
                logits_flat = logits.reshape(-1, logits.size(-1))
                y_flat = y.reshape(-1)
                loss = model.loss_fn(logits_flat, y_flat)
                val_loss_sum += loss
        # 平均 loss
        avg_train_loss = train_loss_sum.item() / len(training_loader)
        avg_val_loss = val_loss_sum.item() / len(val_loader)
        print(f"Epoch {epoch+1:02d} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")
    torch.save(model.state_dict(), "model1104.bin")
    # 训练完成

# 执行推理
def predict(text):
    model = LargeLanguageModel(
        vocab_size=len(vocab),
        n_embed=n_embed,
        n_layer=n_layer
        ).to(device=device)
    model.load_state_dict(torch.load("./model1104.bin"))
    model.eval()

    for i in range(0, 3):
        x = []
        for c in text:
            x.append(vocab[c])
        logits = model.forward(torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0))
        last_logit = logits[0, -1, :]
        next_idx = torch.argmax(last_logit)
        print(f"'{text}'的下一个字符是: '{idx2vocab.get(next_idx.item())}'")
        text = text + idx2vocab.get(next_idx.item())

if __name__ ==  "__main__":
    predict("爱就一个字，今天天气")







