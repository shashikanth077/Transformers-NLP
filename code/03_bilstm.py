"""Recurrent baseline: a bidirectional LSTM trained from scratch (no pre-training)."""
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from utils import Timer, compute_metrics, load_split, save_result, set_seed

set_seed()
train, dev, test = load_split("train"), load_split("dev"), load_split("test")
TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


counter = Counter(tok for s in train["sentence"] for tok in tokenize(s))
vocab = {"<pad>": 0, "<unk>": 1}
for tok, c in counter.items():
    if c >= 2:
        vocab[tok] = len(vocab)


def encode(df):
    X = [torch.tensor([vocab.get(t, 1) for t in tokenize(s)] or [1]) for s in df["sentence"]]
    return X, df["label"].values


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        mask = (x != 0).unsqueeze(-1)
        h = h.masked_fill(~mask, -1e9).max(dim=1).values  # max-pooling over time
        return self.out(self.drop(h))


def batches(X, y, bs, shuffle):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), bs):
        b = idx[i:i + bs]
        yield pad_sequence([X[j] for j in b], batch_first=True), torch.tensor(y[b])


def predict(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in batches(X, np.zeros(len(X), dtype=int), 256, False):
            preds.append(model(xb).argmax(-1))
    return torch.cat(preds).numpy()


Xtr, ytr = encode(train)
Xdv, ydv = encode(dev)
Xte, yte = encode(test)
model = BiLSTM(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
history, best_acc, best_state = [], 0.0, None

with Timer() as t_train:
    for epoch in range(4):
        model.train()
        total = 0.0
        for xb, yb in batches(Xtr, ytr, 64, True):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(yb)
        dev_acc = float((predict(model, Xdv) == ydv).mean())
        history.append({"epoch": epoch + 1, "train_loss": total / len(Xtr), "dev_accuracy": dev_acc})
        print(history[-1], flush=True)
        if dev_acc > best_acc:
            best_acc, best_state = dev_acc, {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
with Timer() as t_inf:
    pred = predict(model, Xte)
save_result("bilstm", {
    "model": "bilstm",
    "family": "recurrent",
    "train_size": len(train),
    "train_time_s": t_train.seconds,
    "latency_ms_per_sample": 1000 * t_inf.seconds / len(test),
    "parameters": sum(p.numel() for p in model.parameters()),
    "history": history,
    **compute_metrics(yte, pred),
    "predictions": pred.tolist(),
})
