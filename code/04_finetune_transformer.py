"""Fine-tune a pre-trained Transformer encoder for binary sentiment classification.

Usage:
    python 04_finetune_transformer.py --model prajjwal1/bert-small --train-size 8000 --epochs 2
"""
import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from utils import Timer, compute_metrics, load_split, save_result, set_seed

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--train-size", type=int, default=8000)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--max-len", type=int, default=64)
p.add_argument("--tag", default=None, help="name of the result file")
p.add_argument("--save-dir", default=None, help="optionally save the fine-tuned model here")
args = p.parse_args()

set_seed()
# CPU by default (the reported results); set the environment variable DEVICE=cuda to use a GPU, e.g. in Google Colab
DEVICE = torch.device(os.environ.get("DEVICE", "cpu"))
train, dev, test = load_split("train"), load_split("dev"), load_split("test")
if args.train_size < len(train):
    train = train.sample(n=args.train_size, random_state=42)

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(DEVICE)


def make_loader(df, shuffle):
    pairs = list(zip(df["sentence"].tolist(), df["label"].tolist()))

    def collate(batch):
        enc = tok([b[0] for b in batch], truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b[1] for b in batch])
        return enc

    return DataLoader(pairs, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)


def predict(loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds.append(model(**batch).logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


train_dl, dev_dl, test_dl = make_loader(train, True), make_loader(dev, False), make_loader(test, False)
steps = args.epochs * len(train_dl)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=math.ceil(0.1 * steps), num_training_steps=steps)

history, best_acc, best_state = [], -1.0, None
with Timer() as t_train:
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)  # cross-entropy loss is computed inside the model
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += out.loss.item() * len(batch["labels"])
            n += len(batch["labels"])
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_dl)} loss {total / n:.4f}", flush=True)
        dev_acc = float((predict(dev_dl) == dev["label"].values).mean())
        history.append({"epoch": epoch + 1, "train_loss": total / n, "dev_accuracy": dev_acc})
        print(history[-1], flush=True)
        if dev_acc > best_acc:
            best_acc, best_state = dev_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
with Timer() as t_inf:
    pred = predict(test_dl)

if args.save_dir:
    model.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)

tag = args.tag or f"ft_{args.model.split('/')[-1]}_{args.train_size}"
save_result(tag, {
    "model": args.model,
    "family": "transformer-finetuned",
    "train_size": len(train),
    "epochs": args.epochs,
    "lr": args.lr,
    "train_time_s": t_train.seconds,
    "latency_ms_per_sample": 1000 * t_inf.seconds / len(test),
    "parameters": sum(p.numel() for p in model.parameters()),
    "history": history,
    **compute_metrics(test["label"].values, pred),
    "predictions": pred.tolist(),
})
