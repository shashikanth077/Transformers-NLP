"""Visualise self-attention of a fine-tuned DistilBERT on example sentences
(helps explain how the Transformer resolves negation and contrast)."""
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import FIG_DIR

NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, output_attentions=True).eval()

sentence = "the plot was not good , but the acting was brilliant"
enc = tok(sentence, return_tensors="pt")
with torch.no_grad():
    out = model(**enc)
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
probs = out.logits.softmax(-1)[0].tolist()
print(f"P(negative)={probs[0]:.3f}  P(positive)={probs[1]:.3f}")

# out.attentions: tuple(layers) of [batch, heads, seq, seq]; average heads in the last layer
att = out.attentions[-1][0].mean(0).numpy()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(att, cmap="Blues")
ax.set_xticks(range(len(tokens)), tokens, rotation=60, ha="right")
ax.set_yticks(range(len(tokens)), tokens)
ax.set_xlabel("attended-to token (key)")
ax.set_ylabel("query token")
ax.set_title("DistilBERT, last layer, mean of 12 heads")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(os.path.join(FIG_DIR, "attention_heatmap.png"), dpi=200)

# [CLS] attention row in every layer -> which words drive the sentence representation
fig, ax = plt.subplots(figsize=(8, 3.2))
cls = torch.stack([a[0].mean(0)[0] for a in out.attentions]).numpy()  # layers x seq
im = ax.imshow(cls, cmap="Oranges", aspect="auto")
ax.set_xticks(range(len(tokens)), tokens, rotation=60, ha="right")
ax.set_yticks(range(cls.shape[0]), [f"L{i + 1}" for i in range(cls.shape[0])])
ax.set_title("Attention from [CLS] to each token, per layer")
fig.colorbar(im, ax=ax, fraction=0.03)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cls_attention_layers.png"), dpi=200)
print("saved attention figures")
