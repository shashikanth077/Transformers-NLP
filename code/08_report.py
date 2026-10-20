"""Aggregate all results into tables and figures used in the paper and the slides."""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import FIG_DIR, RESULTS_DIR, load_split

os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})
# Categorical slots 1-3 of the validated reference palette, one per model family
FAMILY_COLOR = {"classical": "#2a78d6", "recurrent": "#eb6834", "transformer": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"

LABELS = {
    "tfidf_nb": "TF-IDF + Naive Bayes",
    "tfidf_logreg": "TF-IDF + Logistic Regr.",
    "tfidf_svm": "TF-IDF + Linear SVM",
    "bilstm": "BiLSTM (from scratch)",
    "zs_nli_distilroberta": "Zero-shot NLI DistilRoBERTa",
    "pt_twitter_roberta": "Twitter-RoBERTa (off-the-shelf)",
    "pt_distilbert_sst2": "DistilBERT-SST2 (off-the-shelf)",
}


def label_for(tag):
    if tag in LABELS:
        return LABELS[tag]
    m = re.match(r"ft_(.+)_(\d+)$", tag)
    return f"{m.group(1)} fine-tuned ({int(m.group(2)) // 1000}k)" if m else tag


def family_key(fam):
    return "transformer" if fam.startswith("transformer") else fam


main, lc = [], []
for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    tag = os.path.basename(path)[:-5]
    r = json.load(open(path, encoding="utf-8"))
    r["tag"] = tag
    (lc if tag.startswith("lc_") else main).append(r)

df = pd.DataFrame(main)
df["name"] = df["tag"].map(label_for)
df["fam"] = df["family"].map(family_key)
df = df.sort_values("accuracy").reset_index(drop=True)

table = df[["name", "family", "train_size", "parameters", "accuracy", "precision", "recall",
            "f1_macro", "train_time_s", "latency_ms_per_sample"]].copy()
table.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)
print(table.to_string(index=False))

# ---------- Figure 1: accuracy by model ----------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
colors = [FAMILY_COLOR[f] for f in df["fam"]]
ax.barh(df["name"], df["accuracy"] * 100, color=colors, height=0.6, edgecolor="white", linewidth=2)
for i, v in enumerate(df["accuracy"] * 100):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=9, color=INK)
lo = max(0, np.floor(df["accuracy"].min() * 100 / 5) * 5 - 5)
ax.set_xlim(lo, 100)
ax.set_xlabel("Accuracy on SST-2 test set, %")
ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
ax.legend(handles, ["Classical (TF-IDF)", "Recurrent (BiLSTM)", "Transformer"],
          loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_accuracy.png"), dpi=200)

# ---------- Figure 2: accuracy vs. latency ----------
fig, ax = plt.subplots(figsize=(7.2, 4))
for _, r in df.iterrows():
    size = 30 + 250 * np.sqrt(r["parameters"] / df["parameters"].max())
    ax.scatter(r["latency_ms_per_sample"], r["accuracy"] * 100, s=size, color=FAMILY_COLOR[r["fam"]],
               edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(r["name"], (r["latency_ms_per_sample"], r["accuracy"] * 100), fontsize=7.5,
                color=MUTED, xytext=(6, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("Inference latency, ms per sentence (CPU, log scale)")
ax.set_ylabel("Accuracy, %")
ax.set_title("Bubble area proportional to number of parameters", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_accuracy_latency.png"), dpi=200)

# ---------- Figure 3: learning curve ----------
if lc:
    lcd = pd.DataFrame(lc)
    pretty = {"google/bert_uncased_L-4_H-512_A-8": "BERT-small", "google/bert_uncased_L-2_H-128_A-2": "BERT-tiny"}
    lcd["series"] = lcd["model"].map(lambda m: "TF-IDF + Logistic Regr." if m == "tfidf_logreg"
                                     else f"{pretty.get(m, m)} fine-tuned")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    palette = ["#2a78d6", "#1baf7a", "#eb6834"]
    for i, (name, g) in enumerate(lcd.groupby("series")):
        g = g.sort_values("train_size")
        c = FAMILY_COLOR["classical"] if name.startswith("TF-IDF") else palette[1 + i % 2]
        ax.plot(g["train_size"], g["accuracy"] * 100, marker="o", ms=6, lw=2, color=c, label=name)
        last = g.iloc[-1]
        ax.annotate(f"{last['accuracy'] * 100:.1f}", (last["train_size"], last["accuracy"] * 100),
                    xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(sorted(lcd["train_size"].unique()), [str(x) for x in sorted(lcd["train_size"].unique())])
    ax.set_xlabel("Number of labelled training sentences (log scale)")
    ax.set_ylabel("Test accuracy, %")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_learning_curve.png"), dpi=200)
    lcd[["series", "train_size", "accuracy", "f1_macro"]].sort_values(["series", "train_size"]).to_csv(
        os.path.join(RESULTS_DIR, "learning_curve.csv"), index=False)

# ---------- Figure 4: confusion matrices (best classical vs best transformer) ----------
best_cls = df[df["fam"] == "classical"].iloc[-1]
best_tr = df[df["family"] == "transformer-finetuned"].iloc[-1] if (df["family"] == "transformer-finetuned").any() \
    else df[df["fam"] == "transformer"].iloc[-1]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3))
for ax, r in zip(axes, (best_cls, best_tr)):
    cm = np.array(r["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["neg", "pos"])
    ax.set_yticks([0, 1], ["neg", "pos"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title(r["name"], fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_confusion.png"), dpi=200)

# ---------- Linguistic error analysis: negation / contrast subsets ----------
test = load_split("test")
y = test["label"].values
neg_mask = test["sentence"].str.contains(r"\b(?:not|no|never|nothing|n't|nor|neither)\b|n't", regex=True).values
contrast_mask = test["sentence"].str.contains(r"\b(?:but|although|though|yet|however)\b", regex=True).values
rows = []
for _, r in df.iterrows():
    pred = np.array(r["predictions"])
    rows.append({"name": r["name"], "fam": r["fam"],
                 "all": (pred == y).mean(),
                 "negation": (pred[neg_mask] == y[neg_mask]).mean(),
                 "contrast": (pred[contrast_mask] == y[contrast_mask]).mean()})
sub = pd.DataFrame(rows)
sub.to_csv(os.path.join(RESULTS_DIR, "subset_accuracy.csv"), index=False)
print(f"\nnegation subset n={neg_mask.sum()}, contrast subset n={contrast_mask.sum()}")
print(sub.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(sub))
w = 0.26
for k, (col, c) in enumerate((("all", "#c3c2b7"), ("negation", "#2a78d6"), ("contrast", "#eb6834"))):
    ax.bar(x + (k - 1) * w, sub[col] * 100, w, color=c, edgecolor="white", linewidth=1.5,
           label={"all": "All sentences", "negation": f"With negation (n={neg_mask.sum()})",
                  "contrast": f"With contrast (n={contrast_mask.sum()})"}[col])
ax.set_xticks(x, [n.replace(" (", "\n(") for n in sub["name"]], fontsize=7, rotation=0)
ax.set_ylim(50, 100)
ax.set_ylabel("Accuracy, %")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_subsets.png"), dpi=200)

# ---------- Figure 6: fine-tuning dev curves ----------
hist = df[df["history"].notna()] if "history" in df else pd.DataFrame()
if len(hist):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    pal = ["#eb6834", "#1baf7a", "#2a78d6", "#6250d6"]
    for i, (_, r) in enumerate(hist.iterrows()):
        h = pd.DataFrame(r["history"])
        ax.plot(h["epoch"], h["dev_accuracy"] * 100, marker="o", lw=2, ms=6, color=pal[i % 4], label=r["name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dev accuracy, %")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_training_curves.png"), dpi=200)

# Disagreement examples for the error-analysis section
if len(df) >= 2:
    t = np.array(best_tr["predictions"])
    c = np.array(best_cls["predictions"])
    ex = test[(t == y) & (c != y)].head(8).assign(winner=best_tr["name"])
    ex2 = test[(t != y) & (c == y)].head(5).assign(winner=best_cls["name"])
    pd.concat([ex, ex2]).to_csv(os.path.join(RESULTS_DIR, "disagreements.csv"), index=False)
    print(f"\ntransformer right / classical wrong: {((t == y) & (c != y)).sum()}, "
          f"reverse: {((t != y) & (c == y)).sum()}")
# ---------- Statistics: bootstrap 95% CI for accuracy, McNemar test vs best model ----------
from scipy.stats import binomtest  # noqa: E402

rng = np.random.default_rng(42)
B = 2000
idx = rng.integers(0, len(y), size=(B, len(y)))
best = df.iloc[-1]
best_pred = np.array(best["predictions"])
stats = []
for _, r in df.iterrows():
    pred = np.array(r["predictions"])
    correct = (pred == y).astype(float)
    boot = correct[idx].mean(axis=1)
    b01 = int(((best_pred == y) & (pred != y)).sum())  # best right, this wrong
    b10 = int(((best_pred != y) & (pred == y)).sum())  # best wrong, this right
    p = 1.0 if b01 + b10 == 0 else binomtest(min(b01, b10), b01 + b10, 0.5).pvalue  # exact McNemar
    stats.append({"name": r["name"], "accuracy": correct.mean(), "ci_low": np.percentile(boot, 2.5),
                  "ci_high": np.percentile(boot, 97.5), "b01_best_only": b01, "b10_model_only": b10,
                  "mcnemar_p_vs_best": p})
stats = pd.DataFrame(stats)
stats.to_csv(os.path.join(RESULTS_DIR, "statistics.csv"), index=False)
print(f"\nStatistics vs best model ({best['name']}):")
print(stats.to_string(index=False))
print("figures written to", FIG_DIR)
