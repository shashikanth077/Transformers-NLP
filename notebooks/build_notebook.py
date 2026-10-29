"""Generates notebooks/Transformers_NLP_Colab.ipynb (run in Google Colab). Run: python notebooks/build_notebook.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/shashikanth077/Transformers-NLP"
cells = []


def md(text): cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})
def code(text, hide=False):
    meta = {"cellView": "form"} if hide else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)})


md(f"""
# Transformers in NLP: run the whole project in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab/badge.svg)](https://colab.research.google.com/github/shashikanth077/Transformers-NLP/blob/master/notebooks/Transformers_NLP_Colab.ipynb)

**Project:** *Transformers and their application in NLP tasks* (VILNIUS TECH, Shashikanth Hosur Ramegowda and Shaeed Khan).
This notebook runs **the same code as the repository** ([`code/`]({REPO}/tree/master/code)): ten sentiment classifiers on SST-2 (3 TF-IDF models, a BiLSTM, 3 Transformers we fine-tune, 3 ready-made Transformers), then the report with tables, charts and statistics.

### How to use it
1. **Runtime → Run all**, or run the cells one by one from top to bottom (read the text above each cell: it explains what the step does).
2. For speed choose **Runtime → Change runtime type → T4 GPU** (optional). The code is written for the CPU, and a GPU is used only for the fine-tuning and the ready-made models when you allow it in step 2.
3. Choose the mode in step 2: **quick** (a demo in about 10 to 20 minutes) or **full** (the same settings as the paper).

### What to expect (estimates, not measured on every runtime)
| Mode | Colab CPU | Colab T4 GPU |
|---|---|---|
| quick (2,000 training sentences for the three Transformers, no BiLSTM) | roughly 30 to 60 minutes | roughly 10 to 20 minutes |
| full (8,000 sentences, BiLSTM, learning curve, report) | several hours (the laptop in the paper needed about 2.5 to 3 hours with 4 cores) | roughly 30 to 60 minutes |

> **Important for the viva:** the accuracies should be close to the paper (the data split and the random seed are fixed), but **training time and latency are NOT comparable** with the paper's laptop CPU numbers when you use Colab or a GPU. The paper's timings are for an Intel Core i5-8365U laptop CPU without a GPU.
""")

md("## Step 0 · Check the machine")
code('''
import os, platform, sys, time
import torch
print("Python", sys.version.split()[0], "| PyTorch", torch.__version__)
print("CPU cores:", os.cpu_count())
print("GPU available:", torch.cuda.is_available(), "-", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU runtime)")
START = time.time()
''')

md("""
## Step 1 · Get the code and install the libraries
The cell clones the project repository (`code/` has the nine scripts) and installs the few extra libraries Colab does not have. The first run downloads the dataset and the pre-trained models from the Hugging Face Hub (about 1.5 GB in total), which can take a few minutes.
""")
code(f'''
import os
if not os.path.exists("Transformers-NLP"):
    !git clone -q {REPO}.git
%cd Transformers-NLP/code
!pip -q install -r requirements.txt
!mkdir -p results figures data
# keep a copy of the results from the paper (laptop CPU) so that we can compare with the Colab run later
!cp -r results results_paper_laptop 2>/dev/null || true
!ls
''')

md("""
## Step 2 · Choose how to run
* **MODE = quick**: a small demo. The three Transformers are fine-tuned on 2,000 sentences instead of 8,000, the BiLSTM is skipped, and the learning curve uses 500 and 1,000 sentences. The numbers will be lower than in the paper.
* **MODE = full**: exactly the settings of the paper (`run_all.sh`).
* **DEVICE_CHOICE = auto** uses the GPU for fine-tuning and the ready-made models when Colab gives you one; choose **cpu** to run like the paper (slower).
""")
code('''
#@title Settings { display-mode: "form" }
MODE = "quick" #@param ["quick", "full"]
DEVICE_CHOICE = "auto" #@param ["auto", "cpu"]

import os, torch
DEVICE = "cuda" if (DEVICE_CHOICE == "auto" and torch.cuda.is_available()) else "cpu"
os.environ["DEVICE"] = DEVICE            # read by 04_finetune_transformer.py and 05_pretrained_transformers.py
QUICK = MODE == "quick"
N_MAIN = 2000 if QUICK else 8000          # labelled sentences used to fine-tune the three Transformers
LC_SIZES = [500, 1000] if QUICK else [500, 1000, 2000, 4000]   # BERT-small learning-curve sizes (besides N_MAIN)
RUN_BILSTM = not QUICK
print(f"MODE={MODE} | device for Transformers: {DEVICE} | fine-tuning labels: {N_MAIN} | BiLSTM: {RUN_BILSTM}")

def run(cmd):
    """Run a shell command and stream its output into the notebook."""
    print("$", cmd, flush=True)
    get_ipython().system(cmd)
''', hide=True)

md("""
## Step 3 · Data: SST-2 (movie-review sentences, positive or negative)
`01_prepare_data.py` downloads SST-2 and creates three files: **train** (65,349 sentences, what the models learn from), **dev** (2,000, used to choose the best epoch) and **test** (872, the final honest exam that is never used for learning).
""")
code('''
run("python 01_prepare_data.py")
import pandas as pd
for split in ["train", "dev", "test"]:
    df = pd.read_csv(f"data/{split}.csv")
    print(f"{split:5s} {len(df):6d} sentences | positive share {df['label'].mean():.1%} | average length {df['sentence'].str.split().str.len().mean():.1f} words")
pd.read_csv("data/test.csv").sample(5, random_state=1)
''')

md("""
## Step 4 · Classical baselines: TF-IDF + naive Bayes, logistic regression, linear SVM
Each sentence becomes a list of weighted words and word pairs (TF-IDF); a simple classifier learns which weights mean positive or negative. These models train in seconds and are the strong, cheap baselines every Transformer must beat. They ignore word order, so "not good" still contains "good".
""")
code('''
run("python 02_classical_baselines.py")
''')

md("""
## Step 5 · BiLSTM trained from scratch (the pre-Transformer neural approach)
A recurrent network that reads each sentence forwards and backwards, with no pre-training. It runs on the CPU in both modes; in **quick** mode it is skipped to save time (set `RUN_BILSTM = True` in step 2 if you want it).
""")
code('''
if RUN_BILSTM:
    run("python 03_bilstm.py")
else:
    print("BiLSTM skipped in quick mode.")
''')

md("""
## Step 6 · Ready-made Transformers (no training by us)
Three public models evaluated on the 872 test sentences: **DistilBERT-SST2** (already fine-tuned on this task), **Twitter-RoBERTa** (trained on tweets, a different domain) and a **zero-shot NLI model** (never trained for sentiment, asked anyway). They show how far simply reusing a model goes.
""")
code('''
run("python 05_pretrained_transformers.py")
''')

md("""
## Step 7 · Fine-tune three Transformers on our labelled sentences
`04_finetune_transformer.py` takes any pre-trained encoder from the Hugging Face Hub, adds a classification layer and trains the whole network with AdamW (10% warm-up, linear decay) for a few epochs, keeping the **best epoch on the dev set**. The three models differ only in size: BERT-tiny (4 M parameters), BERT-small (29 M) and DistilBERT (67 M).
""")
code('''
FT = [  # (tag, model name on the Hugging Face Hub, learning rate, epochs)
    (f"ft_bert-tiny_{N_MAIN}",  "google/bert_uncased_L-2_H-128_A-2", 2e-4, 3),
    (f"ft_bert-small_{N_MAIN}", "google/bert_uncased_L-4_H-512_A-8", 1e-4, 3),
    (f"ft_distilbert_{N_MAIN}", "distilbert-base-uncased",           5e-5, 2),
]
for tag, model_name, lr, epochs in FT:
    run(f"python 04_finetune_transformer.py --model {model_name} --train-size {N_MAIN} --epochs {epochs} --lr {lr} --tag {tag}")
''')

md("""
## Step 8 · Data efficiency: how many labels do we need?
We train BERT-small and a TF-IDF logistic regression on the *same* random subsets of 500, 1,000, 2,000 … labelled sentences. In the paper, with 500 labels BERT-small reached 75.9% against 61.8% for TF-IDF, because pre-training already taught BERT English.
""")
code('''
run("python 06_learning_curve_classical.py")
for n in LC_SIZES:
    run(f"python 04_finetune_transformer.py --model google/bert_uncased_L-4_H-512_A-8 --train-size {n} --epochs 3 --lr 1e-4 --tag lc_bert-small_{n}")
# the largest point is the main BERT-small run from step 7
!cp results/ft_bert-small_{N_MAIN}.json results/lc_bert-small_{N_MAIN}.json
''')

md("""
## Step 9 · Results: the ten models side by side
The table reads every result file written by the scripts (accuracy and macro-F1 on the 872 test sentences, training time, latency per sentence, number of parameters). Remember: on Colab the **times** are not comparable with the paper's laptop.
""")
code('''
import glob, json, os
import pandas as pd
import matplotlib.pyplot as plt

rows = []
for f in sorted(glob.glob("results/*.json")):
    tag = os.path.basename(f)[:-5]
    if tag.startswith("lc_"):
        continue
    r = json.load(open(f, encoding="utf-8"))
    rows.append({"model": tag, "family": r.get("family"), "labels used": r.get("train_size"), "accuracy %": round(100 * r["accuracy"], 1),
                 "macro-F1 %": round(100 * r["f1_macro"], 1), "train time s": round(r.get("train_time_s", 0)),
                 "latency ms": round(r["latency_ms_per_sample"], 2), "parameters": r.get("parameters")})
table = pd.DataFrame(rows).sort_values("accuracy %", ascending=False).reset_index(drop=True)
display(table)

colors = {"classical": "#2a78d6", "recurrent": "#eb6834"}
fig, ax = plt.subplots(figsize=(8, 0.45 * len(table) + 1))
ax.barh(table["model"][::-1], table["accuracy %"][::-1], color=[colors.get(f, "#1baf7a") for f in table["family"][::-1]])
ax.set_xlim(50, 95); ax.set_xlabel("accuracy on the 872 test sentences, %"); ax.set_title(f"Ten sentiment classifiers (mode: {MODE})")
for i, v in enumerate(table["accuracy %"][::-1]): ax.text(v + 0.3, i, f"{v}", va="center", fontsize=9)
plt.tight_layout(); plt.show()
''')
code('''
# learning curve: BERT-small (pre-trained) against TF-IDF logistic regression on the same subsets
lc = []
for f in sorted(glob.glob("results/lc_*.json")):
    r = json.load(open(f, encoding="utf-8"))
    lc.append({"series": "TF-IDF + logistic regression" if "tfidf" in os.path.basename(f) else "BERT-small (pre-trained)", "labels": r["train_size"], "accuracy": 100 * r["accuracy"]})
lc = pd.DataFrame(lc)
if len(lc):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, g in lc.groupby("series"):
        g = g.sort_values("labels"); ax.plot(g["labels"], g["accuracy"], marker="o", label=name)
        for x, y in zip(g["labels"], g["accuracy"]): ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("labelled training sentences (log scale)"); ax.set_ylabel("test accuracy, %"); ax.legend(); plt.tight_layout(); plt.show()
else:
    print("No learning-curve results yet.")
''')

md("""
## Step 10 · Compare with the paper (laptop CPU) and read the negation results
`results_paper_laptop/` holds the files from the paper. Accuracies should be close (same split, same seed); differences of a few tenths of a point come from different hardware and library versions. **In quick mode the fine-tuned models used fewer labels, so they will be lower on purpose.**
""")
code('''
cmp = []
for f in sorted(glob.glob("results/*.json")):
    tag = os.path.basename(f)[:-5]
    ref = f"results_paper_laptop/{tag}.json"
    if tag.startswith("lc_") or not os.path.exists(ref):
        continue
    a, b = json.load(open(f, encoding="utf-8")), json.load(open(ref, encoding="utf-8"))
    cmp.append({"model": tag, "this run %": round(100 * a["accuracy"], 1), "paper %": round(100 * b["accuracy"], 1),
                "difference": round(100 * (a["accuracy"] - b["accuracy"]), 1)})
display(pd.DataFrame(cmp) if cmp else "Nothing to compare yet (in quick mode the fine-tuned models use different tags).")

# accuracy on the tricky sentences: negation (not, no, never, n't ...) and contrast (but, although, ...)
import numpy as np
test = pd.read_csv("data/test.csv"); y = test["label"].values
neg = test["sentence"].str.contains(r"\\b(?:not|no|never|nothing|n't|nor|neither)\\b|n't", regex=True).values
con = test["sentence"].str.contains(r"\\b(?:but|although|though|yet|however)\\b", regex=True).values
sub = []
for f in sorted(glob.glob("results/*.json")):
    tag = os.path.basename(f)[:-5]
    if tag.startswith("lc_"): continue
    p = np.array(json.load(open(f, encoding="utf-8"))["predictions"])
    sub.append({"model": tag, "all %": round(100 * (p == y).mean(), 1), f"negation % (n={neg.sum()})": round(100 * (p[neg] == y[neg]).mean(), 1), f"contrast % (n={con.sum()})": round(100 * (p[con] == y[con]).mean(), 1)})
display(pd.DataFrame(sub).sort_values("all %", ascending=False).reset_index(drop=True))
''')

md("""
## Step 11 · Look inside the Transformer: attention
`07_attention_visualisation.py` takes the DistilBERT sentiment model and the sentence *"the plot was not good, but the acting was brilliant"* and draws the attention weights of the last layer: which words each word looks at. It illustrates where information can flow between words, but it is **not a proof** that the model understands negation.
""")
code('''
run("python 07_attention_visualisation.py")
from IPython.display import Image, display
display(Image("figures/attention_heatmap.png"))
display(Image("figures/cls_attention_layers.png"))
''')

md("""
## Step 12 · The full report (only in full mode)
`08_report.py` collects every result file and produces the tables, the figures of the paper, the negation and contrast analysis, bootstrap confidence intervals and McNemar tests. It needs all ten models and the learning-curve files, so it is run in **full** mode only.
""")
code('''
if not QUICK:
    run("python 08_report.py")
    from IPython.display import Image, display
    for name in ["fig_accuracy.png", "fig_accuracy_latency.png", "fig_learning_curve.png", "fig_subsets.png"]:
        if os.path.exists(f"figures/{name}"): display(Image(f"figures/{name}"))
else:
    print("Quick mode: the full report (08_report.py) is skipped. Switch MODE to full in step 2 to run it.")
''')

md("""
## Step 13 · Try your own sentence
Type any review and see what a **word-counting model** (TF-IDF + linear SVM, trained here in a few seconds) and a **Transformer** (the ready-made DistilBERT-SST2) say. Try a sentence with negation, for example *"a great ensemble cast can't lift this heartfelt enterprise out of the familiar"* (the true label is negative).
""")
code('''
#@title Your sentence { display-mode: "form" }
SENTENCE = "a great ensemble cast can't lift this heartfelt enterprise out of the familiar" #@param {type:"string"}

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import pipeline

train = pd.read_csv("data/train.csv")
svm = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)), ("clf", LinearSVC(C=0.5))]).fit(train["sentence"], train["label"])
score = float(svm.decision_function([SENTENCE])[0])
print(f"TF-IDF + SVM  : {'positive' if score > 0 else 'negative'}   (score {score:+.2f})")

clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if DEVICE == "cuda" else -1)
out = clf(SENTENCE)[0]
print(f"DistilBERT-SST2: {out['label'].lower()}   (confidence {out['score']:.1%})")
''', hide=True)

md("""
## Step 14 · Save your results
Downloads a zip with the result files, tables and figures of this run (so you can add them to the report or compare later).
""")
code('''
import shutil
shutil.make_archive("/content/colab_results", "zip", ".", "results")
print(f"Total time of this notebook run: {(time.time() - START) / 60:.1f} minutes")
try:
    from google.colab import files
    files.download("/content/colab_results.zip")
except Exception as e:
    print("Not running in Colab; the archive is at /content/colab_results.zip", e)
''')

md(f"""
---
### Notes for the viva
* **Same code as the paper:** the notebook only calls the repository scripts (`01` to `08`); nothing is re-implemented. The only change for Colab is an optional switch `DEVICE=cuda` in scripts 04 and 05; by default they run on the CPU exactly as before.
* **Why numbers can differ slightly:** other hardware, other library versions, and GPU arithmetic. The data split and the random seed (42) are fixed.
* **What stays valid:** the ranking of the models and the conclusions (pre-training helps most when labels are few; the cost is slower and bigger models). The times in the paper belong to the laptop CPU.
* Repository: {REPO}
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": [], "name": "Transformers_NLP_Colab.ipynb", "toc_visible": True}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "Transformers_NLP_Colab.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out, len(cells), "cells")
