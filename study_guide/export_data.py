"""Collects the real experiment data for the interactive study guide -> study_guide/data.json

Needs the project environment (scikit-learn, transformers, torch) and the results in code/results.
Run:  python study_guide/export_data.py
"""
import base64
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
RES = os.path.join(CODE, "results")
OUT = os.path.join(ROOT, "study_guide", "data.json")

# order matters: the 10-character prediction string of every test sentence follows this order
MODELS = [
    ("tfidf_nb", "TF-IDF + Naive Bayes", "classical", "TF-IDF NB"),
    ("tfidf_logreg", "TF-IDF + Logistic Regr.", "classical", "TF-IDF LR"),
    ("tfidf_svm", "TF-IDF + Linear SVM", "classical", "TF-IDF SVM"),
    ("bilstm", "BiLSTM (from scratch)", "recurrent", "BiLSTM"),
    ("ft_bert-tiny_8000", "BERT-tiny fine-tuned (8k)", "transformer", "BERT-tiny"),
    ("ft_bert-small_8000", "BERT-small fine-tuned (8k)", "transformer", "BERT-small"),
    ("ft_distilbert_8000", "DistilBERT fine-tuned (8k)", "transformer", "DistilBERT-FT"),
    ("pt_distilbert_sst2", "DistilBERT-SST2 (ready-made)", "transformer", "DistilBERT-SST2"),
    ("pt_twitter_roberta", "Twitter-RoBERTa (ready-made)", "transformer", "Twitter-RoBERTa"),
    ("zs_nli_distilroberta", "Zero-shot NLI DistilRoBERTa", "transformer", "Zero-shot NLI"),
]
WALK = [0, 4, 627, 110, 398, 481, 527, 183, 271]   # test-set rows used for the sentence walk-through
TOKEN_WORDS = ["unwatchable", "heavy-handed", "perilously", "unflinchingly", "entertaining", "Transformers", "Vilnius", "ChatGPT"]

res = {t: json.load(open(os.path.join(RES, t + ".json"), encoding="utf-8")) for t, *_ in MODELS}
test = pd.read_csv(os.path.join(CODE, "data", "test.csv"))
train = pd.read_csv(os.path.join(CODE, "data", "train.csv"))
y = test.label.values

neg_re = re.compile(r"\b(?:not|no|never|nothing|nor|neither)\b|n't")
con_re = re.compile(r"\b(?:but|although|though|yet|however)\b")
sentences = []
for i, r in test.iterrows():
    p = "".join(str(res[t]["predictions"][i]) for t, *_ in MODELS)
    sentences.append({"s": r.sentence, "y": int(r.label), "p": p,
                      "n": int(bool(neg_re.search(r.sentence))), "c": int(bool(con_re.search(r.sentence)))})

models = []
for t, name, fam, short in MODELS:
    r = res[t]
    models.append({"tag": t, "name": name, "short": short, "family": fam, "train_size": r["train_size"],
                   "acc": r["accuracy"], "prec": r["precision"], "rec": r["recall"], "f1": r["f1_macro"],
                   "train_s": r["train_time_s"], "lat_ms": r["latency_ms_per_sample"], "params": r["parameters"],
                   "history": [h["dev_accuracy"] for h in r.get("history", [])]})

stats = pd.read_csv(os.path.join(RES, "statistics.csv"))
name2tag = {m["name"].replace(" (ready-made)", " (off-the-shelf)"): m["tag"] for m in models}
name2tag.update({m["name"]: m["tag"] for m in models})
stat_rows = {}
for _, r in stats.iterrows():
    tag = name2tag.get(r["name"])
    if tag is None:
        tag = next(m["tag"] for m in models if m["name"].split(" (")[0] == r["name"].split(" (")[0])
    stat_rows[tag] = {"lo": r.ci_low, "hi": r.ci_high}

lc = pd.read_csv(os.path.join(RES, "learning_curve.csv"))
curve = {}
for _, r in lc.iterrows():
    key = "transformer" if r.series.startswith("BERT") else "tfidf"
    curve.setdefault(key, []).append([int(r.train_size), float(r.accuracy)])

# ------------------------------------------------------------ TF-IDF + logistic regression (real weights)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(train.sentence)
lr = LogisticRegression(C=4.0, max_iter=2000).fit(Xtr, train.label)
Xte = vec.transform(test.sentence)
full_pred = lr.predict(Xte)
print("retrained LR accuracy", (full_pred == y).mean(), "(saved model: %.4f)" % res["tfidf_logreg"]["accuracy"])
coef = lr.coef_[0]
terms = np.array(vec.get_feature_names_out())
idf = vec.idf_
order = np.argsort(-np.abs(coef))


# full model, quantised to 3 decimals (stored as int16 / 1000) so that the browser reproduces it exactly
idf_q = np.round(idf * 1000).astype(np.int16)
coef_q = np.round(coef * 1000).astype(np.int16)
bias_q = round(float(lr.intercept_[0]), 3)


def quantised_predict(sentences_):
    index = {t: j for j, t in enumerate(terms)}
    out = []
    for s in sentences_:
        toks = re.findall(r"\b\w\w+\b", s.lower())
        grams = toks + [a + " " + b_ for a, b_ in zip(toks, toks[1:])]
        tf = {}
        for g in grams:
            if g in index:
                tf[g] = tf.get(g, 0) + 1
        w = {g: (1 + np.log(n)) * idf_q[index[g]] / 1000 for g, n in tf.items()}
        norm = np.sqrt(sum(v * v for v in w.values())) or 1.0
        z = bias_q + sum(coef_q[index[g]] / 1000 * v / norm for g, v in w.items())
        out.append(int(z > 0))
    return np.array(out)


agreement = float((quantised_predict(test.sentence) == full_pred).mean())
print("quantised browser model agrees with the sklearn model on", agreement)
assert agreement > 0.995
lr_export = {"terms": "\n".join(terms), "idf": base64.b64encode(idf_q.tobytes()).decode(),
             "coef": base64.b64encode(coef_q.tobytes()).decode(), "bias": bias_q,
             "n_features": int(len(terms)), "agreement": agreement}

# ------------------------------------------------------------ real attention maps (DistilBERT-SST2)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, output_attentions=True).eval()


def attention_for(text):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    att = torch.stack([a[0] for a in out.attentions]).numpy()          # layers x heads x n x n
    q = np.clip(np.round(att * 255), 0, 255).astype(np.uint8)
    p = out.logits.softmax(-1)[0].tolist()
    return toks, base64.b64encode(q.tobytes()).decode(), att.shape, p[1]


walk = []
for i in WALK:
    s = test.sentence[i]
    toks, b64, shape, ppos = attention_for(s)
    walk.append({"i": i, "tokens": toks, "att": b64, "shape": list(shape), "p_pos": round(ppos, 4)})
custom = "the plot was not good , but the acting was brilliant"
toks, b64, shape, ppos = attention_for(custom)
walk.append({"i": -1, "text": custom, "tokens": toks, "att": b64, "shape": list(shape), "p_pos": round(ppos, 4)})

tokenisation = [{"word": w, "pieces": tok.tokenize(w)} for w in TOKEN_WORDS]

data = {"models": models, "stats": stat_rows, "curve": curve, "sentences": sentences, "lr": lr_export,
        "walk": walk, "tokenisation": tokenisation,
        "dataset": {"train": int(len(train)), "dev": 2000, "test": int(len(test)),
                    "train_pos": float(train.label.mean()), "test_pos": float(test.label.mean()),
                    "train_len": float(train.sentence.str.split().str.len().mean()),
                    "test_len": float(test.sentence.str.split().str.len().mean())}}
json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("saved", OUT, round(os.path.getsize(OUT) / 1024), "KB")
