"""Off-the-shelf Transformer models evaluated without any training on our side:

1. distilbert-base-uncased-finetuned-sst-2-english  (in-domain: already fine-tuned on SST-2)
2. cardiffnlp/twitter-roberta-base-sentiment-latest (out-of-domain: trained on tweets, 3 classes)
3. cross-encoder/nli-distilroberta-base              (zero-shot classification through NLI)
"""
import os

import numpy as np
from transformers import pipeline

from utils import Timer, compute_metrics, load_split, save_result, set_seed

set_seed()
DEV = 0 if os.environ.get("DEVICE") == "cuda" else -1   # CPU by default; DEVICE=cuda uses the first GPU (e.g. Google Colab)
test = load_split("test")
texts, y = test["sentence"].tolist(), test["label"].values


def n_params(pipe):
    return sum(p.numel() for p in pipe.model.parameters())


def save(tag, family, model_name, pred, seconds, params):
    save_result(tag, {
        "model": model_name,
        "family": family,
        "train_size": 0,
        "train_time_s": 0.0,
        "latency_ms_per_sample": 1000 * seconds / len(texts),
        "parameters": params,
        **compute_metrics(y, pred),
        "predictions": pred.tolist(),
    })


# 1. In-domain fine-tuned DistilBERT
name = "distilbert-base-uncased-finetuned-sst-2-english"
clf = pipeline("sentiment-analysis", model=name, device=DEV)
with Timer() as t:
    out = clf(texts, batch_size=32, truncation=True)
save("pt_distilbert_sst2", "transformer-pretrained", name,
     np.array([int(o["label"] == "POSITIVE") for o in out]), t.seconds, n_params(clf))

# 2. Out-of-domain Twitter RoBERTa: compare positive vs negative scores only (neutral ignored)
name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
clf = pipeline("sentiment-analysis", model=name, device=DEV)
with Timer() as t:
    out = clf(texts, batch_size=32, truncation=True, top_k=None)
pred = []
for scores in out:
    s = {d["label"].lower(): d["score"] for d in scores}
    pred.append(int(s["positive"] >= s["negative"]))
save("pt_twitter_roberta", "transformer-pretrained", name, np.array(pred), t.seconds, n_params(clf))

# 3. Zero-shot classification via natural language inference
name = "cross-encoder/nli-distilroberta-base"
zs = pipeline("zero-shot-classification", model=name, device=DEV)
with Timer() as t:
    out = zs(texts, candidate_labels=["positive", "negative"],
             hypothesis_template="The sentiment of this movie review is {}.", batch_size=32)
save("zs_nli_distilroberta", "transformer-zero-shot", name,
     np.array([int(o["labels"][0] == "positive") for o in out]), t.seconds, n_params(zs))
