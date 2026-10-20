"""Learning curve for the TF-IDF + Logistic Regression baseline (same sample sizes as the
Transformer learning curve, so the data efficiency of pre-training can be compared)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from utils import compute_metrics, load_split, save_result, set_seed

SIZES = [500, 1000, 2000, 4000, 8000]

set_seed()
train, test = load_split("train"), load_split("test")
for n in SIZES:
    sample = train.sample(n=n, random_state=42)  # identical sample to 04_finetune_transformer.py
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, max_iter=2000)),
    ])
    pipe.fit(sample["sentence"], sample["label"])
    pred = pipe.predict(test["sentence"])
    save_result(f"lc_tfidf_logreg_{n}", {"model": "tfidf_logreg", "train_size": n,
                                          **compute_metrics(test["label"], pred)})
