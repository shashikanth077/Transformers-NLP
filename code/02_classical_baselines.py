"""Classical baselines: TF-IDF features + Naive Bayes / Logistic Regression / Linear SVM."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import Timer, compute_metrics, load_split, save_result, set_seed

set_seed()
train, test = load_split("train"), load_split("test")

models = {
    "tfidf_nb": MultinomialNB(alpha=0.5),
    "tfidf_logreg": LogisticRegression(C=4.0, max_iter=2000),
    "tfidf_svm": LinearSVC(C=0.5),
}

for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])
    with Timer() as t_train:
        pipe.fit(train["sentence"], train["label"])
    with Timer() as t_inf:
        pred = pipe.predict(test["sentence"])
    n_features = len(pipe.named_steps["tfidf"].vocabulary_)
    result = {
        "model": name,
        "family": "classical",
        "train_size": len(train),
        "train_time_s": t_train.seconds,
        "latency_ms_per_sample": 1000 * t_inf.seconds / len(test),
        "parameters": int(n_features * (2 if name == "tfidf_nb" else 1)),
        **compute_metrics(test["label"], pred),
        "predictions": pred.tolist(),
    }
    save_result(name, result)
