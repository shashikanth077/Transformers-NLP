"""Shared helpers: data loading, metrics, timing and result persistence."""
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_split(name: str) -> pd.DataFrame:
    """Return one of: train, dev, test (columns: sentence, label)."""
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def save_result(name: str, result: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {path}")
    summary = {k: v for k, v in result.items() if k not in ("confusion_matrix", "predictions", "history")}
    print(json.dumps(summary, indent=2))
