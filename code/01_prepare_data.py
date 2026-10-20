"""Download GLUE SST-2 and create train / dev / test CSV splits.

SST-2 has no public test labels, so the official validation split (872 sentences)
is used as the held-out TEST set, and 2,000 sentences are carved out of the
training data as a DEV set for model selection / early stopping.
"""
import os

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from utils import DATA_DIR, SEED

os.makedirs(DATA_DIR, exist_ok=True)
ds = load_dataset("nyu-mll/glue", "sst2")
train_full = ds["train"].to_pandas()[["sentence", "label"]]
test = ds["validation"].to_pandas()[["sentence", "label"]]

train, dev = train_test_split(train_full, test_size=2000, random_state=SEED, stratify=train_full["label"])
for name, df in (("train", train), ("dev", dev), ("test", test)):
    df = df.copy()
    df["sentence"] = df["sentence"].str.strip()
    df.to_csv(os.path.join(DATA_DIR, f"{name}.csv"), index=False)
    print(f"{name:5s}: {len(df):6d} rows, positive share = {df['label'].mean():.3f}, "
          f"mean length = {df['sentence'].str.split().str.len().mean():.1f} words")
