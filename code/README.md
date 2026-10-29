# Transformers for NLP: sentiment classification comparison (SST-2)

Code for the group project "Transformers and their application in natural language processing (NLP) tasks"
(Shashikanth Hosur Ramegowda, Shaeed Khan, VILNIUS TECH).

The project compares 10 models on binary sentiment classification of the SST-2 dataset, on CPU only:

| Family | Models |
|---|---|
| Classical | TF-IDF + Naive Bayes, Logistic Regression, Linear SVM |
| Recurrent | BiLSTM trained from scratch |
| Transformer, fine-tuned by us (8k sentences) | BERT-tiny, BERT-small, DistilBERT |
| Transformer, ready-made | DistilBERT-SST2, Twitter-RoBERTa, zero-shot NLI DistilRoBERTa |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Internet access is needed on the first run to download the dataset and the pre-trained models
from the Hugging Face Hub (about 1.5 GB in total).

## Optional: GPU (for example in Google Colab)

The scripts run on the CPU by default, as in the reported results. Setting the environment variable `DEVICE=cuda` makes `04_finetune_transformer.py` and `05_pretrained_transformers.py` use the GPU (see `notebooks/Transformers_NLP_Colab.ipynb`). Times and latencies measured on a GPU are not comparable with the CPU results.

## Run everything

```bash
bash run_all.sh                 # about 2.5-3 hours on a 4-core laptop CPU
```

or run the steps one by one:

| Script | Purpose |
|---|---|
| `01_prepare_data.py` | download SST-2, create `data/train.csv`, `dev.csv`, `test.csv` |
| `02_classical_baselines.py` | TF-IDF + NB / LR / SVM |
| `03_bilstm.py` | BiLSTM baseline (PyTorch) |
| `04_finetune_transformer.py` | fine-tune any Hugging Face encoder, e.g. `--model distilbert-base-uncased --train-size 8000 --epochs 2 --lr 5e-5` |
| `05_pretrained_transformers.py` | off-the-shelf and zero-shot Transformer models |
| `06_learning_curve_classical.py` | classical baseline with 500…8000 training examples |
| `07_attention_visualisation.py` | attention heat-maps of a fine-tuned DistilBERT |
| `08_report.py` | tables (`results/*.csv`), figures (`figures/*.png`), bootstrap CIs, McNemar tests |
| `09_diagrams.py` | architecture and pipeline diagrams |

Every model writes `results/<name>.json` with its metrics, timings, parameter count and predictions.
The test set is the official SST-2 validation split (872 sentences); 2,000 training sentences form the dev set.
