#!/usr/bin/env bash
# Runs the whole experiment end-to-end (CPU, ~2.5-3 h on a 4-core laptop).
# Models are run one after another so that time measurements are not distorted.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
TINY=google/bert_uncased_L-2_H-128_A-2
SMALL=google/bert_uncased_L-4_H-512_A-8
DISTIL=distilbert-base-uncased

[ -f data/test.csv ] || $PY 01_prepare_data.py
$PY 02_classical_baselines.py
$PY 03_bilstm.py
$PY 05_pretrained_transformers.py
$PY 07_attention_visualisation.py
$PY 06_learning_curve_classical.py

# Main comparison: every Transformer fine-tuned on the same 8,000 labelled sentences
$PY 04_finetune_transformer.py --model $TINY   --train-size 8000 --epochs 3 --lr 2e-4 --tag ft_bert-tiny_8000
$PY 04_finetune_transformer.py --model $SMALL  --train-size 8000 --epochs 3 --lr 1e-4 --tag ft_bert-small_8000
$PY 04_finetune_transformer.py --model $DISTIL --train-size 8000 --epochs 2 --lr 5e-5 --tag ft_distilbert_8000

# Learning curve for BERT-small (the 8,000 point is the run above)
for N in 500 1000 2000 4000; do
  $PY 04_finetune_transformer.py --model $SMALL --train-size $N --epochs 3 --lr 1e-4 --tag lc_bert-small_$N
done
cp results/ft_bert-small_8000.json results/lc_bert-small_8000.json

$PY 08_report.py
