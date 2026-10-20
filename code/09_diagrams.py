"""Schematic diagrams for the paper: Transformer encoder and program pipeline."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils import FIG_DIR

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
INK = "#0b0b0b"
os.makedirs(FIG_DIR, exist_ok=True)


def box(ax, x, y, w, h, text, fc, fs=10, ec="#52514e", weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight)


def arrow(ax, x1, y1, x2, y2, style="-|>", color="#52514e", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color,
                                 lw=1.2, connectionstyle=f"arc3,rad={rad}"))


# ---------------- Transformer encoder + classification head ----------------
fig, ax = plt.subplots(figsize=(6.2, 7.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.axis("off")
cx = 4.2
box(ax, cx, 0.7, 6.2, 0.8, "Input tokens:  [CLS]  the  plot  was  not  good  [SEP]", "#f1f0ec", 9.5)
box(ax, cx, 2.0, 6.2, 0.8, "Token embedding  +  Position embedding", "#d6e6f8")
arrow(ax, cx, 1.1, cx, 1.6)
# encoder layer frame
ax.add_patch(FancyBboxPatch((cx - 3.5, 2.9), 7.0, 5.6, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc="none", ec="#2a78d6", lw=1.6, ls="--"))
ax.text(cx + 3.65, 5.7, "× L layers", fontsize=11, color="#2a78d6", va="center", weight="bold")
box(ax, cx, 3.9, 5.0, 0.9, "Multi-head self-attention\nsoftmax(QKᵀ/√dₖ)·V , h heads", "#d4f0e4")
box(ax, cx, 5.2, 5.0, 0.6, "Add & Layer normalisation", "#fde9c8")
box(ax, cx, 6.4, 5.0, 0.8, "Position-wise feed-forward\nGELU(xW₁+b₁)W₂+b₂", "#d4f0e4")
box(ax, cx, 7.65, 5.0, 0.6, "Add & Layer normalisation", "#fde9c8")
arrow(ax, cx, 2.4, cx, 3.45)
arrow(ax, cx, 4.35, cx, 4.9)
arrow(ax, cx, 5.5, cx, 6.0)
arrow(ax, cx, 6.8, cx, 7.35)
# residual connections
arrow(ax, cx - 2.6, 3.1, cx - 2.6, 5.2, style="-", color="#eb6834")
arrow(ax, cx - 2.6, 5.2, cx - 2.5, 5.2, color="#eb6834")
arrow(ax, cx - 2.6, 5.6, cx - 2.6, 7.65, style="-", color="#eb6834")
arrow(ax, cx - 2.6, 7.65, cx - 2.5, 7.65, color="#eb6834")
ax.text(cx - 2.85, 4.2, "residual", rotation=90, fontsize=8.5, color="#eb6834", va="center")
ax.text(cx - 2.85, 6.6, "residual", rotation=90, fontsize=8.5, color="#eb6834", va="center")
box(ax, cx, 9.3, 6.2, 0.8, "Contextual vectors  h[CLS], h₁, …, hₙ", "#f1f0ec")
arrow(ax, cx, 7.95, cx, 8.9)
box(ax, cx, 10.7, 4.2, 0.8, "Classification head:  softmax(W·h[CLS] + b)", "#d6e6f8", 9.5)
arrow(ax, cx, 9.7, cx, 10.3)
box(ax, cx, 12.0, 3.6, 0.8, "P(negative),  P(positive)", "#d4f0e4", 10, weight="bold")
arrow(ax, cx, 11.1, cx, 11.6)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "diagram_encoder.png"), dpi=220)

# ---------------- Program pipeline ----------------
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9.4)
ax.axis("off")
box(ax, 2.0, 7.8, 3.4, 1.0, "Hugging Face Hub\nSST-2 dataset", "#f1f0ec", 9)
box(ax, 2.0, 5.6, 3.4, 1.0, "01_prepare_data.py\ntrain / dev / test CSV", "#d6e6f8", 9)
arrow(ax, 2.0, 7.3, 2.0, 6.1)
models = [
    (7.2, 8.4, "02_classical_baselines.py\nTF-IDF + NB / LR / SVM", "#d6e6f8"),
    (7.2, 6.9, "03_bilstm.py\nBiLSTM from scratch", "#fde2d4"),
    (7.2, 5.4, "04_finetune_transformer.py\nBERT-tiny / small / DistilBERT", "#d4f0e4"),
    (7.2, 3.9, "05_pretrained_transformers.py\noff-the-shelf + zero-shot", "#d4f0e4"),
    (7.2, 2.4, "06_learning_curve_classical.py\n+ 04 with 500…8000 examples", "#ece9fb"),
]
for x, y, t, c in models:
    box(ax, x, y, 4.6, 1.05, t, c, 8.5)
    arrow(ax, 3.7, 5.6, 4.9, y, rad=0.0)
box(ax, 7.2, 0.9, 4.6, 1.0, "07_attention_visualisation.py\nattention heat-maps", "#fdf3c8", 8.5)
box(ax, 12.0, 5.4, 2.6, 1.3, "results/*.json\nmetrics, time,\npredictions", "#f1f0ec", 8.5)
for x, y, t, c in models:
    arrow(ax, 9.5, y, 10.7, 5.4)
box(ax, 14.6, 5.4, 2.4, 1.3, "08_report.py\ntables, figures,\nstatistics", "#d6e6f8", 8.5)
arrow(ax, 13.3, 5.4, 13.4, 5.4)
box(ax, 14.6, 2.6, 2.4, 1.2, "Paper &\npresentation", "#d4f0e4", 9, weight="bold")
arrow(ax, 14.6, 4.75, 14.6, 3.2)
arrow(ax, 9.5, 0.9, 14.6, 2.0, rad=0.2)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "diagram_pipeline.png"), dpi=220)
print("diagrams saved")
