"""Builds the project paper (.docx) from the VGTU cover template + chapter markdown files.

Markup used in ch*.md:
  # / ## / ###  headings (numbered text is written by hand), "#*" = unnumbered chapter heading
  - item        bullet list
  $$ text || (n) equation (rendered from LATEX[n] as an image when available)
  [[TABLE key]] table from TABLES, [[FIG key]] figure from FIGS
"""
import copy
import os
import re
import sys

import docx
import matplotlib
import pandas as pd
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = os.path.join(ROOT, "code")
FIG = os.path.join(CODE, "figures")
RES = os.path.join(CODE, "results")
EQ_DIR = os.path.join(HERE, "eq")
TEMPLATE = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/Admin/Downloads/1Cover_of_Master_Thesis.docx"
OUT = os.path.join(ROOT, "Transformers_in_NLP_Project_Paper.docx")

TITLE = "TRANSFORMERS AND THEIR APPLICATION IN NATURAL LANGUAGE PROCESSING (NLP) TASKS"
SUBTITLE = "A comparative study of Transformer, recurrent and classical models for sentiment classification"
AUTHORS = "Shashikanth Hosur Ramegowda, Shaeed Khan, Rimsha Tariq"
YEAR = "2026"

# ------------------------------------------------------------------ equations
LATEX = {
    "3.1": r"PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d}}\right),\quad PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d}}\right)",
    "3.2": r"Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V",
    "3.3": r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V",
    "3.4": r"\mathrm{softmax}(z)_j=\frac{e^{z_j}}{\sum_{k} e^{z_k}}",
    "3.5": r"\mathrm{head}_i=\mathrm{Attention}(HW_Q^{i},HW_K^{i},HW_V^{i}),\quad \mathrm{MultiHead}(H)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)\,W_O",
    "3.6": r"\mathrm{FFN}(x)=\mathrm{GELU}(xW_1+b_1)\,W_2+b_2",
    "3.7": r"H'=\mathrm{LayerNorm}(H+\mathrm{MultiHead}(H)),\qquad H''=\mathrm{LayerNorm}(H'+\mathrm{FFN}(H'))",
    "3.8": r"\mathrm{LayerNorm}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta",
    "3.9": r"\mathcal{L}_{MLM}=-\sum_{t\in M}\log P(w_t\mid x_{\backslash M})",
    "3.10": r"P(y\mid x)=\mathrm{softmax}(W\,h_{[CLS]}+b)",
    "3.11": r"\mathcal{L}_{CE}=-\frac{1}{N}\sum_{i=1}^{N}\log P(y_i\mid x_i)",
    "3.12": r"m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\ \ v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^{2},\ \ \theta_t=\theta_{t-1}-\eta_t\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}+\lambda\theta_{t-1}\right)",
    "3.13": r"\mathrm{tfidf}(t,d)=\left(1+\log tf(t,d)\right)\cdot\left(\log\frac{1+N}{1+df(t)}+1\right)",
    "3.14": r"\min_{w,b}\ \frac{1}{2}\Vert w\Vert^{2}+C\sum_{i}\max\left(0,\ 1-y_i(w^{T}x_i+b)\right)",
    "3.15": r"f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\quad i_t=\sigma(W_i[h_{t-1},x_t]+b_i),\quad o_t=\sigma(W_o[h_{t-1},x_t]+b_o)",
    "3.16": r"\tilde{c}_t=\tanh(W_c[h_{t-1},x_t]+b_c),\quad c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t,\quad h_t=o_t\odot\tanh(c_t)",
    "3.17": r"\hat{y}=\arg\max_{c\in\{positive,\,negative\}}P(\mathrm{entailment}\mid x,\,h(c))",
    "3.18": r"\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad \mathrm{Precision}=\frac{TP}{TP+FP},\quad \mathrm{Recall}=\frac{TP}{TP+FN}",
    "3.19": r"F_1=\frac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}",
}


def render_eq(num):
    os.makedirs(EQ_DIR, exist_ok=True)
    path = os.path.join(EQ_DIR, f"eq_{num}.png")
    plt.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${LATEX[num]}$", fontsize=12.5)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=False, facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ tables
def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


def load_tables():
    T = {}
    T["team"] = ("Table 0. Distribution of work among the team members",
                 ["Team member", "Main responsibilities"],
                 [["Shashikanth Hosur Ramegowda", "Problem formulation, experiment design, implementation of the Transformer fine-tuning program, Chapters 3.5–3.7 and 4, presentation"],
                  ["Shaeed Khan", "Literature search in Google Scholar and Web of Science, literature review (Chapter 2), classical and BiLSTM baselines, references"],
                  ["Rimsha Tariq", "Introduction and theoretical part on the Transformer architecture (Chapters 1, 3.1–3.4, 3.6), evaluation of ready-made models, error analysis, conclusions"]],
                 [5.0, 12.0])
    T["lit"] = ("Table 2.1. Summary of the reviewed works most relevant to this project",
                ["Authors (year)", "Model / approach", "Task, data", "Main result / relevance"],
                [["Socher et al. (2013)", "Recursive neural tensor network", "SST sentiment, EN", "Created SST; NB ≈ 82%, SVM ≈ 79%, RNTN 85.4% (binary)"],
                 ["Vaswani et al. (2017)", "Transformer (self-attention only)", "Machine translation", "Better BLEU than RNN/CNN at lower training cost"],
                 ["Devlin et al. (2019)", "BERT, masked LM pre-training", "GLUE incl. SST-2", "SST-2 test 93.5% (base) / 94.9% (large)"],
                 ["Liu et al. (2019)", "RoBERTa", "GLUE", "Longer training, more data: SST-2 96.4%"],
                 ["Sanh et al. (2019)", "DistilBERT (distillation)", "GLUE", "40% smaller, 60% faster, ~97% of BERT quality"],
                 ["Turc et al. (2019)", "Compact BERTs (tiny…base)", "GLUE, distillation", "Pre-training compact models matters"],
                 ["Yin et al. (2019)", "Zero-shot via NLI entailment", "Topic, emotion, situation", "Classification without task-labelled data"],
                 ["Brown et al. (2020)", "GPT-3, in-context learning", "Many NLP tasks", "Few-shot learning without fine-tuning"],
                 ["Kapočiūtė-Dzikienė et al. (2019)", "NB, SVM vs LSTM, CNN", "Lithuanian comments, 3 classes", "Traditional NB 0.735 > best deep 0.706 accuracy"],
                 ["Ulčar & Robnik-Šikonja (2021)", "LitLat BERT (XLM-R based)", "LT, LV, EN tasks", "Trilingual model beats massively multilingual"],
                 ["Vileikytė et al. (2024)", "Fine-tuned BERT and T5", "Lithuanian reviews, multi-domain", "High accuracy on clearly polar reviews"],
                 ["Nakvosas et al. (2024)", "Open Llama2 7B/13B for Lithuanian", "LT language modelling", "First open Lithuanian LLMs, 14B+ tokens"],
                 ["Zhang et al. (2024)", "LLMs (ChatGPT etc.) vs fine-tuned", "13 sentiment tasks", "Small fine-tuned models still win on complex tasks"]],
                [3.3, 3.9, 3.6, 6.2])
    T["data"] = ("Table 3.1. Data splits used in the experiment (SST-2)",
                 ["Split", "Source", "Size", "Positive share", "Mean length, words"],
                 data_rows(), [2.0, 5.6, 2.4, 3.0, 4.0])
    T["models"] = ("Table 3.2. Compared models (competitors)",
                   ["Family", "Model", "Pre-trained", "Trained by us", "Parameters"],
                   model_rows(), [3.2, 5.6, 2.4, 2.8, 3.0])
    T["hparams"] = ("Table 3.3. Fine-tuning hyper-parameters",
                    ["Hyper-parameter", "BERT-tiny", "BERT-small", "DistilBERT"],
                    [["Layers / hidden size / heads", "2 / 128 / 2", "4 / 512 / 8", "6 / 768 / 12"],
                     ["Training examples", "8 000", "8 000", "8 000"],
                     ["Epochs (best epoch kept by dev accuracy)", "3", "3", "2"],
                     ["Peak learning rate", "2·10⁻⁴", "1·10⁻⁴", "5·10⁻⁵"],
                     ["Batch size / maximum length", "32 / 64", "32 / 64", "32 / 64"],
                     ["Warm-up / schedule", "10% / linear", "10% / linear", "10% / linear"],
                     ["Weight decay / gradient clipping", "0.01 / 1.0", "0.01 / 1.0", "0.01 / 1.0"]],
                    [6.6, 3.4, 3.4, 3.6])
    T.update(result_tables())
    return T


def data_rows():
    rows = []
    for name, src in (("Train", "SST-2 train minus dev"), ("Dev", "2 000 random from SST-2 train"),
                      ("Test", "SST-2 official validation")):
        df = pd.read_csv(os.path.join(CODE, "data", f"{name.lower()}.csv"))
        rows.append([name, src, fmt_int(len(df)), f"{df['label'].mean() * 100:.1f}%",
                     f"{df['sentence'].str.split().str.len().mean():.1f}"])
    return rows


def params_of(tag, default="–"):
    p = os.path.join(RES, f"{tag}.json")
    if os.path.exists(p):
        import json
        n = json.load(open(p, encoding="utf-8"))["parameters"]
        return f"{n / 1e6:.1f} M" if n >= 1e5 else f"{n / 1e3:.0f} k"
    return default


def model_rows():
    return [
        ["Classical", "TF-IDF + Multinomial Naive Bayes", "no", "yes (65k)", params_of("tfidf_nb") + "*"],
        ["Classical", "TF-IDF + Logistic Regression", "no", "yes (65k)", params_of("tfidf_logreg") + "*"],
        ["Classical", "TF-IDF + Linear SVM", "no", "yes (65k)", params_of("tfidf_svm") + "*"],
        ["Recurrent", "BiLSTM, random embeddings", "no", "yes (65k)", params_of("bilstm")],
        ["Transformer, fine-tuned", "BERT-tiny (Turc et al., 2019)", "yes", "yes (8k)", params_of("ft_bert-tiny_8000")],
        ["Transformer, fine-tuned", "BERT-small (Turc et al., 2019)", "yes", "yes (8k)", params_of("ft_bert-small_8000")],
        ["Transformer, fine-tuned", "DistilBERT-base-uncased", "yes", "yes (8k)", params_of("ft_distilbert_8000")],
        ["Transformer, ready-made", "DistilBERT fine-tuned on SST-2 (public)", "yes", "no", params_of("pt_distilbert_sst2")],
        ["Transformer, ready-made", "Twitter-RoBERTa sentiment (public)", "yes", "no", params_of("pt_twitter_roberta")],
        ["Transformer, zero-shot", "NLI DistilRoBERTa cross-encoder", "yes", "no", params_of("zs_nli_distilroberta")],
    ]


def result_tables():
    T = {}
    p = os.path.join(RES, "summary_table.csv")
    if not os.path.exists(p):
        return T
    s = pd.read_csv(p).sort_values("accuracy", ascending=False)
    rows = []
    for _, r in s.iterrows():
        rows.append([r["name"], fmt_int(r["train_size"]) if r["train_size"] else "0",
                     f"{r['accuracy'] * 100:.1f}", f"{r['precision'] * 100:.1f}", f"{r['recall'] * 100:.1f}",
                     f"{r['f1_macro'] * 100:.1f}",
                     f"{r['train_time_s']:.0f}" if r["train_time_s"] >= 1 else (f"{r['train_time_s']:.1f}" if r["train_time_s"] else "–"),
                     f"{r['latency_ms_per_sample']:.2f}" if r["latency_ms_per_sample"] < 1 else f"{r['latency_ms_per_sample']:.1f}"])
    T["results"] = ("Table 4.1. Results on the SST-2 test set (872 sentences), sorted by accuracy",
                    ["Model", "Train size", "Acc., %", "Prec., %", "Rec., %", "F1, %", "Train time, s", "Latency, ms"],
                    rows, [5.4, 1.7, 1.4, 1.5, 1.4, 1.3, 2.0, 2.0])
    p = os.path.join(RES, "statistics.csv")
    if os.path.exists(p):
        st = pd.read_csv(p).sort_values("accuracy", ascending=False)
        best = st.iloc[0]["name"]
        rows = []
        for _, r in st.iterrows():
            pv = "–" if r["name"] == best else ("< 0.001" if r["mcnemar_p_vs_best"] < 0.001 else f"{r['mcnemar_p_vs_best']:.3f}")
            rows.append([r["name"], f"{r['accuracy'] * 100:.1f}", f"[{r['ci_low'] * 100:.1f}; {r['ci_high'] * 100:.1f}]",
                         "–" if r["name"] == best else str(int(r["b01_best_only"])),
                         "–" if r["name"] == best else str(int(r["b10_model_only"])), pv])
        T["stats"] = (f"Table 4.2. Bootstrap 95% confidence intervals and McNemar test against the best model ({best})",
                      ["Model", "Acc., %", "95% CI", "Only best correct", "Only this correct", "p-value"],
                      rows, [5.6, 1.6, 3.0, 2.3, 2.3, 2.2])
    p = os.path.join(RES, "subset_accuracy.csv")
    if os.path.exists(p):
        sb = pd.read_csv(p).sort_values("all", ascending=False)
        T["subsets"] = ("Table 4.3. Accuracy (%) on linguistically difficult subsets of the test set",
                        ["Model", "All", "Negation", "Contrast", "Δ negation"],
                        [[r["name"], f"{r['all'] * 100:.1f}", f"{r['negation'] * 100:.1f}", f"{r['contrast'] * 100:.1f}",
                          f"{(r['negation'] - r['all']) * 100:+.1f}"] for _, r in sb.iterrows()],
                        [6.6, 2.4, 2.6, 2.6, 2.8])
    p = os.path.join(RES, "learning_curve.csv")
    if os.path.exists(p):
        lc = pd.read_csv(p)
        piv = lc.pivot(index="train_size", columns="series", values="accuracy")
        cols = list(piv.columns)
        rows = []
        for n, r in piv.iterrows():
            rows.append([fmt_int(n)] + [f"{r[c] * 100:.1f}" for c in cols] +
                        ([f"{(r[cols[0]] - r[cols[1]]) * 100:+.1f}"] if len(cols) == 2 else []))
        hdr = ["Training examples"] + [f"{c}, %" for c in cols] + (["Difference, p.p."] if len(cols) == 2 else [])
        T["lc"] = ("Table 4.4. Test accuracy as a function of the number of labelled training examples",
                   hdr, rows, [3.4] + [4.6] * len(cols) + ([3.4] if len(cols) == 2 else []))
    return T


FIGS = {
    "pipeline": ("diagram_pipeline.png", "Fig. 3.1. Structure of the developed program (pipeline of Python modules)", 16.0),
    "encoder": ("diagram_encoder.png", "Fig. 3.2. Transformer encoder with a classification head, as used for sentiment classification", 10.5),
    "accuracy": ("fig_accuracy.png", "Fig. 4.1. Test accuracy of all compared models (colour shows the model family)", 16.0),
    "latency": ("fig_accuracy_latency.png", "Fig. 4.2. Accuracy versus inference latency on CPU (bubble area ∝ number of parameters)", 16.0),
    "training": ("fig_training_curves.png", "Fig. 4.3. Development-set accuracy after each training epoch", 13.0),
    "lc": ("fig_learning_curve.png", "Fig. 4.4. Learning curves: test accuracy versus number of labelled training sentences", 14.0),
    "confusion": ("fig_confusion.png", "Fig. 4.5. Confusion matrices of the best classical model and the best fine-tuned Transformer", 13.0),
    "subsets": ("fig_subsets.png", "Fig. 4.6. Accuracy on all sentences and on sentences with negation and contrast", 16.0),
    "attention": ("attention_heatmap.png", "Fig. 4.7. Self-attention weights of the fine-tuned DistilBERT (last layer, mean over heads)", 11.5),
    "cls": ("cls_attention_layers.png", "Fig. 4.8. Attention of the [CLS] token to every word in each of the 6 layers", 15.0),
}


# ------------------------------------------------------------------ docx helpers
def set_cell_shading(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def add_field(run, instr):
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t")
    txt.text = "Right-click and choose Update Field to build the table of contents." if "TOC" in instr else "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for el in (fld_begin, instr_el, fld_sep, txt, fld_end):
        run._r.append(el)


def make_styles(doc):
    st = doc.styles
    normal = st["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    def pstyle(name, size, bold=False, italic=False, caps=False, before=12, after=6, outline=None,
               align=None, indent=None, keep=True):
        s = st.add_style(name, 1)
        s.base_style = normal
        s.font.name = "Times New Roman"
        s.font.size = Pt(size)
        s.font.bold = bold
        s.font.italic = italic
        s.font.all_caps = caps
        s.font.color.rgb = RGBColor(0, 0, 0)
        pf = s.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.line_spacing = 1.0
        pf.keep_with_next = keep
        if align is not None:
            pf.alignment = align
        if indent is not None:
            pf.first_line_indent = indent
        if outline is not None:
            pPr = s.element.get_or_add_pPr()
            ol = OxmlElement("w:outlineLvl")
            ol.set(qn("w:val"), str(outline))
            pPr.append(ol)
        return s

    pstyle("VG Heading 1", 14, bold=True, caps=True, before=0, after=12, outline=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    pstyle("VG Heading 2", 12, bold=True, before=12, after=6, outline=1)
    pstyle("VG Heading 3", 12, bold=True, italic=True, before=10, after=4, outline=2)
    body = pstyle("VG Body", 12, before=0, after=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=Cm(1.0), keep=False)
    pstyle("VG Caption", 11, before=4, after=10, align=WD_ALIGN_PARAGRAPH.CENTER, keep=False)
    pstyle("VG Table Caption", 11, before=8, after=4, align=WD_ALIGN_PARAGRAPH.LEFT, keep=True)
    pstyle("VG Table Text", 10, before=0, after=0, keep=False)
    pstyle("VG Reference", 11, before=0, after=4, align=WD_ALIGN_PARAGRAPH.JUSTIFY, keep=False)
    return body


def add_bullet_numbering(doc):
    numbering = doc.part.numbering_part.element
    abs_id = 90
    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abs_id))
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    for tag, val in (("w:start", "1"), ("w:numFmt", "bullet"), ("w:lvlText", "–"), ("w:lvlJc", "left")):
        el = OxmlElement(tag)
        el.set(qn("w:val"), val)
        lvl.append(el)
    pPr = OxmlElement("w:pPr")
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "851")
    ind.set(qn("w:hanging"), "284")
    pPr.append(ind)
    lvl.append(pPr)
    abstract.append(lvl)
    first_num = numbering.find(qn("w:num"))
    if first_num is not None:
        first_num.addprevious(abstract)
    else:
        numbering.append(abstract)
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), "90")
    a = OxmlElement("w:abstractNumId")
    a.set(qn("w:val"), str(abs_id))
    num.append(a)
    numbering.append(num)
    return 90


def bullet(doc, text, num_id):
    p = doc.add_paragraph(style="VG Body")
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_after = Pt(3)
    pPr = p._p.get_or_add_pPr()
    numPr = OxmlElement("w:numPr")
    il = OxmlElement("w:ilvl")
    il.set(qn("w:val"), "0")
    ni = OxmlElement("w:numId")
    ni.set(qn("w:val"), str(num_id))
    numPr.append(il)
    numPr.append(ni)
    pPr.append(numPr)
    add_inline(p, text)
    return p


def add_inline(p, text):
    """Supports **bold** spans."""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            p.add_run(part[2:-2]).bold = True
        elif part:
            p.add_run(part)


def add_table(doc, spec):
    caption, header, rows, widths = spec
    doc.add_paragraph(caption, style="VG Table Caption")
    table = doc.add_table(rows=1, cols=len(header))
    table.style = doc.styles["Table Grid"] if "Table Grid" in [s.name for s in doc.styles] else None
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    total = sum(widths)
    scale = 17.0 / total if total > 17.0 else 1.0
    for i, h in enumerate(header):
        c = table.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        c.paragraphs[0].style = doc.styles["VG Table Text"]
        set_cell_shading(c, "D9E2F3")
    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v))
            cells[i].paragraphs[0].style = doc.styles["VG Table Text"]
    for row in table.rows:
        for i, c in enumerate(row.cells):
            c.width = Cm(widths[i] * scale)
    # header row repeats on page break
    trPr = table.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    set_table_borders(table)
    doc.add_paragraph(style="VG Body").paragraph_format.space_after = Pt(2)


def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "808080")
        borders.append(el)
    tblPr.append(borders)
    mar = OxmlElement("w:tblCellMar")
    for edge, v in (("left", "70"), ("right", "70"), ("top", "20"), ("bottom", "20")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), v)
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def add_figure(doc, key):
    fname, caption, width = FIGS[key]
    path = os.path.join(FIG, fname)
    if not os.path.exists(path):
        print("  ! missing figure", fname)
        return
    p = doc.add_paragraph(style="VG Caption")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(path, width=Cm(width))
    doc.add_paragraph(caption, style="VG Caption")


def add_equation(doc, text, num):
    p = doc.add_paragraph(style="VG Body")
    pf = p.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(4)
    pf.space_after = Pt(6)
    pf.tab_stops.add_tab_stop(Cm(8.5), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(17.0), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    if num in LATEX:
        path = render_eq(num)
        from PIL import Image
        w_px, _ = Image.open(path).size
        width_cm = min(w_px / 300 * 2.54, 13.8)
        p.add_run().add_picture(path, width=Cm(width_cm))
    else:
        p.add_run(text).italic = True
    p.add_run(f"\t({num})")


# ------------------------------------------------------------------ cover
def fill_cover(doc):
    body = doc.element.body
    # remove the footnote references (they carry template formatting instructions)
    # remove the "EXAMPLE" WordArt watermark of the template
    MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent"
    for ac in list(body.iter(MC)):
        if "EXAMPLE" in "".join(ac.itertext()):
            run = ac.getparent()
            run.getparent().remove(run) if run.tag == qn("w:r") else run.remove(ac)
    for r in list(body.iter(qn("w:r"))):
        if r.find(qn("w:footnoteReference")) is not None:
            r.getparent().remove(r)
    paras = doc.paragraphs

    def set_text(p, text):
        runs = p.runs
        runs[0].text = text
        for r in runs[1:]:
            r.text = ""

    for p in paras:
        t = p.text
        if t.startswith("Student"):
            set_text(p, AUTHORS)
        elif t.startswith("TITLE OF"):
            set_text(p, TITLE)
        elif t.startswith("Maste"):
            set_text(p, "Group Project Paper")
        elif t.startswith("Vilnius, 202"):
            set_text(p, f"Vilnius, {YEAR}")
        elif t.startswith("DEPARTMENT OF"):
            set_text(p, "DEPARTMENT OF ________________________________________")
        elif "specialisation" in t or "study field" in t:
            set_text(p, re.sub(r"\d$", "", t))
    # subtitle under the title
    for i, p in enumerate(paras):
        if p.text == TITLE:
            nxt = paras[i + 1]
            run = nxt.add_run(SUBTITLE)
            run.font.size = Pt(12)
            run.italic = True
            nxt.alignment = WD_ALIGN_PARAGRAPH.CENTER
            break


def add_page_number_footer(section):
    section.footer.is_linked_to_previous = False
    fp = section.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    run.font.size = Pt(11)
    add_field(run, "PAGE")


# ------------------------------------------------------------------ main
def build():
    doc = docx.Document(TEMPLATE)
    fill_cover(doc)
    make_styles(doc)
    num_id = add_bullet_numbering(doc)
    tables = load_tables()

    # Body section starts on a new page with its own footer (cover has none)
    cover_section = doc.sections[0]
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.left_margin, sec.right_margin = Cm(2.0), Cm(2.0)
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
    add_page_number_footer(sec)
    cover_section.footer.is_linked_to_previous = False

    # Table of contents
    toc_title = doc.add_paragraph("CONTENTS", style="VG Caption")
    toc_title.runs[0].bold = True
    toc_title.runs[0].font.size = Pt(14)
    p = doc.add_paragraph(style="VG Body")
    p.paragraph_format.first_line_indent = Cm(0)
    add_field(p.add_run(), r'TOC \h \z \t "VG Heading 1,1,VG Heading 2,2,VG Heading 3,3"')

    files = sorted(f for f in os.listdir(HERE) if re.match(r"ch\d+_.*\.md$", f))
    for f in files:
        text = open(os.path.join(HERE, f), encoding="utf-8").read()
        blocks = re.split(r"\n\s*\n", text.strip())
        for b in blocks:
            b = b.strip()
            if b.startswith("#* ") or re.match(r"^# ", b):
                title = b.split(" ", 1)[1].strip()
                p = doc.add_paragraph(style="VG Heading 1")
                p.paragraph_format.page_break_before = True
                p.add_run(title)
            elif b.startswith("### "):
                doc.add_paragraph(b[4:].strip(), style="VG Heading 3")
            elif b.startswith("## "):
                doc.add_paragraph(b[3:].strip(), style="VG Heading 2")
            elif b.startswith("[[TABLE"):
                key = b[7:-2].strip()
                if key in tables:
                    add_table(doc, tables[key])
                else:
                    print("  ! missing table", key)
            elif b.startswith("[[FIG"):
                add_figure(doc, b[5:-2].strip())
            elif b.startswith("$$"):
                body_, num = b[2:].split("||")
                add_equation(doc, body_.strip(), num.strip().strip("()"))
            elif b.startswith("- "):
                for line in b.split("\n"):
                    bullet(doc, line[2:].strip(), num_id)
            elif b.startswith("REF "):
                for line in b.split("\n"):
                    rp = doc.add_paragraph(style="VG Reference")
                    rp.paragraph_format.left_indent = Cm(0.75)
                    rp.paragraph_format.first_line_indent = Cm(-0.75)
                    add_inline(rp, line[4:].strip())
            else:
                p = doc.add_paragraph(style="VG Body")
                add_inline(p, " ".join(b.split("\n")))

    # metadata
    cp = doc.core_properties
    cp.title = "Transformers and their application in NLP tasks"
    cp.author = AUTHORS
    cp.subject = "Group project paper, VILNIUS TECH"
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()
