"""Builds the project paper (.docx) on the official VILNIUS TECH Faculty of Fundamental Sciences
template (paper/template/MFT_template_2025.docx).

Chapter sources are the markdown-like files paper/ch*.md:
  # Title        numbered chapter (BD_Skyrius_1_lygio, new page)
  #* Title       unnumbered chapter (BD_Skyrius_nenumeruojamas, new page)
  #A Title       appendix heading
  ## / ###      2nd / 3rd level sections (numbered automatically by the template)
  - item        bullet list            1. item   numbered list
  $$ latex || key            numbered equation (converted to a native Word equation)
  [[TABLE key]] / [[FIG key]]  tables and figures (numbered automatically)
  {tab:key} {fig:key} {eq:key} cross-references, **bold** inline
  REF text      reference list entry
The annotation (abstract) comes from paper/annotation.md.
After building, run word_finalize.ps1 to update the Contents / List of Figures / List of Tables.
"""
import copy
import json
import os
import re

import docx
import latex2mathml.converter
import pandas as pd
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = os.path.join(ROOT, "code")
FIG = os.path.join(CODE, "figures")
RES = os.path.join(CODE, "results")
TEMPLATE = os.path.join(HERE, "template", "MFT_template_2025.docx")
OUT = os.environ.get("PAPER_OUT", os.path.join(ROOT, "docs", "Transformers_in_NLP_Project_Paper.docx"))
NUMERIC = re.compile(r"^[\d\s.,%+\-–<>\[\];/·⁰-₟×kM*()]+$|^(yes|no|yes \(\d+k\))$")
MML2OMML = r"C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL"

AUTHORS = "Shashikanth Hosur Ramegowda, Shaeed Khan, Rimsha Tariq"
TITLE_EN = "TRANSFORMERS AND THEIR APPLICATION IN NATURAL LANGUAGE PROCESSING (NLP) TASKS"
WORK_TYPE = "Group Project Paper"
PROGRAMME = "Engineering of Artificial Intelligence study program, state code 6211BX023"
FIELD = "Informatics"

ACRONYMS = [
    ("AdamW", "Adam optimiser with decoupled weight decay"),
    ("BERT", "Bidirectional Encoder Representations from Transformers"),
    ("BiLSTM", "Bidirectional Long Short-Term Memory network"),
    ("CI", "Confidence interval"),
    ("CLS", "Special classification token of BERT-type models"),
    ("CPU / GPU", "Central / Graphics processing unit"),
    ("GPT", "Generative Pre-trained Transformer"),
    ("LLM", "Large language model"),
    ("MLM", "Masked language modelling"),
    ("NB", "Naive Bayes"),
    ("NLI", "Natural language inference"),
    ("NLP", "Natural language processing"),
    ("RNN", "Recurrent neural network"),
    ("SST-2", "Stanford Sentiment Treebank, binary version"),
    ("SVM", "Support vector machine"),
    ("TF-IDF", "Term frequency – inverse document frequency"),
]

# ---------------------------------------------------------------- equations (LaTeX)
LATEX = {
    "pe": r"PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d}}\right),\quad PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d}}\right)",
    "qkv": r"Q=HW_{Q},\quad K=HW_{K},\quad V=HW_{V}",
    "attn": r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V",
    "softmax": r"\mathrm{softmax}(z)_{j}=\frac{e^{z_{j}}}{\sum_{k}{e^{z_{k}}}}",
    "mha": r"\mathrm{head}_{i}=\mathrm{Attention}(HW_{Q}^{i},HW_{K}^{i},HW_{V}^{i}),\quad \mathrm{MultiHead}(H)=\mathrm{Concat}(\mathrm{head}_{1},\ldots,\mathrm{head}_{h})W_{O}",
    "ffn": r"\mathrm{FFN}(x)=\mathrm{GELU}(xW_{1}+b_{1})W_{2}+b_{2}",
    "res": r"H'=\mathrm{LayerNorm}(H+\mathrm{MultiHead}(H)),\quad H''=\mathrm{LayerNorm}(H'+\mathrm{FFN}(H'))",
    "ln": r"\mathrm{LayerNorm}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta",
    "mlm": r"L_{MLM}=-\sum_{t\in M}{\log P(w_{t}\mid x_{\setminus M})}",
    "cls": r"P(y\mid x)=\mathrm{softmax}(Wh_{CLS}+b)",
    "ce": r"L_{CE}=-\frac{1}{N}\sum_{i=1}^{N}{\log P(y_{i}\mid x_{i})}",
    "adam1": r"m_{t}=\beta_{1}m_{t-1}+(1-\beta_{1})g_{t},\quad v_{t}=\beta_{2}v_{t-1}+(1-\beta_{2})g_{t}^{2}",
    "adam2": r"\theta_{t}=\theta_{t-1}-\eta_{t}\left(\frac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}}+\epsilon}+\lambda\theta_{t-1}\right)",
    "tfidf": r"\mathrm{tfidf}(t,d)=\left(1+\log\left(\mathrm{tf}(t,d)\right)\right)\cdot\left(\log\frac{1+N}{1+\mathrm{df}(t)}+1\right)",
    "svm": r"\mathrm{min}_{w,b}\ \frac{1}{2}\|w\|^{2}+C\sum_{i}{\mathrm{max}\left(0,1-y_{i}(w^{T}x_{i}+b)\right)}",
    "lstm1": r"f_{t}=\sigma(W_{f}[h_{t-1},x_{t}]+b_{f}),\quad i_{t}=\sigma(W_{i}[h_{t-1},x_{t}]+b_{i}),\quad o_{t}=\sigma(W_{o}[h_{t-1},x_{t}]+b_{o})",
    "lstm2": r"\tilde{c}_{t}=\tanh(W_{c}[h_{t-1},x_{t}]+b_{c}),\quad c_{t}=f_{t}\odot c_{t-1}+i_{t}\odot\tilde{c}_{t},\quad h_{t}=o_{t}\odot\tanh(c_{t})",
    "nli": r"\hat{y}=\underset{c\in\{pos,neg\}}{\mathrm{arg\,max}}P(\mathrm{entailment}\mid x,h(c))",
    "metrics": r"\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\quad \mathrm{Precision}=\frac{TP}{TP+FP},\quad \mathrm{Recall}=\frac{TP}{TP+FN}",
    "f1": r"F_{1}=\frac{2\cdot \mathrm{Precision}\cdot \mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}",
}
_XSLT = None


def omml(latex):
    global _XSLT
    if _XSLT is None:
        _XSLT = etree.XSLT(etree.parse(MML2OMML))
    mml = latex2mathml.converter.convert(latex)
    return _XSLT(etree.fromstring(mml.encode("utf-8"))).getroot()


# ---------------------------------------------------------------- tables
def fmt_int(n):
    return f"{int(n):,}".replace(",", " ")


def params_of(tag):
    p = os.path.join(RES, f"{tag}.json")
    if not os.path.exists(p):
        return "–"
    n = json.load(open(p, encoding="utf-8"))["parameters"]
    return f"{n / 1e6:.1f} M" if n >= 1e5 else f"{n / 1e3:.0f} k"


def load_tables():
    T = {}
    T["team"] = ("Distribution of work among the team members",
                 ["Team member", "Main responsibilities"],
                 [["Shashikanth Hosur Ramegowda", "Problem formulation and experiment design; implementation of the Transformer fine-tuning program; practical part (Chapter 3); presentation"],
                  ["Shaeed Khan", "Literature search in Google Scholar and Web of Science; literature review (Chapter 1); classical and BiLSTM baselines; references"],
                  ["Rimsha Tariq", "Introduction; theory of the Transformer architecture and mathematical methods (Chapter 2); evaluation of ready-made models; error analysis; conclusions"]],
                 [5.0, 12.0])
    T["lit"] = ("Summary of the reviewed works most relevant to this project",
                ["Authors (year)", "Model / approach", "Task, data", "Main result / relevance"],
                [["Socher et al. (2013)", "Recursive neural tensor network", "SST sentiment, EN", "Created SST; NB ≈ 82%, SVM ≈ 79%, RNTN 85.4% (binary)"],
                 ["Vaswani et al. (2017)", "Transformer (self-attention only)", "Machine translation", "Better quality than RNN/CNN at lower training cost"],
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
                [3.4, 4.0, 3.6, 6.0])
    rows = []
    for name, src in (("Train", "SST-2 train minus dev"), ("Dev", "2 000 random from SST-2 train"),
                      ("Test", "SST-2 official validation")):
        df = pd.read_csv(os.path.join(CODE, "data", f"{name.lower()}.csv"))
        rows.append([name, src, fmt_int(len(df)), f"{df['label'].mean() * 100:.1f}%",
                     f"{df['sentence'].str.split().str.len().mean():.1f}"])
    T["data"] = ("Data splits used in the experiment (SST-2)",
                 ["Split", "Source", "Size", "Positive share", "Mean length, words"], rows, [2.0, 5.6, 2.4, 3.0, 4.0])
    T["models"] = ("Compared models (competitors)",
                   ["Family", "Model", "Pre-trained", "Trained by us", "Parameters"],
                   [["Classical", "TF-IDF + Multinomial Naive Bayes", "no", "yes (65k)", params_of("tfidf_nb") + "*"],
                    ["Classical", "TF-IDF + Logistic Regression", "no", "yes (65k)", params_of("tfidf_logreg") + "*"],
                    ["Classical", "TF-IDF + Linear SVM", "no", "yes (65k)", params_of("tfidf_svm") + "*"],
                    ["Recurrent", "BiLSTM, random embeddings", "no", "yes (65k)", params_of("bilstm")],
                    ["Transformer, fine-tuned", "BERT-tiny (Turc et al., 2019)", "yes", "yes (8k)", params_of("ft_bert-tiny_8000")],
                    ["Transformer, fine-tuned", "BERT-small (Turc et al., 2019)", "yes", "yes (8k)", params_of("ft_bert-small_8000")],
                    ["Transformer, fine-tuned", "DistilBERT-base-uncased", "yes", "yes (8k)", params_of("ft_distilbert_8000")],
                    ["Transformer, ready-made", "DistilBERT fine-tuned on SST-2 (public)", "yes", "no", params_of("pt_distilbert_sst2")],
                    ["Transformer, ready-made", "Twitter-RoBERTa sentiment (public)", "yes", "no", params_of("pt_twitter_roberta")],
                    ["Transformer, zero-shot", "NLI DistilRoBERTa cross-encoder", "yes", "no", params_of("zs_nli_distilroberta")]],
                   [3.6, 5.8, 2.2, 2.6, 2.8])
    T["hparams"] = ("Fine-tuning hyper-parameters",
                    ["Hyper-parameter", "BERT-tiny", "BERT-small", "DistilBERT"],
                    [["Layers / hidden size / heads", "2 / 128 / 2", "4 / 512 / 8", "6 / 768 / 12"],
                     ["Training examples", "8 000", "8 000", "8 000"],
                     ["Epochs (best epoch kept by dev accuracy)", "3", "3", "2"],
                     ["Peak learning rate", "2·10⁻⁴", "1·10⁻⁴", "5·10⁻⁵"],
                     ["Batch size / maximum length", "32 / 64", "32 / 64", "32 / 64"],
                     ["Warm-up / schedule", "10% / linear", "10% / linear", "10% / linear"],
                     ["Weight decay / gradient clipping", "0.01 / 1.0", "0.01 / 1.0", "0.01 / 1.0"]],
                    [6.8, 3.4, 3.4, 3.4])
    T.update(result_tables())
    return T


def result_tables():
    T = {}
    p = os.path.join(RES, "summary_table.csv")
    if not os.path.exists(p):
        return T
    s = pd.read_csv(p).sort_values("accuracy", ascending=False)
    rows = []
    for _, r in s.iterrows():
        tt = r["train_time_s"]
        rows.append([r["name"], fmt_int(r["train_size"]) if r["train_size"] else "0",
                     f"{r['accuracy'] * 100:.1f}", f"{r['precision'] * 100:.1f}", f"{r['recall'] * 100:.1f}",
                     f"{r['f1_macro'] * 100:.1f}", "–" if not tt else (f"{tt:.0f}" if tt >= 1 else f"{tt:.1f}"),
                     f"{r['latency_ms_per_sample']:.2f}" if r["latency_ms_per_sample"] < 1 else f"{r['latency_ms_per_sample']:.1f}"])
    T["results"] = ("Results on the SST-2 test set (872 sentences), sorted by accuracy",
                    ["Model", "Train size", "Acc., %", "Prec., %", "Rec., %", "F1, %", "Train time, s", "Latency, ms"],
                    rows, [5.4, 1.7, 1.4, 1.5, 1.4, 1.3, 2.0, 2.0])
    p = os.path.join(RES, "statistics.csv")
    if os.path.exists(p):
        st = pd.read_csv(p).sort_values("accuracy", ascending=False)
        best = st.iloc[0]["name"]
        rows = []
        for _, r in st.iterrows():
            isb = r["name"] == best
            pv = "–" if isb else ("< 0.001" if r["mcnemar_p_vs_best"] < 0.001 else f"{r['mcnemar_p_vs_best']:.3f}")
            rows.append([r["name"], f"{r['accuracy'] * 100:.1f}", f"[{r['ci_low'] * 100:.1f}; {r['ci_high'] * 100:.1f}]",
                         "–" if isb else str(int(r["b01_best_only"])), "–" if isb else str(int(r["b10_model_only"])), pv])
        T["stats"] = (f"Bootstrap 95% confidence intervals and McNemar test against the best model",
                      ["Model", "Acc., %", "95% CI", "Only best correct", "Only this correct", "p-value"],
                      rows, [5.8, 1.6, 3.0, 2.3, 2.3, 2.0])
    p = os.path.join(RES, "subset_accuracy.csv")
    if os.path.exists(p):
        sb = pd.read_csv(p).sort_values("all", ascending=False)
        T["subsets"] = ("Accuracy (%) on linguistically difficult subsets of the test set",
                        ["Model", "All", "Negation", "Contrast", "Δ negation, p.p."],
                        [[r["name"], f"{r['all'] * 100:.1f}", f"{r['negation'] * 100:.1f}", f"{r['contrast'] * 100:.1f}",
                          f"{(r['negation'] - r['all']) * 100:+.1f}"] for _, r in sb.iterrows()],
                        [6.8, 2.2, 2.6, 2.6, 2.8])
    p = os.path.join(RES, "learning_curve.csv")
    if os.path.exists(p):
        lc = pd.read_csv(p)
        piv = lc.pivot(index="train_size", columns="series", values="accuracy")
        cols = sorted(piv.columns, key=lambda c: c.startswith("TF"))
        rows = [[fmt_int(n)] + [f"{r[c] * 100:.1f}" for c in cols] + [f"{(r[cols[0]] - r[cols[1]]) * 100:+.1f}"]
                for n, r in piv.iterrows()]
        T["lc"] = ("Test accuracy as a function of the number of labelled training examples",
                   ["Training examples"] + [f"{c}, %" for c in cols] + ["Difference, p.p."], rows, [3.4, 4.8, 4.8, 3.4])
    return T


FIGS = {
    "pipeline": ("diagram_pipeline.png", "Structure of the developed program (pipeline of Python modules)", 15.5),
    "encoder": ("diagram_encoder.png", "Transformer encoder with a classification head, as used for sentiment classification", 9.5),
    "accuracy": ("fig_accuracy.png", "Test accuracy of all compared models (colour shows the model family)", 15.5),
    "latency": ("fig_accuracy_latency.png", "Accuracy versus inference latency on CPU (bubble area proportional to the number of parameters)", 15.5),
    "training": ("fig_training_curves.png", "Development-set accuracy after each training epoch", 12.5),
    "lc": ("fig_learning_curve.png", "Learning curves: test accuracy versus number of labelled training sentences", 13.5),
    "confusion": ("fig_confusion.png", "Confusion matrices of the best classical model and the best fine-tuned Transformer", 12.5),
    "subsets": ("fig_subsets.png", "Accuracy on all sentences and on sentences with negation and contrast", 13.0),
    "attention": ("attention_heatmap.png", "Self-attention weights of DistilBERT for an example sentence (last layer, mean over heads)", 10.0),
    "cls": ("cls_attention_layers.png", "Attention of the [CLS] token to every word in each of the 6 layers", 15.0),
}


# ---------------------------------------------------------------- low-level helpers
def el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


def set_num(p, num_id, ilvl=0):
    pPr = p._p.get_or_add_pPr()
    old = pPr.find(qn("w:numPr"))
    if old is not None:
        pPr.remove(old)
    numPr = el("w:numPr")
    numPr.append(el("w:ilvl", **{"w:val": ilvl}))
    numPr.append(el("w:numId", **{"w:val": num_id}))
    pPr.insert(1 if pPr.find(qn("w:pStyle")) is not None else 0, numPr)


def smart_quotes(text):
    text = re.sub(r'(^|[\s(\[{–—-])"', lambda m: m.group(1) + "“", text)
    text = text.replace('"', "”")
    text = re.sub(r"(^|[\s(\[{])'", lambda m: m.group(1) + "‘", text)
    return text.replace("'", "’")


def add_runs(p, text, size=None, italic=None):
    """Adds text with **bold** spans (typographic quotes)."""
    text = smart_quotes(text)
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        r = p.add_run(part[2:-2] if bold else part)
        r.bold = True if bold else None
        if size:
            r.font.size = Pt(size)
        if italic is not None:
            r.italic = italic


def add_field(run, instr, placeholder=""):
    for tag, extra in (("w:fldChar", {"w:fldCharType": "begin"}), ("w:instrText", None),
                       ("w:fldChar", {"w:fldCharType": "separate"}), ("w:t", None), ("w:fldChar", {"w:fldCharType": "end"})):
        e = el(tag, **(extra or {}))
        if tag == "w:instrText":
            e.set(qn("xml:space"), "preserve")
            e.text = instr
        elif tag == "w:t":
            e.text = placeholder
        run._r.append(e)


class Numbering:
    """Creates bullet / decimal list definitions in the template's numbering part."""

    def __init__(self, doc):
        self.root = doc.part.numbering_part.element
        self.next_abs = 900
        self.next_num = 900
        self.bullet_abs = self._abstract("bullet", "–")
        self.decimal_abs = self._abstract("decimal", "%1.")

    def _abstract(self, fmt, text):
        a = el("w:abstractNum", **{"w:abstractNumId": self.next_abs})
        a.append(el("w:multiLevelType", **{"w:val": "singleLevel"}))
        lvl = el("w:lvl", **{"w:ilvl": 0})
        for tag, val in (("w:start", "1"), ("w:numFmt", fmt), ("w:lvlText", text), ("w:lvlJc", "left")):
            lvl.append(el(tag, **{"w:val": val}))
        pPr = el("w:pPr")
        pPr.append(el("w:ind", **{"w:left": 1077, "w:hanging": 357}))
        lvl.append(pPr)
        a.append(lvl)
        first_num = self.root.find(qn("w:num"))
        (first_num.addprevious(a) if first_num is not None else self.root.append(a))
        self.next_abs += 1
        return self.next_abs - 1

    def new(self, kind):
        """Returns a fresh numId (decimal lists restart at 1 for every new list)."""
        n = el("w:num", **{"w:numId": self.next_num})
        n.append(el("w:abstractNumId", **{"w:val": self.bullet_abs if kind == "bullet" else self.decimal_abs}))
        if kind == "decimal":
            ov = el("w:lvlOverride", **{"w:ilvl": 0})
            ov.append(el("w:startOverride", **{"w:val": 1}))
            n.append(ov)
        self.root.append(n)
        self.next_num += 1
        return self.next_num - 1


# ---------------------------------------------------------------- front matter
def fill_sdts(doc):
    body = doc.element.body
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for sdt in list(body.iter(W + "sdt")):
        txt = "".join(t.text or "" for t in sdt.iter(W + "t")).strip()
        pr = sdt.find(W + "sdtPr")
        value = None
        if txt.startswith("Student name"):
            value = AUTHORS
        elif txt.upper().startswith("TITLE OF THE FINAL THESIS IN LITHUANIAN"):
            # English-only document: remove the Lithuanian title line completely
            holder = sdt.getparent()
            if holder.tag == W + "p":
                holder.getparent().remove(holder)
            else:
                holder.remove(sdt)
            continue
        elif txt.upper().startswith("TITLE OF THE FINAL THESIS IN ENGLISH"):
            value = TITLE_EN
        elif txt.startswith("Select study program"):
            value = PROGRAMME
        elif txt.startswith("Select study field"):
            value = FIELD
        if value is None:
            continue
        ph = pr.find(W + "showingPlcHdr")
        if ph is not None:
            pr.remove(ph)
        content = sdt.find(W + "sdtContent")
        ts = list(content.iter(W + "t"))
        ts[0].text = value
        for t in ts[1:]:
            t.text = ""
        for r in content.iter(W + "r"):
            rpr = r.find(W + "rPr")
            if rpr is not None:
                for child in list(rpr):
                    if child.tag in (W + "rStyle", W + "color", W + "i", W + "iCs"):
                        rpr.remove(child)


W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def remove_second_cover(doc):
    """The first page already carries university, faculty, authors and title; the template's second
    (supervisor) page is not needed for a course project. Everything between the first page break and
    the section break paragraph is removed; the section break itself stays."""
    body = doc.element.body
    kids = list(body)
    i_break = next(i for i, e in enumerate(kids) if e.tag == W_NS + "p" and
                   any(b.get(W_NS + "type") == "page" for b in e.iter(W_NS + "br")))
    i_sect = next(i for i, e in enumerate(kids) if e.tag == W_NS + "p" and e.find(W_NS + "pPr/" + W_NS + "sectPr") is not None)
    for e in kids[i_break:i_sect]:
        body.remove(e)
    # drop bookmarks whose partner was removed
    starts = {b.get(W_NS + "id"): b for b in body.iter(W_NS + "bookmarkStart")}
    ends = {b.get(W_NS + "id"): b for b in body.iter(W_NS + "bookmarkEnd")}
    for i, b in starts.items():
        if i not in ends:
            b.getparent().remove(b)
    for i, b in ends.items():
        if i not in starts:
            b.getparent().remove(b)


def unwrap_content_controls(doc):
    """Replace the template's form fields (content controls) by plain text."""
    for sdt in reversed(list(doc.element.body.iter(W_NS + "sdt"))):
        content = sdt.find(W_NS + "sdtContent")
        parent = sdt.getparent()
        idx = parent.index(sdt)
        for child in list(content):
            parent.insert(idx, child)
            idx += 1
        parent.remove(sdt)


def remove_comments(doc):
    """The template carries review comments of its author; they must not appear in the final document."""
    body = doc.element.body
    for tag in ("commentRangeStart", "commentRangeEnd"):
        for e in list(body.iter(W_NS + tag)):
            e.getparent().remove(e)
    for e in list(body.iter(W_NS + "commentReference")):
        run = e.getparent()
        run.getparent().remove(run)
    for rid, rel in list(doc.part.rels.items()):
        t = rel.reltype.lower()
        if "comment" in t or t.endswith("/people"):
            doc.part.rels.pop(rid)


def front_matter(doc, annotation_paras):
    body = doc.element.body
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    children = list(body)

    def text_of(e):
        return "".join(t.text or "" for t in e.iter(W + "t"))

    # 1) title page: "Master's Final Thesis" -> work type; remove Consultant rows
    for p in doc.paragraphs:
        if p.text.strip().startswith("Master") and "Thesis" in p.text:
            p.runs[0].text = WORK_TYPE
            for r in p.runs[1:]:
                r.text = ""
    # course project: no supervisor / consultant block on the title page
    for t in doc.tables:
        if t.cell(0, 0).text.strip() == "Supervisor":
            t._tbl.getparent().remove(t._tbl)
    # 2) remove the 'objectives' page and the Lithuanian annotation page
    i_obj = next(i for i, e in enumerate(children) if e.tag == W + "sdt" and "objectives" in text_of(e))
    i_en = next(i for i, e in enumerate(children) if e.tag == W + "sdt" and "annotation in English" in text_of(e))
    start = i_obj - 5  # blank + university/faculty/department header lines of that page
    for e in children[start:i_en - 1]:
        body.remove(e)
    # 3) English annotation -> our annotation
    sdt = next(e for e in body if e.tag == W + "sdt" and "annotation in English" in text_of(e))
    anchor = sdt
    head = doc.add_paragraph()
    r = head.add_run("Annotation")
    r.bold = True
    r.font.size = Pt(14)
    anchor.addnext(head._p)
    anchor = head._p
    for text in annotation_paras:
        p = doc.add_paragraph(style="BD_pagrindinis_tekstas")
        add_runs(p, text)
        anchor.addnext(p._p)
        anchor = p._p
    body.remove(sdt)
    # 4) lists of figures/tables and acronyms on one page: drop the page break before "Acronyms"
    kids = list(body)
    i_acr = next(i for i, e in enumerate(kids) if text_of(e).strip() == "Acronyms and Terms")
    for e in kids[max(0, i_acr - 3):i_acr]:
        for br in e.iter(W + "br"):
            if br.get(W + "type") == "page":
                br.getparent().remove(br)
    # 5) acronyms table
    acr = next(t for t in doc.tables if t.cell(0, 0).text.strip() == "DB")
    proto = copy.deepcopy(acr.rows[0]._tr)
    for row in list(acr.rows):
        row._tr.getparent().remove(row._tr)
    for a, b in ACRONYMS:
        tr = copy.deepcopy(proto)
        acr._tbl.append(tr)
        row = acr.rows[-1]
        for cell, val in zip(row.cells, (a, b)):
            ts = list(cell._tc.iter(W + "t"))
            ts[0].text = val
            for t in ts[1:]:
                t.text = ""


def tidy_annotation_page(doc):
    """Remove empty filler paragraphs between the title page and the end of the annotation page."""
    body = doc.element.body
    kids = list(body)
    sect_idx = [i for i, e in enumerate(kids) if e.tag == W_NS + "p" and e.find(W_NS + "pPr/" + W_NS + "sectPr") is not None]
    for e in kids[sect_idx[0] + 1:sect_idx[1]]:
        has_content = e.find(".//" + W_NS + "t") is not None
        special = e.find(".//" + W_NS + "br") is not None or e.find(".//" + W_NS + "drawing") is not None             or e.find(W_NS + "pPr/" + W_NS + "sectPr") is not None
        if e.tag == W_NS + "p" and not has_content and not special:
            body.remove(e)


def strip_template_body(doc):
    """Deletes the sample chapters (from 'Introduction' to the end), keeps the final sectPr.
    Returns the numPr numId used by the template's reference list."""
    body = doc.element.body
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    kids = list(body)
    i_intro = next(i for i, e in enumerate(kids) if e.tag == W + "p" and
                   "".join(t.text or "" for t in e.iter(W + "t")).strip() == "Introduction" and
                   e.find(".//" + W + "pStyle") is not None)
    ref_num = None
    for e in kids[i_intro:]:
        ps = e.find(".//" + W + "pStyle")
        if ref_num is None and ps is not None and ps.get(W + "val") == "ListParagraph":
            ni = e.find(".//" + W + "numId")
            if ni is not None:
                ref_num = ni.get(W + "val")
    for e in kids[i_intro:]:
        if e.tag != W + "sectPr":
            body.remove(e)
    return ref_num


def fix_toc_fields(doc):
    """The template's TOC codes use ';' as list separator, which fails on systems with ',' as separator.
    Use separator-free codes instead: outline levels for Contents, SEQ captions for the lists."""
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for it in doc.element.body.iter(W + "instrText"):
        t = it.text or ""
        if "TOC" not in t:
            continue
        if "BD_paveiklsiuko_pavadinimas" in t:
            it.text = r' TOC \h \z \c "Figure" '
        elif "BD_lenteles_pavadinimas" in t:
            it.text = r' TOC \h \z \c "Table" '
        else:
            it.text = r' TOC \o "1-2" \h \z \u '


def single_spacing(doc):
    """The assignment requires single line spacing (the template default is 1.5)."""
    for name in ("Normal", "BD_paveiklsiuko_pavadinimas", "List Paragraph"):
        pf = doc.styles[name].paragraph_format
        pf.line_spacing = 1.0
    doc.styles["Normal"].paragraph_format.space_after = Pt(6)
    doc.styles["BD_paveiklsiuko_pavadinimas"].paragraph_format.space_after = Pt(10)


# ---------------------------------------------------------------- body builders
class Builder:
    def __init__(self, doc, tables, ref_num):
        self.doc = doc
        self.tables = tables
        self.ref_num = ref_num
        self.num = Numbering(doc)
        self.fig_no = self.tab_no = self.eq_no = 0
        self.labels = {}

    # numbering pre-pass so that cross-references can point forward
    def scan(self, blocks):
        f = t = e = 0
        for b in blocks:
            if b.startswith("[[FIG"):
                f += 1
                self.labels["fig:" + b[5:-2].strip()] = f
            elif b.startswith("[[TABLE"):
                t += 1
                self.labels["tab:" + b[7:-2].strip()] = t
            elif b.startswith("$$"):
                e += 1
                self.labels["eq:" + b.split("||")[1].strip()] = e

    def refs(self, text):
        def rep(m):
            key = m.group(1)
            if key not in self.labels:
                print("  ! unknown reference", key)
                return "?"
            return str(self.labels[key])
        return re.sub(r"\{((?:fig|tab|eq):[\w-]+)\}", rep, text)

    def para(self, text, style="BD_pagrindinis_tekstas"):
        p = self.doc.add_paragraph(style=style)
        add_runs(p, self.refs(text))
        return p

    def heading(self, text, style, page_break=False):
        p = self.doc.add_paragraph(text, style=style)
        if page_break:
            p.paragraph_format.page_break_before = True
        return p

    def appendix_heading(self, text):
        p = self.doc.add_paragraph(text, style="BD_priedo_Skyrius_1_lygio")
        set_num(p, 0)
        return p

    def listing(self, lines, kind):
        num_id = self.num.new(kind)
        for line in lines:
            p = self.doc.add_paragraph(style="BD_pagrindinis_tekstas")
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_after = Pt(3)
            set_num(p, num_id)
            add_runs(p, self.refs(line))

    @staticmethod
    def caption_label(p, label, seq, number):
        """'Fig. N. ' with N as a SEQ field, so Word can build the List of Figures / Tables."""
        r = p.add_run(f"{label} ")
        r.bold, r.italic = True, False
        r = p.add_run()
        r.bold, r.italic = True, False
        add_field(r, f" SEQ {seq} \\* ARABIC ", str(number))
        r = p.add_run(". ")
        r.bold, r.italic = True, False

    def figure(self, key):
        fname, caption, width = FIGS[key]
        path = os.path.join(FIG, fname)
        if not os.path.exists(path):
            print("  ! missing figure", fname)
            return
        self.fig_no += 1
        p = self.doc.add_paragraph(style="BD_paveiksliukas")
        p.paragraph_format.space_before = Pt(6)
        p.add_run().add_picture(path, width=Cm(width))
        c = self.doc.add_paragraph(style="BD_paveiklsiuko_pavadinimas")
        set_num(c, 0)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.caption_label(c, "Fig.", "Figure", self.fig_no)
        r = c.add_run(caption)
        r.italic = False

    def table(self, key):
        if key not in self.tables:
            print("  ! missing table", key)
            return
        caption, header, rows, widths = self.tables[key]
        self.tab_no += 1
        c = self.doc.add_paragraph(style="BD_lenteles_pavadinimas")
        set_num(c, 0)
        c.paragraph_format.space_before = Pt(6)
        self.caption_label(c, "Table", "Table", self.tab_no)
        r = c.add_run(caption)
        r.italic = False
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = self.doc.styles["BD_lentele"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        total = sum(widths)
        scale = 17.0 / total if total > 17.0 else 1.0
        for i, h in enumerate(header):
            cell = t.rows[0].cells[i]
            p = cell.paragraphs[0]
            p.style = self.doc.styles["BD_lenteles_tekstas"]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run(h).bold = True
            tcPr = cell._tc.get_or_add_tcPr()
            tcPr.append(el("w:shd", **{"w:val": "clear", "w:color": "auto", "w:fill": "C6D9F1"}))
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                p = cells[i].paragraphs[0]
                p.style = self.doc.styles["BD_lenteles_tekstas"]
                p.add_run(str(v))
                if i > 0 and NUMERIC.match(str(v)):
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for row in t.rows:
            for i, cell in enumerate(row.cells):
                cell.width = Cm(widths[i] * scale)
        trPr = t.rows[0]._tr.get_or_add_trPr()
        trPr.append(el("w:tblHeader", **{"w:val": "true"}))
        # keep each table on one page together with its caption
        for k, row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(el("w:cantSplit", **{"w:val": "true"}))
            if k < len(t.rows) - 1:
                for cell in row.cells:
                    for cp in cell.paragraphs:
                        cp.paragraph_format.keep_with_next = True
        spacer = self.doc.add_paragraph(style="BD_pagrindinis_tekstas")
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.line_spacing = Pt(8)

    def equation(self, key):
        self.eq_no += 1
        if self.doc.paragraphs:
            self.doc.paragraphs[-1].paragraph_format.keep_with_next = True
        p = self.doc.add_paragraph(style="BD_formule")
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(6)
        p.add_run("\t")
        p._p.append(omml(LATEX[key]))
        r = p.add_run(f"\t({self.eq_no})")
        r.italic = False
        r.font.size = Pt(12)

    def reference(self, text):
        p = self.doc.add_paragraph(style="List Paragraph")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.75)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if self.ref_num:
            set_num(p, self.ref_num)
        add_runs(p, text, size=11)

    def run(self, blocks):
        self.scan(blocks)
        for b in blocks:
            if b.startswith("#A "):
                self.appendix_heading(b[3:].strip())
            elif b.startswith("#* "):
                self.heading(b[3:].strip(), "BD_Skyrius_nenumeruojamas", page_break=True)
            elif b.startswith("# "):
                self.heading(b[2:].strip(), "BD_Skyrius_1_lygio", page_break=True)
            elif b.startswith("### "):
                self.heading(b[4:].strip(), "BD_Skyrius_3_lygio")
            elif b.startswith("## "):
                self.heading(b[3:].strip(), "BD_Skyrius_2_lygio")
            elif b.startswith("[[TABLE"):
                self.table(b[7:-2].strip())
            elif b.startswith("[[FIG"):
                self.figure(b[5:-2].strip())
            elif b.startswith("$$"):
                self.equation(b.split("||")[1].strip())
            elif b.startswith("- "):
                self.listing([l[2:].strip() for l in b.split("\n")], "bullet")
            elif re.match(r"^\d+\. ", b):
                self.listing([re.sub(r"^\d+\.\s+", "", l).strip() for l in b.split("\n")], "decimal")
            elif b.startswith("REF "):
                for line in b.split("\n"):
                    self.reference(line[4:].strip())
            else:
                self.para(" ".join(b.split("\n")))


def read_blocks():
    blocks = []
    for f in sorted(x for x in os.listdir(HERE) if re.match(r"ch\d+_.*\.md$", x)):
        text = open(os.path.join(HERE, f), encoding="utf-8").read()
        blocks += [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return blocks


def build():
    doc = docx.Document(TEMPLATE)
    remove_second_cover(doc)
    fill_sdts(doc)
    annotation = [b.strip().replace("\n", " ") for b in
                  re.split(r"\n\s*\n", open(os.path.join(HERE, "annotation.md"), encoding="utf-8").read().strip())]
    front_matter(doc, annotation)
    tidy_annotation_page(doc)
    ref_num = strip_template_body(doc)
    unwrap_content_controls(doc)
    remove_comments(doc)
    for h in list(doc.element.body.iter(W_NS + "highlight")):
        h.getparent().remove(h)
    single_spacing(doc)
    fix_toc_fields(doc)
    Builder(doc, load_tables(), ref_num).run(read_blocks())
    cp = doc.core_properties
    cp.title = "Transformers and their application in NLP tasks"
    cp.author = AUTHORS
    cp.last_modified_by = AUTHORS
    cp.subject = "Group project paper, VILNIUS TECH, Faculty of Fundamental Sciences"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()
