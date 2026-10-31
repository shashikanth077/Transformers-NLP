"""Builds docs/Transformers_in_NLP_Presentation.pptx on the official VILNIUS TECH 4:3 template (slides/template/*.potx).
The 11 slides keep the content, the order and the speaker notes of the project presentation; all numbers come from
study_guide/data.json (the project's real results). Run: python slides/build_deck.py"""
import copy, json, os, zipfile
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)   # HERE = slides/
TPL = os.path.join(ROOT, "slides", "template", "VILNIUS_TECH_standard_4x3.potx")
import tempfile
BASE = os.path.join(tempfile.gettempdir(), "vgtu_base_template.pptx")   # temporary copy of the template with a presentation content type
OUT = os.path.join(ROOT, "docs", "Transformers_in_NLP_Presentation.pptx")
D = json.load(open(os.path.join(ROOT, "study_guide", "data.json"), encoding="utf-8"))
ASSETS = os.path.join(HERE, "assets")
NOTES = json.load(open(os.path.join(ASSETS, "notes.json"), encoding="utf-8"))
M = {m["tag"]: m for m in D["models"]}
acc = lambda t: M[t]["acc"] * 100

# template palette (theme of the VILNIUS TECH standard)
BLUE, ORANGE, CYAN, GREEN = RGBColor(0x0B, 0x4D, 0xC7), RGBColor(0xF6, 0x8B, 0x28), RGBColor(0x3E, 0xB8, 0xD8), RGBColor(0x00, 0xDD, 0xA5)
BLACK, WHITE, GREY, LIGHT = RGBColor(0, 0, 0), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0x59, 0x59, 0x59), RGBColor(0xEE, 0xF2, 0xFB)
PALE_BLUE, PALE_ORANGE = RGBColor(0xE6, 0xEE, 0xFB), RGBColor(0xFE, 0xF0, 0xE4)
FAMILY_COLOR = {"classical": CYAN, "recurrent": ORANGE, "transformer": BLUE}


# ------------------------------------------------------------------ base file: the template with its 19 sample slides removed
def make_base():
    with zipfile.ZipFile(TPL) as zin, zipfile.ZipFile(BASE, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":   # a .potx is a template; python-pptx needs the presentation type
                data = data.replace(b"presentationml.template.main+xml", b"presentationml.presentation.main+xml")
            zout.writestr(item, data)


make_base()
prs = Presentation(BASE)
sld_ids = prs.slides._sldIdLst
for sid in list(sld_ids):
    prs.part.drop_rel(sid.rId); sld_ids.remove(sid)
LAYOUT = {l.name: l for l in prs.slide_layouts}


# ------------------------------------------------------------------ helpers
def new_slide(layout, title=None, title_size=28):
    s = prs.slides.add_slide(LAYOUT[layout])
    for ph in list(s.placeholders):                     # keep only the placeholders we fill
        if ph.placeholder_format.type is not None and ph.placeholder_format.idx == 0 and title is not None:
            continue
        if ph.placeholder_format.idx == 0 and title is None:
            continue
        ph._element.getparent().remove(ph._element)
    if title is not None:
        t = s.shapes.title; t.text = title
        t.left, t.top, t.width, t.height = Inches(0.47), Inches(0.35), Inches(7.75), Inches(1.45)   # keep clear of the logo
        t.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in t.text_frame.paragraphs:
            for r in p.runs: r.font.size = Pt(title_size); r.font.bold = True
    return s


def text(s, x, y, w, h, paras, size=14, color=BLACK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=4):
    """paras: str or list of str / (str, dict) where dict may hold size, bold, color, bullet."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    items = paras if isinstance(paras, list) else [paras]
    for i, it in enumerate(items):
        t, o = (it if isinstance(it, tuple) else (it, {}))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align); p.space_after = Pt(o.get("space", space))
        runs = t if isinstance(t, list) else [(t, {})]
        for rt, ro in runs:
            r = p.add_run(); r.text = rt
            r.font.size = Pt(ro.get("size", o.get("size", size))); r.font.bold = ro.get("bold", o.get("bold", bold))
            r.font.color.rgb = ro.get("color", o.get("color", color))
    return tb


def box(s, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None: sh.line.fill.background()
    else: sh.line.color.rgb = line; sh.line.width = Pt(1)
    sh.shadow.inherit = False
    return sh


def badge(s, x, y, n, fill=BLUE, d=0.42, size=14):
    sh = box(s, x, y, d, d, fill, shape=MSO_SHAPE.OVAL)
    tf = sh.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = str(n)
    r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHITE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return sh


def picture(s, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def table(s, x, y, w, rows, col_w, size=12, row_h=0.42):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    for j, cw in enumerate(col_w): tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.text = ""
            c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(val); r.font.size = Pt(size)
            r.font.bold = (i == 0); r.font.color.rgb = WHITE if i == 0 else BLACK
            c.fill.solid(); c.fill.fore_color.rgb = BLUE if i == 0 else (LIGHT if i % 2 == 0 else WHITE)
    return shp


def set_notes(s, idx):
    s.notes_slide.notes_text_frame.text = NOTES[idx]


# ------------------------------------------------------------------ 1  title (blue)
s = prs.slides.add_slide(LAYOUT["1_Titulinis"])
for ph in list(s.placeholders):
    if ph.placeholder_format.idx == 0:
        ph.text_frame.text = "Transformers and their application in NLP tasks"
        for p in ph.text_frame.paragraphs:
            for r in p.runs: r.font.size = Pt(34); r.font.bold = True
    elif ph.placeholder_format.idx == 10:
        ph.text_frame.text = "VILNIUS TECH · Group project"
    else:
        ph._element.getparent().remove(ph._element)
text(s, 0.34, 2.55, 9.2, 0.9, ["Do Transformers pay off on a laptop CPU?", ("A comparison of 10 sentiment classifiers on SST-2", {"size": 16, "bold": False})], size=20, color=WHITE, bold=True, space=2)
text(s, 0.34, 6.95, 9.3, 0.4, "Shashikanth Hosur Ramegowda  ·  Shaeed Khan", size=13, color=WHITE)
set_notes(s, 0)

# ------------------------------------------------------------------ 2  reading word by word vs all at once
s = new_slide("1_Tekstas", "From reading word by word to reading all at once")
for x, head, col, pale, items in [
    (0.47, "Recurrent networks (LSTM)", ORANGE, PALE_ORANGE, ["Process tokens one after another", "n sequential steps: slow to train", "Distant words interact only through a long chain of states", "Trained from scratch for each task"]),
    (5.13, "Transformer (2017)", BLUE, PALE_BLUE, ["Every token attends to every other token", "All positions computed in parallel (GPU or CPU)", "Pre-train once on huge unlabelled text, then fine-tune", "Basis of BERT, GPT, T5 and LitLat BERT"])]:
    box(s, x, 1.95, 4.4, 4.35, pale)
    box(s, x, 1.95, 4.4, 0.7, col)
    text(s, x + 0.2, 1.95, 4.0, 0.7, head, size=18, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.25, 2.9, 3.95, 3.3, [("•  " + t, {"space": 12}) for t in items], size=16)
set_notes(s, 1)

# ------------------------------------------------------------------ 3  self-attention
s = new_slide("1_Tekstas", "Self-attention: each word looks at the whole sentence")
picture(s, os.path.join(ASSETS, "arch_diagram.png"), 0.5, 1.9, h=4.75)
box(s, 4.85, 1.95, 4.68, 0.95, PALE_BLUE)
text(s, 4.95, 1.98, 4.5, 0.55, "Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V", size=15, bold=True, color=BLUE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 4.95, 2.5, 4.5, 0.35, "queries · keys → weights → weighted mix of values", size=12, color=GREY, align=PP_ALIGN.CENTER)
steps = [("Sub-word tokens + position embeddings", "WordPiece: “unwatchable” → un ##watch ##able"),
         ("Multi-head attention", "h heads learn different relations (negation scope, syntax)"),
         ("Feed-forward + residual + LayerNorm", "Stacked L times (2, 4 or 6 layers in our models)"),
         ("[CLS] vector → softmax classifier", "Whole network fine-tuned with cross-entropy + AdamW")]
for i, (a, b) in enumerate(steps):
    y = 3.15 + i * 0.84
    badge(s, 4.85, y + 0.04, i + 1, fill=BLUE if i % 2 == 0 else ORANGE, d=0.38, size=12)
    text(s, 5.35, y - 0.02, 4.2, 0.8, [(a, {"bold": True, "size": 14, "space": 1}), (b, {"size": 12, "color": GREY})])
set_notes(s, 2)

# ------------------------------------------------------------------ 4  literature
s = new_slide("1_Tekstas", "Literature: what is known, and what is missing")
for i, (h, models, task, col) in enumerate([("Encoder-only", "BERT, DistilBERT, RoBERTa", "understanding, classification", GREEN),
                                             ("Decoder-only", "GPT, Llama", "text generation", CYAN),
                                             ("Encoder–decoder", "T5, BART", "translation, summaries", BLUE)]):
    x = 0.47 + i * 3.1
    box(s, x, 1.85, 2.96, 1.2, LIGHT); box(s, x, 1.85, 0.09, 1.2, col)
    text(s, x + 0.22, 1.9, 2.7, 0.4, h, size=15, bold=True, color=col if col != GREEN else RGBColor(0x00, 0x9E, 0x72))
    text(s, x + 0.22, 2.3, 2.7, 0.7, [(models, {"bold": True, "size": 12, "space": 2}), (task, {"size": 12, "color": GREY})])
text(s, 0.47, 3.3, 5.3, 0.45, "What the studies show", size=16, bold=True, color=BLUE)
facts = [("Transformers are the most accurate.", " Fine-tuned BERT-type models exceed 90% on SST-2; classical models reach about 79–83%."),
         ("Without pre-training, small data is a problem.", " In Lithuania a simple naive Bayes (0.735) beat LSTM and CNN networks (0.706), Kapočiūtė-Dzikienė et al. (2019)."),
         ("Lithuanian now has pre-trained models.", " LitLat BERT (2021), BERT and T5 for reviews (2024), the first open Llama2 models (2024).")]
for i, (b, rest) in enumerate(facts):
    y = 3.85 + i * 0.95
    badge(s, 0.47, y + 0.02, i + 1, fill=BLUE, d=0.36, size=12)
    text(s, 0.95, y - 0.02, 4.85, 0.95, [([(b, {"bold": True}), (rest, {})], {})], size=13)
box(s, 5.95, 3.3, 3.58, 3.2, PALE_ORANGE); box(s, 5.95, 3.3, 3.58, 0.07, ORANGE)
text(s, 6.12, 3.45, 3.3, 0.45, "What is missing: our gap", size=15, bold=True, color=RGBColor(0xB8, 0x44, 0x1B))
text(s, 6.12, 3.95, 3.3, 2.5, [("These studies report accuracy. They rarely tell a student with a laptop:", {"space": 6}),
                              ("•  What does the better model cost? (time, speed, size)", {"space": 4}),
                              ("•  How many labelled sentences does it need?", {"space": 4}),
                              ("•  Is a ready-made model good enough?", {"space": 8}),
                              ("We measure all three, on one laptop CPU.", {"bold": True, "color": RGBColor(0xB8, 0x44, 0x1B)})], size=12)
set_notes(s, 3)

# ------------------------------------------------------------------ 5  experiment design
s = new_slide("1_Tekstas", "Experiment: 10 competitors, one dataset, one laptop CPU")
for i, (big, cap) in enumerate([("SST-2", "movie-review sentences, positive or negative"), ("65k / 2k / 872", "train / development / test sentences"), ("i5-8365U", "4-core laptop CPU, 16 GB RAM, no GPU")]):
    y = 1.9 + i * 1.45
    box(s, 0.47, y, 3.05, 1.25, LIGHT); box(s, 0.47, y, 0.09, 1.25, BLUE)
    text(s, 0.7, y + 0.1, 2.75, 0.55, big, size=22, bold=True, color=BLUE)
    text(s, 0.7, y + 0.66, 2.75, 0.55, cap, size=11.5, color=GREY)
groups = [("Classical", "TF-IDF + Naive Bayes · Logistic Regression · Linear SVM", CYAN),
          ("Recurrent", "BiLSTM trained from scratch", ORANGE),
          ("Fine-tuned by us (8k labels)", "BERT-tiny 4M · BERT-small 29M · DistilBERT 67M", BLUE),
          ("Ready-made, no training", "DistilBERT-SST2 · Twitter-RoBERTa · zero-shot NLI", GREEN)]
for i, (h, sub, col) in enumerate(groups):
    y = 1.9 + i * 0.98
    box(s, 3.85, y + 0.08, 0.34, 0.34, col, shape=MSO_SHAPE.OVAL)
    text(s, 4.35, y - 0.02, 5.2, 0.9, [(h, {"bold": True, "size": 15, "space": 1}), (sub, {"size": 12.5, "color": GREY})])
box(s, 0.47, 5.95, 9.06, 0.75, LIGHT)
text(s, 0.6, 5.98, 8.8, 0.7, [([("Criteria: ", {"bold": True, "color": BLUE}), ("accuracy and macro-F1 · training time · latency per sentence · parameters · negation and contrast subsets · learning curve · bootstrap CI and McNemar test", {})], {})], size=12.5, anchor=MSO_ANCHOR.MIDDLE)
set_notes(s, 4)

# ------------------------------------------------------------------ 6  implementation
s = new_slide("1_Tekstas", "Implementation: a reproducible Python pipeline")
picture(s, os.path.join(ASSETS, "pipeline.png"), 0.47, 1.95, w=5.75)
box(s, 6.45, 1.95, 3.08, 4.4, LIGHT); box(s, 6.45, 1.95, 3.08, 0.07, BLUE)
text(s, 6.6, 2.12, 2.85, 4.2, [("PyTorch · Hugging Face Transformers · scikit-learn", {"bold": True, "color": BLUE, "space": 10}),
                              ("•  One generic fine-tuning script for any encoder", {"space": 8}),
                              ("•  AdamW, 10% warm-up, linear decay, best epoch by development accuracy", {"space": 8}),
                              ("•  Every model saves metrics, timings and predictions as JSON", {"space": 8}),
                              ("•  run_all.sh reproduces all results (about 2.5 hours on a laptop)", {"space": 8})], size=12.5)
set_notes(s, 5)

# ------------------------------------------------------------------ 7  ranking chart
s = new_slide("1_Tekstas", "Fine-tuned DistilBERT: +5.9 points over classical")
order = sorted(M.values(), key=lambda m: m["acc"])
SHORT = {"tfidf_nb": "TF-IDF + Naive Bayes", "tfidf_logreg": "TF-IDF + Logistic Regr.", "tfidf_svm": "TF-IDF + Linear SVM", "bilstm": "BiLSTM (from scratch)", "ft_bert-tiny_8000": "BERT-tiny fine-tuned (8k)", "ft_bert-small_8000": "BERT-small fine-tuned (8k)", "ft_distilbert_8000": "DistilBERT fine-tuned (8k)", "pt_distilbert_sst2": "DistilBERT-SST2 (off-the-shelf)", "pt_twitter_roberta": "Twitter-RoBERTa (off-the-shelf)", "zs_nli_distilroberta": "Zero-shot NLI DistilRoBERTa"}
cd = CategoryChartData(); cd.categories = [SHORT[m["tag"]] for m in order]; cd.add_series("Accuracy, %", [round(m["acc"] * 100, 1) for m in order])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.4), Inches(1.85), Inches(6.2), Inches(4.85), cd)
ch = gf.chart; ch.has_title = False; ch.has_legend = False; ch.font.size = Pt(10.5)
pl = ch.plots[0]; pl.gap_width = 45; pl.has_data_labels = True
pl.data_labels.number_format = '0.0"%"'; pl.data_labels.number_format_is_linked = False; pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END; pl.data_labels.font.size = Pt(10)
va = ch.value_axis; va.minimum_scale = 60; va.maximum_scale = 95; va.has_major_gridlines = False; va.tick_labels.font.size = Pt(9)
for i, m in enumerate(order):
    pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = FAMILY_COLOR[m["family"]]
for i, (big, cap, col) in enumerate([("88.8%", "DistilBERT fine-tuned on 8,000 labels", BLUE), ("82.9%", "best classical: TF-IDF + Linear SVM", CYAN), ("81.8%", "zero-shot: no labels at all", GREEN)]):
    y = 1.95 + i * 1.55
    box(s, 6.85, y, 2.7, 1.35, LIGHT); box(s, 6.85, y, 0.09, 1.35, col)
    text(s, 7.05, y + 0.08, 2.4, 0.6, big, size=26, bold=True, color=BLUE if col == BLUE else (RGBColor(0x00, 0x9E, 0x72) if col == GREEN else RGBColor(0x1F, 0x9B, 0xBA)))
    text(s, 7.05, y + 0.72, 2.4, 0.6, cap, size=11.5, color=GREY)
text(s, 0.47, 6.72, 6.2, 0.3, "Bars: classical (cyan), recurrent (orange), transformer (blue). Test set: 872 sentences.", size=10, color=GREY)
set_notes(s, 6)

# ------------------------------------------------------------------ 8  price table
s = new_slide("1_Tekstas", "The price: slower training and inference, bigger models")
rows = [["Model", "Accuracy, %", "Train time", "Latency, ms", "Params"]]
for t, lab in [("tfidf_svm", "TF-IDF + Linear SVM"), ("bilstm", "BiLSTM (from scratch)"), ("ft_bert-tiny_8000", "BERT-tiny fine-tuned (8k)"), ("ft_bert-small_8000", "BERT-small fine-tuned (8k)"), ("ft_distilbert_8000", "DistilBERT fine-tuned (8k)"), ("pt_distilbert_sst2", "DistilBERT-SST2 (off-the-shelf)"), ("zs_nli_distilroberta", "Zero-shot NLI DistilRoBERTa")]:
    m = M[t]
    tt = "–" if m["train_s"] == 0 else ("%d s" % m["train_s"] if m["train_s"] < 60 else ("%d min" % (m["train_s"] / 60) if m["train_s"] < 3600 else "%.1f h" % (m["train_s"] / 3600)))
    pp = ("%.1f M" % (m["params"] / 1e6)) if m["params"] >= 1e6 else ("%d k" % (m["params"] / 1e3))
    rows.append([lab, "%.1f" % (m["acc"] * 100), tt, ("%.2f" % m["lat_ms"]) if m["lat_ms"] < 1 else ("%.1f" % m["lat_ms"]), pp])
table(s, 0.47, 1.9, 6.0, rows, [2.5, 0.9, 0.95, 0.95, 0.7], size=11, row_h=0.46)
box(s, 6.75, 1.9, 2.78, 1.9, LIGHT); box(s, 6.75, 1.9, 2.78, 0.07, ORANGE)
text(s, 6.92, 2.05, 2.5, 0.7, "×710", size=30, bold=True, color=ORANGE)
text(s, 6.92, 2.8, 2.5, 1.0, "slower inference of DistilBERT vs TF-IDF, still only about 40 ms per sentence", size=12, color=GREY)
box(s, 6.75, 4.0, 2.78, 2.5, LIGHT); box(s, 6.75, 4.0, 2.78, 0.07, BLUE)
text(s, 6.92, 4.15, 2.5, 0.5, "Fine-tuning is affordable", size=15, bold=True, color=BLUE)
text(s, 6.92, 4.7, 2.5, 1.7, "Training on the laptop CPU took from 2 minutes to 2.6 hours. No GPU is needed for a few thousand examples.", size=12, color=GREY)
set_notes(s, 7)

# ------------------------------------------------------------------ 9  learning curve
s = new_slide("1_Tekstas", "Pre-training is the key: fewer labels, higher accuracy")
cd = CategoryChartData(); cd.categories = ["500", "1,000", "2,000", "4,000", "8,000"]
cd.add_series("BERT-small (pre-trained)", [round(v * 100, 1) for _, v in D["curve"]["transformer"]])
cd.add_series("TF-IDF + logistic regression", [round(v * 100, 1) for _, v in D["curve"]["tfidf"]])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.4), Inches(1.85), Inches(6.2), Inches(4.85), cd)
ch = gf.chart; ch.has_title = False; ch.font.size = Pt(10.5); ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(10.5)
va = ch.value_axis; va.minimum_scale = 55; va.maximum_scale = 90; va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xDD, 0xDD); va.tick_labels.font.size = Pt(9)
ch.category_axis.tick_labels.font.size = Pt(10); ch.category_axis.has_title = True; ch.category_axis.axis_title.text_frame.text = "labelled training sentences"
ch.category_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
for ser, col in zip(ch.plots[0].series, [BLUE, ORANGE]):
    ser.format.line.color.rgb = col; ser.format.line.width = Pt(2.75); ser.smooth = False
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col; ser.marker.format.line.color.rgb = col
    ser.data_labels.show_value = True; ser.data_labels.font.size = Pt(9); ser.data_labels.position = XL_LABEL_POSITION.ABOVE
for i, (big, cap, col) in enumerate([("4,000", "labels: BERT-small already matches the best classical model trained on all 65k (82.7% vs 82.9%)", BLUE), ("+14 p.p.", "advantage with only 500 labels; the gap shrinks as data grows", ORANGE)]):
    y = 1.95 + i * 2.35
    box(s, 6.85, y, 2.7, 2.1, LIGHT); box(s, 6.85, y, 0.09, 2.1, col)
    text(s, 7.05, y + 0.1, 2.4, 0.7, big, size=26, bold=True, color=col)
    text(s, 7.05, y + 0.85, 2.4, 1.2, cap, size=11.5, color=GREY)
set_notes(s, 8)

# ------------------------------------------------------------------ 10  negation and contrast
s = new_slide("1_Tekstas", "Context matters: negation and contrast sentences")
cd = CategoryChartData(); cd.categories = ["TF-IDF + Linear SVM", "BiLSTM", "DistilBERT fine-tuned", "DistilBERT-SST2"]
tags = ["tfidf_svm", "bilstm", "ft_distilbert_8000", "pt_distilbert_sst2"]
cd.add_series("All sentences", [round(acc(t), 1) for t in tags])
cd.add_series("Negation", [74.5, 78.8, 85.3, 87.0]); cd.add_series("Contrast", [77.7, 77.7, 85.4, 86.2])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.85), Inches(5.3), Inches(4.85), cd)
ch = gf.chart; ch.has_title = False; ch.font.size = Pt(10); ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(10)
va = ch.value_axis; va.minimum_scale = 60; va.maximum_scale = 95; va.has_major_gridlines = False; va.tick_labels.font.size = Pt(9)
ch.category_axis.tick_labels.font.size = Pt(9)
pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -5; pl.has_data_labels = True; pl.data_labels.font.size = Pt(8); pl.data_labels.number_format = "0"; pl.data_labels.number_format_is_linked = False
for ser, col in zip(pl.series, [BLUE, ORANGE, CYAN]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
picture(s, os.path.join(ASSETS, "attention.png"), 5.95, 1.9, w=3.6)
text(s, 5.95, 5.3, 3.6, 1.0, "“the plot was not good, but the acting was brilliant”: DistilBERT attention, last layer", size=11, color=GREY)
set_notes(s, 9)

# ------------------------------------------------------------------ 11  conclusions
s = new_slide("1_Tekstas", "Conclusions and recommendations")
items = ["Fine-tuned Transformers are the most accurate: 88.8% vs 82.9% (classical) and 82.8% (BiLSTM)",
         "Pre-training gives data efficiency: BERT-small with 4,000 labels matches classical models trained on 65k; DistilBERT with 8,000 beats them",
         "Cost is real but manageable: minutes of CPU fine-tuning, milliseconds per sentence",
         "Reuse first: an in-domain public model gives 91.1% with zero training; zero-shot (81.8%) and out-of-domain models are useful but weaker",
         "For Lithuanian: fine-tune LitLat BERT or XLM-R on a small labelled set, the next step of this work"]
for i, t in enumerate(items):
    y = 1.9 + i * 0.86
    badge(s, 0.47, y + 0.04, i + 1, fill=BLUE if i % 2 == 0 else ORANGE, d=0.42, size=14)
    text(s, 1.07, y, 8.45, 0.8, t, size=14.5, anchor=MSO_ANCHOR.TOP)
text(s, 0.47, 6.45, 9.0, 0.5, "Thank you! Questions?", size=20, bold=True, color=BLUE)
set_notes(s, 10)

# ------------------------------------------------------------------ properties and save
prs.core_properties.author = "Shashikanth Hosur Ramegowda, Shaeed Khan"
prs.core_properties.title = "Transformers and their application in NLP tasks"
prs.core_properties.subject = "Group project, VILNIUS TECH, Engineering of Artificial Intelligence"
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
