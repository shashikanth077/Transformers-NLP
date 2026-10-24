"""Builds docs/Literature_Gap_Explained.pdf (A4 landscape): the gap in the four Lithuanian-related articles, why small data hurts,
and what this project did, step by step, with real numbers from the project results.
Run: python study_guide/gap_pdf/build.py   (needs pandas and Microsoft Edge)"""
import html, json, os, re, subprocess, tempfile, textwrap, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = os.path.join(ROOT, "docs", "Literature_Gap_Explained.pdf")
esc = html.escape
D = json.load(open(os.path.join(ROOT, "study_guide", "data.json"), encoding="utf-8"))
MOD = {m["tag"]: m for m in D["models"]}; TAGS = [m["tag"] for m in D["models"]]
BLUE, GREEN, RED, AMBER, VIOLET, INK, GREY = "#2a78d6", "#159a6a", "#c0392b", "#c27a10", "#6250d6", "#1d1d3a", "#6b6b88"

# ---- real numbers computed for this document
tr = pd.read_csv(os.path.join(ROOT, "code", "data", "train.csv")); te = pd.read_csv(os.path.join(ROOT, "code", "data", "test.csv"))
tk = lambda s: re.findall(r"[a-z']+", str(s).lower())
test_tokens = [w for s in te["sentence"] for w in tk(s)]
oov = {}
for n in (500, 2000, 8000, len(tr)):
    sub = tr.sample(n=n, random_state=42) if n < len(tr) else tr
    vocab = {w for s in sub["sentence"] for w in tk(s)}
    oov[n] = (len(vocab), sum(1 for w in test_tokens if w not in vocab) / len(test_tokens))
CURVE_T, CURVE_C = D["curve"]["transformer"], D["curve"]["tfidf"]
SENT = next(s for s in D["sentences"] if s["s"].startswith("a great ensemble cast ca n't lift"))
assert SENT["y"] == 0


def wrap(s, n): return textwrap.wrap(s, n)


def box(x, y, w, h, fill, stroke, lines, fs=12.5, head=None, hc=None, tc=INK, rx=12):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    yy = y + 2
    if head: yy = y + 20; s += f'<text x="{x + 12}" y="{yy}" font-size="{fs + 1}" font-weight="800" fill="{hc or stroke}">{esc(head)}</text>'; yy += 8
    for ln in lines:
        yy += fs + 3.5; s += f'<text x="{x + 12}" y="{yy}" font-size="{fs}" fill="{tc}">{esc(ln)}</text>'
    return s


def arrow(x1, y1, x2, y2, c=GREY, w=2.4):
    return f'<path d="M{x1} {y1} C{(x1 + x2) / 2} {y1}, {(x1 + x2) / 2} {y2}, {x2} {y2}" fill="none" stroke="{c}" stroke-width="{w}" marker-end="url(#ar{c[1:]})"/>'


def markers(*cs): return "<defs>" + "".join(f'<marker id="ar{c[1:]}" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0 0L12 6L0 12z" fill="{c}"/></marker>' for c in cs) + "</defs>"


# ------------------------------------------------------------------ page 1: the gap map
def gap_map():
    s = f'<svg viewBox="0 0 1100 690" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">{markers(GREY, RED, GREEN, AMBER)}'
    for x, t, c in [(20, "1  WHAT THE FOUR ARTICLES DID", BLUE), (350, "2  WHAT EACH ONE LEAVES OPEN", AMBER), (660, "3  THE GAP THEY SHARE", RED), (880, "4  WHAT WE DID", GREEN)]:
        s += f'<text x="{x}" y="22" font-size="13" font-weight="800" letter-spacing="1.2" fill="{c}">{t}</text>'
    arts = [("Kapočiūtė-Dzikienė et al. (2019)", ["Compared naive Bayes with LSTM and CNN on", "Lithuanian comments (3 classes).", "Result: NB 0.735 beat CNN 0.706."], ["Showed that networks without pre-training lose on", "small data, but did not test a pre-trained", "Transformer or measure cost."]),
            ("Ulčar & Robnik-Šikonja (2021)", ["Built LitLat BERT for Lithuanian, Latvian", "and English.", "Result: corpus and vocabulary size matter."], ["Focus is building the model; behaviour with", "few labelled examples and cost on ordinary", "hardware are not the question."]),
            ("Vileikytė et al. (2024)", ["Fine-tuned multilingual BERT and T5 on", "Lithuanian online reviews.", "Result: high accuracy on clear reviews."], ["Reports accuracy; training time, latency, size", "and a comparison with cheap classical", "models are not the focus."]),
            ("Nakvosas et al. (2024)", ["Released the first open Llama2-based Lithuanian", "models (7 B and 13 B parameters).", "Result: new models and benchmarks."], ["Needs large hardware; says little about what a", "small team on a laptop should choose or", "how many labels it needs."])]
    for i, (h, a, b) in enumerate(arts):
        y = 38 + i * 142
        s += box(20, y, 310, 124, "#eaf1fc", BLUE, a, 12, head=h, hc=BLUE)
        s += box(350, y, 290, 124, "#fff4de", AMBER, b, 12, head="Problem / limit", hc=AMBER)
        s += arrow(330, y + 62, 350, y + 62, BLUE)
        s += arrow(640, y + 62, 660, 310, AMBER)
    s += box(660, 38, 200, 552, "#fdeceb", RED, [], 12)
    gaps = [("GAP 1", ["Nobody measures quality,", "cost and data size", "together under one", "protocol on ordinary", "hardware."]), ("GAP 2", ["How many labels does a", "pre-trained model need", "compared with a", "classical one?", "(small data question)"]), ("GAP 3", ["Ready-made models vs", "a model fine-tuned on", "your own data: seldom", "compared fairly."])]
    yy = 66
    for h, ls in gaps:
        s += f'<text x="676" y="{yy}" font-size="14" font-weight="800" fill="{RED}">{h}</text>'
        for ln in ls: yy += 17; s += f'<text x="676" y="{yy + 6}" font-size="12.5" fill="{INK}">{esc(ln)}</text>'
        yy += 42
    s += arrow(860, 310, 880, 310, RED, 3.4)
    steps = ["Ask 5 research questions", "Take one dataset: SST-2", "Build 10 models, 4 families", "Same laptop, same data, same rules", "Measure accuracy, time, speed, size", "Add data-size curve and statistics", "Recommend what to use when"]
    s += f'<rect x="880" y="38" width="205" height="552" rx="12" fill="#e6f6ee" stroke="{GREEN}" stroke-width="2"/>'
    for i, t in enumerate(steps):
        y = 60 + i * 57
        s += f'<circle cx="904" cy="{y + 14}" r="12" fill="{GREEN}"/><text x="904" y="{y + 19}" text-anchor="middle" font-size="12" font-weight="800" fill="#fff">{i + 1}</text>'
        for k, ln in enumerate(wrap(t, 22)): s += f'<text x="926" y="{y + 12 + k * 15}" font-size="12.5" fill="{INK}" font-weight="600">{esc(ln)}</text>'
    s += f'<rect x="20" y="598" width="1065" height="90" rx="12" fill="#ece9fd" stroke="{VIOLET}" stroke-width="2"/>'
    s += f'<text x="36" y="622" font-size="13" font-weight="800" fill="{VIOLET}">THEORY IN ONE LINE (explained on page 2)</text>'
    t1 = "A network trained from scratch must learn the language AND the task from the few labelled examples it has. A pre-trained Transformer already knows the language, so it only has to learn the task."
    t2 = f"With 500 labelled sentences about {oov[500][1] * 100:.0f}% of the words in our test set were never seen in training: a model trained from scratch has no idea what they mean."
    for k, ln in enumerate(wrap(t1, 135) + wrap(t2, 135)): s += f'<text x="36" y="{641 + k * 15}" font-size="12.2" fill="{INK}">{esc(ln)}</text>'
    return s + "</svg>"


# ------------------------------------------------------------------ page 2: why small data
def curve_svg():
    W, H, L, R, T_, B = 520, 280, 46, 14, 20, 40; xs = [500, 1000, 2000, 4000, 8000]
    X = lambda n: L + (__import__("math").log(n) - __import__("math").log(500)) / (__import__("math").log(8000) - __import__("math").log(500)) * (W - L - R)
    Y = lambda v: T_ + (1 - (v - 0.55) / 0.35) * (H - T_ - B)
    s = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif"><g stroke="#d9dcef"><line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}"/><line x1="{L}" y1="{T_}" x2="{L}" y2="{H - B}"/></g>'
    for v in (0.6, 0.7, 0.8, 0.9): s += f'<text x="6" y="{Y(v) + 4}" font-size="11" fill="{GREY}">{round(v * 100)}%</text><line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#d9dcef" stroke-dasharray="2 4"/>'
    for n in xs: s += f'<text x="{X(n)}" y="{H - 22}" font-size="11" text-anchor="middle" fill="{GREY}">{n}</text>'
    for data, c, nm, dy in [(CURVE_T, GREEN, "BERT-small (pre-trained)", -9), (CURVE_C, BLUE, "TF-IDF + logistic regression (from scratch)", 17)]:
        s += f'<polyline fill="none" stroke="{c}" stroke-width="3.2" points="{" ".join(f"{X(n)},{Y(v)}" for n, v in data)}"/>'
        for n, v in data: s += f'<circle cx="{X(n)}" cy="{Y(v)}" r="4.5" fill="{c}"/><text x="{X(n)}" y="{Y(v) + dy}" font-size="10.5" text-anchor="middle" fill="{c}" font-weight="700">{v * 100:.1f}</text>'
    s += f'<text x="{L + 8}" y="{T_ - 4}" font-size="12" font-weight="800" fill="{GREEN}">BERT-small (pre-trained)</text><text x="{L + 200}" y="{T_ - 4}" font-size="12" font-weight="800" fill="{BLUE}">TF-IDF (from scratch)</text>'
    s += f'<text x="{(L + W) / 2}" y="{H - 4}" font-size="11" text-anchor="middle" fill="{GREY}">number of labelled training sentences (log scale)</text><text x="{L + 150}" y="{Y(0.585)}" font-size="12" font-weight="800" fill="{RED}">gap at 500 labels: +{(CURVE_T[0][1] - CURVE_C[0][1]) * 100:.1f} points</text>'
    return s + "</svg>"


def oov_svg():
    W, H = 520, 140; s = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif">'
    for i, (n, (voc, o)) in enumerate(oov.items()):
        y = 16 + i * 33
        s += f'<text x="8" y="{y + 15}" font-size="12" fill="{INK}">{n if n < 60000 else "all 65,349"} labels</text><rect x="120" y="{y}" width="{o * 190 / 0.32}" height="20" rx="5" fill="{RED}" opacity="{0.45 + o:.2f}"/><text x="{128 + o * 190 / 0.32}" y="{y + 15}" font-size="12.5" font-weight="800" fill="{INK}">{o * 100:.1f}% unseen</text><text x="{W - 2}" y="{y + 15}" text-anchor="end" font-size="11" fill="{GREY}">{voc:,} known words</text>'
    return s + "</svg>"


def from_scratch_vs_pretrained():
    s = f'<svg viewBox="0 0 560 250" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">{markers(GREY, BLUE, GREEN)}'
    s += f'<rect x="4" y="4" width="270" height="242" rx="12" fill="#eaf1fc" stroke="{BLUE}" stroke-width="2"/><text x="139" y="26" text-anchor="middle" font-size="13.5" font-weight="800" fill="{BLUE}">FROM SCRATCH</text>'
    s += box(24, 40, 230, 42, "#fff", BLUE, ["500 labelled sentences"], 12.5) + arrow(139, 84, 139, 104, BLUE) + box(24, 106, 230, 54, "#fff", BLUE, ["must learn word meanings", "AND the sentiment task"], 12)
    s += arrow(139, 162, 139, 182, BLUE) + box(24, 184, 230, 52, "#fdeceb", RED, ["many words unseen,", "easy to memorise: weak"], 12)
    s += f'<rect x="286" y="4" width="270" height="242" rx="12" fill="#e6f6ee" stroke="{GREEN}" stroke-width="2"/><text x="421" y="26" text-anchor="middle" font-size="13.5" font-weight="800" fill="{GREEN}">PRE-TRAINED, THEN FINE-TUNED</text>'
    s += box(306, 40, 230, 42, "#fff", GREEN, ["huge unlabelled text (once)"], 12.5) + arrow(421, 84, 421, 104, GREEN) + box(306, 106, 230, 54, "#fff", GREEN, ["already knows words and", "grammar; learns the task"], 12)
    s += arrow(421, 162, 421, 182, GREEN) + box(306, 184, 230, 52, "#e2f5ec", GREEN, ["few labels are enough:", "much stronger"], 12)
    return s + "</svg>"


# ------------------------------------------------------------------ page 3: step by step with a real sentence
rows = []
labels = {"tfidf_nb": "TF-IDF + naive Bayes", "tfidf_logreg": "TF-IDF + logistic regression", "tfidf_svm": "TF-IDF + linear SVM", "bilstm": "BiLSTM (from scratch)", "ft_bert-tiny_8000": "BERT-tiny fine-tuned (8,000 labels)", "ft_bert-small_8000": "BERT-small fine-tuned (8,000 labels)", "ft_distilbert_8000": "DistilBERT fine-tuned (8,000 labels)", "pt_distilbert_sst2": "DistilBERT-SST2 (ready-made)", "pt_twitter_roberta": "Twitter-RoBERTa (ready-made)", "zs_nli_distilroberta": "Zero-shot NLI (no labels)"}
for t in TAGS:
    pred = int(SENT["p"][TAGS.index(t)]); ok = pred == SENT["y"]
    rows.append(f"<tr><td>{labels[t]}</td><td style='text-align:center'>{'positive' if pred else 'negative'}</td><td style='text-align:center;color:{GREEN if ok else RED};font-weight:800'>{'✓ correct' if ok else '✗ wrong'}</td></tr>")
n_ok = sum(1 for t in TAGS if int(SENT["p"][TAGS.index(t)]) == SENT["y"])

CSS = """
@page { size: A4 landscape; margin: 9mm 10mm 10mm 10mm; @bottom-center { content: counter(page); font: 8pt 'Segoe UI', sans-serif; color: #777; } }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font: 9.6pt/1.5 'Segoe UI', 'Nirmala UI', sans-serif; color: #1d1d3a; }
.page { page-break-after: always; break-after: page; } .page:last-child { page-break-after: auto; }
h1 { font-size: 17pt; margin: 0 0 1mm; color: #2e2a6b; } h1 small { font-size: 9.5pt; font-weight: 400; color: #6b6b88; margin-left: 3mm; } h2 { font-size: 12pt; margin: 3mm 0 1.2mm; color: #6250d6; } p { margin: 0 0 1.8mm; }
svg { max-width: 100%; height: auto; display: block; margin: 0 auto; } .p1 svg { max-height: 175mm; } .cols svg { max-height: 62mm; } .cols.top svg { max-height: 56mm; } .p2 .box { margin: 1.3mm 0; padding: 1.8mm 3.2mm; font-size: 9pt; } .p2 p { font-size: 8.5pt; } table { font-size: 8.2pt; } .step p { font-size: 8.3pt !important; } .p3 .box { font-size: 8.4pt; padding: 1.6mm 3mm; }
.box { border-radius: 2.5mm; padding: 2.2mm 3.6mm; margin: 1.8mm 0; break-inside: avoid; } .box.pl { background: #e6f6ee; border-left: 4px solid #159a6a; } .box.mt { background: #ece9fd; border-left: 4px solid #6250d6; } .box.wn { background: #fff3dc; border-left: 4px solid #c27a10; } .box.rw { background: #fff1e8; border-left: 4px solid #d9541f; }
.box b.h { display: block; font-size: 7.6pt; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .4mm; } .pl b.h { color: #159a6a; } .mt b.h { color: #6250d6; } .wn b.h { color: #c27a10; } .rw b.h { color: #d9541f; }
.f { font-family: 'Cambria Math', Cambria, serif; font-size: 10.5pt; background: #f1f1f9; border-radius: 2mm; padding: 1.3mm 3mm; margin: 1.2mm 0; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; } table { border-collapse: collapse; width: 100%; font-size: 8.8pt; } th { background: #2e2a6b; color: #fff; text-align: left; padding: 1.1mm 2mm; } td { padding: 1mm 2mm; border-bottom: 1px solid #d9dcef; } tr:nth-child(even) td { background: #f8f8fd; }
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; } .step { border: 1px solid #d9dcef; border-radius: 3mm; padding: 2.2mm 3mm; background: #fff; break-inside: avoid; } .step b.n { display: inline-block; width: 5.2mm; height: 5.2mm; border-radius: 50%; background: #159a6a; color: #fff; text-align: center; font-size: 8pt; line-height: 5.2mm; margin-right: 1.5mm; } .step p { margin: 1mm 0 0; font-size: 8.8pt; } .step .ex { color: #6250d6; font-size: 8.5pt; margin-top: 1mm; }
code { font-family: Consolas, monospace; font-size: .9em; background: #f1f1f9; padding: 0 .3em; border-radius: 3px; }
"""
P = []
P.append(f'<div class="page p1"><h1>The literature gap, drawn<small>four Lithuanian-related articles, what they leave open, and what this project did</small></h1><svg style="display:none"></svg>{gap_map()}</div>')

P.append(f"""<div class="page p2"><h1>Why does small data hurt?<small>theory first, then our own numbers</small></h1>
<div class="cols top"><div>
<div class="box pl"><b class="h">In plain words</b>A classifier learns from examples. If it has only a few examples it cannot see most words, and a big model can simply memorise the few sentences it has been shown. It then looks perfect on its training data and fails on new sentences. A simple model with few parts (naive Bayes) cannot memorise much, so it is more careful and often wins when data are scarce.</div>
<div class="box mt"><b class="h">The theory (bias and variance)</b>The error of a model on new data has two parts:
<div class="f">expected error ≈ bias² + variance + noise</div>
<b>Bias</b> is error from a model that is too simple. <b>Variance</b> is error from a model that is too flexible: it changes a lot depending on which few examples it saw. A network with millions of free numbers has low bias but <b>high variance</b>, and variance only falls when the number of training examples grows. A naive Bayes model has few numbers and strong assumptions: higher bias, <b>low variance</b>, so it does better with little data. This is exactly the 2019 Lithuanian result (naive Bayes 0.735 against CNN 0.706).</div>
<div class="box mt"><b class="h">What pre-training changes</b>Pre-training spends the “learning words and grammar” cost once, on huge unlabelled text. Fine-tuning then only has to learn the sentiment task, so the variance problem shrinks even with few labels. That is why the pre-trained Transformer in the Lithuanian papers (LitLat BERT, mBERT, T5) works where a scratch network does not.</div>
</div><div>{from_scratch_vs_pretrained()}
<h2 style="margin-top:2mm">Our data, part 1: unseen words</h2>{oov_svg()}<p style="font-size:8pt;color:#6b6b88;margin:0">Share of the running words in our 872 test sentences that never occur in a random training subset (SST-2, seed 42). Computed for this document.</p>
<h2 style="margin-top:2mm">Our data, part 2: accuracy against the number of labels</h2>{curve_svg()}</div></div></div>""")

P.append(f"""<div class="page p3"><h1>What we did, step by step<small>with one real sentence followed through the experiment</small></h1>
<div class="box rw"><b class="h">The running example (true label: negative)</b><b style="font-size:11pt">“a great ensemble cast ca n't lift this heartfelt enterprise out of the familiar .”</b><br>A word-counting model sees <i>great</i> and <i>heartfelt</i> (both positive) and misses that “can't lift … out of” turns the sentence negative. This is the kind of sentence a good model must understand.</div>
<div class="steps">
<div class="step"><p><b class="n">1</b><b>Find the gap</b></p><p>Read the four articles and the Google Scholar and Web of Science results; note what each leaves open.</p><div class="ex">Gap: no joint measure of quality, cost and data size.</div></div>
<div class="step"><p><b class="n">2</b><b>Ask five questions</b></p><p>RQ1 accuracy, RQ2 cost, RQ3 data size, RQ4 ready-made models, RQ5 negation and contrast.</p><div class="ex">Each question has one measurement.</div></div>
<div class="step"><p><b class="n">3</b><b>Choose the data</b></p><p>SST-2 movie reviews: 65,349 training, 2,000 development, 872 test sentences.</p><div class="ex">Our example is one of the 872 test sentences.</div></div>
<div class="step"><p><b class="n">4</b><b>Build ten models</b></p><p>TF-IDF with 3 classifiers, a BiLSTM, 3 Transformers we fine-tune, 3 ready-made ones.</p><div class="ex">Four families, one fair set-up.</div></div>
<div class="step"><p><b class="n">5</b><b>Same rules for all</b></p><p>One laptop CPU (no GPU), one data split, one random seed, the same test set.</p><div class="ex">So differences come from the model, not the set-up.</div></div>
<div class="step"><p><b class="n">6</b><b>Measure quality</b></p><p>Accuracy, macro-F1 and accuracy on negation (184) and contrast (130) sentences.</p><div class="ex">See the table: {n_ok} of 10 models get the example right.</div></div>
<div class="step"><p><b class="n">7</b><b>Measure cost and data size</b></p><p>Training time, speed per sentence, model size, and a learning curve from 500 to 8,000 labels.</p><div class="ex">500 labels: BERT-small 75.9% vs TF-IDF 61.8%.</div></div>
<div class="step"><p><b class="n">8</b><b>Check and conclude</b></p><p>Confidence intervals and McNemar's test; then practical advice for what to use when.</p><div class="ex">Fine-tuned DistilBERT beats TF-IDF, p &lt; 0.001.</div></div>
</div>
<div class="cols" style="margin-top:2.5mm"><div><h2>Step 6 in practice: the ten models on the example sentence</h2><table><thead><tr><th>Model</th><th>Predicts</th><th>Result</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div><h2>What this shows</h2>
<div class="box pl"><b class="h">Why the answers differ</b>All three TF-IDF models and the BiLSTM, which learn only from our labelled sentences, call it positive. The three larger Transformers and the zero-shot model get it right: from pre-training they already know how “can't lift” and “familiar” behave in English. The two small fine-tuned models (BERT-tiny, BERT-small) have less capacity to use that knowledge and also miss it.</div>
<div class="box mt"><b class="h">The same story in numbers</b>Accuracy on negation sentences: TF-IDF + SVM {MOD['tfidf_svm']['acc'] * 100:.1f}% overall but 74.5% on negation; DistilBERT-SST2 {MOD['pt_distilbert_sst2']['acc'] * 100:.1f}% overall and 87.0% on negation. Bag-of-words models lose 8 to 9 points on negation, large Transformers 2 to 4.</div>
<div class="box wn"><b class="h">The price</b>The better models are 20 to 2,000 times slower per sentence and much larger. The gap we filled is exactly this trade-off: quality, cost and data size, measured together.</div></div></div></div>""")

P.append("""<div class="page"><h1>How to write and verify the gap<small>for the literature review and for the viva</small></h1>
<div class="cols"><div>
<h2>Three sentences that state the gap</h2>
<div class="box pl"><b class="h">Template with our content</b>1. <b>What the literature shows:</b> studies of Lithuanian sentiment classification show that pre-trained models (LitLat BERT, multilingual BERT and T5, Lithuanian Llama2) work, and that neural networks without pre-training do not beat classical methods on small data (naive Bayes 0.735 against CNN 0.706).<br>2. <b>What it lacks:</b> these studies report accuracy; they do not measure training and inference cost, model size, or how many labelled examples a pre-trained model needs, and they seldom compare ready-made models with a fine-tuned in-domain model under one protocol.<br>3. <b>What we do:</b> this project compares ten models of four families under identical CPU-only conditions, measuring quality, cost, data efficiency and robustness.</div>
<h2>Why this is a real gap and not just a different question</h2>
<p>A reader who must choose a model for a real task needs three numbers together: how good, how expensive, how much labelled data. The articles give the first. We add the other two on the same data.</p>
</div><div>
<h2>How to find it yourself in Google Scholar and Web of Science</h2>
<table><tbody>
<tr><td><b>1 Search</b></td><td><code>("transformer" OR "BERT" OR "self-attention") AND ("sentiment analysis" OR "text classification")</code>; add <code>AND ("Lithuanian" OR "Baltic")</code> for the local part.</td></tr>
<tr><td><b>2 Filter</b></td><td>Google Scholar: since 2019, sort by relevance. Web of Science: Document type Article, category Computer Science, Artificial Intelligence.</td></tr>
<tr><td><b>3 Screen</b></td><td>Read titles and abstracts; keep papers that report a classifier result on a text task.</td></tr>
<tr><td><b>4 Mine the gap sentences</b></td><td>In each paper's introduction and conclusion look for “however”, “few studies”, “limited”, “future work”. Copy each into a table: author, year, what they did, what they left out.</td></tr>
<tr><td><b>5 Find the pattern</b></td><td>When several papers leave out the same thing, that is the gap.</td></tr>
<tr><td><b>6 Track citations</b></td><td>Use “Cited by” for newer work and the reference list for older work.</td></tr>
<tr><td><b>7 Verify</b></td><td>Open the original of every article you cite and check the numbers (for example 0.735 and 0.706) before they go into the paper.</td></tr></tbody></table>
<div class="box wn"><b class="h">Important</b>The “what each article leaves open” boxes on page 1 are our reading of the articles as summarised in our literature review. Confirm each one against the original text before presenting it as a fact.</div>
</div></div></div>""")

HTML = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>The literature gap, explained</title><style>{CSS}</style></head><body>{''.join(P)}</body></html>"
hp = os.path.join(HERE, "gap.html"); open(hp, "w", encoding="utf-8").write(HTML)
prof = tempfile.mkdtemp(prefix="edge_pdf_"); tmp = os.path.join(tempfile.gettempdir(), "gap_tmp.pdf")
if os.path.exists(tmp): os.remove(tmp)
subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-first-run", "--no-pdf-header-footer", f"--print-to-pdf={tmp}", "--virtual-time-budget=10000", "file:///" + hp.replace("\\", "/")], capture_output=True, text=True, timeout=240)
last = -1
for _ in range(60):
    if os.path.exists(tmp):
        sz = os.path.getsize(tmp)
        if sz > 0 and sz == last: break
        last = sz
    time.sleep(1)
os.replace(tmp, OUT); print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
