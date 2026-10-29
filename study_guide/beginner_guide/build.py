"""Builds docs/Project_Beginner_Guide_End_to_End.pdf: a beginner-friendly study guide for the course project,
with real-world examples, the project end to end, a paper map, a slide-by-slide speaking guide and tips for the professor.
All numbers come from study_guide/data.json (the project's real results).
Run: python study_guide/beginner_guide/build.py   (system Python with pandas; needs Microsoft Edge for HTML -> PDF)"""
import base64, html, json, math, os, subprocess, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT = os.path.join(ROOT, "docs", "Project_Beginner_Guide_End_to_End.pdf")
esc = html.escape
D = json.load(open(os.path.join(ROOT, "study_guide", "data.json"), encoding="utf-8"))
M = D["models"]; TAGS = [m["tag"] for m in M]; MOD = {m["tag"]: m for m in M}; S = D["sentences"]
BLUE, GREEN, RED, AMBER, VIOLET, INK, GREY, ORANGE = "#2a78d6", "#159a6a", "#c0392b", "#c27a10", "#6250d6", "#1d1d3a", "#6b6b88", "#d9541f"
FAMC = {"classical": BLUE, "recurrent": ORANGE, "transformer": GREEN}
pc = lambda v, d=1: f"{v * 100:.{d}f}%"


def sub_acc(i, f):
    rows = [s for s in S if f(s)]; return sum(1 for s in rows if s["p"][i] == str(s["y"])) / len(rows)


for i, m in enumerate(M):
    m["neg"] = sub_acc(i, lambda s: s["n"]); m["con"] = sub_acc(i, lambda s: s["c"]); m["ci"] = (D["stats"][m["tag"]]["lo"], D["stats"][m["tag"]]["hi"])
A = lambda t: MOD[t]["acc"]
dropv = lambda t: round(MOD[t]["acc"] * 100, 1) - round(MOD[t]["neg"] * 100, 1)
CT, CC = D["curve"]["transformer"], D["curve"]["tfidf"]
SHORT = {"tfidf_nb": "TF-IDF + naive Bayes", "tfidf_logreg": "TF-IDF + logistic reg.", "tfidf_svm": "TF-IDF + linear SVM", "bilstm": "BiLSTM (scratch)", "ft_bert-tiny_8000": "BERT-tiny (fine-tuned)", "ft_bert-small_8000": "BERT-small (fine-tuned)", "ft_distilbert_8000": "DistilBERT (fine-tuned)", "pt_distilbert_sst2": "DistilBERT-SST2 (ready)", "pt_twitter_roberta": "Twitter-RoBERTa (ready)", "zs_nli_distilroberta": "Zero-shot NLI"}
fmt_t = lambda s: "none" if s < 1 else f"{s:.0f} s" if s < 60 else f"{s / 60:.0f} min" if s < 3600 else f"{s / 3600:.1f} h"
fmt_p = lambda p: f"{p / 1e6:.1f} M" if p >= 1e6 else f"{p / 1e3:.0f} k"


# ------------------------------------------------------------------ figures
def bars_svg():
    order = sorted(M, key=lambda m: -m["acc"]); W, H = 560, 26 * len(order) + 24
    s = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif">'
    for i, m in enumerate(order):
        y = 8 + i * 26; x0 = 190; w = (m["acc"] - 0.5) / 0.45 * 300
        lo, hi = (m["ci"][0] - 0.5) / 0.45 * 300, (m["ci"][1] - 0.5) / 0.45 * 300
        s += f'<text x="{x0 - 8}" y="{y + 14}" text-anchor="end" font-size="11.5" fill="{INK}">{SHORT[m["tag"]]}</text><rect x="{x0}" y="{y}" width="{w}" height="18" rx="4" fill="{FAMC[m["family"]]}" opacity=".88"/><rect x="{x0 + lo}" y="{y + 7}" width="{hi - lo}" height="4" fill="{INK}" opacity=".55"/><text x="{x0 + w + 6}" y="{y + 14}" font-size="11.5" font-weight="800" fill="{INK}">{m["acc"] * 100:.1f}%</text>'
    s += f'<text x="190" y="{H - 2}" font-size="9.5" fill="{GREY}">dark whisker = 95% confidence interval; axis starts at 50%</text>'
    return s + "</svg>"


def scatter_svg():
    W, H, L, R, T, B = 560, 300, 50, 14, 14, 40
    X = lambda ms: L + (math.log10(ms) - math.log10(0.03)) / (math.log10(150) - math.log10(0.03)) * (W - L - R)
    Y = lambda a: T + (1 - (a - 0.74) / 0.20) * (H - T - B)
    s = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif"><g stroke="#d9dcef"><line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}"/><line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}"/></g>'
    for a in (0.76, 0.8, 0.84, 0.88, 0.92): s += f'<text x="6" y="{Y(a) + 4}" font-size="10.5" fill="{GREY}">{round(a * 100)}%</text><line x1="{L}" x2="{W - R}" y1="{Y(a)}" y2="{Y(a)}" stroke="#d9dcef" stroke-dasharray="2 4"/>'
    for ms in (0.1, 1, 10, 100): s += f'<text x="{X(ms)}" y="{H - 24}" font-size="10.5" text-anchor="middle" fill="{GREY}">{ms:g} ms</text>'
    offs = {"tfidf_nb": (10, 4, "start"), "tfidf_logreg": (10, 18, "start"), "tfidf_svm": (10, -9, "start"), "bilstm": (0, 19, "middle"), "pt_twitter_roberta": (-9, -8, "end"), "zs_nli_distilroberta": (-9, 16, "end"), "pt_distilbert_sst2": (-9, -10, "end"), "ft_distilbert_8000": (-9, 17, "end"), "ft_bert-small_8000": (-9, -9, "end"), "ft_bert-tiny_8000": (9, 5, "start")}
    for m in M:
        x, y = X(m["lat_ms"]), Y(m["acc"]); dx, dy, an = offs[m["tag"]]
        s += f'<circle cx="{x}" cy="{y}" r="6" fill="{FAMC[m["family"]]}"/><text x="{x + dx}" y="{y + dy}" font-size="10" text-anchor="{an}" fill="{INK}">{SHORT[m["tag"]]}</text>'
    s += f'<text x="{(L + W) / 2}" y="{H - 6}" font-size="11" text-anchor="middle" fill="{GREY}">time to classify one sentence on the laptop CPU (log scale; further right = slower)</text><text x="14" y="{T + 6}" font-size="11" fill="{GREY}" transform="rotate(-90 14 {T + 100})">accuracy</text>'
    s += f'<text x="{W - 20}" y="{T + 16}" text-anchor="end" font-size="11" font-weight="800" fill="{GREEN}">better ↑</text><text x="{L + 8}" y="{H - 48}" font-size="11" font-weight="800" fill="{GREEN}">← cheaper</text>'
    return s + "</svg>"


def curve_svg():
    W, H, L, R, T, B = 540, 270, 44, 14, 20, 38; xs = [500, 1000, 2000, 4000, 8000]
    X = lambda n: L + (math.log(n) - math.log(500)) / (math.log(8000) - math.log(500)) * (W - L - R); Y = lambda v: T + (1 - (v - 0.55) / 0.35) * (H - T - B)
    s = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif"><g stroke="#d9dcef"><line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}"/><line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}"/></g>'
    for v in (0.6, 0.7, 0.8, 0.9): s += f'<text x="6" y="{Y(v) + 4}" font-size="10.5" fill="{GREY}">{round(v * 100)}%</text><line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="#d9dcef" stroke-dasharray="2 4"/>'
    for n in xs: s += f'<text x="{X(n)}" y="{H - 22}" font-size="10.5" text-anchor="middle" fill="{GREY}">{n}</text>'
    for data, c, nm, dy in [(CT, GREEN, "BERT-small (pre-trained)", -9), (CC, BLUE, "TF-IDF + logistic regression", 17)]:
        s += f'<polyline fill="none" stroke="{c}" stroke-width="3" points="{" ".join(f"{X(n)},{Y(v)}" for n, v in data)}"/>' + "".join(f'<circle cx="{X(n)}" cy="{Y(v)}" r="4.5" fill="{c}"/><text x="{X(n)}" y="{Y(v) + dy}" font-size="10" text-anchor="middle" fill="{c}" font-weight="700">{v * 100:.1f}</text>' for n, v in data)
    s += f'<text x="{L + 8}" y="{T - 5}" font-size="11.5" font-weight="800" fill="{GREEN}">BERT-small (pre-trained)</text><text x="{L + 190}" y="{T - 5}" font-size="11.5" font-weight="800" fill="{BLUE}">TF-IDF + logistic regression</text><text x="{(L + W) / 2}" y="{H - 4}" font-size="11" text-anchor="middle" fill="{GREY}">labelled training sentences (log scale)</text>'
    return s + "</svg>"


def drop_svg():
    tags = ["tfidf_svm", "bilstm", "ft_bert-small_8000", "ft_distilbert_8000", "pt_distilbert_sst2"]; W, H = 560, 36 * len(tags) + 30
    s = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif">'
    for i, t in enumerate(tags):
        m = MOD[t]; y = 8 + i * 36; x0 = 170; f = lambda v: (v - 0.6) / 0.35 * 280
        s += f'<text x="{x0 - 8}" y="{y + 12}" text-anchor="end" font-size="11.5" fill="{INK}">{SHORT[t]}</text><rect x="{x0}" y="{y}" width="{f(m["acc"])}" height="13" rx="3" fill="{FAMC[m["family"]]}" opacity=".9"/><text x="{x0 + f(m["acc"]) + 5}" y="{y + 11}" font-size="10.5" fill="{INK}">overall {m["acc"] * 100:.1f}%</text>'
        s += f'<rect x="{x0}" y="{y + 15}" width="{f(m["neg"])}" height="13" rx="3" fill="{RED}" opacity=".78"/><text x="{x0 + f(m["neg"]) + 5}" y="{y + 26}" font-size="10.5" fill="{INK}">with negation {m["neg"] * 100:.1f}%  (drop {dropv(t):.1f})</text>'
    return s + f'<text x="170" y="{H - 4}" font-size="9.5" fill="{GREY}">axis starts at 60%</text></svg>'


def attention_svg():
    w = next(e for e in D["walk"] if e["i"] == -1); L_, H_, n, _ = w["shape"]; raw = base64.b64decode(w["att"]); toks = w["tokens"]
    mat = [[sum(raw[((L_ - 1) * H_ + h) * n * n + i * n + j] for h in range(H_)) / H_ / 255 for j in range(n)] for i in range(n)]
    cell, lab = 30, 78; Wd = lab + n * cell + 8; Ht = 70 + n * cell
    s = f'<svg viewBox="0 0 {Wd} {Ht}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif">'
    for j, t in enumerate(toks): s += f'<text transform="translate({lab + j * cell + cell / 2 + 3},62) rotate(-55)" font-size="10.5" fill="{INK}">{esc(t)}</text>'
    for i, t in enumerate(toks):
        s += f'<text x="{lab - 5}" y="{70 + i * cell + cell / 2 + 4}" text-anchor="end" font-size="10.5" fill="{INK}">{esc(t)}</text>'
        for j in range(n):
            v = mat[i][j]; s += f'<rect x="{lab + j * cell}" y="{70 + i * cell}" width="{cell - 1}" height="{cell - 1}" rx="3" fill="rgba(98,80,214,{min(1, v * 1.5) * .92:.3f})"/>' + (f'<text x="{lab + j * cell + cell / 2 - .5}" y="{70 + i * cell + cell / 2 + 3}" text-anchor="middle" font-size="8.5" fill="{"#fff" if v > .3 else INK}">{round(v * 100)}</text>' if v >= .1 else "")
    return s + "</svg>"


def flow_svg():
    steps = [("1 Find the gap", "read articles", BLUE), ("2 Questions", "five RQs", BLUE), ("3 Data", "SST-2", BLUE), ("4 Ten models", "four families", ORANGE), ("5 Measure", "accuracy, cost, data", ORANGE), ("6 Check", "statistics", ORANGE), ("7 Explain", "paper, slides, guides", GREEN)]
    s = f'<svg viewBox="0 0 800 110" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, sans-serif"><defs><marker id="a" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto"><path d="M0 0L11 5.5L0 11z" fill="{GREY}"/></marker></defs>'
    for i, (a, b, c) in enumerate(steps):
        x = 6 + i * 113; s += f'<rect x="{x}" y="22" width="102" height="62" rx="10" fill="{c}"/><text x="{x + 51}" y="48" text-anchor="middle" font-size="12.5" font-weight="800" fill="#fff">{a}</text><text x="{x + 51}" y="68" text-anchor="middle" font-size="10.5" fill="#fff">{b}</text>'
        if i < 6: s += f'<path d="M{x + 102} 53 L{x + 111} 53" stroke="{GREY}" stroke-width="2" marker-end="url(#a)"/>'
    s += f'<text x="6" y="14" font-size="11" font-weight="800" fill="{BLUE}">PLAN</text><text x="344" y="14" font-size="11" font-weight="800" fill="{ORANGE}">BUILD AND MEASURE</text><text x="682" y="14" font-size="11" font-weight="800" fill="{GREEN}">DELIVER</text>'
    return s + "</svg>"


# ------------------------------------------------------------------ the worked example sentence
EX = next(s for s in S if s["s"].startswith("a great ensemble cast ca n't lift"))
ex_rows = "".join(f"<tr><td>{SHORT[t]}</td><td>{'positive' if int(EX['p'][TAGS.index(t)]) else 'negative'}</td><td style='color:{GREEN if int(EX['p'][TAGS.index(t)]) == EX['y'] else RED};font-weight:800'>{'✓' if int(EX['p'][TAGS.index(t)]) == EX['y'] else '✗'}</td></tr>" for t in TAGS)

# ------------------------------------------------------------------ small html helpers
def card(kind, head, body): return f'<div class="box {kind}"><b class="h">{head}</b>{body}</div>'
def concept(title, plain, real, proj, extra=""): return f'<h3>{title}</h3>' + card("pl", "In plain words", plain) + card("rw", "Real-world example", real) + card("pj", "Where it appears in our project", proj) + extra
def tbl(head, rows, cls=""): return f'<table class="{cls}"><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr></thead><tbody>{"".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)}</tbody></table>'


model_rows = [[f"<span class='dot' style='background:{FAMC[m['family']]}'></span>{SHORT[m['tag']]}", m["family"].capitalize(), fmt_p(m["params"]), ("0" if m["train_size"] == 0 else f"{m['train_size']:,}"), pc(m["acc"]), fmt_t(m["train_s"]), f"{m['lat_ms']:.2f} ms"] for m in M]
ci_rows = [[SHORT[m["tag"]], pc(m["acc"]), f"{m['ci'][0] * 100:.1f} to {m['ci'][1] * 100:.1f}", pc(m["neg"]), pc(m["con"])] for m in sorted(M, key=lambda m: -m["acc"])]

P = []
# ============================================================ cover
P.append(f"""<div class="page cover"><div class="tag">Beginner's study guide · end to end</div><h1>Transformers in NLP:<br>Understand the Project, Present It with Confidence</h1>
<h2>From “what is sentiment analysis?” to “how do I answer the professor?”, with real-world examples, the real numbers of our experiment, a paper map, a slide-by-slide speaking guide and tips for the questions.</h2>
<div class="ex"><b>The project in one sentence</b><br>We compared ten ways of classifying movie-review sentences as positive or negative, from simple word counting to Transformers, on one ordinary laptop, and measured not only <i>how accurate</i> each is but also <i>how much it costs</i> and <i>how many labelled examples it needs</i>.</div>
<p style="margin-top:9mm;color:#dcdcf6;max-width:150mm">Team: Shashikanth Hosur Ramegowda and Shaeed Khan · VILNIUS TECH, Faculty of Fundamental Sciences, Engineering of Artificial Intelligence</p>
<div class="foot">All numbers come from the project's real results (SST-2 test set, 872 sentences, one laptop CPU). Companion material: paper (25 pages), presentation (11 slides), interactive study guide, project map with 54 interview questions.</div></div>""")

# ============================================================ how to use + one page overview
P.append(f"""<div class="page"><h1 class="t">How to use this guide</h1>
<div class="cols" style="grid-template-columns:repeat(3,1fr)"><div class="card2"><b>If you have 1 day</b><p class="sm">Read Part 2 (the project end to end) and Part 4 (explaining to the professor). Skim the concept boxes you do not know. Rehearse the slides once with a timer.</p></div>
<div class="card2"><b>If you have 3 days</b><p class="sm">Day 1: Part 1 concepts. Day 2: Part 2 and the paper map. Day 3: slide guide, questions, two full rehearsals.</p></div>
<div class="card2"><b>If you have a week</b><p class="sm">Do the 3-day plan, then add the interactive study guide (test your own sentences on all ten models), the project map quiz mode, and a mock viva with a friend.</p></div></div>
<h2>Every concept box has three parts</h2>
{card("pl", "In plain words", "The idea with no jargon, as you would explain it to a friend.")}{card("rw", "Real-world example", "Something from daily life (food-delivery reviews, a bus pass counter, cricket scores) that makes the idea stick.")}{card("pj", "Where it appears in our project", "Exactly where this idea shows up in our experiment, paper or slides, with the real numbers.")}
<h2>The project on one page</h2>
<figure>{flow_svg()}</figure>
{tbl(["Question we asked", "How we answered it", "Short answer"], [
 ["Do Transformers really beat older methods?", "Ten models, same data, same laptop", f"Yes: fine-tuned DistilBERT {pc(A('ft_distilbert_8000'))} vs best classical {pc(A('tfidf_svm'))}"],
 ["What does the improvement cost?", "Measured training time, speed, size", f"20 to 2,000 times slower per sentence; DistilBERT needs {fmt_t(MOD['ft_distilbert_8000']['train_s'])} to fine-tune on the CPU"],
 ["How much labelled data is needed?", "Learning curve, 500 to 8,000 labels", f"With 500 labels BERT-small {pc(CT[0][1])} vs TF-IDF {pc(CC[0][1])}"],
 ["Is reusing a public model enough?", "Three ready-made models", f"Best: {pc(A('pt_distilbert_sst2'))} with no training at all"],
 ["Are Transformers better on tricky sentences?", "Negation and contrast subsets", f"TF-IDF + SVM drops {dropv('tfidf_svm'):.1f} points on negation, DistilBERT-SST2 {dropv('pt_distilbert_sst2'):.1f}"]])}
<div class="box wn"><b class="h">The motto of the project</b>Explain it. Measure it. Make it useful for everyone. We explain how Transformers work, we measure honestly when they are worth their cost, and we turn what we learned into material others can use.</div></div>""")

# ============================================================ PART 1 concepts
P.append(f"""<div class="page"><div class="part">PART 1 · CONCEPTS FROM ZERO</div><h1 class="t">1. The problem: sentiment analysis</h1>
{concept("What is sentiment analysis?",
 "Teaching a computer to decide whether a piece of text is positive or negative. It is a <b>classification</b> task: the answer is one of a few labels.",
 "A restaurant on a food-delivery app gets 5,000 reviews a month. Nobody can read them all. A sentiment model reads every review and reports “78% positive, 22% negative”, and flags the negative ones about delivery delays.",
 "Our task is exactly this, with movie-review sentences. Each sentence is labelled <b>1 (positive)</b> or <b>0 (negative)</b>. The data set is called <b>SST-2</b> (Stanford Sentiment Treebank, two classes).")}
<h3>Why is it harder than it looks?</h3>
{tbl(["Sentence", "A word-counting model thinks", "The truth", "Why it is hard"], [
 ["The biryani was great.", "positive", "positive", "easy: one positive word"],
 ["The biryani was <b>not</b> great.", "positive (sees “great”)", "negative", "<b>negation</b> flips the meaning"],
 ["The food was good, <b>but</b> the delivery was terribly late.", "mixed", "negative overall", "<b>contrast</b>: the part after “but” decides"],
 ["Oh great, another two-hour wait.", "positive (sees “great”)", "negative", "<b>sarcasm</b>: needs real understanding"]])}
<div class="box pj"><b class="h">Where it appears in our project</b>We specifically measure accuracy on <b>184 test sentences with negation</b> (not, no, never, n't …) and <b>130 with contrast</b> (but, although, though, yet, however). This is research question RQ5.</div>
<h3>The data: SST-2</h3>
<p>Movie-review sentences from Rotten Tomatoes. Think of three kinds of exam papers:</p>
{tbl(["Part", "Size", "Real-world picture", "Used for"], [["Training set", "65,349 sentences", "practice papers", "the model learns from these"], ["Development set", "2,000 sentences", "mock exam", "choosing the best epoch (when to stop)"], ["Test set", "872 sentences", "the real exam: questions never seen", "the final honest score; never used for learning"]])}
<p class="sm">Training sentences are short phrases (about 9.4 words on average); test sentences are full sentences (about 19.5 words). So models must generalise from phrases to whole sentences, which makes the task a fair challenge.</p></div>""")

P.append(f"""<div class="page"><div class="part">PART 1 · CONCEPTS FROM ZERO</div><h1 class="t">2. From text to numbers, and the classical methods</h1>
{concept("Tokens, vocabulary and ids",
 "A computer cannot read letters. We cut a sentence into pieces called <b>tokens</b> (mostly words) and give every token a number from a fixed list called the <b>vocabulary</b>.",
 "A bus-pass counter writes every stop name in a register with a serial number: Majestic = 1, Mysuru Road = 2 … Staff then talk in numbers. Words that are not in the register are “unknown”.",
 "Classical models use words and word pairs as features. The Transformers use <b>WordPiece</b> pieces: “unwatchable” becomes <i>un ##watch ##able</i>, so even unseen words can be understood from their pieces.")}
{concept("Bag of words and TF-IDF",
 "<b>Bag of words</b> counts how often each word occurs and forgets the order. <b>TF-IDF</b> improves this: common words like “the” get a low weight, rare informative words get a high weight.",
 "Three delivery reviews: (1) “biryani was great”, (2) “delivery was late”, (3) “the delivery was great”. The word “was” appears in all three, so it tells us nothing. “biryani” appears once, so it is informative.",
 "All three of our classical models use TF-IDF features (about 73,000 word and word-pair weights).",
 extra=tbl(["Word", "Reviews containing it (of 3)", "IDF = ln((1+3)/(1+df)) + 1", "Meaning"], [["was", "3", "1.00", "common: low weight"], ["delivery", "2", "1.29", "medium"], ["biryani", "1", "1.69", "rare: high weight"]]))}
{concept("Naive Bayes, logistic regression, linear SVM",
 "Three simple ways to turn word weights into a decision. Each word votes a little for “positive” or “negative”, and the votes are added up.",
 "A panel of judges where each judge only knows a few words: “great” gets +2 votes, “late” gets −2. The sum decides. Quick and cheap, but the judges ignore how words combine.",
 f"Our best classical model, TF-IDF + linear SVM, reaches <b>{pc(A('tfidf_svm'))}</b>, trains in 2 seconds and classifies a sentence in 0.06 ms. It is the strong, cheap baseline every other model must beat.",
 extra=card("wn", "Their weakness", "Because they ignore word order, “not great” still contains “great”. On our negation sentences the SVM drops from 82.9% to 74.5%."))}
</div>""")

P.append(f"""<div class="page"><div class="part">PART 1 · CONCEPTS FROM ZERO</div><h1 class="t">3. Neural networks, reading in order, and the Transformer</h1>
{concept("How a neural network learns",
 "A network is a long chain of simple calculations with many adjustable numbers (<b>weights</b>). Learning is a loop: guess, measure how wrong the guess was (<b>loss</b>), work out which weights to change (<b>backpropagation</b>), change them a little (<b>gradient descent</b>), repeat thousands of times.",
 "A cricket coach tells a batter: “your shot was too early”. The batter adjusts a little and tries again. After thousands of balls the timing improves. The weights are the batter's habits; the loss is the coach's complaint.",
 "We fine-tune three Transformers with an optimiser called <b>AdamW</b> and choose the best epoch on the development set.")}
{concept("Reading in order: RNN, LSTM, BiLSTM",
 "A recurrent network reads one word at a time and carries a memory forward. An <b>LSTM</b> adds gates so it can keep important things longer. A <b>BiLSTM</b> reads the sentence both forwards and backwards.",
 "Reading a long WhatsApp message aloud and trying to remember its start by the time you reach the end. The longer it is, the more you forget, and you cannot start the last line before finishing the first.",
 f"Our BiLSTM (2.0 M parameters, trained from scratch) reaches {pc(A('bilstm'))}, almost the same as the SVM ({pc(A('tfidf_svm'))}). It is better on negation ({pc(MOD['bilstm']['neg'])} vs {pc(MOD['tfidf_svm']['neg'])}) because it reads word order, but without pre-training it brings no overall gain.")}
{concept("The Transformer and self-attention",
 "The Transformer (Vaswani et al., 2017) does not read word by word. Every word looks at <b>all</b> the other words at once and decides which ones matter for understanding it. This “looking” is called <b>self-attention</b>. Because all words are processed together, training is fast and the model can be made very large.",
 "A group discussion: when one person speaks, everyone listens, but they pay most attention to the people whose points relate to theirs. In “the plot was <b>not good</b>, but the acting was <b>brilliant</b>”, the word “good” pays attention to “not” and the word “brilliant” attends to “acting”.",
 "Three of our models are Transformer <i>encoders</i> (BERT-tiny, BERT-small, DistilBERT). For classification we take the special <b>[CLS]</b> vector, which summarises the whole sentence, and add one layer that outputs positive or negative. The formula on slide 3 is <b>softmax(QKᵀ/√d)·V</b>: Q = what each word is looking for, K = what each word offers, V = the content it passes on.")}
</div>""")

P.append(f"""<div class="page"><div class="part">PART 1 · CONCEPTS FROM ZERO</div><h1 class="t">4. What attention looks like, and what to say about it</h1>
<p>This map shows the <b>real attention</b> of the DistilBERT model (last layer, averaged over its heads) for the sentence “the plot was not good, but the acting was brilliant”. Each row is a word that is looking; each column is a word it looks at; numbers are percent and each row adds up to 100.</p>
<figure class="att">{attention_svg()}<figcaption>Real attention from DistilBERT-SST2 (the same sentence as slide 10). Cell darkness = share of attention.</figcaption></figure>
<div class="box wn"><b class="h">How to talk about this figure honestly (important in the viva)</b>Notice that a lot of attention goes to the comma and the [SEP] marker, which act as “rest” positions, and that some words attend to their neighbours (for example “acting” and “brilliant”). Say: “The map shows where information <i>can</i> flow between words.” Do <b>not</b> say: “this proves the model understands negation”. Jain and Wallace (2019) warned that attention weights are an illustration, not a proof of reasoning. Our slide uses the same wording on purpose.</div>
{concept("Pre-training, fine-tuning, zero-shot and ready-made models",
 "<b>Pre-training</b>: the model reads a huge amount of text once (for BERT, by guessing hidden words) and learns English in general. <b>Fine-tuning</b>: we then train it a little on our labelled sentences so it learns our task. <b>Ready-made</b>: someone else has already fine-tuned a model and we simply use it. <b>Zero-shot</b>: a model never trained for sentiment is asked to decide anyway.",
 "A fresh graduate who has studied for years (pre-training) joins a company and learns its processes in a few weeks (fine-tuning). A ready-made model is an experienced colleague you borrow. A zero-shot model is a smart person doing a task for the first time with only the instructions.",
 f"Fine-tuned by us on 8,000 sentences: BERT-tiny, BERT-small, DistilBERT. Ready-made: DistilBERT-SST2 ({pc(A('pt_distilbert_sst2'))}), Twitter-RoBERTa ({pc(A('pt_twitter_roberta'))}, a different domain), and a zero-shot NLI model ({pc(A('zs_nli_distilroberta'))} with <b>no</b> labelled examples).")}
</div>""")

P.append(f"""<div class="page"><div class="part">PART 1 · CONCEPTS FROM ZERO</div><h1 class="t">5. How to judge models: the measurements</h1>
{tbl(["Measurement", "Plain meaning", "Real-world picture", "Our use"], [
 ["Accuracy", "share of test sentences classified correctly", "marks in the exam out of 872", "main quality number"],
 ["Macro-F1", "accuracy that treats both classes equally", "marks that cannot be gamed by always answering the common class", "checked alongside accuracy (nearly identical here)"],
 ["Training time", "seconds or hours to build the model", "how long the student studies", "RQ2: cost"],
 ["Latency", "milliseconds to classify one sentence", "how long the counter clerk takes per customer", "RQ2: cost of using the model"],
 ["Parameters", "number of adjustable numbers", "size of the clerk's rulebook", "memory and size"],
 ["Negation / contrast accuracy", "accuracy only on the tricky sentences", "marks on the hardest questions", "RQ5"],
 ["Learning curve", "accuracy as the number of labels grows", "marks after 10, 50, 100 practice papers", "RQ3: data efficiency"]])}
{concept("Confidence interval and McNemar's test",
 "A test set is only a sample. If we had picked 872 different sentences the accuracy would be slightly different. The <b>95% confidence interval</b> is a range that shows how much it could wobble. <b>McNemar's test</b> compares two models on the <i>same</i> sentences and asks: is one really better, or is the difference luck?",
 "Two batters average 41 and 43 over 10 matches. Is the second one truly better? With so few matches you cannot tell. Over 200 matches you could. The interval is the batting average with an honest “plus or minus”.",
 f"We resampled the test set many times (bootstrap) to get an interval for every model; for example DistilBERT-SST2 {pc(A('pt_distilbert_sst2'))}, interval {MOD['pt_distilbert_sst2']['ci'][0] * 100:.1f} to {MOD['pt_distilbert_sst2']['ci'][1] * 100:.1f}. McNemar's test shows our fine-tuned DistilBERT is significantly better than the best classical model (p &lt; 0.001).")}
<div class="box wn"><b class="h">Rule of thumb for 872 test sentences</b>The wobble of an accuracy near 85% is about ±2.4 points. Treat differences under 2 to 3 points between single models as <i>not clearly real</i>; larger gaps (88.8% vs 82.9%) are clearly real.</div></div>""")

# ============================================================ PART 2 project end to end
P.append(f"""<div class="page"><div class="part">PART 2 · THE PROJECT END TO END</div><h1 class="t">6. The gap and the five research questions</h1>
<p><b>The gap in one paragraph.</b> Studies of Lithuanian sentiment classification show that pre-trained models (LitLat BERT, multilingual BERT and T5, Lithuanian Llama2) work, and that neural networks without pre-training do not beat classical methods on small data (naive Bayes 0.735 against CNN 0.706). But they report accuracy; they seldom measure cost (time, speed, size), how many labels a pre-trained model needs, or compare ready-made models with a fine-tuned in-domain model under one protocol. Our project measures all of these together.</p>
<div class="box rw"><b class="h">Real-world example of the gap</b>A small company in Vilnius must choose a sentiment model for customer reviews and has one laptop. The papers tell it “BERT works”. They do not tell it how long it will take on a laptop, how many reviews it must label first, or whether it can just download a ready-made model. We answer those questions.</div>
{tbl(["RQ", "Question", "What we measured", "Where in the paper"], [
 ["RQ1", "How much more accurate are Transformers?", "accuracy and macro-F1 of ten models", "Chapter 3, quality section"],
 ["RQ2", "What does the improvement cost?", "training time, latency, parameters", "Chapter 3, cost section"],
 ["RQ3", "How does the advantage depend on labels?", "learning curve 500 to 8,000 labels", "Chapter 3, data efficiency"],
 ["RQ4", "Ready-made vs trained by us?", "three ready-made models vs fine-tuned ones", "Chapter 3, ready-made models"],
 ["RQ5", "Better on negation and contrast?", "accuracy on 184 + 130 tricky sentences", "Chapter 3, linguistic subsets"]])}
<h2>The ten competitors</h2>
{tbl(["Model", "Family", "Parameters", "Training labels", "Accuracy", "Training time", "Latency"], model_rows)}
<p class="sm">Parameters for the TF-IDF models are the number of word weights (73 k to 147 k), not a neural network. Latency = time to classify one sentence (batches of 32).</p></div>""")

P.append(f"""<div class="page"><div class="part">PART 2 · THE PROJECT END TO END</div><h1 class="t">7. The pipeline: what each script does</h1>
<p>The program is nine small Python scripts that pass files to each other. You can explain it as “data in, ten models, results out, report”.</p>
{tbl(["Script", "What it does (plain words)", "What it produces"], [
 ["01_prepare_data.py", "Downloads SST-2 and splits it into training, development and test sentences", "train.csv, dev.csv, test.csv"],
 ["02_classical_baselines.py", "Turns sentences into TF-IDF numbers and trains naive Bayes, logistic regression, linear SVM", "3 result files"],
 ["03_bilstm.py", "Builds a vocabulary and trains the BiLSTM from scratch", "1 result file"],
 ["04_finetune_transformer.py", "Takes any pre-trained model name and a number of labels, adds a classification layer and fine-tunes it (AdamW, warm-up, linear decay)", "BERT-tiny, BERT-small, DistilBERT results + learning-curve runs"],
 ["05_pretrained_transformers.py", "Runs the three ready-made models on the test sentences with no training", "3 result files"],
 ["06_learning_curve_classical.py", "Repeats the 500 to 8,000 label experiment for TF-IDF", "learning_curve.csv"],
 ["07_attention_visualisation.py", "Extracts attention weights of a fine-tuned model for an example sentence", "attention figure"],
 ["08_report.py", "Collects all result files; makes tables, figures, subset analysis, confidence intervals and McNemar tests", "tables and charts for the paper"],
 ["09_diagrams.py", "Draws the pipeline and architecture diagrams", "figures"]])}
<div class="box pl"><b class="h">Why this design is good (say it to the professor)</b>Every model writes the <b>same kind of result file</b> including its individual predictions. That makes the study reproducible and lets us compute paired statistics afterwards without retraining. A single command, <code>run_all.sh</code>, reproduces everything in about 2.5 to 3 hours on the laptop.</div>
<h2>The set-up that makes the comparison fair</h2>
{tbl(["Fixed for all models", "Value"], [["Computer", "Intel Core i5-8365U laptop, 4 cores, 16 GB RAM, <b>no GPU</b>"], ["Data split", "same train / development / test sentences for every model"], ["Random seed", "42 for splits, sampling and initialisation"], ["Fine-tuning labels", "8,000 sentences for each of the three fine-tuned Transformers"], ["Test set", "the 872 official validation sentences, never used for learning or choosing settings"]])}
<div class="box rw"><b class="h">Real-world example</b>A fair race: same track, same day, same stopwatch. If one runner had a bicycle the times would mean nothing. Running everything on one CPU and one split is our “same track”.</div></div>""")

top3 = sorted(M, key=lambda m: -m["acc"])[:3]
P.append(f"""<div class="page"><div class="part">PART 2 · THE PROJECT END TO END</div><h1 class="t">8. The results, one question at a time</h1>
<h2>RQ1 · Accuracy</h2>
<figure>{bars_svg()}</figure>
<p><b>Plain meaning:</b> the ready-made DistilBERT is best ({pc(A('pt_distilbert_sst2'))}). Our DistilBERT, fine-tuned on only 8,000 sentences, reaches {pc(A('ft_distilbert_8000'))}, which is {(A('ft_distilbert_8000') - A('tfidf_svm')) * 100:.1f} points above the best classical model although that model saw 65,349 sentences. The BiLSTM ({pc(A('bilstm'))}) gives no advantage over the SVM. Tiny Transformers are not enough: BERT-tiny is the weakest model ({pc(A('ft_bert-tiny_8000'))}).</p>
<div class="box pj"><b class="h">What to say</b>“A Transformer fine-tuned on eight times less data beat the best classical model, and the difference is statistically significant. Pre-training is the reason.”</div>
<h2>RQ2 · Cost: the accuracy–speed trade-off</h2>
<figure>{scatter_svg()}</figure>
<p><b>Plain meaning:</b> the best models are on the right (slow). TF-IDF answers in 0.06 ms; DistilBERT in about 39 to 45 ms; the zero-shot NLI model in 107 ms. That is 20 to 2,000 times slower. Fine-tuning DistilBERT on the CPU took {fmt_t(MOD['ft_distilbert_8000']['train_s'])}; the SVM took 2 seconds. For most real uses, 40 ms per sentence is still perfectly fast, so the extra cost is usually acceptable.</p></div>""")

P.append(f"""<div class="page"><div class="part">PART 2 · THE PROJECT END TO END</div><h1 class="t">8. The results (continued)</h1>
<h2>RQ3 · Data efficiency</h2>
<figure>{curve_svg()}</figure>
<p><b>Plain meaning:</b> with only 500 labelled sentences BERT-small scores {pc(CT[0][1])} against {pc(CC[0][1])} for TF-IDF: <b>+{(CT[0][1] - CC[0][1]) * 100:.1f} points</b>. With 4,000 labels it already matches the classical model trained on 65,349 (82.7% vs 82.9%). The gap shrinks as labels grow, but it never disappears in the range we tested. <b>Why:</b> BERT-small already knows English from pre-training; TF-IDF must learn every word from our labels (with 500 labels, about 29% of the words in the test sentences were never seen).</p>
<h2>RQ4 · Ready-made models</h2>
<p><b>Plain meaning:</b> the cheapest way to a good model is to reuse one. DistilBERT-SST2 gives {pc(A('pt_distilbert_sst2'))} with zero training. Twitter-RoBERTa ({pc(A('pt_twitter_roberta'))}), trained on tweets, still works well on movie reviews but loses a little because of the domain change. The zero-shot NLI model ({pc(A('zs_nli_distilroberta'))}) uses <b>no</b> labelled data at all and still matches classical models, though it is the slowest.</p>
<h2>RQ5 · Negation and contrast</h2>
<figure>{drop_svg()}</figure>
<p><b>Plain meaning:</b> bag-of-words models lose 8 to 9 points on sentences with negation; the large Transformers lose only 2 to 4 (DistilBERT-SST2: {pc(A('pt_distilbert_sst2'))} overall, {pc(MOD['pt_distilbert_sst2']['neg'])} on negation; SVM: {pc(A('tfidf_svm'))} and {pc(MOD['tfidf_svm']['neg'])}). Self-attention lets “not” modify “good” even when the words are far apart.</p>
<div class="box wn"><b class="h">Honest note</b>The small fine-tuned models (BERT-tiny, BERT-small) are <i>not</i> better on tricky sentences. The advantage on negation belongs to the larger Transformers.</div></div>""")

P.append(f"""<div class="page"><div class="part">PART 2 · THE PROJECT END TO END</div><h1 class="t">9. One sentence through all ten models</h1>
<div class="box rw"><b class="h">The sentence (true label: negative)</b><b style="font-size:11pt">“a great ensemble cast ca n't lift this heartfelt enterprise out of the familiar .”</b><br>A word-counting model sees <i>great</i> and <i>heartfelt</i> (positive words) and misses that “can't lift … out of the familiar” makes it negative.</div>
<div class="cols"><div>{tbl(["Model", "Predicts", ""], [[r for r in row] for row in [[SHORT[t], 'positive' if int(EX['p'][TAGS.index(t)]) else 'negative', ('✓' if int(EX['p'][TAGS.index(t)]) == EX['y'] else '✗')] for t in TAGS]])}</div>
<div><div class="box pl"><b class="h">What this shows</b>The three TF-IDF models and the BiLSTM, which learn only from our labelled sentences, call it positive. The larger Transformers and the zero-shot model get it right: from pre-training they already know how “can't lift” and “familiar” behave in English. The two small fine-tuned models have less capacity to use that knowledge and also miss it.</div>
<div class="box wn"><b class="h">Use it in the talk</b>One real sentence is worth a page of numbers. Show it before the results slide: “Here is why accuracy differs.”</div></div></div>
<h2>Confidence intervals and tricky-sentence accuracy for every model</h2>
{tbl(["Model", "Accuracy", "95% interval", "Negation (184)", "Contrast (130)"], ci_rows)}
<p class="sm">Intervals of the top models overlap (for example 86.6 to 90.7 for fine-tuned DistilBERT and 89.1 to 92.9 for DistilBERT-SST2), so we do not claim that the ready-made model is clearly better than our fine-tuned one.</p></div>""")

P.append(f"""<div class="page"><div class="part">PART 2 · THE PROJECT END TO END</div><h1 class="t">10. Conclusions, recommendations and limits</h1>
<div class="cols"><div>
<h3>What we found</h3>
<ol class="sm"><li>Fine-tuned Transformers are the most accurate: {pc(A('ft_distilbert_8000'))} against {pc(A('tfidf_svm'))} (classical) and {pc(A('bilstm'))} (BiLSTM).</li><li>Pre-training gives data efficiency: BERT-small with 4,000 labels matches classical models trained on 65 k.</li><li>The cost is real but manageable: minutes to hours of CPU fine-tuning, milliseconds per sentence.</li><li>Reuse first: a ready-made in-domain model gives {pc(A('pt_distilbert_sst2'))} with no training.</li><li>Large Transformers handle negation and contrast much better.</li></ol>
</div><div>
<h3>What we recommend</h3>
<ol class="sm"><li>Look for a public model already fine-tuned for your task and domain.</li><li>If none exists, fine-tune a distilled Transformer such as DistilBERT on a few thousand labels. This works even without a GPU.</li><li>With no labels at all, a zero-shot model is a usable start.</li><li>Keep TF-IDF only when inference must be extremely cheap or the hardware is very limited.</li></ol>
</div></div>
<h3>Limits we state openly (this builds trust with the professor)</h3>
{tbl(["Limit", "Why it matters", "What we would do"], [["One dataset (English movie reviews)", "results may differ for other domains and languages", "repeat on other datasets, Lithuanian and Kannada text"], ["One random seed", "run-to-run variation is not measured; we use bootstrap intervals instead", "repeat with several seeds and report the spread"], ["872 test sentences", "differences under 2 to 3 points are not clearly real", "use bigger test sets and paired tests for close pairs"], ["CPU-only timing", "GPU timing would change the ratios", "measure on a GPU as well"], ["8,000 fine-tuning labels", "other amounts could change the ranking of small models", "extend the learning curve to the fine-tuned DistilBERT"]])}
<div class="box pj"><b class="h">Future work (one sentence for the closing slide)</b>The most natural continuation is the same procedure for Lithuanian texts with LitLat BERT or XLM-RoBERTa, with several random seeds and GPU timing.</div></div>""")

# ============================================================ PART 3 paper and slides
P.append(f"""<div class="page"><div class="part">PART 3 · THE PAPER AND THE PRESENTATION</div><h1 class="t">11. The paper, chapter by chapter</h1>
<p>The paper has 25 pages in the VILNIUS TECH template. Use this map to find things quickly and to know what each part is for.</p>
{tbl(["Part", "Pages", "What is inside", "If the professor asks…"], [
 ["Title page, contents", "1 to 3", "title, authors, table of contents, list of abbreviations", "—"],
 ["Introduction", "4 to 5", "why the topic matters, research aim, objectives, methods, work distribution (two students)", "“What is the aim?”: to analyse the Transformer and evaluate it experimentally against classical and recurrent methods for sentiment classification."],
 ["1. Literature review", "6 to 8", "search strategy (Google Scholar, Web of Science), history from word vectors to Transformers, model families, Lithuanian research, summary table, the gap", "“What did you read and what is missing?” Use the gap paragraph from section 6."],
 ["2. Research methodology", "9 to 16", "problem, dataset, ten models, criteria, program design, <b>the Transformer architecture</b> (tokenisation, positional encoding, attention, multi-head, feed-forward, complexity), mathematics of training and evaluation, protocol and threats to validity", "“Explain attention.” Section 2.6, pages 12 to 14. “How did you evaluate?” Section 2.7, pages 14 to 16."],
 ["3. Practical part", "17 to 22", "implementation, quality results, cost, data efficiency, ready-made models, negation and contrast, comparison with previous research", "“What did you find?” Use section 8 of this guide."],
 ["Conclusions", "23", "five numbered conclusions and practical recommendations", "“What do you recommend?”"],
 ["References", "24 to 25", "38 entries", "“Where is this from?”"]])}
<div class="box pl"><b class="h">How to read the paper in 30 minutes</b>1) Introduction (aim and objectives). 2) The summary table in the literature review and its conclusions. 3) Sections 2.2 and 2.3 (data and models). 4) Section 2.6 headings only. 5) The results tables and the five conclusions. Then read the slides next to them: each slide matches one part of the paper.</div>
<div class="box wn"><b class="h">Numbers you must know by heart</b>872 test sentences · 65,349 training · 10 models · 4 families · 8,000 fine-tuning labels · best {pc(A('pt_distilbert_sst2'))} (ready-made) · {pc(A('ft_distilbert_8000'))} (ours) · {pc(A('tfidf_svm'))} (best classical) · +{(CT[0][1] - CC[0][1]) * 100:.1f} points at 500 labels · 20 to 2,000× slower · laptop i5-8365U, no GPU.</div></div>""")

slides = [
 ("1", "Shashikanth", "30 s", "Title: introduce yourselves, the topic, and the question.", "“Transformers power BERT, GPT and almost every modern NLP system. Our question: what do they really give, and at what cost, on an ordinary laptop with limited labelled data?”", "The question (not a number).", "Why a laptop?", "Smile, state the question slowly; do not rush the opening."),
 ("2", "Shashikanth", "45 s", "Show the shift from reading word by word to reading all at once.", "“Before 2017 models read a sentence word by word. The Transformer removed recurrence and kept only attention: every word looks at every other word, all at the same time. That allowed huge pre-training: BERT, GPT.”", "2017, attention only.", "What is the difference from an LSTM?", "Use the analogy: a group discussion instead of passing a message down a line."),
 ("3", "Shashikanth", "60 s", "Explain self-attention and the architecture simply.", "“Each word asks a question (query), every word offers a label (key) and content (value). The softmax of the dot products gives weights; the output is a weighted mix of the values. Several heads run in parallel, layers are stacked. For classification we take the [CLS] vector.”", "softmax(QKᵀ/√d)·V.", "Why divide by √d?", "Point at Q, K, V on the slide as you say them; do not read the formula."),
 ("4", "Shaeed", "60 s", "Literature: what exists and what is missing.", "“We searched Google Scholar and Web of Science. Three families: encoder-only, decoder-only, encoder-decoder. In Lithuania, early work showed deep nets without pre-training lose to naive Bayes; newer work uses LitLat BERT and Llama2. The gap: cost and small-data behaviour on ordinary hardware.”", "NB 0.735 vs CNN 0.706.", "What exactly is the gap?", "End with the gap sentence; it sets up everything that follows."),
 ("5", "Shaeed", "45 s", "The experiment design.", "“SST-2 movie reviews. The official validation set is our test set; 2,000 training sentences are the development set. Ten models in four groups, all on the same laptop CPU so the times are comparable.”", "65k / 2k / 872; 10 models.", "Why is the validation set your test set?", "Explain train, development and test with the exam analogy."),
 ("6", "Shashikanth", "40 s", "Implementation: a reproducible pipeline.", "“Nine scripts, one generic fine-tuning script that takes any model name from the Hugging Face hub. Every model saves metrics, timings and predictions, so the report script computes tables and statistics. run_all.sh reproduces everything.”", "~2.5 h to reproduce.", "How reproducible is it?", "Mention the fixed seed (42) and saved predictions."),
 ("7", "Shashikanth", "60 s", "Result 1: accuracy.", "“Classical models reach about 82 to 83%, the BiLSTM 82.8%. Our fine-tuned DistilBERT reaches 88.8% with only 8,000 labels. The ready-made DistilBERT-SST2 reaches 91.1%. Zero-shot, with no labels, still gets 81.8%.”", f"{pc(A('ft_distilbert_8000'))} vs {pc(A('tfidf_svm'))}.", "Is 88.8 vs 91.1 a real difference?", "Say the three big numbers once, then stop talking and let them look."),
 ("8", "Shashikanth", "45 s", "Result 2: the price.", "“The gain is not free: TF-IDF trains in seconds and answers in 0.06 milliseconds; the Transformers need minutes to hours of CPU training and milliseconds per sentence. For most applications that is still acceptable.”", "20 to 2,000× slower.", "Is the cost worth it?", "Read one row of the table (SVM vs DistilBERT) instead of the whole table."),
 ("9", "Shaeed", "50 s", "Result 3: data efficiency.", "“With only 500 labels BERT-small is 14 points better than TF-IDF, because it already knows English from pre-training. With 4,000 labels it matches the classical model trained on 65,000. This agrees with the Lithuanian finding that pre-training, not the neural network as such, matters on small data.”", "+14 points at 500 labels.", "Why does small data hurt?", "Use the “graduate vs fresher” analogy for pre-training."),
 ("10", "Shaeed", "45 s", "Result 4: negation and contrast.", "“Bag-of-words models lose the most on sentences with negation and contrast, because ‘not good’ contains ‘good’. Transformers keep their accuracy much better. The heat map shows how attention links the words, but it is an illustration, not a proof of reasoning.”", "8 to 9 vs 2 to 4 points lost.", "Does attention explain the decision?", "Use the real example sentence (section 9 of this guide) before the map."),
 ("11", "Shaeed", "60 s", "Conclusions and recommendation.", "“Fine-tuned Transformers are most accurate; pre-training gives data efficiency; the cost is real but manageable. Recommendation: first try a public in-domain model; otherwise fine-tune a compact Transformer on a few thousand labels; keep TF-IDF only when latency or hardware are extremely limited. Next: Lithuanian data with LitLat BERT.”", "91.1% with zero training.", "What would you do differently?", "Finish with the recommendation, then say “thank you” and invite questions."),
]
srows = "".join(f"<tr><td><b>{a}</b></td><td>{b}<br><span class='m'>{c}</span></td><td>{d}</td><td>{e}</td><td>{f}</td><td>{g}</td><td>{h}</td></tr>" for a, b, c, d, e, f, g, h in slides)
P.append(f"""<div class="page"><div class="part">PART 3 · THE PAPER AND THE PRESENTATION</div><h1 class="t">12. The presentation, slide by slide</h1>
<p>11 slides, about <b>9 minutes</b> in total (Shashikanth about 4 min 40 s, Shaeed about 4 min 20 s). Practise saying the quoted text in your own words, not word for word.</p>
<div class="slidetbl">{tbl(["#", "Speaker, time", "Goal of the slide", "What to say (sample)", "Key number", "Likely question", "Tip"], [[a, f"{b}<br><span class='m'>{c}</span>", d, e, f, g, h] for a, b, c, d, e, f, g, h in slides[:4]])}</div></div>""")
P.append(f"""<div class="page"><div class="part">PART 3 · THE PAPER AND THE PRESENTATION</div><h1 class="t">12. The presentation (continued)</h1>
<div class="slidetbl">{tbl(["#", "Speaker, time", "Goal of the slide", "What to say (sample)", "Key number", "Likely question", "Tip"], [[a, f"{b}<br><span class='m'>{c}</span>", d, e, f, g, h] for a, b, c, d, e, f, g, h in slides[4:8]])}</div></div>""")
P.append(f"""<div class="page"><div class="part">PART 3 · THE PAPER AND THE PRESENTATION</div><h1 class="t">12. The presentation (continued)</h1>
<div class="slidetbl">{tbl(["#", "Speaker, time", "Goal of the slide", "What to say (sample)", "Key number", "Likely question", "Tip"], [[a, f"{b}<br><span class='m'>{c}</span>", d, e, f, g, h] for a, b, c, d, e, f, g, h in slides[8:]])}</div>
<h2>Handover lines between the two speakers</h2>
{tbl(["After slide", "From → to", "Say"], [["3", "Shashikanth → Shaeed", "“That is how the architecture works. Shaeed will now show what the literature says and what is missing.”"], ["5", "Shaeed → Shashikanth", "“Now Shashikanth will explain how we implemented the experiment and show the results.”"], ["8", "Shashikanth → Shaeed", "“So accuracy and cost are clear. Shaeed will show what happens with fewer labels and on harder sentences.”"]])}
<h2>Timing and rehearsal plan</h2>
{tbl(["Rehearsal", "Goal", "How"], [["1: read aloud", "know the content", "read the sample text for each slide once, slowly"], ["2: own words", "natural speech", "close the guide; explain each slide in your own words; time it"], ["3: full run with a timer", "7 to 10 minutes", "use the rehearsal timer in the interactive study guide; target 9 minutes"], ["4: with questions", "calm answers", "a friend asks 10 questions from section 14; practise the ABC method"]])}
<div class="box wn"><b class="h">If you are running out of time</b>Shorten slide 6 (implementation) and slide 8 (cost) to one sentence each. Never skip slide 11: the recommendation is what the professor remembers.</div></div>""")

# ============================================================ PART 4 explaining to the professor
P.append(f"""<div class="page"><div class="part">PART 4 · EXPLAINING IT TO THE PROFESSOR</div><h1 class="t">13. The storyline and the golden rules</h1>
<h2>The five-sentence storyline (memorise this)</h2>
<ol><li><b>Problem:</b> Transformers are everywhere in NLP, but nobody tells a student with a laptop what they really give and cost.</li><li><b>Gap:</b> the literature reports accuracy; cost and small-data behaviour are rarely measured together.</li><li><b>Method:</b> ten models, one dataset, one laptop, same rules.</li><li><b>Results:</b> Transformers are clearly more accurate and need far fewer labels, at 20 to 2,000 times the inference cost.</li><li><b>So what:</b> reuse a ready-made model first, fine-tune a small Transformer second, keep TF-IDF for extreme cost limits.</li></ol>
<h2>A 30-second opening you can say</h2>
<div class="say">“Good morning. We are Shashikanth and Shaeed. Today's NLP systems are built on Transformers, but a student with a normal laptop wants to know: how much better are they than simple methods, and what do they cost? We compared ten sentiment classifiers on one laptop and measured accuracy, cost and the number of labels needed. Here is what we found.”</div>
<h2>Golden rules</h2>
<div class="cols"><div>{tbl(["Do", ""], [["Say one number per slide, slowly", "✓"], ["Explain with one real sentence or analogy", "✓"], ["Say “our results suggest” for close differences", "✓"], ["State limits yourself, before they are asked", "✓"], ["Look at the audience, point at the chart", "✓"], ["Hand over cleanly to your partner", "✓"]])}</div>
<div>{tbl(["Do not", ""], [["Read the slide word for word", "✗"], ["Say “proves” (say “shows” or “suggests”)", "✗"], ["Claim attention “explains” the model", "✗"], ["Compare 88.8 and 91.1 as if the gap were certain", "✗"], ["Hide a limitation or guess a number", "✗"], ["Talk over your partner or interrupt", "✗"]])}</div></div>
<h2>How to explain any chart in three steps</h2>
<div class="cols"><div class="card2"><b>1 Axes</b><p class="sm">“On this axis is accuracy, on this one the number of labels.”</p></div><div class="card2"><b>2 Pattern</b><p class="sm">“The green line is always above the blue one; the gap is largest on the left.”</p></div><div class="card2"><b>3 Meaning</b><p class="sm">“So pre-training matters most when labels are scarce.”</p></div></div>
<h2>Honest wording table</h2>
{tbl(["Instead of…", "Say…"], [["“Transformers are always better.”", "“In our experiment the larger Transformers were clearly more accurate; the tiny ones were not.”"], ["“The attention map proves understanding.”", "“The map illustrates where information can flow.”"], ["“Our model is the best.”", "“The ready-made DistilBERT was best; ours, trained on 8,000 labels, was close and significantly above the classical models.”"], ["“We don't know.”", "“We did not test that; we would expect X because Y, and we would test it by Z.”"]])}</div>""")

qa = [
 ("What is the aim of the project?", "To analyse the Transformer architecture and to measure experimentally how much better it is than classical and recurrent methods for sentiment classification, including its cost, data needs and robustness."),
 ("Why sentiment analysis and why SST-2?", "It is a common real task with published results for every model family, and it contains hard cases (negation, contrast) that test understanding of context. SST-2 is the standard benchmark for it."),
 ("What is a Transformer in simple words?", "A network where every word looks at all other words at once (self-attention) instead of reading in order. That allows fast parallel training and very large pre-training."),
 ("What is pre-training versus fine-tuning?", "Pre-training teaches general English from huge unlabelled text, once. Fine-tuning then teaches our specific task with a few thousand labelled sentences. That is why 8,000 labels were enough for 88.8%."),
 ("Why did you use the validation set as the test set?", "GLUE does not publish test labels. We use the official validation set as held-out test data and never use it for learning; a separate development set of 2,000 training sentences is used to choose the best epoch."),
 ("Why only a CPU?", "It is what students and many small deployments really have. The limit is that GPU timing would differ and the ratios between models might change; we state this as a limitation."),
 ("Which model is best and by how much?", f"The ready-made DistilBERT-SST2 ({pc(A('pt_distilbert_sst2'))}). Our fine-tuned DistilBERT ({pc(A('ft_distilbert_8000'))}) is {(A('ft_distilbert_8000') - A('tfidf_svm')) * 100:.1f} points above the best classical model ({pc(A('tfidf_svm'))}), a significant difference (McNemar p < 0.001)."),
 ("Is 88.8% versus 91.1% a real difference?", "Their 95% intervals overlap (86.6 to 90.7 and 89.1 to 92.9), so we do not claim a clear difference. The ready-made model is probably a little better, which is plausible because it was fine-tuned on the full training set."),
 ("Why did the BiLSTM not beat TF-IDF?", f"It was trained from scratch without pre-training, so it must learn language and task from the same labels. It reached {pc(A('bilstm'))} against {pc(A('tfidf_svm'))}; it is better on negation but not overall. This agrees with the Lithuanian study."),
 ("Why was BERT-tiny so weak?", f"It has only 4.4 million parameters, so its capacity is small. Size matters: tiny {pc(A('ft_bert-tiny_8000'))}, small {pc(A('ft_bert-small_8000'))}, DistilBERT {pc(A('ft_distilbert_8000'))} with the same training recipe."),
 ("What does the learning curve show?", f"With 500 labels BERT-small beats TF-IDF by {(CT[0][1] - CC[0][1]) * 100:.1f} points; with 4,000 it matches the classical model trained on 65 k. Pre-training makes the model data-efficient."),
 ("What is the cost of using a Transformer?", "20 to 2,000 times slower per sentence and much larger. On a laptop CPU that is about 40 ms per sentence for DistilBERT, which is acceptable for most applications."),
 ("What is a confidence interval?", "A range showing how much the accuracy could change if we had a different sample of test sentences; we get it by resampling the test set many times (bootstrap)."),
 ("What is McNemar's test?", "A test for two models on the same sentences: it looks only at the sentences where exactly one model is right and asks whether the split is more uneven than chance would give."),
 ("How reproducible is your work?", "All stages are scripts with a fixed seed (42), every model saves its predictions, and run_all.sh reproduces everything in about 2.5 to 3 hours. One limit: only one seed, so run-to-run variation is not measured."),
 ("Explain self-attention.", "Each word makes a query, every word offers a key and a value. The match between query and keys, turned into weights by softmax, tells how much of each word's value to take. The result is a mix of the words that matter for that word."),
 ("Why divide by √d in the attention formula?", "Dot products of long vectors become large and make the softmax almost 0 or 1, which stops learning. Dividing by √d keeps the scores moderate."),
 ("What are your limitations?", "One dataset and language, one seed, a small test set (872), CPU-only timing, and a single fine-tuning size (8,000). We state them and describe how to remove each."),
 ("What would you do next?", "Repeat with several seeds, add GPU timing, apply the same procedure to Lithuanian (LitLat BERT, XLM-RoBERTa) and other datasets, and add an error analysis of sentences all models get wrong."),
 ("What did each of you do?", "Prepare this together beforehand. Suggested honest split in the paper: Shashikanth the design, the fine-tuning program, the Transformer theory and the practical part; Shaeed the introduction, the literature review, the classical and BiLSTM baselines, the error analysis and conclusions."),
]
P.append(f"""<div class="page"><div class="part">PART 4 · EXPLAINING IT TO THE PROFESSOR</div><h1 class="t">14. Questions you may get, and how to answer</h1>
<div class="box pl"><b class="h">The ABC method for any question</b><b>A</b>cknowledge (“Good question.”), give a <b>B</b>rief answer in one or two sentences, <b>C</b>onnect it to evidence (“…as the learning curve on slide 9 shows”). If you do not know: “We did not test that. We would expect X because Y, and we would check it by Z.” Never guess a number.</div>
{tbl(["Question", "Beginner-friendly answer"], [[f"<b>{q}</b>", a] for q, a in qa[:10]])}</div>""")
P.append(f"""<div class="page"><div class="part">PART 4 · EXPLAINING IT TO THE PROFESSOR</div><h1 class="t">14. Questions (continued)</h1>
{tbl(["Question", "Beginner-friendly answer"], [[f"<b>{q}</b>", a] for q, a in qa[10:]])}
<p class="sm">More questions with answers (54 in total, with a quiz mode) are in the interactive page <code>Project_Map_and_Interview_Prep.html</code>.</p></div>""")

P.append(f"""<div class="page"><div class="part">PART 4 · EXPLAINING IT TO THE PROFESSOR</div><h1 class="t">15. Day-of checklist and the one-page cheat sheet</h1>
<div class="cols"><div>
<h3>The day before</h3>
<ul class="chk"><li>Two full timed rehearsals (target 9 minutes, never above 10)</li><li>Agree who answers which kind of question</li><li>Read the paper's introduction and conclusions once more</li><li>Charge the laptop; copy files to a USB stick</li></ul>
<h3>Bring and open</h3>
<ul class="chk"><li>Presentation (PPTX) and a PDF copy as a backup</li><li>The paper (PDF) open at the results and the architecture pages</li><li>The interactive study guide and project map (work offline)</li><li>Water, a pen, a watch for timing</li></ul>
<h3>During the talk</h3>
<ul class="chk"><li>Slow start; one clear sentence for the question</li><li>Eye contact; do not read the slides</li><li>Hand over with the prepared lines</li><li>End with the recommendation and “thank you”</li></ul>
<h3>During questions</h3>
<ul class="chk"><li>Listen to the end; repeat the question if unclear</li><li>ABC method; short answers</li><li>Let your partner help if the question is theirs</li><li>Be honest about limits</li></ul>
</div><div>
<h3>One-page cheat sheet</h3>
{tbl(["Item", "Value"], [["Task", "binary sentiment, SST-2"], ["Splits", "65,349 / 2,000 / 872"], ["Models", "10: 3 TF-IDF, 1 BiLSTM, 3 fine-tuned, 3 ready-made"], ["Best (ready-made)", f"DistilBERT-SST2 {pc(A('pt_distilbert_sst2'))}"], ["Best trained by us", f"DistilBERT {pc(A('ft_distilbert_8000'))} (8,000 labels, {fmt_t(MOD['ft_distilbert_8000']['train_s'])})"], ["Best classical", f"TF-IDF + SVM {pc(A('tfidf_svm'))} (2 s, 0.06 ms)"], ["BiLSTM", f"{pc(A('bilstm'))}"], ["Zero-shot", f"{pc(A('zs_nli_distilroberta'))}, no labels"], ["500-label gap", f"BERT-small {pc(CT[0][1])} vs TF-IDF {pc(CC[0][1])}"], ["Negation drop", f"SVM {dropv('tfidf_svm'):.1f}, DistilBERT-SST2 {dropv('pt_distilbert_sst2'):.1f} points"], ["Cost", "20 to 2,000× slower; DistilBERT ~40 ms per sentence"], ["Hardware", "i5-8365U, 4 cores, 16 GB, no GPU, seed 42"], ["Test noise", "about ±2.4 points on 872 sentences"]])}
<div class="box pl"><b class="h">Our motto</b>Explain it. Measure it. Make it useful for everyone.</div></div></div></div>""")

glossary = [("Accuracy", "share of test sentences classified correctly"), ("Attention", "each word looking at other words and taking a weighted mix of their content"), ("Backpropagation", "working out how each weight should change to reduce the error"), ("Bag of words", "counting words and ignoring their order"), ("BERT", "a Transformer encoder pre-trained by guessing hidden words"), ("BiLSTM", "an LSTM that reads a sentence both forwards and backwards"), ("[CLS]", "a special token whose vector summarises the whole sentence for classification"), ("Confidence interval", "a range showing how much a result could wobble because of the test sample"), ("DistilBERT", "a smaller, faster BERT made by distillation (67 M parameters)"), ("Fine-tuning", "training a pre-trained model briefly on your own labelled task"), ("Latency", "time to classify one sentence"), ("Learning curve", "accuracy as the number of training labels grows"), ("LSTM", "a recurrent network with gates that keep long-term memory"), ("Macro-F1", "average of the F1 scores of the two classes"), ("McNemar's test", "paired test comparing two models on the same sentences"), ("Pre-training", "learning general language from huge unlabelled text"), ("Ready-made model", "a public model already trained for a task"), ("SST-2", "movie-review sentences labelled positive or negative"), ("Self-attention", "attention among the words of the same sentence"), ("TF-IDF", "word weights that favour rare, informative words"), ("Token", "a piece of text (word or word part) the model reads"), ("Transformer", "a network based on self-attention that processes all words at once"), ("Zero-shot", "using a model for a task it was not trained on, with no examples")]
P.append(f"""<div class="page"><div class="part">GLOSSARY AND NEXT STEPS</div><h1 class="t">16. Glossary</h1>
<div class="gl2">{''.join(f"<div class='gl'><b>{a}</b><span>{b}</span></div>" for a, b in glossary)}</div>
<h2>Where to practise next</h2>
{tbl(["Material", "Use it for"], [["Project_Study_Guide.html", "test your own sentences on all ten models; real attention maps; quiz; rehearsal timer"], ["Project_Map_and_Interview_Prep.html", "the clickable project flow and 54 interview questions with a quiz mode"], ["Transformers_in_NLP_Presentation.pptx", "the 11 slides with speaker notes"], ["Transformers_in_NLP_Project_Paper.pdf", "the full written work (25 pages)"], ["Literature_Gap_Explained.pdf", "the gap in the literature, with theory and an example"], ["Transformer_Encoder_Decoder_Flow.pdf", "how the encoder and decoder work, with real numbers"]])}
<div class="box wn"><b class="h">A last thought</b>You do not need to know everything. You need to explain clearly what you did, say honestly what the numbers show, and admit calmly what you did not test. That is what a professor wants to see.</div></div>""")

CSS = """
@page { size: A4; margin: 13mm 13mm 14mm 13mm; @bottom-center { content: counter(page); font: 9pt 'Segoe UI', sans-serif; color: #777; } }
@page :first { margin: 0; @bottom-center { content: none; } }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font: 9.3pt/1.45 'Segoe UI', 'Nirmala UI', sans-serif; color: #1d1d3a; }
.page { page-break-after: always; break-after: page; } .page:last-child { page-break-after: auto; }
.part { font-size: 8.3pt; letter-spacing: .12em; font-weight: 700; color: #6250d6; margin-bottom: .5mm; }
h1.t { font-size: 16pt; margin: 0 0 2mm; color: #2e2a6b; border-bottom: 2px solid #d9dcef; padding-bottom: 1.5mm; } h2 { font-size: 11.3pt; margin: 2.6mm 0 1.1mm; color: #6250d6; } h3 { font-size: 10.5pt; margin: 2.6mm 0 1mm; color: #2e2a6b; }
p { margin: 0 0 2mm; } .sm { font-size: 8.8pt; color: #4a4a68; } .m { color: #6b6b88; font-size: 8pt; }
.cover { height: 297mm; padding: 26mm 20mm 18mm; background: linear-gradient(160deg, #16163f 0%, #2e2a6b 60%, #3a3590 100%); color: #fff; position: relative; overflow: hidden; }
.cover .tag { letter-spacing: .14em; text-transform: uppercase; font-size: 9.5pt; color: #b9b8e8; margin-bottom: 11mm; } .cover h1 { font-size: 30pt; line-height: 1.15; margin: 0 0 6mm; } .cover h2 { font-size: 13pt; font-weight: 400; color: #dcdcf6; border: 0; margin: 0 0 11mm; line-height: 1.55; }
.cover .ex { background: rgba(255,255,255,.1); border-radius: 4mm; padding: 5mm 6mm; font-size: 11pt; max-width: 160mm; line-height: 1.6; } .cover .foot { position: absolute; left: 20mm; right: 20mm; bottom: 16mm; font-size: 9pt; color: #b9b8e8; border-top: 1px solid rgba(255,255,255,.25); padding-top: 4mm; }
.box { border-radius: 2.5mm; padding: 1.9mm 3.6mm; margin: 1.7mm 0; break-inside: avoid; } .box b.h { display: block; font-size: 7.8pt; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .5mm; }
.box.pl { background: #e6f6ee; border-left: 4px solid #159a6a; } .pl b.h { color: #159a6a; } .box.rw { background: #fff1e8; border-left: 4px solid #d9541f; } .rw b.h { color: #d9541f; } .box.pj { background: #ece9fd; border-left: 4px solid #6250d6; } .pj b.h { color: #6250d6; } .box.wn { background: #fff3dc; border-left: 4px solid #c27a10; } .wn b.h { color: #c27a10; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 1.2mm 0 2mm; } th { background: #2e2a6b; color: #fff; text-align: left; padding: 1.2mm 2mm; } td { padding: 1.1mm 2mm; border-bottom: 1px solid #d9dcef; vertical-align: top; } tr { break-inside: avoid; } tr:nth-child(even) td { background: #f8f8fd; }
.slidetbl table { font-size: 8.3pt; } .slidetbl td { padding: 1.8mm 1.8mm; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; align-items: start; } .card2 { border: 1px solid #d9dcef; border-radius: 3mm; padding: 2.4mm 3.4mm; background: #fff; break-inside: avoid; }
figure { margin: 2mm 0; text-align: center; break-inside: avoid; } figure svg { max-width: 100%; height: auto; max-height: 64mm; display: block; margin: 0 auto; } figure.att svg { max-height: 112mm; } figcaption { font-size: 8.3pt; color: #5a5977; margin-top: 1mm; }
code { font-family: Consolas, monospace; font-size: .9em; background: #f1f1f9; padding: 0 .3em; border-radius: 3px; }
.dot { display: inline-block; width: 2.4mm; height: 2.4mm; border-radius: 50%; margin-right: 1.2mm; vertical-align: middle; }
.say { border-left: 4px solid #6250d6; background: #f3f1fe; padding: 2.6mm 4mm; border-radius: 0 2.5mm 2.5mm 0; font-style: italic; margin: 1.5mm 0; }
ul.chk { list-style: none; padding: 0; margin: 0 0 1.5mm; } ul.chk li { padding-left: 5mm; position: relative; margin: .7mm 0; font-size: 9pt; } ul.chk li::before { content: "☐"; position: absolute; left: 0; color: #159a6a; }
.gl { border-bottom: 1px solid #e3e5f3; padding: .6mm 0; break-inside: avoid; } .gl b { color: #2e2a6b; display: block; font-size: 9pt; } .gl span { font-size: 8.6pt; color: #4a4a68; }
ol { margin: 1mm 0 2mm; padding-left: 5mm; } .gl2 { column-count: 2; column-gap: 5mm; } li { margin: .8mm 0; }
"""
HTML = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Beginner's study guide: Transformers in NLP</title><style>{CSS}</style></head><body>{''.join(P)}</body></html>"
hp = os.path.join(HERE, "guide.html"); open(hp, "w", encoding="utf-8").write(HTML)
prof = tempfile.mkdtemp(prefix="edge_pdf_"); tmp = os.path.join(tempfile.gettempdir(), "beginner_guide_tmp.pdf")
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
