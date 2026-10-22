"""Builds docs/Transformer_Encoder_Decoder_Flow.pdf: diagrams + a step-by-step real-world example (real NLLB tensors from real.json).
Run: python study_guide/transformer_pdf/build.py   (needs Microsoft Edge for HTML -> PDF)"""
import html, json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R = json.load(open(os.path.join(HERE, "real.json"), encoding="utf-8"))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT_PDF = os.path.join(ROOT, "docs", "Transformer_Encoder_Decoder_Flow.pdf")
CFG = R["cfg"]
BLUE, GREEN, PURPLE, AMBER, ORANGE, GREY, INK = "#2a78d6", "#159a6a", "#6250d6", "#c27a10", "#d9541f", "#6b6b88", "#1d1d3a"
esc = html.escape
clean = lambda t: t.replace("▁", "") or "▁"
SRC = [clean(t) for t in R["src_tok"]]; OUT = [clean(t) for t in R["out_tok"]]; DEC_IN = [clean(t) for t in R["dec_in_tok"]]
S, T = len(SRC), len(OUT)


# ------------------------------------------------------------------ svg helpers
def box(x, y, w, h, text, fill, stroke=None, tc="#fff", fs=12, rx=9, sub=None, dash=False):
    st = f'stroke="{stroke or fill}" stroke-width="1.6"' + (' stroke-dasharray="6 4"' if dash else "")
    t = f'<text x="{x + w / 2}" y="{y + h / 2 + (-3 if sub else 4)}" text-anchor="middle" font-size="{fs}" font-weight="700" fill="{tc}">{text}</text>'
    if sub: t += f'<text x="{x + w / 2}" y="{y + h / 2 + 12}" text-anchor="middle" font-size="{fs - 2.5}" fill="{tc}" opacity=".9">{sub}</text>'
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" {st}/>{t}'


def arrow(pts, color=INK, w=1.8, dash=False, head=True):
    d = "M" + " L".join(f"{x} {y}" for x, y in pts)
    extra = (' stroke-dasharray="5 4"' if dash else "") + (' marker-end="url(#ar' + color[1:] + ')"' if head else "")
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{extra}/>'


def markers(*colors):
    return "<defs>" + "".join(f'<marker id="ar{c[1:]}" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9z" fill="{c}"/></marker>' for c in colors) + "</defs>"


def badge(x, y, n, color=INK):
    return f'<circle cx="{x}" cy="{y}" r="11" fill="{color}"/><text x="{x}" y="{y + 4}" text-anchor="middle" font-size="11" font-weight="800" fill="#fff">{n}</text>'


def txt(x, y, s, fs=11, color=INK, anchor="middle", weight=400, kn=False):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" fill="{color}" font-weight="{weight}"{" class=kn" if kn else ""}>{s}</text>'


def heat(mat, rows, cols, cell=26, title=None, color=(98, 80, 214), lab_w=78, mask=False):
    n, m = len(rows), len(cols); W = lab_w + m * cell + 4; Hh = 52 + n * cell
    s = f'<svg viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">'
    for j, c in enumerate(cols):
        s += f'<text transform="translate({lab_w + j * cell + cell / 2 + 3},46) rotate(-55)" font-size="9.5" fill="{INK}">{esc(c)}</text>'
    for i, rname in enumerate(rows):
        s += f'<text x="{lab_w - 4}" y="{52 + i * cell + cell / 2 + 3}" text-anchor="end" font-size="9.5" fill="{INK}">{esc(rname)}</text>'
        for j in range(m):
            v = mat[i][j]; masked = mask and j > i
            fill = "#e9e9f1" if masked else f"rgba({color[0]},{color[1]},{color[2]},{min(1, v) * .92:.3f})"
            s += f'<rect x="{lab_w + j * cell}" y="{52 + i * cell}" width="{cell - 1}" height="{cell - 1}" rx="3" fill="{fill}"/>'
            if not masked and v >= 0.12: s += f'<text x="{lab_w + j * cell + cell / 2 - .5}" y="{52 + i * cell + cell / 2 + 3}" text-anchor="middle" font-size="8.5" fill="{"#fff" if v > .45 else INK}">{round(v * 100)}</text>'
            if masked: s += f'<text x="{lab_w + j * cell + cell / 2 - .5}" y="{52 + i * cell + cell / 2 + 3}" text-anchor="middle" font-size="8" fill="#a0a0b8">×</text>'
    return s + "</svg>"


# ------------------------------------------------------------------ diagrams
def d_big():
    c = [BLUE, GREEN, PURPLE, AMBER, ORANGE, GREY, INK]
    s = f'<svg viewBox="0 0 800 905" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">{markers(*c)}'
    # encoder column
    s += f'<rect x="30" y="215" width="300" height="470" rx="16" fill="#eaf1fc" stroke="{BLUE}" stroke-width="2" stroke-dasharray="7 5"/>' + txt(180, 238, "ENCODER LAYER  × 12", 14, BLUE, weight=800)
    s += box(60, 600, 240, 52, "Multi-head self-attention", PURPLE, sub="every word looks at every word") + box(60, 548, 240, 36, "Add & LayerNorm", GREY, fs=12)
    s += box(60, 466, 240, 52, "Feed-forward network", AMBER, sub="per word: 1024 → " + str(CFG["ffn"]) + " → 1024") + box(60, 414, 240, 36, "Add & LayerNorm", GREY, fs=12)
    s += txt(180, 385, "↑ same block repeated 12 times (own weights each)", 10, BLUE)
    s += arrow([(180, 600), (180, 586)], INK) + arrow([(180, 548), (180, 520)], INK) + arrow([(180, 466), (180, 452)], INK)
    s += arrow([(180, 672), (180, 654)], INK)
    # residual skips
    s += arrow([(180, 668), (312, 668), (312, 566), (302, 566)], GREY, 1.4, True) + arrow([(180, 532), (312, 532), (312, 432), (302, 432)], GREY, 1.4, True)
    s += txt(322, 620, "skip", 9, GREY, "start")
    s += box(60, 735, 240, 40, "Embedding + positional encoding", BLUE, sub=None, fs=11.5)
    s += arrow([(180, 735), (180, 676)], INK)
    s += box(60, 800, 240, 34, "Tokenizer: text → token IDs", GREY, fs=11.5) + arrow([(180, 800), (180, 777)], INK)
    s += txt(180, 862, "English input:", 11, GREY) + txt(180, 880, "“I am going to Mysuru tomorrow.”", 13, INK, weight=700)
    s += arrow([(180, 862 - 11), (180, 836)], INK)
    s += box(60, 160, 240, 40, "Encoder output (memory)", BLUE, sub=f"{S} words × 1024 numbers", fs=12) + arrow([(180, 414), (180, 202)], INK, 2.2)
    # decoder column
    s += f'<rect x="450" y="215" width="300" height="470" rx="16" fill="#e8f6f0" stroke="{GREEN}" stroke-width="2" stroke-dasharray="7 5"/>' + txt(600, 238, "DECODER LAYER  × 12", 14, GREEN, weight=800)
    s += box(480, 612, 240, 52, "Masked self-attention", PURPLE, sub="looks only at words already written") + box(480, 566, 240, 34, "Add & LayerNorm", GREY, fs=12)
    s += box(480, 490, 240, 52, "Cross-attention", ORANGE, sub="Q from decoder · K, V from encoder") + box(480, 444, 240, 34, "Add & LayerNorm", GREY, fs=12)
    s += box(480, 366, 240, 50, "Feed-forward network", AMBER, sub="1024 → " + str(CFG["ffn"]) + " → 1024") + box(480, 320, 240, 34, "Add & LayerNorm", GREY, fs=12)
    s += txt(600, 296, "↑ same block repeated 12 times", 10, GREEN)
    s += arrow([(600, 612), (600, 602)], INK) + arrow([(600, 566), (600, 544)], INK) + arrow([(600, 490), (600, 480)], INK) + arrow([(600, 444), (600, 418)], INK) + arrow([(600, 366), (600, 356)], INK)
    s += arrow([(600, 700), (600, 666)], INK)
    s += box(480, 735, 240, 40, "Embedding + positional encoding", GREEN, fs=11.5) + arrow([(600, 735), (600, 686)], INK)
    s += box(480, 800, 240, 34, "Previous tokens (right-shifted)", GREY, fs=11.5) + arrow([(600, 800), (600, 777)], INK)
    s += txt(600, 862, "Kannada written so far:", 11, GREY) + txt(600, 880, "&lt;/s&gt;  kan_Knda  ನಾ  ಳೆ  …", 12, INK, weight=700, kn=True)
    s += arrow([(600, 851), (600, 836)], INK)
    # memory -> cross attention
    s += arrow([(300, 180), (390, 180), (390, 516), (478, 516)], ORANGE, 2.6) + txt(396, 345, "K, V", 11, ORANGE, "start", 800)
    s += txt(396, 360, "memory goes to", 9.5, ORANGE, "start") + txt(396, 372, "every decoder", 9.5, ORANGE, "start") + txt(396, 384, "layer", 9.5, ORANGE, "start")
    # output head
    s += box(480, 150, 240, 34, "Linear: 1024 → 256,206 scores", GREEN, fs=11.5) + box(480, 100, 240, 34, "Softmax: probabilities", GREEN, fs=11.5)
    s += arrow([(600, 320), (600, 300), (600, 186)], INK, 2.2, head=True) + arrow([(600, 150), (600, 136)], INK)
    s += box(480, 40, 240, 40, "Next-token probabilities", INK, sub="ನಾ 57.7% · ನಾನು 35.4% · …", fs=12) + arrow([(600, 100), (600, 82)], INK)
    s += arrow([(722, 60), (768, 60), (768, 817), (722, 817)], GREEN, 2.2, True) + f'<text transform="translate(786,440) rotate(90)" text-anchor="middle" font-size="10.5" fill="{GREEN}">chosen token is appended; repeat until &lt;/s&gt;</text>'
    # badges
    for (x, y, n, col) in [(300, 880 - 18, 1, GREY), (44, 755, 2, BLUE), (44, 626, 3, PURPLE), (44, 566, 4, GREY), (44, 492, 5, AMBER), (44, 432, 6, GREY), (44, 180, 7, BLUE),
                           (464, 755, 8, GREEN), (464, 638, 9, PURPLE), (464, 516, 10, ORANGE), (464, 391, 11, AMBER), (464, 166, 12, GREEN), (464, 116, 13, GREEN)]:
        s += badge(x, y, n, col)
    s += txt(180, 128, "RUNS ONCE per sentence", 11, BLUE, weight=800) + txt(600, 22, "RUNS ONCE PER OUTPUT TOKEN", 11, GREEN, weight=800)
    return s + "</svg>"


def d_flow():
    c = [BLUE, GREEN, INK]
    s = f'<svg viewBox="0 0 760 560" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">{markers(*c)}'
    s += f'<rect x="8" y="8" width="744" height="206" rx="14" fill="#eaf1fc"/>' + txt(24, 30, "PHASE A: THE ENCODER READS (once)", 13, BLUE, "start", 800)
    labs = [("1 Tokenize", "10 tokens"), ("2 Embed + pos.", "10 × 1024"), ("3 Self-attention", "mix words"), ("4 Add & Norm", "keep + stabilise"), ("5 Feed-forward", "think per word"), ("6 Add & Norm", ""), ("7 Repeat ×12", "→ memory")]
    xs = [20 + i * 105 for i in range(7)]
    for (a, b), x in zip(labs, xs):
        fill = PURPLE if "attention" in a else AMBER if "Feed" in a else GREY if "Norm" in a else BLUE
        s += box(x, 60, 96, 62, a, fill, fs=11, sub=b)
    for x in xs[:-1]: s += arrow([(x + 96, 91), (x + 105, 91)], INK, 1.8)
    s += txt(380, 150, "“I am going to Mysuru tomorrow.”  →  10 pieces: eng_Latn · I · am · going · to · Mys · uru · tomorrow · . · &lt;/s&gt;", 11, INK)
    s += txt(380, 172, "Result: one 1024-number vector per word, each now aware of the whole sentence (the “memory”).", 11, INK, weight=700)
    s += txt(380, 196, "Steps 3 to 6 are one encoder layer; the real model stacks 12 of them.", 10.5, GREY)
    s += f'<rect x="8" y="232" width="744" height="316" rx="14" fill="#e8f6f0"/>' + txt(24, 254, "PHASE B: THE DECODER WRITES (loop, one token per turn)", 13, GREEN, "start", 800)
    labs2 = [("8 Embed + position", "tokens so far"), ("9 Masked self-attn", "look back only"), ("10 Cross-attention", "look at memory"), ("11 Feed-forward", "+ Add & Norm"), ("12 Linear + softmax", "256,206 scores"), ("13 Pick token", "append, repeat")]
    xs2 = [20 + i * 122 for i in range(6)]
    for (a, b), x in zip(labs2, xs2):
        fill = PURPLE if "Masked" in a else ORANGE if "Cross" in a else AMBER if "Feed" in a else GREEN if ("Linear" in a or "Pick" in a) else BLUE
        s += box(x, 284, 112, 62, a, fill, fs=11, sub=b)
    for x in xs2[:-1]: s += arrow([(x + 112, 315), (x + 122, 315)], INK, 1.8)
    s += arrow([(xs2[-1] + 56, 348), (xs2[-1] + 56, 372), (xs2[0] + 56, 372), (xs2[0] + 56, 348)], GREEN, 2.2)
    s += txt(380, 392, "loop until the model picks &lt;/s&gt;", 11, GREEN, weight=700)
    rows = [("turn 1", "input: &lt;/s&gt; kan_Knda", "→ picks", "ನಾ", "57.7%"), ("turn 2", "input: … ನಾ", "→ picks", "ಳೆ", "95.1%"), ("turn 3", "input: … ನಾ ಳೆ", "→ picks", "ನಾನು", "56.2%"), ("…", "", "", "", ""), ("turn 11", "input: … ುತ್ತಿದ್ದೇನೆ .", "→ picks", "&lt;/s&gt;", "stop")]
    for i, (a, b, c_, d, e) in enumerate(rows):
        y = 424 + i * 24
        s += txt(30, y, a, 11, GREEN, "start", 800) + txt(100, y, b, 11, INK, "start", kn=True) + txt(330, y, c_, 11, GREY, "start") + txt(385, y, d, 12, INK, "start", 800, kn=True) + txt(470, y, e, 11, GREY, "start")
    s += txt(380, 540, "Real probabilities from NLLB-200 for this exact sentence (greedy decoding).", 10.5, GREY)
    return s + "</svg>"


def d_encoder_layer():
    s = f'<svg viewBox="0 0 760 625" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">{markers(INK, GREY, PURPLE)}'
    X, W = 170, 260
    s += box(X, 10, W, 40, "Output of this layer", BLUE, sub=f"{S} × 1024, goes to the next layer") + box(X, 80, W, 36, "Add (skip connection)", GREY, fs=12)
    s += box(X, 146, W, 64, "Feed-forward network", AMBER, sub=f"1024 → {CFG['ffn']} → 1024", fs=13) + txt(X + W / 2, 202, "each word processed alone, ReLU in the middle", 9.5, "#fff")
    s += box(X, 236, W, 34, "LayerNorm", GREY, fs=12) + box(X, 296, W, 36, "Add (skip connection)", GREY, fs=12)
    s += f'<rect x="{X}" y="362" width="{W}" height="110" rx="10" fill="#f1eefc" stroke="{PURPLE}" stroke-width="1.8"/>' + txt(X + W / 2, 380, "Multi-head self-attention", 12.5, PURPLE, weight=800)
    for k, nm in enumerate(["Q = X·Wq", "K = X·Wk", "V = X·Wv"]): s += box(X + 10 + k * 82, 390, 76, 26, nm, PURPLE, fs=10)
    s += box(X + 10, 424, W - 20, 38, "softmax(Q·Kᵀ / √64) · V", PURPLE, sub="16 heads at once, joined by Wo", fs=11)
    s += box(X, 500, W, 34, "LayerNorm", GREY, fs=12) + box(X, 566, W, 40, "Input X", BLUE, sub=f"{S} word vectors × 1024")
    c = X + W / 2
    for y1, y2 in [(566, 536), (500, 474), (362, 334), (296, 272), (236, 212), (146, 118), (80, 52)]: s += arrow([(c, y1), (c, y2)], INK, 2)
    s += arrow([(c, 551), (490, 551), (490, 314), (X + W + 2, 314)], GREY, 1.8, True) + arrow([(c, 283), (530, 283), (530, 98), (X + W + 2, 98)], GREY, 1.8, True)
    s += txt(500, 440, "skip 1:", 10.5, GREY, "start", 700) + txt(500, 454, "the input X is added back", 10, GREY, "start") + txt(540, 200, "skip 2:", 10.5, GREY, "start", 700) + txt(540, 214, "the middle result is", 10, GREY, "start") + txt(540, 228, "added back", 10, GREY, "start")
    for (x, y, n, col) in [(X - 22, 586, 2, BLUE), (X - 22, 417, 3, PURPLE), (X - 22, 314, 4, GREY), (X - 22, 178, 5, AMBER), (X - 22, 98, 6, GREY), (X - 22, 30, 7, BLUE)]: s += badge(x, y, n, col)
    s += txt(X - 40, 417, "each word asks a Query,", 10, GREY, "end") + txt(X - 40, 431, "compares it to all Keys,", 10, GREY, "end") + txt(X - 40, 445, "takes a mix of Values", 10, GREY, "end")
    s += txt(X - 40, 520, "NLLB normalises BEFORE each", 9.5, GREY, "end") + txt(X - 40, 533, "sub-layer (pre-norm); the 2017", 9.5, GREY, "end") + txt(X - 40, 546, "paper does it AFTER the Add", 9.5, GREY, "end")
    return s + "</svg>"


def d_decoder_layer():
    s = f'<svg viewBox="0 0 760 672" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">{markers(INK, GREY, ORANGE, BLUE)}'
    X, W = 200, 260; c = X + W / 2
    s += box(X, 10, W, 36, "Output → next decoder layer", GREEN, sub="after layer 12: linear + softmax", fs=12) + box(X, 70, W, 30, "Add (skip)", GREY, fs=11.5)
    s += box(X, 122, W, 48, "Feed-forward network", AMBER, sub=f"1024 → {CFG['ffn']} → 1024, per position", fs=12.5) + box(X, 192, W, 28, "LayerNorm", GREY, fs=11.5) + box(X, 242, W, 30, "Add (skip)", GREY, fs=11.5)
    s += box(X, 294, W, 64, "Cross-attention", ORANGE, sub="Q: Kannada side · K, V: encoder memory", fs=12.5) + box(X, 380, W, 28, "LayerNorm", GREY, fs=11.5) + box(X, 430, W, 30, "Add (skip)", GREY, fs=11.5)
    s += box(X, 482, W, 60, "Masked self-attention", PURPLE, sub="Q, K, V from the Kannada side;", fs=12.5) + txt(c, 533, "future words are masked out", 9.5, "#fff") + box(X, 564, W, 28, "LayerNorm", GREY, fs=11.5)
    s += box(X, 614, W, 40, "Input: Kannada tokens so far", GREEN, sub="embedded + position", fs=12)
    for y1, y2 in [(614, 594), (564, 544), (482, 462), (430, 410), (380, 360), (294, 274), (242, 222), (192, 172), (122, 102), (70, 48)]: s += arrow([(c, y1), (c, y2)], INK, 2)
    s += box(560, 296, 170, 60, "Encoder memory", BLUE, sub=f"{S} × 1024 (fixed)", fs=12.5) + arrow([(560, 326), (X + W + 2, 326)], ORANGE, 3) + txt(512, 318, "K, V", 11.5, ORANGE, weight=800) + txt(645, 288, "from the encoder (computed once)", 9.5, BLUE)
    s += arrow([(c, 603), (140, 603), (140, 445), (X - 2, 445)], GREY, 1.7, True) + arrow([(c, 419), (100, 419), (100, 257), (X - 2, 257)], GREY, 1.7, True) + arrow([(c, 231), (60, 231), (60, 85), (X - 2, 85)], GREY, 1.7, True)
    s += txt(52, 330, "skip", 10, GREY, "end", 700) + txt(52, 343, "connections", 9.5, GREY, "end")
    for (x, y, n, col) in [(X + W + 24, 512, 9, PURPLE), (X + W + 24, 326, 10, ORANGE), (X + W + 24, 146, 11, AMBER)]: s += badge(x, y, n, col)
    return s + "</svg>"


def d_causal(n=5):
    s = f'<svg viewBox="0 0 330 250" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">'
    labels = ["⟨start⟩", "ನಾ", "ಳೆ", "ನಾನು", "ಮ"][:n]
    s += txt(180, 16, "who may look at whom (rows look at columns)", 10.5, GREY)
    for i in range(n):
        s += f'<text x="62" y="{74 + i * 30}" text-anchor="end" font-size="11" class="kn" fill="{INK}">{labels[i]}</text><text transform="translate({92 + i * 30},58) rotate(-40)" font-size="11" class="kn" fill="{INK}">{labels[i]}</text>'
        for j in range(n):
            ok = j <= i; s += f'<rect x="{70 + j * 30}" y="{58 + i * 30}" width="28" height="28" rx="4" fill="{PURPLE if ok else "#e9e9f1"}" opacity="{1 if ok else 1}"/><text x="{84 + j * 30}" y="{76 + i * 30}" text-anchor="middle" font-size="12" fill="{"#fff" if ok else "#9a9ab0"}">{"✓" if ok else "×"}</text>'
    s += txt(165, 232, "× = blocked (score set to −∞ before softmax, so weight = 0)", 10, GREY)
    return s + "</svg>"


def d_training():
    s = f'<svg viewBox="0 0 760 270" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">{markers(INK, GREEN, ORANGE)}'
    s += f'<rect x="6" y="6" width="364" height="258" rx="12" fill="#eaf1fc"/><rect x="390" y="6" width="364" height="258" rx="12" fill="#e8f6f0"/>'
    s += txt(188, 28, "TRAINING: teacher forcing (one parallel pass)", 12.5, BLUE, weight=800) + txt(572, 28, "INFERENCE: the loop (one pass per token)", 12.5, GREEN, weight=800)
    s += box(24, 50, 330, 34, "Decoder input (the correct answer, shifted right):", GREY, fs=10.5)
    s += txt(189, 106, "⟨start⟩  ನಾನು  ನಾಳೆ  ಮೈಸೂರಿಗೆ  ಹೋಗುತ್ತೇನೆ", 11.5, INK, weight=700, kn=True) + arrow([(189, 114), (189, 140)], INK)
    s += box(24, 142, 330, 40, "Decoder predicts at EVERY position at once", GREEN, fs=11.5) + arrow([(189, 182), (189, 204)], INK)
    s += txt(189, 224, "target:  ನಾನು  ನಾಳೆ  ಮೈಸೂರಿಗೆ  ಹೋಗುತ್ತೇನೆ  ⟨end⟩", 11.5, INK, weight=700, kn=True) + txt(189, 246, "loss = −mean ln p(correct word); a wrong guess cannot spoil the next input", 9.5, GREY)
    s += box(408, 50, 330, 34, "Start with only ⟨start⟩; model writes ONE token", GREY, fs=10.5) + arrow([(573, 84), (573, 106)], INK)
    s += box(408, 108, 330, 34, "Append the token it chose to the input", GREEN, fs=11.5) + arrow([(573, 142), (573, 164)], INK)
    s += box(408, 166, 330, 34, "Run the decoder again on the longer input", GREEN, fs=11.5) + arrow([(573, 200), (573, 216)], INK)
    s += txt(573, 234, "stop when ⟨end⟩ is chosen; the encoder is NOT re-run", 10.5, INK, weight=700) + txt(573, 252, "(its memory is reused every turn)", 9.5, GREY)
    return s + "</svg>"


def d_variants():
    s = f'<svg viewBox="0 0 760 170" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Nirmala UI, sans-serif">{markers(INK)}'
    for k, (x, name, parts, ex, col) in enumerate([(10, "Encoder only", "reads, understands", "BERT, DistilBERT: classify a review, tag names, search", BLUE), (265, "Decoder only", "writes continuation", "GPT family and most chatbots: next word prediction", GREEN), (520, "Encoder + decoder", "reads, then writes", "NLLB, T5, Whisper: translation, summaries, speech", PURPLE)]):
        s += f'<rect x="{x}" y="8" width="230" height="154" rx="12" fill="{col}" opacity=".12"/>' + txt(x + 115, 32, name, 14, col, weight=800) + txt(x + 115, 52, parts, 11, INK)
        if k == 0: s += box(x + 55, 70, 120, 34, "Encoder", BLUE, fs=12)
        elif k == 1: s += box(x + 55, 70, 120, 34, "Decoder", GREEN, fs=12)
        else: s += box(x + 14, 70, 94, 34, "Encoder", BLUE, fs=12) + box(x + 122, 70, 94, 34, "Decoder", GREEN, fs=12) + arrow([(x + 108, 87), (x + 122, 87)], ORANGE)
        for i, line in enumerate(ex.split(": ")): s += txt(x + 115, 126 + i * 15, line, 9.8, GREY)
    return s + "</svg>"


# ------------------------------------------------------------------ page content
enc1, enc12 = R["enc_self"]["0"], R["enc_self"]["11"]
decs, cross = R["dec_self"]["11"], R["cross"]["11"]
srcw = SRC; real_src = [i for i, t in enumerate(R["src_tok"]) if t not in ("eng_Latn", "</s>")]


def look(mat, i):  # most-attended other word for position i (excluding self and the language/end markers), plus share on markers
    cands = [(mat[i][j], j) for j in real_src if j != i]; best = max(cands)
    return srcw[best[1]], best[0]


def marker_share(row): return sum(row[j] for j in range(len(row)) if j not in real_src)


tbl_enc = "".join(f"<tr><td>{esc(SRC[i])}</td><td>{esc(look(enc1, i)[0])} <span class='m'>({look(enc1, i)[1] * 100:.0f}%)</span></td><td>{esc(look(enc12, i)[0])} <span class='m'>({look(enc12, i)[1] * 100:.0f}%)</span></td></tr>" for i in real_src)
cross_rows = []
for t in range(T):
    row = cross[t + 1] if len(cross) > t + 1 else cross[t]; srcrow = row[:S]
    cands = [(srcrow[j], j) for j in real_src]; w, j = max(cands)
    cross_rows.append(f"<tr><td class='kn'>{esc(OUT[t])}</td><td>{esc(SRC[j])} <span class='m'>({w * 100:.0f}%)</span></td><td>{marker_share(srcrow) * 100:.0f}%</td></tr>")
steps_rows = "".join(f"<tr><td>{i + 1}</td><td class='kn' style='font-size:8.5pt'>{esc(' '.join(clean(p) for p in s['prefix'][-4:]))}{'…' if len(s['prefix']) > 4 else ''}</td><td class='kn'><b>{esc(clean(s['chosen']))}</b> {s['top'][0][1] * 100:.1f}%</td><td class='kn'>{esc(clean(s['top'][1][0]))} {s['top'][1][1] * 100:.1f}%</td><td class='kn'>{esc(clean(s['top'][2][0]))} {s['top'][2][1] * 100:.1f}%</td></tr>" for i, s in enumerate(R["steps"]))
emb_rows = "".join(f"<tr><td>{i + 1}</td><td>{esc(SRC[i])}</td><td>{R['src_ids'][i]}</td><td class='n'>{', '.join(f'{v:+.2f}' for v in R['emb_first4'][i])}</td><td class='n'>{', '.join(f'{v:+.2f}' for v in R['x0_first4'][i])}</td></tr>" for i in range(S))

CSS = """
@page { size: A4; margin: 13mm 13mm 14mm 13mm; @bottom-center { content: counter(page); font: 9pt 'Segoe UI', sans-serif; color: #777; } }
@page :first { margin: 0; @bottom-center { content: none; } }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font: 10pt/1.55 'Segoe UI', 'Nirmala UI', sans-serif; color: #1d1d3a; }
.kn { font-family: 'Nirmala UI', 'Noto Sans Kannada', sans-serif; }
.page { page-break-after: always; break-after: page; } .page:last-child { page-break-after: auto; }
h1.t { font-size: 22pt; line-height: 1.15; margin: 0 0 3mm; } h2 { font-size: 15pt; margin: 0 0 2mm; color: #2e2a6b; border-bottom: 2px solid #d9dcef; padding-bottom: 1.5mm; } h3 { font-size: 11pt; margin: 4mm 0 1mm; color: #6250d6; }
p { margin: 0 0 2.2mm; } .m { color: #6b6b88; } .small { font-size: 8.8pt; color: #4a4a68; }
.cover { height: 297mm; padding: 28mm 20mm 18mm; background: linear-gradient(160deg, #16163f 0%, #2e2a6b 60%, #3a3590 100%); color: #fff; position: relative; overflow: hidden; }
.cover .tag { letter-spacing: .14em; text-transform: uppercase; font-size: 9.5pt; color: #b9b8e8; margin-bottom: 12mm; } .cover h1 { font-size: 34pt; line-height: 1.12; margin: 0 0 6mm; } .cover h2 { font-size: 15pt; font-weight: 400; color: #dcdcf6; border: 0; margin: 0 0 14mm; line-height: 1.5; }
.cover .ex { background: rgba(255,255,255,.1); border-radius: 4mm; padding: 5mm 6mm; font-size: 12pt; max-width: 150mm; } .cover .ex b { color: #ffd98a; } .cover .foot { position: absolute; left: 20mm; right: 20mm; bottom: 16mm; font-size: 9.5pt; color: #b9b8e8; border-top: 1px solid rgba(255,255,255,.25); padding-top: 4mm; }
.box { border-radius: 3mm; padding: 3mm 4.5mm; margin: 3mm 0; break-inside: avoid; } .box.pl { background: #e6f6ee; border-left: 4px solid #159a6a; } .box.rw { background: #fff1e8; border-left: 4px solid #d9541f; } .box.mt { background: #ece9fd; border-left: 4px solid #6250d6; } .box.wn { background: #fff3dc; border-left: 4px solid #c27a10; }
.box b.h { display: block; font-size: 8pt; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .6mm; }
.pl b.h { color: #159a6a; } .rw b.h { color: #d9541f; } .mt b.h { color: #6250d6; } .wn b.h { color: #c27a10; }
.f { font-family: 'Cambria Math', Cambria, serif; font-size: 11pt; background: #f1f1f9; border-radius: 2mm; padding: 1.8mm 3.5mm; margin: 1.5mm 0; line-height: 1.8; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 2mm 0 3mm; break-inside: auto; } th { background: #2e2a6b; color: #fff; text-align: left; padding: 1.4mm 2mm; } td { padding: 1.3mm 2mm; border-bottom: 1px solid #d9dcef; vertical-align: top; } tr { break-inside: avoid; } tr:nth-child(even) td { background: #f8f8fd; } td.n { font-variant-numeric: tabular-nums; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; } .cols3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3mm; } .card { border: 1px solid #d9dcef; border-radius: 3mm; padding: 2.5mm 3.5mm; background: #fff; break-inside: avoid; }
svg { max-width: 100%; height: auto; display: block; margin: 0 auto; } figure.sm svg { max-height: 118mm; } figure.md svg { max-height: 150mm; } figure.dd svg { max-height: 112mm; } figure { margin: 2mm 0; text-align: center; break-inside: avoid; } figcaption { font-size: 8.5pt; color: #5a5977; margin-top: 1mm; }
.num { display: inline-block; width: 5mm; height: 5mm; border-radius: 50%; background: #1d1d3a; color: #fff; text-align: center; font-size: 8pt; line-height: 5mm; font-weight: 700; margin-right: 1.5mm; }
.chip { display: inline-block; border: 1px solid #c8cbe6; border-radius: 2mm; padding: 0 2mm; margin: 0 .8mm .8mm 0; background: #fff; font-size: 9pt; }
"""

P = []
P.append(f"""<div class="page cover"><div class="tag">Step-by-step visual guide</div><h1>The Transformer:<br>Encoder and Decoder, Flow by Flow</h1>
<h2>How a sentence travels through the network, drawn as diagrams and followed with real numbers from a real translation model.</h2>
<div class="ex"><b>Worked example</b><br>A visitor in Mysuru asks, in English:<br><i>“I am going to Mysuru tomorrow.”</i><br><span class="kn">The Transformer answers in Kannada: ನಾಳೆ ನಾನು ಮೈಸೂರಿಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ.</span></div>
<p style="margin-top:12mm;color:#dcdcf6;max-width:150mm">Inside: the big architecture diagram, a 13-step flow, one encoder layer and one decoder layer opened up, the attention matrices of the real model, the token-by-token decoding loop, and how training differs from use.</p>
<div class="foot">Numbers are from NLLB-200 (distilled 600M, Meta AI): {CFG['enc_layers']} encoder + {CFG['dec_layers']} decoder layers, width {CFG['d_model']}, {CFG['heads']} attention heads, {CFG['params'] / 1e6:.0f} million parameters. Diagrams are original.</div></div>""")

P.append(f"""<div class="page"><h2>1. The job and the two halves</h2>
<div class="box rw"><b class="h">Real-world picture</b>An interpreter team at the Mysuru bus stand. The <b>reader</b> (encoder) listens to the whole English sentence once and writes detailed notes on every word, including how the words relate. The <b>writer</b> (decoder) then produces the Kannada <em>one word at a time</em>; before each word she re-reads the reader’s notes and her own sentence so far.</div>
<div class="cols"><div class="card"><b style="color:#2a78d6">Encoder: understands</b><p class="small">Input: all source words together. Output: one vector (1024 numbers) per word, each already informed by every other word. Runs <b>once</b>.</p></div>
<div class="card"><b style="color:#159a6a">Decoder: writes</b><p class="small">Input: the words written so far + the encoder’s notes. Output: probabilities for the next word. Runs <b>once per output word</b>, in a loop.</p></div></div>
<h3>Our example in tokens</h3><p>Models do not read words; they read <b>tokens</b> (word pieces) and each token has an ID. The real NLLB tokenizer cuts the sentence into {S} pieces:</p>
<p>{''.join(f"<span class='chip'>{esc(t)}</span>" for t in R['src_tok'])}</p>
<p class="small">“▁” marks the start of a word. “Mys” + “uru” shows a rare name split into pieces; <code>eng_Latn</code> tells the model the source language and <code>&lt;/s&gt;</code> marks the end. IDs: {', '.join(str(i) for i in R['src_ids'])}.</p>
<p>The Kannada answer is also written in pieces, one per loop turn ({T} tokens including the end marker):</p><p class="kn">{''.join(f"<span class='chip'>{esc(t)}</span>" for t in R['out_tok'])}</p>
<figure>{d_variants()}<figcaption>The same two halves, used alone or together: this guide follows the full encoder + decoder (translation).</figcaption></figure>
<div class="box wn"><b class="h">Reading the diagrams</b><span style="color:#2a78d6"><b>Blue</b></span> = encoder, <span style="color:#159a6a"><b>green</b></span> = decoder, <span style="color:#6250d6"><b>purple</b></span> = self-attention, <span style="color:#d9541f"><b>orange</b></span> = cross-attention (decoder looks at the encoder), <span style="color:#c27a10"><b>amber</b></span> = feed-forward, grey = add &amp; normalise and plumbing. Numbered circles match the 13 steps on the next pages.</div></div>""")

P.append(f"""<div class="page"><h2>2. The complete architecture</h2>
<figure>{d_big()}<figcaption>Encoder (left) runs once; its output is given to every decoder layer (orange arrow). The decoder (right) runs in a loop, feeding each chosen token back in. “Add &amp; LayerNorm” is drawn as in the 2017 paper; NLLB applies the LayerNorm before each sub-layer instead (see the layer diagrams).</figcaption></figure></div>""")

P.append(f"""<div class="page"><h2>3. The whole flow in 13 steps</h2>
<figure class='sm'>{d_flow()}</figure>
<div class="box pl"><b class="h">In plain words</b>Phase A turns the English sentence into “understood notes” (steps 1 to 7). Phase B uses those notes to write Kannada token by token (steps 8 to 13). Steps 8 to 13 repeat until the model chooses the end token. The encoder is never run again.</div>
<table><thead><tr><th>#</th><th>What happens</th><th>Shape in the real model</th></tr></thead><tbody>
<tr><td>1</td><td>Text is cut into tokens and each gets an ID</td><td>{S} integers</td></tr><tr><td>2</td><td>IDs become vectors (embedding) and a position signal is added</td><td>{S} × {CFG['d_model']}</td></tr>
<tr><td>3</td><td>Self-attention: each word gathers information from all words</td><td>{S} × {CFG['d_model']} (16 heads of 64)</td></tr><tr><td>4</td><td>Add the skip connection, normalise</td><td>unchanged</td></tr>
<tr><td>5</td><td>Feed-forward network, word by word</td><td>{CFG['d_model']} → {CFG['ffn']} → {CFG['d_model']}</td></tr><tr><td>6</td><td>Add skip, normalise; steps 3 to 6 form one layer, repeated {CFG['enc_layers']} times</td><td>{S} × {CFG['d_model']}</td></tr>
<tr><td>7</td><td>Encoder output = memory handed to the decoder</td><td>{S} × {CFG['d_model']}</td></tr><tr><td>8</td><td>Kannada tokens so far: embed + position</td><td>t × {CFG['d_model']}</td></tr>
<tr><td>9</td><td>Masked self-attention over the Kannada written so far</td><td>t × {CFG['d_model']}</td></tr><tr><td>10</td><td>Cross-attention: look at the encoder memory</td><td>t × {CFG['d_model']}</td></tr>
<tr><td>11</td><td>Feed-forward (each decoder layer has steps 9 to 11, repeated {CFG['dec_layers']} times)</td><td>t × {CFG['d_model']}</td></tr><tr><td>12</td><td>Linear layer + softmax over the whole vocabulary (last position only is needed)</td><td>{CFG['vocab']:,} probabilities</td></tr>
<tr><td>13</td><td>Choose a token (greedy: the most probable), append, loop</td><td>t grows by 1</td></tr></tbody></table></div>""")

P.append(f"""<div class="page"><h2>4. Steps 1 and 2: tokens become vectors, plus position</h2>
<div class="box pl"><b class="h">In plain words</b>The network only handles numbers. Each token ID selects a row of 1024 learned numbers (its <b>embedding</b>), a point in a space where words with related meaning sit near each other. But the network looks at all words at the same time, so it would not know word order; therefore a <b>position signal</b> is added so that “Mysuru” as word 6 differs from “Mysuru” as word 2.</div>
<div class="box mt"><b class="h">The maths</b>For token ID <i>x<sub>i</sub></i> at position <i>i</i>, the input to layer 1 is
<div class="f"><i>X<sub>i</sub></i> = √<i>d</i> · <i>E</i>[<i>x<sub>i</sub></i>] + <i>PE</i>(<i>i</i>), &nbsp;&nbsp; <i>d</i> = {CFG['d_model']}, √<i>d</i> = {R['embed_scale']:.0f}</div>
<div class="f"><i>PE</i>(<i>pos</i>, 2<i>k</i>) = sin( <i>pos</i> / 10000<sup>2<i>k</i>/<i>d</i></sup> ) &nbsp;&nbsp; <i>PE</i>(<i>pos</i>, 2<i>k</i>+1) = cos( <i>pos</i> / 10000<sup>2<i>k</i>/<i>d</i></sup> )</div>
Each dimension pair is a sine/cosine wave of a different wavelength, so every position gets a unique pattern and nearby positions get similar patterns. (NLLB stores all the sines in the first half of the 1024 numbers and the cosines in the second half; same idea.)</div>
<h3>Real numbers (first 4 of 1024 dimensions)</h3>
<table><thead><tr><th>#</th><th>token</th><th>ID</th><th>embedding (scaled), first 4</th><th>after adding position, first 4</th></tr></thead><tbody>{emb_rows}</tbody></table>
<p class="small">Compare the last two columns: the position signal shifts each vector. Both columns come from the real model’s input layer (positions here use the model’s own sinusoidal table).</p>
<div class="box rw"><b class="h">Real-world picture</b>Each word is written on a card with 1024 attributes; then a numbered sticker (its position) is stuck on the card so the card keeps its place in the line even after the cards are shuffled into one pile.</div></div>""")

P.append(f"""<div class="page"><h2>5. Steps 3 to 7: inside an encoder layer</h2>
<figure class='md'>{d_encoder_layer()}<figcaption>Diagram of one encoder layer. Skip connections (grey) add each sub-layer’s input back to its output.</figcaption></figure>
<div class="box pl"><b class="h">Step 3: self-attention, in plain words</b>Every word asks a question (<b>Q</b>uery), every word offers a label (<b>K</b>ey) and content (<b>V</b>alue). A word compares its query with all keys; the better the match, the more of that word’s value it takes. After this, “Mysuru” carries information about “going” and “tomorrow”.</div>
<div class="box mt"><b class="h">The maths</b>With <i>X</i> the {S} × 1024 input, learned matrices <i>W<sub>Q</sub></i>, <i>W<sub>K</sub></i>, <i>W<sub>V</sub></i> (1024 × 1024):
<div class="f"><i>Q</i> = <i>X W<sub>Q</sub></i>, &nbsp; <i>K</i> = <i>X W<sub>K</sub></i>, &nbsp; <i>V</i> = <i>X W<sub>V</sub></i> &nbsp;&nbsp;&nbsp; Attention = softmax( <i>Q K</i><sup>T</sup> / √<i>d<sub>k</sub></i> ) · <i>V</i></div>
The split into 16 heads gives each head <i>d<sub>k</sub></i> = 64; each head learns its own kind of relationship; the 16 results are joined and mixed by one more matrix. <b>Step 4 (skip):</b> <i>Y</i> = <i>X</i> + Attention( LayerNorm(<i>X</i>) ). <b>Step 5:</b> FFN(<i>u</i>) = <i>W</i><sub>2</sub> ReLU(<i>W</i><sub>1</sub><i>u</i> + <i>b</i><sub>1</sub>) + <i>b</i><sub>2</sub> with 1024 → {CFG['ffn']} → 1024, applied to each word separately. <b>Step 6 (skip):</b> <i>Z</i> = <i>Y</i> + FFN( LayerNorm(<i>Y</i>) ). The 2017 paper normalises after the Add instead (post-norm); NLLB, like most recent models, normalises before each sub-layer (pre-norm). Both use the same ingredients.</div></div>""")

P.append(f"""<div class="page"><h2>6. The real encoder attention for our sentence</h2>
<p>These are the actual attention weights (average over the 16 heads, in percent) of the real model on “I am going to Mysuru tomorrow.” Row = the word that is looking; column = the word it looks at; each row adds up to 100.</p>
<div class="cols"><figure>{heat(enc1, SRC, SRC, 25)}<figcaption>Encoder layer 1 (first layer)</figcaption></figure><figure>{heat(enc12, SRC, SRC, 25)}<figcaption>Encoder layer 12 (last layer)</figcaption></figure></div>
<h3>Where each word looks most (excluding itself and the markers)</h3>
<table><thead><tr><th>word</th><th>layer 1 looks at</th><th>layer 12 looks at</th></tr></thead><tbody>{tbl_enc}</tbody></table>
<div class="box wn"><b class="h">How to read this honestly</b>This is an average over 16 heads, so it blurs different specialisations. Early layers lean on neighbouring words and the end marker; later layers mix more widely. Attention maps show where information <em>can</em> flow; they are not a full explanation of the model’s reasoning. A large share often sits on the end marker <code>&lt;/s&gt;</code> and the language tag, which act as “rest” positions.</div>
<div class="box pl"><b class="h">Step 7: the encoder output (the memory)</b>Think of twelve rounds of a meeting: in each round every word listens to all the others (attention), then thinks on its own (feed-forward). After round 12 the {S} vectors (10 × 1024) are the <b>memory</b>: computed once, kept, and read by the decoder in every layer at every loop turn.</div></div>""")

P.append(f"""<div class="page"><h2>7. Steps 8 to 11: inside a decoder layer</h2>
<figure class='dd'>{d_decoder_layer()}<figcaption>One decoder layer: masked self-attention, cross-attention to the encoder memory, feed-forward.</figcaption></figure>
<div class="cols"><div><div class="box pl"><b class="h">Step 9: masked self-attention</b>The decoder is writing left to right, so when it works on word <i>t</i> it may only look at words 1…<i>t</i>. Looking at later words would be cheating during training. A <b>mask</b> blocks them: their scores are set to −∞ before the softmax so their weight is 0.</div>
<div class="f" style="font-size:10pt">softmax( <i>Q K</i><sup>T</sup> / √<i>d<sub>k</sub></i> + <i>M</i> ) <i>V</i>, &nbsp; <i>M<sub>ij</sub></i> = 0 if <i>j</i> ≤ <i>i</i>, else −∞</div></div>
<figure>{d_causal()}</figure></div>
<div class="box pl"><b class="h">Step 10: cross-attention, the bridge</b>Same attention formula, but the three ingredients come from two places: the <b>query</b> from the Kannada written so far, the <b>keys and values</b> from the encoder memory. This is how the writer “looks at the reader’s notes”.
<div class="f"><i>Q</i> = <i>Y</i><sub>dec</sub> <i>W<sub>Q</sub></i>, &nbsp; <i>K</i> = <i>M</i><sub>enc</sub> <i>W<sub>K</sub></i>, &nbsp; <i>V</i> = <i>M</i><sub>enc</sub> <i>W<sub>V</sub></i></div></div></div>""")

P.append(f"""<div class="page"><h2>8. The real cross-attention: Kannada looks back at English</h2>
<p>Real weights of decoder layer 12 (average over heads, percent), computed while the model processes its own output. Rows are the Kannada pieces the decoder is working on, columns the English pieces in the encoder memory.</p>
<figure>{heat(cross, DEC_IN, SRC, 25, lab_w=96)}<figcaption>Cross-attention, decoder layer 12. The first two rows are the start tokens (&lt;/s&gt;, kan_Knda).</figcaption></figure>
<h3>Which English word does each Kannada piece look at most?</h3>
<table><thead><tr><th>Kannada piece (being predicted)</th><th>most-attended English word</th><th>weight on markers</th></tr></thead><tbody>{''.join(cross_rows)}</tbody></table>
<div class="box wn"><b class="h">Honest reading</b>Attention often rests heavily on the end marker; only the share left over is informative, and averaging 16 heads blurs it. In the table, the two pieces of ನಾಳೆ (“tomorrow”) give “tomorrow” the top share among the real words, and the place-name pieces (ಮ, ೈಸ, ೂ, ರಿಗೆ) look at “Mys” and “uru”, as a human would hope. Other rows are less intuitive (for example ನಾನು, “I”, looks mostly at “Mys”). That is what the model really does: attention weights are evidence of where information can flow, not a complete explanation.</div></div>""")

P.append(f"""<div class="page"><h2>9. Steps 12 and 13: from vectors to the next word</h2>
<div class="box pl"><b class="h">In plain words</b>The decoder’s final vector for the last position is turned into one score for every token in the vocabulary ({CFG['vocab']:,} of them) by a linear layer; <b>softmax</b> converts the scores into probabilities that sum to 1. The most probable token is chosen (greedy decoding), added to the decoder input, and the loop runs again.</div>
<div class="box mt"><b class="h">The maths</b><div class="f"><i>z</i> = <i>W<sub>out</sub></i> <i>h<sub>t</sub></i> &nbsp;&nbsp; <i>p<sub>k</sub></i> = e<sup><i>z<sub>k</sub></i></sup> / Σ<sub><i>j</i></sub> e<sup><i>z<sub>j</sub></i></sup> &nbsp;&nbsp; next token = argmax<sub><i>k</i></sub> <i>p<sub>k</sub></i></div>
<i>W<sub>out</sub></i> is 1024 × {CFG['vocab']:,}; in NLLB it shares its values with the token embedding table.</div>
<h3>The real decoding loop for our sentence</h3>
<table><thead><tr><th>turn</th><th>last tokens fed in</th><th>chosen (probability)</th><th>2nd choice</th><th>3rd choice</th></tr></thead><tbody>{steps_rows}</tbody></table>
<p class="small m">Pieces that begin with a vowel sign are shown with a dotted circle by some fonts; they are fragments of a Kannada word, not separate letters.</p><p>Result: <b class="kn" style="font-size:12pt">{esc(R['text'])}</b> &nbsp; <span class="m">(“Tomorrow I am going to Mysuru.”)</span></p>
<div class="box rw"><b class="h">Real-world picture</b>The writer at the bus stand pens one syllable at a time. At turn 1 she is 58% sure the sentence starts “ನಾ…” (the start of ನಾನು, “I”) and 35% sure it starts with the full word ನಾನು; at turn 2 she is 95% sure the piece after ನಾ is ಳೆ (ನಾಳೆ, “tomorrow”). Each choice is then fed back in, so the next prediction depends on everything written so far.</div>
<div class="box wn"><b class="h">Why the model put “tomorrow” first</b>Kannada word order is flexible. Both “ನಾನು ನಾಳೆ ಮೈಸೂರಿಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ” and “ನಾಳೆ ನಾನು …” are correct; a model trained on real text picked the one it found most probable. Beam search or sampling could pick the other. Pieces such as ಮ + ೈಸ + ೂ + ರಿಗೆ spell ಮೈಸೂರಿಗೆ (“to Mysuru”) one fragment at a time.</div></div>""")

P.append(f"""<div class="page"><h2>10. Training versus using the model</h2>
<figure>{d_training()}</figure>
<div class="box pl"><b class="h">Teacher forcing</b>During training we know the correct translation, so we feed it (shifted by one) to the decoder and ask for the next word at <b>every position at once</b>. This is fast, and one wrong guess cannot spoil the later inputs. During real use there is no answer to feed, so the decoder must use its <b>own</b> previous words, one turn at a time.</div>
<div class="box mt"><b class="h">The loss and the update</b><div class="f"><i>L</i> = −(1/<i>T</i>) Σ<sub><i>t</i></sub> ln <i>p<sub>t</sub></i>( <i>y<sub>t</sub></i> ) &nbsp;&nbsp; then backpropagation and an optimiser step <i>w</i> ← <i>w</i> − η·Adam(∂<i>L</i>/∂<i>w</i>)</div>
At training time the causal mask (step 9) is what makes the parallel pass legitimate: each position still sees only the earlier words. At inference, caching the keys and values of earlier tokens avoids recomputing them every turn; the encoder memory is computed once.</div>
<h3>Cost and scale</h3>
<table><thead><tr><th>Quantity</th><th>Our example</th><th>Why it matters</th></tr></thead><tbody>
<tr><td>Encoder passes</td><td>1</td><td>Source never changes</td></tr><tr><td>Decoder passes</td><td>{T} (one per output token)</td><td>Longer answers cost more turns</td></tr>
<tr><td>Self-attention comparisons per layer</td><td>{S} × {S} = {S * S} (encoder)</td><td>Grows with the square of sentence length</td></tr>
<tr><td>Parameters</td><td>{CFG['params']:,}</td><td>Mostly in the feed-forward layers and the {CFG['vocab']:,}-token embedding table</td></tr></tbody></table></div>""")

P.append(f"""<div class="page"><h2>11. Cheat sheet</h2>
<table><thead><tr><th>Part</th><th>Reads</th><th>Mixes information by</th><th>Produces</th></tr></thead><tbody>
<tr><td><b style="color:#2a78d6">Encoder layer</b></td><td>all source words at once</td><td>self-attention (all ↔ all), then feed-forward</td><td>refined vector per source word</td></tr>
<tr><td><b style="color:#159a6a">Decoder: masked self-attention</b></td><td>target words so far</td><td>attention to earlier target words only</td><td>context of what is already written</td></tr>
<tr><td><b style="color:#d9541f">Decoder: cross-attention</b></td><td>query: target; keys/values: encoder memory</td><td>attention from target to source</td><td>the source information relevant to the next word</td></tr>
<tr><td><b style="color:#c27a10">Feed-forward</b></td><td>each position alone</td><td>two linear layers with ReLU</td><td>per-word processing</td></tr>
<tr><td>Add &amp; LayerNorm</td><td>sub-layer input and output</td><td>skip connection, then rescale</td><td>stable, trainable deep stacks</td></tr>
<tr><td>Linear + softmax</td><td>last decoder vector</td><td>one score per vocabulary token</td><td>probabilities for the next token</td></tr></tbody></table>
<h3>Questions people ask</h3>
<div class="cols"><div class="card"><b>Why “Q, K, V”?</b><p class="small">Query = what I am looking for, Key = what I offer to be found by, Value = what I give if chosen. Matching Q against K decides how much of each V is taken.</p></div>
<div class="card"><b>Why divide by √d<sub>k</sub>?</b><p class="small">Dot products of long vectors grow large and push the softmax to near 0/1, which stalls learning. Dividing keeps the scores moderate.</p></div>
<div class="card"><b>Why skip connections?</b><p class="small">They give the signal and the gradient a direct path through the stack, so 12 layers can train.</p></div>
<div class="card"><b>Is the decoder slower?</b><p class="small">Yes: one pass per output token. The encoder reads the whole input in one parallel pass.</p></div></div>
<div class="box mt"><b class="h">One line to remember</b>The encoder turns the source sentence into notes (once). The decoder writes the answer one token at a time, each time consulting its own text so far (masked self-attention) and the notes (cross-attention), then choosing the most probable next token.</div>
<p class="small m" style="margin-top:6mm">Sources: Vaswani et al., “Attention Is All You Need” (2017). Real tensors: NLLB-200 distilled 600M, Meta AI (research and education use). Companion pages in this repository: Transformers_Explained_Overview.html and NMT_Pipeline_Explained.html.</p></div>""")

HTML = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Transformer encoder and decoder flow</title><style>{CSS}</style></head><body>{''.join(P)}</body></html>"
html_path = os.path.join(HERE, "flow.html")
open(html_path, "w", encoding="utf-8").write(HTML)

profile = tempfile.mkdtemp(prefix="edge_pdf_")
tmp_pdf = os.path.join(tempfile.gettempdir(), "transformer_flow_tmp.pdf")
if os.path.exists(tmp_pdf): os.remove(tmp_pdf)
cmd = [EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-first-run", "--no-pdf-header-footer", f"--print-to-pdf={tmp_pdf}", "--virtual-time-budget=10000", "file:///" + html_path.replace("\\", "/")]
subprocess.run(cmd, capture_output=True, text=True, timeout=240)
last = -1
for _ in range(60):
    if os.path.exists(tmp_pdf):
        sz = os.path.getsize(tmp_pdf)
        if sz > 0 and sz == last: break
        last = sz
    time.sleep(1)
os.replace(tmp_pdf, OUT_PDF)
print("wrote", OUT_PDF, os.path.getsize(OUT_PDF) // 1024, "KB")
