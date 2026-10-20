"""Inline SVG diagrams with Kannada labels for the Kannada study notes."""

F = 'font-family="Nirmala UI, Segoe UI, sans-serif"'


def _box(x, y, w, h, text, fill="#fff", stroke="#9aa0cf", color="#1d1d3a", size=15, weight=600, sub=None, rx=10):
    t = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    if sub:
        t += f'<text x="{x + w / 2}" y="{y + h / 2 - 3}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{color}" {F}>{text}</text>'
        t += f'<text x="{x + w / 2}" y="{y + h / 2 + 15}" text-anchor="middle" font-size="{size - 4}" fill="{color}" opacity=".78" {F}>{sub}</text>'
    else:
        t += f'<text x="{x + w / 2}" y="{y + h / 2 + size * .35}" text-anchor="middle" font-size="{size}" font-weight="{weight}" fill="{color}" {F}>{text}</text>'
    return t


def _arrow(x1, y1, x2, y2, color="#6a6c9b", w=2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" marker-end="url(#ah)"/>')


DEFS = '<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#6a6c9b"/></marker></defs>'


def attention_kn():
    """Illustrative attention for the Kannada sentence 'ಈ ಸಿನಿಮಾ ಕೆಟ್ಟದ್ದಲ್ಲ' ('this movie is not bad')."""
    toks = ["ಈ", "ಸಿನಿಮಾ", "ಕೆಟ್ಟದ್ದು", "ಅಲ್ಲ"]
    xs = [95, 255, 430, 590]
    s = f'<svg viewBox="0 0 700 250" xmlns="http://www.w3.org/2000/svg">{DEFS}'
    w = {0: 0.05, 1: 0.20, 3: 0.60}              # attention paid by "ಕೆಟ್ಟದ್ದು" (illustrative numbers)
    for j, v in w.items():
        x2 = xs[j]; x1 = xs[2]
        mid = (x1 + x2) / 2
        s += f'<path d="M{x1},118 Q{mid},{118 - 85 - abs(x1 - x2) * .1} {x2},118" fill="none" stroke="#6250d6" stroke-width="{2 + v * 14}" opacity=".78" stroke-linecap="round"/>'
        s += f'<text x="{mid}" y="{118 - 52 - abs(x1 - x2) * .08}" text-anchor="middle" font-size="15" font-weight="700" fill="#6250d6" {F}>{int(v * 100)}%</text>'
    s += f'<path d="M{xs[2] - 18},122 q-26,38 18,38 q44,0 18,-38" fill="none" stroke="#6250d6" stroke-width="{2 + .15 * 14}" opacity=".78"/><text x="{xs[2]}" y="188" text-anchor="middle" font-size="14" font-weight="700" fill="#6250d6" {F}>15%</text>'
    for t, x in zip(toks, xs):
        hl = t == "ಕೆಟ್ಟದ್ದು"
        s += _box(x - 62, 118, 124, 46, t, fill="#fff3dc" if hl else "#fff", stroke="#d9541f" if hl else "#9aa0cf", size=19)
    s += f'<text x="{xs[2]}" y="212" text-anchor="middle" font-size="13" fill="#d9541f" {F}>↑ ಈ ಪದ ಉಳಿದ ಪದಗಳನ್ನು ನೋಡುತ್ತಿದೆ</text>'
    s += f'<text x="350" y="240" text-anchor="middle" font-size="13" fill="#5a5977" {F}>(ಸಂಖ್ಯೆಗಳು ಕಲಿಕೆಗಾಗಿ ಕೊಟ್ಟ ಉದಾಹರಣೆ ಮಾತ್ರ, ನಿಜವಾದ ಮಾಡೆಲ್‌ನ ಅಂಕೆಗಳಲ್ಲ)</text></svg>'
    return s


def sequential_vs_parallel():
    toks = ["ನಾನು", "ನಿನ್ನೆ", "ಮೈಸೂರಿಗೆ", "ಹೋದೆ"]
    xs = [60, 215, 370, 525]
    s = f'<svg viewBox="0 0 700 330" xmlns="http://www.w3.org/2000/svg">{DEFS}'
    s += f'<text x="10" y="24" font-size="16" font-weight="800" fill="#d9541f" {F}>ಎಲ್‌ಎಸ್‌ಟಿಎಂ: ಒಂದರ ನಂತರ ಒಂದು (ಸಾಲಿನಲ್ಲಿ ಟಿಕೆಟ್ ಕೌಂಟರ್)</text>'
    for i, (t, x) in enumerate(zip(toks, xs)):
        s += _box(x, 40, 110, 42, t, fill="#fde9e0", stroke="#d9541f", size=17)
        s += f'<text x="{x + 55}" y="104" text-anchor="middle" font-size="13" fill="#d9541f" {F}>ಹೆಜ್ಜೆ {i + 1}</text>'
        if i < 3:
            s += _arrow(x + 112, 61, xs[i + 1] - 3, 61, "#d9541f")
    s += f'<text x="10" y="150" font-size="16" font-weight="800" fill="#159a6a" {F}>ಟ್ರಾನ್ಸ್‌ಫಾರ್ಮರ್: ಎಲ್ಲರೂ ಒಮ್ಮೆಲೇ (ಹತ್ತು ಕೌಂಟರ್‌ಗಳು)</text>'
    for i in range(4):
        for j in range(i + 1, 4):
            x1, x2 = xs[i] + 55, xs[j] + 55
            if j - i == 1:
                s += f'<line x1="{x1 + 30}" y1="212" x2="{x2 - 30}" y2="212" stroke="#159a6a" stroke-width="2" opacity=".7"/>'
            else:
                mid = (x1 + x2) / 2; h = 28 * (j - i)
                s += f'<path d="M{x1},190 Q{mid},{190 - h} {x2},190" fill="none" stroke="#159a6a" stroke-width="2" opacity=".6"/>'
    for t, x in zip(toks, xs):
        s += _box(x, 190, 110, 42, t, fill="#e2f5ec", stroke="#159a6a", size=17)
    s += f'<text x="350" y="276" text-anchor="middle" font-size="15" fill="#159a6a" font-weight="700" {F}>ಪ್ರತಿ ಪದಕ್ಕೂ ಪ್ರತಿ ಪದದ ಜೊತೆ ನೇರ ಸಂಪರ್ಕ: ಒಂದೇ ಹೆಜ್ಜೆ</text>'
    s += f'<text x="350" y="304" text-anchor="middle" font-size="13" fill="#5a5977" {F}>ವೆಚ್ಚ: 4 ಪದ → 16 ಜೋಡಿ; 100 ಪದ → 10,000 ಜೋಡಿ. ಅಂದರೆ ಉದ್ದ ವಾಕ್ಯಗಳಿಗೆ ಅಟೆನ್ಷನ್ ದುಬಾರಿ</text></svg>'
    return s


def architecture_kn():
    s = f'<svg viewBox="0 0 560 560" xmlns="http://www.w3.org/2000/svg">{DEFS}'
    y = 505
    def blk(text, y, fill, stroke, sub=None):
        return _box(120, y, 320, 44, text, fill=fill, stroke=stroke, size=15, sub=sub)
    s += blk("ಇನ್‌ಪುಟ್ ವಾಕ್ಯ → ಟೋಕನ್‌ಗಳು", 505, "#f3f4fb", "#9aa0cf", None)
    s += _arrow(280, 505, 280, 487)
    s += blk("ಎಂಬೆಡ್ಡಿಂಗ್ + ಸ್ಥಾನ ಸಂಖ್ಯೆ", 443, "#e8effd", "#2a78d6")
    s += _arrow(280, 443, 280, 425)
    s += '<rect x="95" y="190" width="370" height="233" rx="16" fill="none" stroke="#2a78d6" stroke-width="2.2" stroke-dasharray="7 5"/>'
    s += f'<text x="372" y="184" font-size="15" font-weight="800" fill="#2a78d6" {F}>× L ಪದರಗಳು (layers)</text>'
    s += blk("ಮಲ್ಟಿ-ಹೆಡ್ ಅಟೆನ್ಷನ್", 365, "#e2f5ec", "#159a6a")
    s += _arrow(280, 365, 280, 349)
    s += blk("ಸೇರಿಸು + ಸಮತೋಲನ (Add & Norm)", 305, "#fff3dc", "#d9a441")
    s += _arrow(280, 305, 280, 289)
    s += blk("ಫೀಡ್-ಫಾರ್ವರ್ಡ್ (ಆಲೋಚನೆ)", 245, "#e2f5ec", "#159a6a")
    s += _arrow(280, 245, 280, 229)
    s += blk("ಸೇರಿಸು + ಸಮತೋಲನ (Add & Norm)", 197, "#fff3dc", "#d9a441")
    s += _arrow(280, 190, 280, 172)
    s += blk("[CLS] ಸಾರಾಂಶ ವೆಕ್ಟರ್", 128, "#f3f4fb", "#9aa0cf")
    s += _arrow(280, 128, 280, 110)
    s += blk("ವರ್ಗೀಕರಣ ಪದರ: softmax", 66, "#e8effd", "#2a78d6")
    s += _arrow(280, 66, 280, 50)
    s += _box(120, 6, 320, 40, "P(ಧನಾತ್ಮಕ) , P(ಋಣಾತ್ಮಕ)", fill="#e2f5ec", stroke="#159a6a", size=15, weight=800)
    s += f'<text x="10" y="287" font-size="12" fill="#d9541f" {F} transform="rotate(-90 14 287)" text-anchor="middle">ಶಾರ್ಟ್‌ಕಟ್ (residual)</text>'
    s += '</svg>'
    return s


def pipeline_kn():
    names = [("1", "ಪ್ರಶ್ನೆ", "RQ1–5"), ("2", "ಸಾಹಿತ್ಯ ಸಮೀಕ್ಷೆ", "Scholar, WoS"), ("3", "ಡೇಟಾ", "SST-2 ವಿಭಜನೆ"), ("4", "ಪದ ಎಣಿಕೆ", "TF-IDF"), ("5", "ಎಲ್‌ಎಸ್‌ಟಿಎಂ", "BiLSTM"),
             ("6", "ಫೈನ್-ಟ್ಯೂನ್", "BERT / DistilBERT"), ("7", "ರೆಡಿಮೇಡ್", "ಜೀರೋ-ಶಾಟ್"), ("8", "ಮೌಲ್ಯಮಾಪನ", "CI, McNemar"), ("9", "ವರದಿ", "ಪೇಪರ್, ಸ್ಲೈಡ್")]
    cols = {"1": "#e8effd", "2": "#e8effd", "3": "#f3f4fb", "4": "#dfeafb", "5": "#fde9e0", "6": "#e2f5ec", "7": "#e2f5ec", "8": "#ece9fd", "9": "#fff3dc"}
    s = f'<svg viewBox="0 0 700 270" xmlns="http://www.w3.org/2000/svg">{DEFS}'
    pos = []
    for i, (n, t, sub) in enumerate(names):
        row, col = (0, i) if i < 5 else (1, i - 5)
        x = 12 + col * 136 + (68 if row else 0); y = 20 + row * 125
        pos.append((x, y))
        s += _box(x, y, 126, 80, t, fill=cols[n], stroke="#9aa0cf", size=14, sub=sub)
        s += f'<circle cx="{x + 14}" cy="{y + 2}" r="13" fill="#2e2a6b"/><text x="{x + 14}" y="{y + 7}" text-anchor="middle" font-size="14" font-weight="800" fill="#fff" {F}>{n}</text>'
    for i in range(8):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if i == 4:
            s += f'<path d="M{x1 + 62},{y1 + 82} L{x1 + 62},{y1 + 112} L{x2 + 62},{y1 + 112} L{x2 + 62},{y2 - 2}" fill="none" stroke="#6a6c9b" stroke-width="2" marker-end="url(#ah)"/>'
        else:
            s += _arrow(x1 + 126, y1 + 40, x2 - 3, y2 + 40)
    s += f'<text x="350" y="262" text-anchor="middle" font-size="13" fill="#5a5977" {F}>ಒಂದು ಹೆಜ್ಜೆಯ ಫಲಿತಾಂಶ (JSON ಫೈಲ್) ಮುಂದಿನ ಹೆಜ್ಜೆಗೆ ಆಧಾರ</text></svg>'
    return s


def pretrain_kn():
    cols = [
        ("1 · ಪೂರ್ವ-ತರಬೇತಿ", "(Pre-training)", ["ಸಾಮಾನ್ಯ ವಿದ್ಯಾಭ್ಯಾಸ", "SSLC → PUC → ಡಿಗ್ರಿ"], ["ಬೇರೆಯವರು ಮಾಡಿದ್ದು:", "ದಿನಗಟ್ಟಲೆ, ಬಹಳ GPU"], "#e8effd", "#2a78d6"),
        ("2 · ಫೈನ್-ಟ್ಯೂನಿಂಗ್", "(Fine-tuning)", ["ಕೆಲಸದ ತರಬೇತಿ", "ಕಂಪನಿ ಸೇರಿದ ಮೇಲೆ 2 ವಾರ"], ["ನಾವು ಮಾಡಿದ್ದು:", "ನಿಮಿಷ–ಗಂಟೆ, ಲ್ಯಾಪ್‌ಟಾಪ್"], "#e2f5ec", "#159a6a"),
        ("3 · ಬಳಕೆ", "(Inference)", ["ಕೆಲಸ ಮಾಡುವುದು", "ವಿಮರ್ಶೆ → ಒಳ್ಳೆಯದೋ ಕೆಟ್ಟದೋ?"], ["ಮಿಲಿಸೆಕೆಂಡ್‌ಗಳಲ್ಲಿ", "ಉತ್ತರ"], "#fff3dc", "#d9a441"),
    ]
    s = f'<svg viewBox="0 0 700 270" xmlns="http://www.w3.org/2000/svg">{DEFS}'
    for i, (t, en, a_lines, c_lines, fill, st) in enumerate(cols):
        x = 8 + i * 232
        s += f'<rect x="{x}" y="10" width="216" height="222" rx="14" fill="{fill}" stroke="{st}" stroke-width="2"/>'
        s += f'<text x="{x + 108}" y="40" text-anchor="middle" font-size="16" font-weight="800" fill="{st}" {F}>{t}</text>'
        s += f'<text x="{x + 108}" y="60" text-anchor="middle" font-size="12" fill="#5a5977" {F}>{en}</text>'
        y = 98
        for k, line in enumerate(a_lines):
            s += f'<text x="{x + 108}" y="{y}" text-anchor="middle" font-size="{14 if k == 0 else 12}" font-weight="{700 if k == 0 else 400}" fill="#1d1d3a" {F}>{line}</text>'
            y += 24
        y += 22
        for line in c_lines:
            s += f'<text x="{x + 108}" y="{y}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{st}" {F}>{line}</text>'
            y += 22
        if i < 2:
            s += _arrow(x + 218, 120, x + 230, 120, "#6a6c9b", 3)
    s += f'<text x="350" y="256" text-anchor="middle" font-size="12.5" fill="#5a5977" {F}>ರೆಡಿಮೇಡ್ ಮಾಡೆಲ್ = ಬೇರೆಯವರು ಹಂತ 1 ಮತ್ತು 2 ಮುಗಿಸಿದ್ದನ್ನು ನೇರವಾಗಿ ಬಳಸುವುದು</text></svg>'
    return s


def kannada_words():
    s = f'<svg viewBox="0 0 700 270" xmlns="http://www.w3.org/2000/svg">{DEFS}'
    s += _box(250, 100, 200, 56, "ಮನೆ", fill="#fff3dc", stroke="#d9541f", size=24, weight=800, sub="house (ಮೂಲ ಪದ)")
    forms = [("ಮನೆಗೆ", "to the house", 20, 10), ("ಮನೆಯಲ್ಲಿ", "in the house", 245, 6), ("ಮನೆಯಿಂದ", "from the house", 480, 10),
             ("ಮನೆಯವರು", "people of the house", 20, 196), ("ಮನೆಯಲ್ಲಿದ್ದರು", "they were at home", 245, 200), ("ಮನೆಮಾತು", "a household word", 480, 196)]
    for t, en, x, y in forms:
        s += _box(x, y, 200, 56, t, fill="#fff", stroke="#9aa0cf", size=18, sub=en)
        cx, cy = x + 100, (y + 56 if y < 100 else y)
        s += _arrow(350 + (cx - 350) * .35, 128 + (cy - 128) * .35, cx - (cx - 350) * .12, cy - (cy - 128) * .0 + (4 if y < 100 else -4), "#6a6c9b", 1.8)
    s += '</svg>'
    return s


def split_bar_kn(train, dev, test):
    tot = train + dev + test
    w = 640
    wt = w * 0.78
    s = f'<svg viewBox="0 0 700 190" xmlns="http://www.w3.org/2000/svg">'
    s += f'<rect x="30" y="30" width="{wt}" height="46" rx="8" fill="#2a78d6"/><text x="{30 + wt / 2}" y="59" text-anchor="middle" font-size="16" font-weight="800" fill="#fff" {F}>ತರಬೇತಿ (train) {train:,}</text>'
    s += f'<rect x="{30 + wt + 6}" y="30" width="48" height="46" rx="8" fill="#6250d6"/><text x="{30 + wt + 30}" y="59" text-anchor="middle" font-size="13" font-weight="800" fill="#fff" {F}>{dev:,}</text>'
    s += f'<rect x="{30 + wt + 60}" y="30" width="38" height="46" rx="8" fill="#d9541f"/><text x="{30 + wt + 79}" y="59" text-anchor="middle" font-size="12" font-weight="800" fill="#fff" {F}>{test}</text>'
    s += f'<text x="30" y="108" font-size="14" font-weight="700" fill="#2a78d6" {F}>ಪಠ್ಯಪುಸ್ತಕ + ಅಭ್ಯಾಸ: ಮಾಡೆಲ್ ಇದರಿಂದ ಕಲಿಯುತ್ತದೆ</text>'
    s += f'<text x="30" y="132" font-size="14" font-weight="700" fill="#6250d6" {F}>ಮಾದರಿ ಪ್ರಶ್ನೆಪತ್ರಿಕೆ (dev, {dev:,}): ಯಾವ ಎಪಾಕ್ ಉತ್ತಮ ಎಂದು ಆರಿಸಲು</text>'
    s += f'<text x="30" y="156" font-size="14" font-weight="700" fill="#d9541f" {F}>ನಿಜವಾದ ಬೋರ್ಡ್ ಪರೀಕ್ಷೆ (test, {test}): ಒಮ್ಮೆ ಮಾತ್ರ ಬಳಕೆ, ಮೊದಲೇ ನೋಡುವುದಿಲ್ಲ</text>'
    s += f'<text x="30" y="180" font-size="12" fill="#5a5977" {F}>(ಪಟ್ಟಿಯ ಅಗಲ ಪ್ರಮಾಣಕ್ಕೆ ಅನುಗುಣವಾಗಿಲ್ಲ — dev ಮತ್ತು test ತುಂಬಾ ಚಿಕ್ಕವು)</text></svg>'
    return s


def cover_art():
    pts = [(60, 120), (170, 40), (300, 110), (420, 30), (520, 130), (360, 200), (190, 190)]
    s = '<svg width="560" height="260" viewBox="0 0 580 250" xmlns="http://www.w3.org/2000/svg">'
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            s += f'<line x1="{pts[i][0]}" y1="{pts[i][1]}" x2="{pts[j][0]}" y2="{pts[j][1]}" stroke="#fff" stroke-opacity=".16" stroke-width="2"/>'
    for (x, y), c in zip(pts, ["#1baf7a", "#8f87ff", "#f2b33d", "#1baf7a", "#ee7a4b", "#8f87ff", "#2a78d6"]):
        s += f'<circle cx="{x}" cy="{y}" r="15" fill="{c}"/><circle cx="{x}" cy="{y}" r="24" fill="{c}" opacity=".22"/>'
    return s + '</svg>'
