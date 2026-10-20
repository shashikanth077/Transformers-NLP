"""Builds the Kannada study notes (PDF) from the real experiment data.

  python study_guide/kannada/build_kannada_notes.py

Steps: read study_guide/data.json -> compute all numbers -> write notes_kn.html -> print to PDF with
Microsoft Edge (headless; correct Kannada shaping, fonts embedded) -> second pass fills the table of
contents with real page numbers -> docs/Transformers_NLP_Kannada_Notes.pdf
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import pymupdf
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
DOCS = os.path.join(ROOT, "docs")
HTML = os.path.join(HERE, "notes_kn.html")
PDF_OUT = os.path.join(DOCS, "Transformers_NLP_Kannada_Notes.pdf")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

import kn_content_a as A  # noqa: E402
import kn_content_b as B  # noqa: E402

CHAPTERS = [
    ("c1", "ಅಧ್ಯಾಯ 1 · ಎನ್ಎಲ್ಪಿ ಅಂದರೆ ಏನು?", "WHAT IS NLP?"),
    ("c2", "ಅಧ್ಯಾಯ 2 · ನಮ್ಮ ಪ್ರಾಜೆಕ್ಟ್: ಒಂದು ಕತೆ", "THE PROJECT AS A STORY"),
    ("c3", "ಅಧ್ಯಾಯ 3 · ಹಳೆಯ ವಿಧಾನಗಳು", "OLDER APPROACHES"),
    ("c4", "ಅಧ್ಯಾಯ 4 · ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್: ಐದು ಐಡಿಯಾ", "HOW A TRANSFORMER WORKS"),
    ("c5", "ಅಧ್ಯಾಯ 5 · ಕನ್ನಡ ಮತ್ತು ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್", "KANNADA AND TRANSFORMERS"),
    ("c6", "ಅಧ್ಯಾಯ 6 · ನಮ್ಮ ಪ್ರಯೋಗ ಹೇಗೆ ನಡೆಯಿತು", "THE EXPERIMENT, STEP BY STEP"),
    ("c7", "ಅಧ್ಯಾಯ 7 · ಫಲಿತಾಂಶಗಳು: ಸಂಖ್ಯೆಗಳು ಏನು ಹೇಳುತ್ತವೆ?", "RESULTS"),
    ("c8", "ಅಧ್ಯಾಯ 8 · ಯಾವ ಮಾಡೆಲ್ ಯಾವಾಗ?", "WHICH MODEL WHEN"),
    ("c9", "ಅಧ್ಯಾಯ 9 · ವೈವಾ ಪ್ರಶ್ನೆ–ಉತ್ತರ", "LIKELY EXAMINER QUESTIONS"),
    ("c10", "ಅಧ್ಯಾಯ 10 · ಪದಕೋಶ", "GLOSSARY"),
    ("c11", "ಅಧ್ಯಾಯ 11 · ಸ್ವಯಂ ಪರೀಕ್ಷೆ ಮತ್ತು ಭಾಷಣ ಅಭ್ಯಾಸ", "SELF-TEST AND SPEECH PRACTICE"),
    ("c12", "ಅಧ್ಯಾಯ 12 · ಒಂದು ಪುಟದ ಸಾರಾಂಶ", "ONE-PAGE REVISION SHEET"),
]

PAGE_BREAK_CHAPTERS = ["c1", "c7", "c10", "c12"]   # heavy chapters start on a new page; the others flow on

# Kannada abbreviations / loanwords that need a zero-width non-joiner so letters do not fuse into conjuncts
ZWNJ = "\u200c"
FIXES = {
    "ಎನ್ಎಲ್ಪಿ": f"ಎನ್{ZWNJ}ಎಲ್{ZWNJ}ಪಿ", "ಎಲ್ಎಸ್ಟಿಎಂ": f"ಎಲ್{ZWNJ}ಎಸ್{ZWNJ}ಟಿಎಂ", "ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್": f"ಟ್ರಾನ್ಸ್{ZWNJ}ಫಾರ್ಮರ್",
    "ಇನ್ಪುಟ್": f"ಇನ್{ZWNJ}ಪುಟ್", "ಟ್ರಾನ್ಸ್ಲೇಟ್": f"ಟ್ರಾನ್ಸ್{ZWNJ}ಲೇಟ್", "ಶಾರ್ಟ್ಕಟ್": f"ಶಾರ್ಟ್{ZWNJ}ಕಟ್", "ಲ್ಯಾಪ್ಟಾಪ್": f"ಲ್ಯಾಪ್{ZWNJ}ಟಾಪ್",
    "ಚಾಟ್ಬಾಟ್": f"ಚಾಟ್{ZWNJ}ಬಾಟ್", "ಸ್ಟಾರ್ಟ್ಅಪ್": f"ಸ್ಟಾರ್ಟ್{ZWNJ}ಅಪ್", "ಕೆಎಸ್ಆರ್ಟಿಸಿ": f"ಕೆ{ZWNJ}ಎಸ್{ZWNJ}ಆರ್{ZWNJ}ಟಿ{ZWNJ}ಸಿ", "ಬಿಎಂಟಿಸಿ": f"ಬಿ{ZWNJ}ಎಂ{ZWNJ}ಟಿ{ZWNJ}ಸಿ",
    "ಸಾಫ್ಟ್ಮ್ಯಾಕ್ಸ್": f"ಸಾಫ್ಟ್{ZWNJ}ಮ್ಯಾಕ್ಸ್", "ಮೆಕ್ನೆಮಾರ್": f"ಮೆಕ್{ZWNJ}ನೆಮಾರ್", "ಡಿಸ್ಟಿಲ್ಬರ್ಟ್": f"ಡಿಸ್ಟಿಲ್{ZWNJ}ಬರ್ಟ್", "ಡೇಟಾಸೆಟ್": "ಡೇಟಾಸೆಟ್",
    "ಬೈ-ಡೈರೆಕ್ಷನಲ್": "ಬೈ-ಡೈರೆಕ್ಷನಲ್", "ಫೀಡ್-ಫಾರ್ವರ್ಡ್": "ಫೀಡ್-ಫಾರ್ವರ್ಡ್", "ಸ್ಮಾರ್ಟ್ವಾಚ್": f"ಸ್ಮಾರ್ಟ್{ZWNJ}ವಾಚ್", "ರಿಕಾಲ್": "ರಿಕಾಲ್",
}


def fix_text(html):
    html = html.replace(ZWNJ, "")
    for k, v in FIXES.items():
        html = html.replace(k, v)
    return html


def kn_time(s):
    if s < 1:
        return "ಇಲ್ಲ"
    if s < 60:
        return f"{round(s)} ಸೆಕೆಂಡ್"
    if s < 3600:
        return f"{round(s / 60)} ನಿಮಿಷ"
    return f"{s / 3600:.1f} ಗಂಟೆ"


def numbers():
    data = json.load(open(os.path.join(ROOT, "study_guide", "data.json"), encoding="utf-8"))
    models = data["models"]
    M = {m["tag"]: m for m in models}
    order = [m["tag"] for m in models]
    sent = data["sentences"]
    ds = data["dataset"]
    N = {"acc": {}, "lat": {}, "params": {}, "train": {}, "labels": {}, "ci": {}}
    for t, m in M.items():
        N["acc"][t] = f"{m['acc'] * 100:.1f}"
        N["lat"][t] = f"{m['lat_ms']:.2f}" if m["lat_ms"] < 1 else f"{m['lat_ms']:.1f}"
        N["params"][t] = f"{m['params'] / 1e6:.1f} M" if m["params"] >= 1e5 else f"{round(m['params'] / 1e3)} k"
        N["train"][t] = kn_time(m["train_s"])
        N["labels"][t] = f"{m['train_size']:,}" if m["train_size"] else "0 (ತರಬೇತಿ ಇಲ್ಲ)"
        st = data["stats"][t]
        N["ci"][t] = (f"{st['lo'] * 100:.1f}", f"{st['hi'] * 100:.1f}")
    N["rank"] = sorted(M, key=lambda t: -M[t]["acc"])
    N["n_train"], N["n_dev"], N["n_test"] = f"{ds['train']:,}", f"{ds['dev']:,}", f"{ds['test']:,}"
    N["n_train_i"], N["n_dev_i"], N["n_test_i"] = ds["train"], ds["dev"], ds["test"]
    N["train_len"], N["test_len"] = f"{ds['train_len']:.1f}", f"{ds['test_len']:.1f}"
    N["lr_features"] = f"{data['lr']['n_features']:,}"
    N["gain"] = f"{round(float(N['acc']['ft_distilbert_8000']) - float(N['acc']['tfidf_svm']), 1):.1f}"
    N["latratio"] = int(round(M["ft_distilbert_8000"]["lat_ms"] / M["tfidf_svm"]["lat_ms"] / 10) * 10)

    def acc_on(rows, i):
        return sum(r["p"][i] == str(r["y"]) for r in rows) / len(rows)

    neg = [r for r in sent if r["n"]]
    con = [r for r in sent if r["c"]]
    N["neg_n"], N["con_n"] = len(neg), len(con)
    N["allwrong"] = sum(all(r["p"][i] != str(r["y"]) for i in range(len(order))) for r in sent)
    minus = "\u2212"
    sub = []
    for i, t in enumerate(order):
        allv, ng, cn = acc_on(sent, i), acc_on(neg, i), acc_on(con, i)
        sub.append((allv, t, f"{allv * 100:.1f}", f"{ng * 100:.1f}", f"{cn * 100:.1f}", f"{minus}{abs((ng - allv) * 100):.1f}", f"{minus}{abs((cn - allv) * 100):.1f}"))
    N["sub"] = [(t, a, b, c, d, e) for _, t, a, b, c, d, e in sorted(sub, key=lambda x: -x[0])]

    def mcn(ta, tb):
        ia, ib = order.index(ta), order.index(tb)
        a_only = sum(r["p"][ia] == str(r["y"]) and r["p"][ib] != str(r["y"]) for r in sent)
        b_only = sum(r["p"][ib] == str(r["y"]) and r["p"][ia] != str(r["y"]) for r in sent)
        n, k = a_only + b_only, min(a_only, b_only)
        p = binomtest(k, n, 0.5).pvalue if n else 1.0
        return a_only, b_only, ("< 0.001" if p < 0.001 else f"{p:.2f}".rstrip("0").rstrip("."))
    N["mc"] = {"ft_svm": mcn("ft_distilbert_8000", "tfidf_svm"), "lstm_svm": mcn("bilstm", "tfidf_svm")}

    ct = {n: a for n, a in data["curve"]["transformer"]}
    cf = {n: a for n, a in data["curve"]["tfidf"]}
    N["lc500_t"], N["lc500_c"] = f"{ct[500] * 100:.1f}", f"{cf[500] * 100:.1f}"
    N["lc500_gain"] = f"{round(float(N['lc500_t']) - float(N['lc500_c']), 1):.1f}"
    N["lc4000_t"] = f"{ct[4000] * 100:.1f}"
    N["lc_rows"] = "".join(f"<tr><td class='num'>{n:,}</td><td class='num'>{ct[n] * 100:.1f}</td><td class='num'>{cf[n] * 100:.1f}</td><td class='num'>+{round(float(f'{ct[n] * 100:.1f}') - float(f'{cf[n] * 100:.1f}'), 1):.1f}</td></tr>" for n in sorted(ct))
    custom = next(w for w in data["walk"] if w["i"] == -1)
    N["walk_ppos"] = f"{custom['p_pos'] * 100:.0f}"
    return N


def assemble(N, toc_pages):
    toc = '<ul class="toc">' + "".join(f'<li><span class="t">{t}</span><span class="p">{toc_pages.get(cid, "") if toc_pages else ""}</span></li>' for cid, t, _ in CHAPTERS) + "</ul>"
    body = A.cover(N) + A.howto(N).replace("{{TOC}}", toc) + A.ch1(N) + A.ch2(N) + A.ch3(N) + A.ch4(N) + A.ch5(N) + B.ch6(N) + B.ch7(N) + B.ch8(N) + B.ch9(N) + B.ch10(N) + B.ch11(N) + B.ch12(N)
    for cid in PAGE_BREAK_CHAPTERS:
        body = body.replace(f'class="ch" id="{cid}"', f'class="ch pb" id="{cid}"')
    css = open(os.path.join(HERE, "notes.css"), encoding="utf-8").read()
    html = f'<!doctype html><html lang="kn"><head><meta charset="utf-8"><title>ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಮತ್ತು ಎನ್ಎಲ್ಪಿ: ಕನ್ನಡದಲ್ಲಿ ಅಧ್ಯಯನ ಟಿಪ್ಪಣಿಗಳು</title><style>{css}</style></head><body>{body}</body></html>'
    return fix_text(html)


def render(html_text, pdf_path):
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html_text)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    url = "file:///" + HTML.replace("\\", "/")
    profile = os.path.join(tempfile.gettempdir(), "kn_notes_edge_profile")   # separate profile: do not hand off to a running Edge
    cmd = [EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-first-run", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", "--virtual-time-budget=10000", url]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    # the browser's main process may exit before its child has finished writing: wait for a stable file
    last, stable = -1, 0
    for _ in range(240):
        size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else -1
        stable = stable + 1 if size > 0 and size == last else 0
        last = size
        if stable >= 3:
            break
        time.sleep(0.5)
    if stable < 3:
        raise RuntimeError(f"Edge did not produce a PDF (exit {res.returncode}): {res.stderr[-600:]}")


def find_pages(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = {}
    for cid, _, sub in CHAPTERS:
        key = re.sub(r"\s+", "", sub)
        for i, p in enumerate(doc):
            txt = re.sub(r"\s+", "", p.get_text())
            if key in txt:
                pages[cid] = i + 1
                break
    return pages, len(doc)


if __name__ == "__main__":
    N = numbers()
    tmp = os.path.join(HERE, "_pass1.pdf")
    render(assemble(N, None), tmp)
    pages, n1 = find_pages(tmp)
    print("pass 1:", n1, "pages;", pages)
    for attempt in range(3):
        render(assemble(N, pages), PDF_OUT)
        pages2, n2 = find_pages(PDF_OUT)
        print(f"pass {attempt + 2}:", n2, "pages;", pages2)
        if pages2 == pages:
            break
        pages = pages2
    os.remove(tmp)
    missing = [c for c, _, _ in CHAPTERS if c not in pages]
    print("chapters not found in PDF:", missing or "none")
    print("saved", PDF_OUT, os.path.getsize(PDF_OUT) // 1024, "KB")
