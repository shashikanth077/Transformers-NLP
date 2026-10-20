"""Builds the self-contained interactive study guide: docs/Project_Study_Guide.html

Inputs : study_guide/src/*  and  study_guide/data.json (written by export_data.py)
Output : one HTML file with all CSS, JavaScript and data inlined (works offline, no server needed).
Run    : python study_guide/build.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(os.path.dirname(HERE), "docs", "Project_Study_Guide.html")
JS_ORDER = ["content_a.js", "content_b.js", "core.js", "charts.js", "sections.js", "main.js"]


def read(name):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        return f.read()


data = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
data_js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
shell = read("shell.html")
page = (shell.replace("/*__CSS__*/", read("style.css"))
        .replace("<!--__SECTIONS__-->", read("sections.html"))
        .replace("/*__DATA__*/", data_js)
        .replace("/*__JS__*/", "\n".join(read(n).replace("</script", "<\\/script") for n in JS_ORDER)))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(page)
print("saved", OUT, round(os.path.getsize(OUT) / 1024), "KB")
