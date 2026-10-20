"""Build the single-file offline page docs/Transformers_Explained_Overview.html from study_guide/overview/src + data.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
rd = lambda *p: open(os.path.join(*p), encoding="utf-8").read()

data = json.loads(rd(HERE, "data.json"))
guide = json.loads(rd(ROOT, "study_guide", "data.json"))
# classification-head demo: real P(positive) of the project's fine-tuned DistilBERT on SST-2 test sentences
data["cls"] = [{"text": guide["sentences"][w["i"]]["s"], "y": guide["sentences"][w["i"]]["y"], "p": round(w["p_pos"], 4)} for w in guide["walk"] if w["i"] >= 0]

page = rd(HERE, "src", "shell.html")
page = page.replace("/*__CSS__*/", rd(HERE, "src", "style.css"))
page = page.replace("<!--__BODY__-->", rd(HERE, "src", "body.html"))
page = page.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\/"))
page = page.replace("/*__JS__*/", rd(HERE, "src", "app.js").replace("</script", "<\/script"))
out = os.path.join(ROOT, "docs", "Transformers_Explained_Overview.html")
open(out, "w", encoding="utf-8").write(page)
print("wrote", out, round(len(page) / 1024), "KB")
