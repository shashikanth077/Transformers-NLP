"""Builds docs/Project_Map_and_Interview_Prep.html from src.html and the real results in study_guide/data.json."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
d = json.load(open(os.path.join(ROOT, "study_guide", "data.json"), encoding="utf-8"))
S = d["sentences"]
models = []
for i, m in enumerate(d["models"]):
    acc_on = lambda f: sum(1 for s in S if f(s) and s["p"][i] == str(s["y"])) / max(1, sum(1 for s in S if f(s)))
    ci = d["stats"][m["tag"]]
    models.append({"tag": m["tag"], "name": m["name"], "short": m["short"], "family": m["family"], "acc": m["acc"], "train_s": m["train_s"], "lat_ms": m["lat_ms"], "params": m["params"],
                   "ci": [ci["lo"], ci["hi"]], "neg": acc_on(lambda s: s["n"]), "con": acc_on(lambda s: s["c"])})
data = {"models": models, "curve": d["curve"]}
page = open(os.path.join(HERE, "src.html"), encoding="utf-8").read().replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
out = os.path.join(ROOT, "docs", "Project_Map_and_Interview_Prep.html")
open(out, "w", encoding="utf-8").write(page); print("wrote", out, len(page) // 1024, "KB")
