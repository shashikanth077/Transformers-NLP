// Builds docs/NMT_Pipeline_Explained.html from src.html + corpus/seq2seq/pipeline code + nllb.json + toy.json
// Order: node study_guide/nmt/train_toy.js ; python study_guide/nmt/export_nllb.py ; node study_guide/nmt/build.js
const fs = require("fs"), path = require("path");
const rd = f => fs.readFileSync(path.join(__dirname, f), "utf8");
const engine = ["corpus.js", "seq2seq.js", "pipeline.js"].map(rd).join("\n");
const page = rd("src.html").replace("/*__ENGINE__*/", () => engine).replace("/*__NLLB__*/", () => rd("nllb.json").trim()).replace("/*__TOY__*/", () => rd("toy.json").trim());
const out = path.join(__dirname, "..", "..", "docs", "NMT_Pipeline_Explained.html");
fs.writeFileSync(out, page); console.log("wrote", out, Math.round(page.length / 1024), "KB");
