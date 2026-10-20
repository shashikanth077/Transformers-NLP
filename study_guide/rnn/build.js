// Builds docs/RNN_LSTM_GRU_Explained.html from src.html + engine.js + weights.json (run train_explorer_weights.js first)
const fs = require("fs"), path = require("path");
const rd = f => fs.readFileSync(path.join(__dirname, f), "utf8");
const page = rd("src.html").replace("/*__ENGINE__*/", () => rd("engine.js")).replace("/*__WEIGHTS__*/", () => rd("weights.json").trim());
const out = path.join(__dirname, "..", "..", "docs", "RNN_LSTM_GRU_Explained.html");
fs.writeFileSync(out, page); console.log("wrote", out, Math.round(page.length / 1024), "KB");
