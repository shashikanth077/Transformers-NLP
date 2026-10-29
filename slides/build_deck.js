// Builds the 7–10 minute presentation from the experiment results.
// Run: node build_deck.js   (NODE_PATH must point to a node_modules with pptxgenjs, react-icons, react, react-dom, sharp)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "code", "results");
const FIG = path.join(ROOT, "code", "figures");
const OUT = path.join(ROOT, "docs", "Transformers_in_NLP_Presentation.pptx");
const APPLY_THEME = process.env.APPLY_THEME;

// ---------------------------------------------------------------- data
function readCsv(file) {
  const txt = fs.readFileSync(path.join(RES, file), "utf8").trim().split(/\r?\n/);
  const parse = (line) => {
    const out = []; let cur = ""; let q = false;
    for (const ch of line) {
      if (ch === '"') q = !q; else if (ch === "," && !q) { out.push(cur); cur = ""; } else cur += ch;
    }
    out.push(cur); return out;
  };
  const hdr = parse(txt[0]);
  return txt.slice(1).map((l) => Object.fromEntries(parse(l).map((v, i) => [hdr[i], isNaN(+v) || v === "" ? v : +v])));
}
const summary = readCsv("summary_table.csv").sort((a, b) => b.accuracy - a.accuracy);
const lc = readCsv("learning_curve.csv");
const subsets = readCsv("subset_accuracy.csv");
const stats = readCsv("statistics.csv");
const pct = (x, d = 1) => (x * 100).toFixed(d);
const byName = (n) => summary.find((r) => r.name === n);
const family = (r) => (r.family.startsWith("transformer") ? "transformer" : r.family);

const bestCls = summary.filter((r) => r.family === "classical")[0];
const bestFt = summary.filter((r) => r.family === "transformer-finetuned")[0];
const bilstm = summary.find((r) => r.family === "recurrent");
const zeroShot = summary.find((r) => r.family === "transformer-zero-shot");
const ptSst2 = summary.find((r) => r.name.startsWith("DistilBERT-SST2"));
const twitter = summary.find((r) => r.name.startsWith("Twitter"));

// ---------------------------------------------------------------- theme
const THEME = {
  name: "Attention",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16163F", lt1: "FFFFFF", dk2: "2E2A6B", lt2: "EEF0FA",
    accent1: "1BAF7A", accent2: "2A78D6", accent3: "EB6834", accent4: "F2B33D",
    accent5: "6250D6", accent6: "8A8AA8", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const H = THEME.colors; // hex for hex-only options
const FAM = { classical: H.accent2, recurrent: H.accent3, transformer: H.accent1 };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Transformers and their application in NLP tasks";
pres.author = "Shashikanth Hosur Ramegowda, Shaeed Khan";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "DARK",
  background: { color: H.dk1 },
  objects: [],
  slideNumber: { x: 12.4, y: 6.95, w: 0.6, h: 0.35, color: "B8B8D0", fontSize: 10 },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: "Cambria", fontSize: 32, bold: true, color: H.dk1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Transformers in NLP · VILNIUS TECH", options: { x: 0.6, y: 6.95, w: 6, h: 0.35, fontSize: 10, color: "8A8AA8", margin: 0 } } },
  ],
  slideNumber: { x: 12.4, y: 6.95, w: 0.6, h: 0.35, color: "8A8AA8", fontSize: 10 },
});

// ---------------------------------------------------------------- helpers
async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function title(slide, text) { slide.addText(text, { placeholder: "title" }); }
function body(slide, text, opts) {
  slide.addText(text, { isTextBox: true, fontFace: "Calibri", fontSize: 16, color: H.dk1, margin: 0, valign: "top", ...opts });
}
function card(slide, x, y, w, h, fill = H.lt2, name = "card") {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
}
async function iconCircle(slide, Comp, x, y, d, bg, fg = "FFFFFF") {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg } });
  slide.addImage({ data: await icon(Comp, fg), x: x + d * 0.24, y: y + d * 0.24, w: d * 0.52, h: d * 0.52 });
}
function img(slide, file, x, y, w, h) {
  slide.addImage({ path: path.join(FIG, file), x, y, w, h });
}
function imgFit(slide, file, x, y, maxW, maxH) {
  const meta = require("child_process").execSync(`node -e "require('sharp')('${path.join(FIG, file).replace(/\\/g, "/")}').metadata().then(m=>console.log(m.width+','+m.height))"`).toString().trim().split(",");
  const ar = +meta[0] / +meta[1];
  let w = maxW, h = maxW / ar;
  if (h > maxH) { h = maxH; w = maxH * ar; }
  slide.addImage({ path: path.join(FIG, file), x: x + (maxW - w) / 2, y, w, h });
}

(async () => {
  // ============================================================ 1. Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
  s.addText("VILNIUS TECH · Group project", { isTextBox: true, x: 0.8, y: 0.8, w: 8, h: 0.4, fontSize: 14, color: "B8B8D0", margin: 0 });
  s.addText("Transformers and their application in NLP tasks", { isTextBox: true, x: 0.8, y: 1.6, w: 8.2, h: 2.0, fontFace: "Cambria", fontSize: 42, bold: true, color: "FFFFFF", margin: 0, valign: "top" });
  s.addText("Do Transformers pay off on a laptop CPU? A comparison of 10 sentiment classifiers on SST-2", { isTextBox: true, x: 0.8, y: 3.75, w: 7.8, h: 0.9, fontSize: 18, color: "D6D6EE", margin: 0, italic: true });
  s.addText([
    { text: "Shashikanth Hosur Ramegowda", options: { breakLine: true } },
    { text: "Shaeed Khan" },
  ], { isTextBox: true, x: 0.8, y: 5.2, w: 6, h: 1.2, fontSize: 16, color: "FFFFFF", margin: 0, paraSpaceAfter: 4 });
  // motif: attention grid
  const words = ["not", "good", "but", "great"];
  const att = [[0.55, 0.35, 0.05, 0.05], [0.6, 0.3, 0.05, 0.05], [0.1, 0.2, 0.3, 0.4], [0.05, 0.05, 0.3, 0.6]];
  const gx = 9.3, gy = 2.0, cell = 0.78;
  words.forEach((w, i) => {
    s.addText(w, { isTextBox: true, x: gx - 1.0, y: gy + i * cell, w: 0.9, h: cell, fontSize: 13, color: "D6D6EE", align: "right", valign: "middle", margin: 0 });
    s.addText(w, { isTextBox: true, x: gx + i * cell, y: gy - 0.45, w: cell, h: 0.4, fontSize: 13, color: "D6D6EE", align: "center", margin: 0 });
    att[i].forEach((v, j) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: gx + j * cell + 0.04, y: gy + i * cell + 0.04, w: cell - 0.08, h: cell - 0.08, rectRadius: 0.08, fill: { color: H.accent1, transparency: Math.round(100 - v * 160 < 5 ? 5 : 100 - v * 160) }, line: { color: H.accent1, transparency: 60 } }));
  });
  s.addText("self-attention weights", { isTextBox: true, x: gx, y: gy + 4 * cell + 0.1, w: 3.2, h: 0.35, fontSize: 12, color: "B8B8D0", align: "center", margin: 0, italic: true });
  s.addNotes("Speaker: Shashikanth (≈30 s). Introduce the team and the topic: Transformers are the architecture behind BERT, GPT and almost every modern NLP system. Our question: what do they really give, and at what cost, when you only have a normal laptop and limited labelled data?");

  // ============================================================ 2. Why Transformers
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  title(s, "From reading word by word to reading all at once");
  card(s, 0.6, 1.5, 5.85, 4.9);
  card(s, 6.85, 1.5, 5.85, 4.9, "E6F6EF");
  await iconCircle(s, fa.FaHourglassHalf, 0.95, 1.85, 0.8, H.accent3);
  await iconCircle(s, fa.FaBolt, 7.2, 1.85, 0.8, H.accent1);
  s.addText("Recurrent networks (LSTM)", { isTextBox: true, x: 1.95, y: 1.95, w: 4.3, h: 0.6, fontFace: "Cambria", fontSize: 22, bold: true, color: H.dk1, margin: 0, valign: "middle" });
  s.addText("Transformer (2017)", { isTextBox: true, x: 8.2, y: 1.95, w: 4.3, h: 0.6, fontFace: "Cambria", fontSize: 22, bold: true, color: H.dk1, margin: 0, valign: "middle" });
  body(s, [
    { text: "Process tokens one after another", options: { bullet: true, breakLine: true } },
    { text: "n sequential steps – slow to train", options: { bullet: true, breakLine: true } },
    { text: "Distant words interact only through a long chain of states", options: { bullet: true, breakLine: true } },
    { text: "Trained from scratch for each task", options: { bullet: true } },
  ], { x: 1.0, y: 2.95, w: 5.2, h: 3.2, paraSpaceAfter: 10 });
  body(s, [
    { text: "Every token attends to every other token", options: { bullet: true, breakLine: true } },
    { text: "All positions computed in parallel on GPU/CPU", options: { bullet: true, breakLine: true } },
    { text: "Pre-train once on huge unlabelled text, then fine-tune", options: { bullet: true, breakLine: true } },
    { text: "Basis of BERT, GPT, T5, ChatGPT, LitLat BERT…", options: { bullet: true } },
  ], { x: 7.25, y: 2.95, w: 5.2, h: 3.2, paraSpaceAfter: 10 });
  s.addNotes("Speaker: Shashikanth (≈45 s). Before 2017 NLP models read sentences word by word. The Transformer by Vaswani et al. removed recurrence and kept only attention. That gave parallel training and made huge pre-training possible: BERT, GPT and others. Aim of the project: explain the architecture and measure its benefit experimentally.");

  // ============================================================ 3. Self-attention
  pres.addSection({ title: "Theory" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Theory" });
  title(s, "Self-attention: each word looks at the whole sentence");
  imgFit(s, "diagram_encoder.png", 0.6, 1.35, 4.6, 5.45);
  card(s, 5.6, 1.5, 7.1, 1.35, H.lt2);
  s.addText("Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V", { isTextBox: true, x: 5.85, y: 1.6, w: 6.7, h: 0.7, fontFace: "Cambria", fontSize: 24, bold: true, color: H.dk1, margin: 0, align: "center", valign: "middle" });
  s.addText("queries · keys → weights → weighted mix of values", { isTextBox: true, x: 5.85, y: 2.25, w: 6.7, h: 0.45, fontSize: 14, color: "52516E", margin: 0, align: "center", italic: true });
  const steps = [
    [fa.FaCubes, "Sub-word tokens + position embeddings", "WordPiece: “unwatchable” → un ##watch ##able"],
    [fa.FaProjectDiagram, "Multi-head attention", "h heads learn different relations (negation scope, syntax)"],
    [fa.FaLayerGroup, "Feed-forward + residual + LayerNorm", "Stacked L times (2 / 4 / 6 layers in our models)"],
    [fa.FaFlagCheckered, "[CLS] vector → softmax classifier", "Whole network fine-tuned with cross-entropy + AdamW"],
  ];
  for (let i = 0; i < steps.length; i++) {
    const y = 3.15 + i * 0.92;
    await iconCircle(s, steps[i][0], 5.65, y, 0.62, i % 2 ? H.accent2 : H.accent1);
    s.addText(steps[i][1], { isTextBox: true, x: 6.5, y: y - 0.04, w: 6.2, h: 0.36, fontSize: 16, bold: true, color: H.dk1, margin: 0 });
    s.addText(steps[i][2], { isTextBox: true, x: 6.5, y: y + 0.32, w: 6.2, h: 0.32, fontSize: 13, color: "52516E", margin: 0 });
  }
  s.addNotes("Speaker: Shashikanth (≈60 s). Explain Q, K, V: each word asks a question (query), every word offers a key, the softmax of their dot products gives attention weights, and the output is a weighted mix of values. Several heads run in parallel. Layers are stacked with residual connections and layer normalisation. For classification we take the [CLS] vector and add a softmax layer.");

  // ============================================================ 4. Literature
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Theory" });
  title(s, "Literature: what is known, and what is missing");
  // three model families in one compact row
  const fams = [
    ["Encoder-only", "BERT, DistilBERT, RoBERTa", "understanding, classification", H.accent1],
    ["Decoder-only", "GPT, Llama", "text generation", H.accent2],
    ["Encoder–decoder", "T5, BART", "translation, summaries", H.accent5],
  ];
  fams.forEach((f, i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.4, 3.8, 0.95, H.lt2);
    s.addText(f[0], { isTextBox: true, x: x + 0.25, y: 1.48, w: 3.3, h: 0.4, fontFace: "Cambria", fontSize: 17, bold: true, color: f[3], margin: 0 });
    s.addText([{ text: f[1], options: { bold: true, color: H.dk1 } }, { text: "  →  " + f[2], options: { color: "52516E" } }], { isTextBox: true, x: x + 0.25, y: 1.9, w: 3.4, h: 0.4, fontSize: 12.5, margin: 0, valign: "middle" });
  });
  // left: what the studies show (three plain sentences)
  card(s, 0.6, 2.6, 7.4, 4.1, H.lt2);
  await iconCircle(s, fa.FaBookOpen, 0.85, 2.78, 0.6, H.accent2);
  s.addText("What the studies show", { isTextBox: true, x: 1.6, y: 2.78, w: 5.5, h: 0.6, fontFace: "Cambria", fontSize: 20, bold: true, color: H.dk1, margin: 0, valign: "middle" });
  const facts = [
    ["1", "Transformers are the most accurate.", " Fine-tuned BERT-type models exceed 90% on SST-2; classical models reach about 79–83%."],
    ["2", "Without pre-training, small data is a problem.", " In Lithuania a simple naive Bayes (0.735) beat LSTM and CNN networks (0.706) (Kapočiūtė-Dzikienė et al., 2019)."],
    ["3", "Lithuanian now has pre-trained models.", " LitLat BERT (2021), BERT and T5 for reviews (2024), the first open Llama2 models (2024)."],
  ];
  facts.forEach((f, i) => {
    const y = 3.6 + i * 1.02;
    s.addShape(pres.shapes.OVAL, { x: 0.85, y: y + 0.05, w: 0.42, h: 0.42, fill: { color: H.dk2 }, line: { color: H.dk2 } });
    s.addText(f[0], { isTextBox: true, x: 0.85, y: y + 0.05, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
    s.addText([{ text: f[1], options: { bold: true, color: H.dk1 } }, { text: f[2], options: { color: "3B3A55" } }], { isTextBox: true, x: 1.45, y, w: 6.35, h: 0.95, fontSize: 14.5, margin: 0, valign: "top" });
  });
  // right: the gap in plain words
  card(s, 8.3, 2.6, 4.4, 4.1, "FDF1E8");
  await iconCircle(s, fa.FaSearch, 8.55, 2.78, 0.6, H.accent3);
  s.addText("What is missing: our gap", { isTextBox: true, x: 9.3, y: 2.78, w: 3.3, h: 0.6, fontFace: "Cambria", fontSize: 18, bold: true, color: H.dk1, margin: 0, valign: "middle" });
  s.addText([
    { text: "These studies report accuracy. They rarely tell a student with a laptop:", options: { breakLine: true, color: H.dk1, paraSpaceAfter: 6 } },
    { text: "What does the better model cost? (training time, speed, size)", options: { bullet: true, breakLine: true, color: H.dk1 } },
    { text: "How many labelled sentences does it need?", options: { bullet: true, breakLine: true, color: H.dk1 } },
    { text: "Is a ready-made model good enough?", options: { bullet: true, breakLine: true, color: H.dk1, paraSpaceAfter: 8 } },
    { text: "We measure all three, on one laptop CPU.", options: { bold: true, color: "B8441B" } },
  ], { isTextBox: true, x: 8.55, y: 3.55, w: 3.95, h: 3.0, fontSize: 14.5, margin: 0, valign: "top", paraSpaceAfter: 4 });
  s.addNotes("Speaker: Shaeed (≈60 s). In one sentence: the papers tell us which model is accurate, but not what it costs or how many labels it needs. We searched Google Scholar and Web of Science (transformer OR BERT AND sentiment / text classification, plus Lithuanian). There are three model families; we use encoder-only models for classification. What the studies show: first, Transformers are the most accurate, above 90% on SST-2 against 79 to 83% for classical models. Second, without pre-training small data is a problem: in Lithuania naive Bayes at 0.735 beat LSTM and CNN at 0.706. Third, Lithuanian now has pre-trained models, LitLat BERT and BERT/T5, and open Llama2 models. What is missing, our gap: these studies report accuracy, but a student with a laptop also needs to know the cost, how many labels are needed, and whether a ready-made model is enough. We measure all three on one laptop CPU.");

  // ============================================================ 5. Experiment design
  pres.addSection({ title: "Experiment" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiment" });
  title(s, "Experiment: 10 competitors, one dataset, one laptop CPU");
  const nTrain = 65349;
  const tiles = [
    ["SST-2", "movie-review sentences, positive / negative"],
    [`${(nTrain / 1000).toFixed(0)}k / 2k / 872`, "train / dev / test sentences"],
    ["i5-8365U", "4-core laptop CPU, 16 GB RAM, no GPU"],
  ];
  tiles.forEach((t, i) => {
    const y = 1.5 + i * 1.65;
    card(s, 0.6, y, 4.2, 1.4, H.lt2);
    s.addText(t[0], { isTextBox: true, x: 0.85, y: y + 0.12, w: 3.8, h: 0.7, fontFace: "Cambria", fontSize: 28, bold: true, color: H.dk2, margin: 0 });
    s.addText(t[1], { isTextBox: true, x: 0.85, y: y + 0.82, w: 3.8, h: 0.45, fontSize: 13, color: "52516E", margin: 0 });
  });
  const groups = [
    ["Classical", "TF-IDF + Naive Bayes · Logistic Regression · Linear SVM", FAM.classical, fa.FaCalculator],
    ["Recurrent", "BiLSTM trained from scratch", FAM.recurrent, fa.FaRedo],
    ["Fine-tuned by us (8k labels)", "BERT-tiny 4M · BERT-small 29M · DistilBERT 67M", FAM.transformer, fa.FaTools],
    ["Ready-made, no training", "DistilBERT-SST2 · Twitter-RoBERTa · zero-shot NLI", H.accent5, fa.FaBoxOpen],
  ];
  for (let i = 0; i < groups.length; i++) {
    const y = 1.5 + i * 1.08;
    await iconCircle(s, groups[i][3], 5.25, y + 0.1, 0.7, groups[i][2]);
    s.addText(groups[i][0], { isTextBox: true, x: 6.15, y: y + 0.02, w: 6.5, h: 0.42, fontSize: 17, bold: true, color: H.dk1, margin: 0 });
    s.addText(groups[i][1], { isTextBox: true, x: 6.15, y: y + 0.44, w: 6.5, h: 0.4, fontSize: 14, color: "52516E", margin: 0 });
  }
  body(s, "Criteria: accuracy, macro-F1 · training time · latency per sentence · parameters · negation & contrast subsets · learning curve · bootstrap CI + McNemar test", { x: 5.25, y: 5.95, w: 7.45, h: 0.75, fontSize: 13, color: "52516E", italic: true });
  s.addNotes("Speaker: Shaeed (≈45 s). Dataset: Stanford Sentiment Treebank, binary. The official validation set is our test set, 2,000 training sentences are the dev set. Ten models in four groups. Everything runs on the same laptop CPU so times are comparable.");

  // ============================================================ 6. Program
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Experiment" });
  title(s, "Implementation: a reproducible Python pipeline");
  imgFit(s, "diagram_pipeline.png", 0.5, 1.35, 8.3, 5.4);
  card(s, 9.1, 1.5, 3.6, 5.2, H.lt2);
  body(s, [
    { text: "PyTorch · Hugging Face Transformers · scikit-learn", options: { bold: true, breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "One generic fine-tuning script for any encoder", options: { bullet: true, breakLine: true } },
    { text: "AdamW, 10% warm-up, linear decay, best epoch by dev accuracy", options: { bullet: true, breakLine: true } },
    { text: "Every model saves metrics, timings and predictions as JSON", options: { bullet: true, breakLine: true } },
    { text: "run_all.sh reproduces all results (~2.5 h)", options: { bullet: true } },
  ], { x: 9.35, y: 1.75, w: 3.15, h: 4.8, fontSize: 14, paraSpaceAfter: 6 });
  s.addNotes("Speaker: Shashikanth (≈40 s). The program is a pipeline of nine scripts. The key one is the fine-tuning script: give it any model name from the Hugging Face hub and a number of training examples. All results are stored with predictions, so the report script computes tables, figures and statistical tests.");

  // ============================================================ 7. Results accuracy (native chart)
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, `Fine-tuned DistilBERT: +${(+pct(bestFt.accuracy) - +pct(bestCls.accuracy)).toFixed(1)} points over classical`);
  const asc = [...summary].reverse();
  const famOrder = ["classical", "recurrent", "transformer"];
  const series = famOrder.map((f) => ({
    name: f === "classical" ? "Classical" : f === "recurrent" ? "Recurrent" : "Transformer",
    labels: asc.map((r) => r.name),
    values: asc.map((r) => (family(r) === f ? +(r.accuracy * 100).toFixed(1) : null)),
  }));
  s.addChart(pres.charts.BAR, series, {
    x: 0.5, y: 1.35, w: 8.4, h: 5.45, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 45,
    chartColors: [FAM.classical, FAM.recurrent, FAM.transformer],
    valAxisMinVal: 50, valAxisMaxVal: 100, valAxisMajorUnit: 10,
    showValue: true, dataLabelPosition: "inEnd", dataLabelColor: "FFFFFF", dataLabelFontSize: 10, dataLabelFormatCode: "0.0",
    catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, catAxisLabelColor: "52516E", valAxisLabelColor: "8A8AA8",
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
    valGridLine: { color: "E4E3EF", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt",
    showValAxisTitle: true, valAxisTitle: "Test accuracy, %", valAxisTitleFontSize: 11, valAxisTitleColor: "52516E",
  });
  const call = [
    [pct(bestFt.accuracy), `${bestFt.name}`, H.accent1],
    [pct(bestCls.accuracy), `best classical: ${bestCls.name}`, H.accent2],
    [pct(zeroShot.accuracy), "zero-shot – no labels at all", H.accent5],
  ];
  call.forEach((c, i) => {
    const y = 1.45 + i * 1.75;
    card(s, 9.2, y, 3.5, 1.55, H.lt2);
    s.addText(c[0] + "%", { isTextBox: true, x: 9.4, y: y + 0.1, w: 3.1, h: 0.8, fontFace: "Cambria", fontSize: 36, bold: true, color: c[2], margin: 0 });
    s.addText(c[1], { isTextBox: true, x: 9.4, y: y + 0.9, w: 3.1, h: 0.5, fontSize: 12, color: "52516E", margin: 0 });
  });
  s.addNotes(`Speaker: Shashikanth (≈60 s). Ranking of all ten models. Classical TF-IDF models reach ${pct(summary.filter(r => r.family === "classical").slice(-1)[0].accuracy)}–${pct(bestCls.accuracy)}%, the BiLSTM ${pct(bilstm.accuracy)}%, our fine-tuned Transformers up to ${pct(bestFt.accuracy)}% with only 8,000 labelled sentences. The ready-made DistilBERT-SST2 reaches ${pct(ptSst2.accuracy)}%. Zero-shot, with no labels at all, still gets ${pct(zeroShot.accuracy)}%.`);

  // ============================================================ 8. Cost
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "The price: slower training and inference, bigger models");
  const rowsCost = [["Model", "Acc., %", "Train time", "Latency, ms", "Params"]];
  const fmtTime = (t) => (t === 0 ? "–" : t < 60 ? `${t.toFixed(0)} s` : t < 3600 ? `${(t / 60).toFixed(0)} min` : `${(t / 3600).toFixed(1)} h`);
  const fmtPar = (p) => (p >= 1e6 ? `${(p / 1e6).toFixed(1)} M` : `${(p / 1e3).toFixed(0)} k`);
  const pick = [bestCls, bilstm, ...summary.filter((r) => r.family === "transformer-finetuned").sort((a, b) => a.parameters - b.parameters), ptSst2, zeroShot];
  pick.forEach((r) => rowsCost.push([r.name, pct(r.accuracy), fmtTime(r.train_time_s), r.latency_ms_per_sample < 1 ? r.latency_ms_per_sample.toFixed(2) : r.latency_ms_per_sample.toFixed(1), fmtPar(r.parameters)]));
  s.addTable(rowsCost.map((row, i) => row.map((v, j) => ({
    text: String(v),
    options: { bold: i === 0, color: i === 0 ? "FFFFFF" : H.dk1, fill: { color: i === 0 ? H.dk2 : i % 2 ? "FFFFFF" : H.lt2 }, fontSize: 13, align: j === 0 ? "left" : "center", valign: "middle" },
  }))), { x: 0.6, y: 1.5, w: 8.0, colW: [3.4, 1.0, 1.2, 1.3, 1.1], rowH: 0.52, border: { type: "solid", pt: 0.5, color: "D8D8E8" }, fontFace: "Calibri" });
  const ratio = (bestFt.latency_ms_per_sample / bestCls.latency_ms_per_sample);
  card(s, 9.0, 1.5, 3.7, 2.35, "FDF1E8");
  s.addText(`×${ratio >= 100 ? Math.round(ratio / 10) * 10 : ratio.toFixed(0)}`, { isTextBox: true, x: 9.25, y: 1.6, w: 3.3, h: 1.0, fontFace: "Cambria", fontSize: 48, bold: true, color: H.accent3, margin: 0 });
  s.addText(`slower inference of ${bestFt.name.split(" ")[0]} vs TF-IDF – still only ${bestFt.latency_ms_per_sample.toFixed(0)} ms per sentence`, { isTextBox: true, x: 9.25, y: 2.65, w: 3.3, h: 1.0, fontSize: 13, color: "52516E", margin: 0 });
  card(s, 9.0, 4.1, 3.7, 2.55, "E6F6EF");
  s.addText("Fine-tuning is affordable", { isTextBox: true, x: 9.25, y: 4.25, w: 3.3, h: 0.5, fontSize: 16, bold: true, color: H.dk1, margin: 0 });
  s.addText(`Training on the laptop CPU took ${fmtTime(summary.filter((r) => r.family === "transformer-finetuned").reduce((m, r) => Math.min(m, r.train_time_s), 1e9))} – ${fmtTime(summary.filter((r) => r.family === "transformer-finetuned").reduce((m, r) => Math.max(m, r.train_time_s), 0))}; no GPU needed for a few thousand examples.`, { isTextBox: true, x: 9.25, y: 4.8, w: 3.3, h: 1.7, fontSize: 13, color: "52516E", margin: 0 });
  s.addNotes("Speaker: Shashikanth (≈45 s). The gain is not free: a TF-IDF model trains in seconds and classifies in microseconds; Transformers need minutes to tens of minutes of CPU training and milliseconds per sentence. For most applications that is still perfectly acceptable.");

  // ============================================================ 9. Learning curve (native line chart)
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Pre-training is the key: fewer labels, higher accuracy");
  const lcSeries = [...new Set(lc.map((r) => r.series))].sort((a, b) => (a.startsWith("TF") ? 1 : -1));
  const sizes = [...new Set(lc.map((r) => r.train_size))].sort((a, b) => a - b);
  s.addChart(pres.charts.LINE, lcSeries.map((n) => ({
    name: n, labels: sizes.map(String),
    values: sizes.map((z) => +(lc.find((r) => r.series === n && r.train_size === z).accuracy * 100).toFixed(1)),
  })), {
    x: 0.5, y: 1.35, w: 8.4, h: 5.45, chartColors: [FAM.transformer, FAM.classical], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 9,
    valAxisMinVal: 60, valAxisMaxVal: 95, valAxisMajorUnit: 5,
    showValue: true, dataLabelPosition: "t", dataLabelFontSize: 10, dataLabelColor: "52516E", dataLabelFormatCode: "0.0", dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: "52516E", valAxisLabelColor: "8A8AA8", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E4E3EF", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: "Labelled training sentences", catAxisTitleFontSize: 12, catAxisTitleColor: "52516E",
    showValAxisTitle: true, valAxisTitle: "Test accuracy, %", valAxisTitleFontSize: 12, valAxisTitleColor: "52516E",
  });
  const lcT = (n, z) => lc.find((r) => r.series === n && r.train_size === z).accuracy;
  const tn = lcSeries.find((n) => !n.startsWith("TF")), cn = lcSeries.find((n) => n.startsWith("TF"));
  const fullCls = bestCls.accuracy;
  const firstBeat = sizes.find((z) => lcT(tn, z) >= fullCls - 0.005);
  card(s, 9.2, 1.5, 3.5, 2.4, "E6F6EF");
  s.addText(firstBeat ? firstBeat.toLocaleString("en-US") : "–", { isTextBox: true, x: 9.45, y: 1.6, w: 3.0, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true, color: H.accent1, margin: 0 });
  s.addText(`labels: BERT-small already matches the best classical model trained on all ${(nTrain / 1000).toFixed(0)}k (${pct(lcT(tn, firstBeat))}% vs ${pct(fullCls)}%)`, { isTextBox: true, x: 9.45, y: 2.5, w: 3.0, h: 1.3, fontSize: 13, color: "52516E", margin: 0 });
  card(s, 9.2, 4.15, 3.5, 2.5, H.lt2);
  s.addText(`+${((lcT(tn, sizes[0]) - lcT(cn, sizes[0])) * 100).toFixed(0)} p.p.`, { isTextBox: true, x: 9.45, y: 4.25, w: 3.0, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true, color: H.dk2, margin: 0 });
  s.addText(`advantage with only ${sizes[0]} labels – the gap shrinks as data grows`, { isTextBox: true, x: 9.45, y: 5.15, w: 3.0, h: 1.3, fontSize: 13, color: "52516E", margin: 0 });
  s.addNotes("Speaker: Shaeed (≈50 s). Same random subsets for both models. With very few labels TF-IDF cannot learn the vocabulary, while BERT-small already knows English from pre-training. This confirms the Lithuanian finding that pre-training, not the neural network as such, is what makes the difference on small data.");

  // ============================================================ 10. Negation + attention
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Context matters: negation and contrast sentences");
  const subSel = [bestCls, bilstm, bestFt, ptSst2].map((r) => subsets.find((x) => x.name === r.name));
  s.addChart(pres.charts.BAR, [
    { name: "All", labels: subSel.map((r) => r.name.replace(/ \(.*\)/, "")), values: subSel.map((r) => +(r.all * 100).toFixed(1)) },
    { name: "Negation", labels: subSel.map((r) => r.name.replace(/ \(.*\)/, "")), values: subSel.map((r) => +(r.negation * 100).toFixed(1)) },
    { name: "Contrast", labels: subSel.map((r) => r.name.replace(/ \(.*\)/, "")), values: subSel.map((r) => +(r.contrast * 100).toFixed(1)) },
  ], {
    x: 0.5, y: 1.35, w: 7.0, h: 5.45, barDir: "col", barGapWidthPct: 60,
    chartColors: ["B8B8D0", H.accent2, H.accent3], valAxisMinVal: 60, valAxisMaxVal: 100, valAxisMajorUnit: 10,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: "52516E", dataLabelFormatCode: "0", dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 10, catAxisLabelColor: "52516E", valAxisLabelColor: "8A8AA8", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E4E3EF", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt",
    showValAxisTitle: true, valAxisTitle: "Accuracy, %", valAxisTitleFontSize: 11, valAxisTitleColor: "52516E",
  });
  imgFit(s, "attention_heatmap.png", 7.8, 1.35, 4.9, 4.6);
  s.addText("“the plot was not good, but the acting was brilliant” – DistilBERT attention, last layer", { isTextBox: true, x: 7.8, y: 6.05, w: 4.9, h: 0.6, fontSize: 11, color: "52516E", margin: 0, italic: true, align: "center" });
  s.addNotes("Speaker: Shaeed (≈45 s). Bag-of-words models lose most on sentences with negation and contrast, because 'not good' contains the word 'good'. Transformers keep their accuracy much better. The heat-map shows how attention links the words in such a sentence – but, as Jain and Wallace warn, attention is an illustration, not a proof of reasoning.");

  // ============================================================ 11. Conclusions
  pres.addSection({ title: "Conclusions" });
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusions" });
  s.addText("Conclusions & recommendations", { isTextBox: true, x: 0.8, y: 0.55, w: 11.7, h: 0.9, fontFace: "Cambria", fontSize: 34, bold: true, color: "FFFFFF", margin: 0 });
  const concl = [
    [fa.FaTrophy, `Fine-tuned Transformers are the most accurate: ${pct(bestFt.accuracy)}% vs ${pct(bestCls.accuracy)}% (classical) and ${pct(bilstm.accuracy)}% (BiLSTM)`],
    [fa.FaDatabase, "Pre-training gives data efficiency: BERT-small with 4,000 labels matches classical models trained on 65k; DistilBERT with 8,000 beats them"],
    [fa.FaStopwatch, "Cost is real but manageable: minutes of CPU fine-tuning, milliseconds per sentence"],
    [fa.FaBoxOpen, `Reuse first: an in-domain public model gives ${pct(ptSst2.accuracy)}% with zero training; zero-shot (${pct(zeroShot.accuracy)}%) and out-of-domain models are useful but weaker`],
    [fa.FaFlag, "For Lithuanian: fine-tune LitLat BERT / XLM-R on a small labelled set – the next step of this work"],
  ];
  for (let i = 0; i < concl.length; i++) {
    const y = 1.7 + i * 1.0;
    await iconCircle(s, concl[i][0], 0.8, y, 0.68, i === 4 ? H.accent3 : H.accent1);
    s.addText(concl[i][1], { isTextBox: true, x: 1.75, y: y - 0.05, w: 10.8, h: 0.8, fontSize: 17, color: "FFFFFF", margin: 0, valign: "middle" });
  }
  s.addText("Thank you! Questions?", { isTextBox: true, x: 0.8, y: 6.6, w: 8, h: 0.5, fontFace: "Cambria", fontSize: 18, italic: true, color: "B8B8D0", margin: 0 });
  s.addNotes("Speaker: Shaeed (≈60 s). Summarise the answers to the research questions and give the practical recommendation: first try a public in-domain model, otherwise fine-tune a compact Transformer on a few thousand labels; use TF-IDF only when latency or hardware are extremely limited. Future work: Lithuanian data with LitLat BERT. Invite questions.");

  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("saved", OUT);
})();
