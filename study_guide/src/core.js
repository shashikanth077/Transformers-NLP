"use strict";
/* Core helpers: DOM, storage, data decoding, key numbers, statistics, theme, navigation */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k, d) { try { const v = localStorage.getItem("tsg:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("tsg:" + k, JSON.stringify(v)); } catch (e) { /* storage unavailable: page still works */ } },
  clear() { try { Object.keys(localStorage).filter(k => k.startsWith("tsg:")).forEach(k => localStorage.removeItem(k)); } catch (e) { /* ignore */ } },
};
const debounce = (fn, ms = 180) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const fmtInt = n => Math.round(n).toLocaleString("en-US");
const pct1 = x => (x * 100).toFixed(1);
const fmtLat = ms => (ms < 1 ? ms.toFixed(2) : ms.toFixed(1));
const fmtParams = p => (p >= 1e6 ? (p / 1e6).toFixed(1) + " M" : Math.round(p / 1e3) + " k");
const fmtTime = s => (s < 1 ? "–" : s < 60 ? Math.round(s) + " s" : s < 3600 ? Math.round(s / 60) + " min" : (s / 3600).toFixed(1) + " h");

/* ---------- data ---------- */
const M = DATA.models;                                  // order = order of the characters in sentence.p
const MI = Object.fromEntries(M.map((m, i) => [m.tag, i]));
const SENT = DATA.sentences;
const FAMCLR = { classical: "var(--classical)", recurrent: "var(--recurrent)", transformer: "var(--transformer)" };
const FAMNAME = { classical: "Classical", recurrent: "Recurrent", transformer: "Transformer" };
const isRight = (row, mi) => row.p[mi] === String(row.y);
const nRight = row => { let c = 0; for (let i = 0; i < M.length; i++) if (isRight(row, i)) c++; return c; };
const accOn = (rows, mi) => rows.length ? rows.filter(r => isRight(r, mi)).length / rows.length : NaN;

function b64bytes(b64) { const bin = atob(b64); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }

/* ---------- statistics: exact McNemar (two-sided binomial test with p = 0.5) ---------- */
const logFact = (() => { const a = [0]; for (let i = 1; i <= 2000; i++) a[i] = a[i - 1] + Math.log(i); return a; })();
function mcnemar(a, b) {
  let only_a = 0, only_b = 0;
  for (const r of SENT) { const ca = isRight(r, a), cb = isRight(r, b); if (ca && !cb) only_a++; else if (!ca && cb) only_b++; }
  const n = only_a + only_b, k = Math.min(only_a, only_b);
  let p = 1;
  if (n > 0) {
    let cdf = 0;
    for (let i = 0; i <= k; i++) cdf += Math.exp(logFact[n] - logFact[i] - logFact[n - i] - n * Math.LN2);
    p = Math.min(1, 2 * cdf);
  }
  return { only_a, only_b, n, p };
}
const fmtP = p => (p < 0.001 ? "< 0.001" : p.toFixed(3));

/* ---------- TF-IDF + logistic regression (the project's real model, decoded lazily) ---------- */
let _lr = null;
function LR() {
  if (_lr) return _lr;
  const i16 = b64 => { const u = b64bytes(b64); return new Int16Array(u.buffer, u.byteOffset, u.byteLength / 2); };
  const terms = DATA.lr.terms.split("\n");
  const idf = i16(DATA.lr.idf), coef = i16(DATA.lr.coef);
  const index = new Map(); terms.forEach((t, i) => index.set(t, i));
  _lr = { terms, idf, coef, index, bias: DATA.lr.bias };
  return _lr;
}
function bowAnalyse(text) {
  const m = LR();
  const toks = (text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || []);   // same tokens as scikit-learn's \b\w\w+\b (Unicode-aware)
  const grams = toks.concat(toks.slice(1).map((t, i) => toks[i] + " " + t));
  const tf = new Map();
  grams.forEach(g => { if (m.index.has(g)) tf.set(g, (tf.get(g) || 0) + 1); });
  const w = new Map(); let sq = 0;
  tf.forEach((n, g) => { const v = (1 + Math.log(n)) * m.idf[m.index.get(g)] / 1000; w.set(g, v); sq += v * v; });
  const norm = Math.sqrt(sq) || 1;
  const parts = []; let z = m.bias;
  w.forEach((v, g) => { const c = m.coef[m.index.get(g)] / 1000 * v / norm; z += c; parts.push({ gram: g, c, bigram: g.includes(" ") }); });
  parts.sort((a, b) => Math.abs(b.c) - Math.abs(a.c));
  return { tokens: toks, parts, bias: m.bias, z, p: 1 / (1 + Math.exp(-z)), unknown: toks.filter(t => !m.index.has(t)) };
}

/* ---------- attention (real DistilBERT-SST2 maps) ---------- */
function attOf(entry) {
  if (!entry._u8) entry._u8 = b64bytes(entry.att);
  const [L, H, n] = entry.shape;
  return (layer, head, i, j) => {
    if (head === -1) { let s = 0; for (let h = 0; h < H; h++) s += entry._u8[((layer * H + h) * n + i) * n + j]; return s / H / 255; }
    return entry._u8[((layer * H + head) * n + i) * n + j] / 255;
  };
}

/* ---------- key numbers (all text below uses these, so nothing drifts from the results) ---------- */
const K = {};
(function buildK() {
  M.forEach(m => {
    const k = m.tag.replace(/-/g, "_");
    K["acc_" + k] = pct1(m.acc); K["lat_" + k] = fmtLat(m.lat_ms); K["params_" + k] = fmtParams(m.params);
    K["train_" + k] = fmtTime(m.train_s);
  });
  const d = DATA.dataset;
  K.n_train = fmtInt(d.train); K.n_dev = fmtInt(d.dev); K.n_test = fmtInt(d.test); K.lr_features = fmtInt(DATA.lr.n_features);
  K.best_cls_acc = K.acc_tfidf_svm;
  K.gain_ft_cls = (+K.acc_ft_distilbert_8000 - +K.best_cls_acc).toFixed(1);
  K.train_h_distil = (M[MI.ft_distilbert_8000].train_s / 3600).toFixed(1);
  const at = (arr, n) => arr.find(p => p[0] === n)[1];
  const ct = DATA.curve.transformer, cf = DATA.curve.tfidf;
  K.lc500_t = pct1(at(ct, 500)); K.lc500_c = pct1(at(cf, 500));
  K.lc500_gain = (+K.lc500_t - +K.lc500_c).toFixed(1);
  K.lc4000_t = pct1(at(ct, 4000)); K.lc8000_t = pct1(at(ct, 8000)); K.lc8000_c = pct1(at(cf, 8000));
  K.neg_n = SENT.filter(r => r.n).length; K.con_n = SENT.filter(r => r.c).length;
  K.n_all_wrong = SENT.filter(r => nRight(r) === 0).length;
  const mc = mcnemar(MI.ft_distilbert_8000, MI.tfidf_svm);
  K.mc_ft_svm_b = mc.only_a; K.mc_ft_svm_c = mc.only_b; K.mc_ft_svm_p = (mc.p < 0.001 ? "&lt; 0.001" : "= " + mc.p.toFixed(3));
  K.latratio = Math.round(M[MI.ft_distilbert_8000].lat_ms / M[MI.tfidf_svm].lat_ms / 10) * 10;
})();
function bindK(root = document) { $$("[data-k]", root).forEach(e => { const v = K[e.dataset.k]; if (v == null) console.warn("missing K", e.dataset.k); else e.textContent = v; }); }
const resolve = v => (typeof v === "function" ? v(K) : v);

/* ---------- tooltip ---------- */
const tip = {
  el: null,
  show(evt, html) {
    const t = this.el || (this.el = $("#tip")); t.innerHTML = html; t.style.opacity = 1;
    const x = (evt.clientX ?? 0) + 14, y = (evt.clientY ?? 0) + 14, w = t.offsetWidth, h = t.offsetHeight;
    t.style.left = Math.max(8, Math.min(x, innerWidth - w - 8)) + "px";
    t.style.top = Math.max(8, Math.min(y, innerHeight - h - 8)) + "px";
  },
  hide() { (this.el || (this.el = $("#tip"))).style.opacity = 0; },
};
function attachTip(node, htmlFn) {
  node.addEventListener("mousemove", e => tip.show(e, htmlFn()));
  node.addEventListener("mouseleave", () => tip.hide());
  node.addEventListener("click", e => { tip.show(e, htmlFn()); setTimeout(() => tip.hide(), 2600); });
}

/* ---------- code blocks with copy button ---------- */
function codeBlock(file, text) {
  const wrap = document.createElement("div");
  wrap.innerHTML = `<div class="codefile"><span>${esc(file || "terminal")}</span><button class="copy" type="button">Copy</button></div><pre>${esc(text)}</pre>`;
  $(".copy", wrap).addEventListener("click", async e => {
    const b = e.currentTarget;
    try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; }
    catch (err) {
      const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); b.textContent = "Copied"; } catch (e2) { b.textContent = "Select & copy"; }
      ta.remove();
    }
    setTimeout(() => (b.textContent = "Copy"), 1400);
  });
  return wrap;
}

/* ---------- theme, navigation, progress ---------- */
const SECTIONS = [
  ["start", "Start here"], ["story", "The real-world story"], ["pipeline", "The project, step by step"], ["theory", "How Transformers work"],
  ["lab", "Sentence lab"], ["arena", "Model arena"], ["results", "Results explorer"], ["choose", "Choose a model"],
  ["lit", "Literature & Lithuania"], ["defend", "Defence prep"], ["present", "Presentation coach"], ["files", "Code & files"], ["glossary", "Glossary"],
];
function setupShell() {
  const root = document.documentElement;
  const effective = () => (root.dataset.theme === "dark" || (root.dataset.theme === "auto" && matchMedia("(prefers-color-scheme: dark)").matches)) ? "dark" : "light";
  root.dataset.theme = store.get("theme", "auto");
  $("#themeBtn").addEventListener("click", () => { const t = effective() === "dark" ? "light" : "dark"; root.dataset.theme = t; store.set("theme", t); window.dispatchEvent(new Event("themechange")); });
  $("#printBtn").addEventListener("click", () => window.print());
  $("#resetBtn").addEventListener("click", () => { if (confirm("Reset your progress, checklist and quiz score in this browser?")) { store.clear(); location.reload(); } });

  const nav = $("#nav"), side = $("#side");
  SECTIONS.forEach(([id, label], i) => {
    const a = document.createElement("a"); a.href = "#" + id; a.dataset.id = id;
    a.innerHTML = `<span class="no"><span>${i + 1}</span></span><span>${esc(label)}</span>`;
    a.addEventListener("click", () => { side.classList.remove("open"); $("#menuBtn").setAttribute("aria-expanded", "false"); });
    nav.appendChild(a);
  });
  $("#menuBtn").addEventListener("click", () => { const o = side.classList.toggle("open"); $("#menuBtn").setAttribute("aria-expanded", String(o)); });

  const seen = new Set(store.get("seen", []));
  const paint = () => {
    $$("#nav a").forEach(a => a.classList.toggle("seen", seen.has(a.dataset.id)));
    $("#progBar").style.width = (seen.size / SECTIONS.length * 100) + "%";
    $("#progLabel").textContent = `${seen.size} of ${SECTIONS.length} sections visited`;
  };
  paint();
  const obs = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (!en.isIntersecting) return;
      const id = en.target.id; seen.add(id); store.set("seen", [...seen]);
      $$("#nav a").forEach(a => a.classList.toggle("active", a.dataset.id === id)); paint();
    });
  }, { rootMargin: "-25% 0px -65% 0px" });
  $$(".sec").forEach(s => obs.observe(s));
}
