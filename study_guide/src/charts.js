"use strict";
/* SVG charts (no libraries). Each chart re-renders to the container width so text stays readable on phones. */

const SVGNS = "http://www.w3.org/2000/svg";
function S(tag, attrs = {}, parent) { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function T(text, attrs, parent) { const e = S("text", attrs, parent); e.textContent = text; return e; }
function svgRoot(el, W, H, label) {
  el.innerHTML = "";
  const s = S("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart", role: "img", "aria-label": label });
  s.style.maxWidth = "100%"; s.style.height = "auto";
  el.appendChild(s); return s;
}
const widthOf = el => Math.max(300, Math.floor(el.clientWidth || 700));
const lin = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
const logS = (d0, d1, r0, r1) => v => r0 + (Math.log(v) - Math.log(d0)) / (Math.log(d1) - Math.log(d0)) * (r1 - r0);
const famColor = m => FAMCLR[m.family];
function modelTip(m) {
  return `<b>${esc(m.name)}</b><br>Accuracy ${pct1(m.acc)}% · F1 ${pct1(m.f1)}%<br>Latency ${fmtLat(m.lat_ms)} ms · ${fmtParams(m.params)} parameters<br>Training: ${m.train_size ? fmtInt(m.train_size) + " labels, " + fmtTime(m.train_s) : "none (ready-made)"}`;
}

/* 1 · accuracy dot plot with 95% bootstrap intervals */
function chartAccuracy(el) {
  const W = widthOf(el), narrow = W < 560;
  const rows = [...M].sort((a, b) => b.acc - a.acc);
  const left = narrow ? 112 : 196, right = 18, top = 6, rh = 32, bottom = 36, H = top + rows.length * rh + bottom;
  const sx = lin(70, 95, left, W - right);
  const s = svgRoot(el, W, H, "Accuracy of the ten models with 95% confidence intervals");
  for (let v = 70; v <= 95; v += 5) { S("line", { x1: sx(v), x2: sx(v), y1: top, y2: H - bottom, class: "grid" }, s); T(v + "%", { x: sx(v), y: H - bottom + 18, "text-anchor": "middle", class: "muted-t" }, s); }
  rows.forEach((m, i) => {
    const y = top + i * rh + rh / 2, st = DATA.stats[m.tag];
    T(narrow ? m.short : m.name, { x: left - 10, y: y + 4, "text-anchor": "end" }, s);
    S("line", { x1: sx(70), x2: sx(95), y1: y, y2: y, class: "grid" }, s);
    S("line", { x1: sx(st.lo * 100), x2: sx(st.hi * 100), y1: y, y2: y, stroke: famColor(m), "stroke-width": 3, "stroke-linecap": "round", opacity: .45 }, s);
    const dot = S("circle", { cx: sx(m.acc * 100), cy: y, r: 7, fill: famColor(m), stroke: "var(--surface)", "stroke-width": 2 }, s);
    T(pct1(m.acc), { x: sx(st.hi * 100) + 8, y: y + 4, class: "muted-t" }, s);
    const hit = S("rect", { x: 0, y: y - rh / 2, width: W, height: rh, fill: "transparent" }, s);
    attachTip(hit, () => modelTip(m) + `<br>95% interval ${pct1(st.lo)}–${pct1(st.hi)}%`);
  });
  T("Accuracy on the 872 test sentences (axis starts at 70%)", { x: (left + W - right) / 2, y: H - 4, "text-anchor": "middle", class: "muted-t" }, s);
}

/* 2 · cost scatter: accuracy vs latency (log) */
function chartCost(el) {
  const W = widthOf(el), narrow = W < 560;
  const left = 48, right = narrow ? 14 : 30, top = 12, bottom = 44, H = Math.round(Math.max(300, W * (narrow ? .9 : .52)));
  const sx = logS(0.03, 250, left, W - right), sy = lin(74, 93, H - bottom, top);
  const maxP = Math.max(...M.map(m => m.params));
  const s = svgRoot(el, W, H, "Accuracy versus inference latency");
  for (let v = 74; v <= 92; v += 2) { S("line", { x1: left, x2: W - right, y1: sy(v), y2: sy(v), class: "grid" }, s); T(v, { x: left - 8, y: sy(v) + 4, "text-anchor": "end", class: "muted-t" }, s); }
  [0.1, 1, 10, 100].forEach(v => { S("line", { x1: sx(v), x2: sx(v), y1: top, y2: H - bottom, class: "grid" }, s); T(v + " ms", { x: sx(v), y: H - bottom + 18, "text-anchor": "middle", class: "muted-t" }, s); });
  T("Inference time per sentence on CPU (log scale; slower →)", { x: (left + W - right) / 2, y: H - 6, "text-anchor": "middle", class: "muted-t" }, s);
  T("Accuracy %", { x: 12, y: top + 8, class: "muted-t" }, s);
  const offs = { tfidf_svm: [10, -9, "start"], tfidf_nb: [11, 4, "start"], tfidf_logreg: [10, 17, "start"] };
  [...M].sort((a, b) => b.params - a.params).forEach(m => {
    const cx = sx(m.lat_ms), cy = sy(m.acc * 100), r = 5 + 13 * Math.sqrt(m.params / maxP);
    const g = S("g", {}, s);
    S("circle", { cx, cy, r, fill: famColor(m), "fill-opacity": .85, stroke: "var(--surface)", "stroke-width": 2 }, g);
    const o = offs[m.tag] || (cx > W * .62 ? [-r - 6, 4, "end"] : [r + 6, 4, "start"]);
    T(m.short, { x: cx + o[0], y: cy + o[1], "text-anchor": o[2], class: "muted-t" }, g);
    attachTip(g, () => modelTip(m));
  });
}

/* 3 · learning curve (log x), with a movable marker */
function interp(curve, n) {
  const pts = curve.slice().sort((a, b) => a[0] - b[0]);
  if (n <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) if (n <= pts[i][0]) { const t = (Math.log(n) - Math.log(pts[i - 1][0])) / (Math.log(pts[i][0]) - Math.log(pts[i - 1][0])); return pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1]); }
  return pts[pts.length - 1][1];
}
function labelsNeeded(curve, acc) {          // inverse interpolation along a (monotone) learning curve
  const pts = curve.slice().sort((a, b) => a[0] - b[0]);
  if (acc <= pts[0][1]) return pts[0][0];
  for (let i = 1; i < pts.length; i++) if (acc <= pts[i][1]) { const t = (acc - pts[i - 1][1]) / (pts[i][1] - pts[i - 1][1]); return Math.exp(Math.log(pts[i - 1][0]) + t * (Math.log(pts[i][0]) - Math.log(pts[i - 1][0]))); }
  return Infinity;
}
function chartLC(el, n) {
  const W = widthOf(el), narrow = W < 560;
  const left = 46, right = 16, top = 14, bottom = 44, H = Math.round(Math.max(280, W * (narrow ? .8 : .46)));
  const sx = logS(400, 10000, left, W - right), sy = lin(55, 90, H - bottom, top);
  const s = svgRoot(el, W, H, "Learning curves of BERT-small and TF-IDF");
  for (let v = 55; v <= 90; v += 5) { S("line", { x1: left, x2: W - right, y1: sy(v), y2: sy(v), class: "grid" }, s); T(v, { x: left - 8, y: sy(v) + 4, "text-anchor": "end", class: "muted-t" }, s); }
  [500, 1000, 2000, 4000, 8000].forEach(v => { S("line", { x1: sx(v), x2: sx(v), y1: top, y2: H - bottom, class: "grid" }, s); T(fmtInt(v), { x: sx(v), y: H - bottom + 18, "text-anchor": "middle", class: "muted-t" }, s); });
  T("Labelled training sentences (log scale)", { x: (left + W - right) / 2, y: H - 6, "text-anchor": "middle", class: "muted-t" }, s);
  const ref = +K.best_cls_acc;
  S("line", { x1: left, x2: W - right, y1: sy(ref), y2: sy(ref), stroke: "var(--classical)", "stroke-dasharray": "6 5", "stroke-width": 1.5 }, s);
  T(`best classical model on all ${K.n_train} sentences: ${ref}%`, { x: left + 8, y: sy(ref) - 7, "text-anchor": "start", style: "fill:var(--classical)" }, s);
  const draw = (curve, color, name) => {
    const pts = curve.slice().sort((a, b) => a[0] - b[0]);
    S("polyline", { points: pts.map(p => `${sx(p[0])},${sy(p[1] * 100)}`).join(" "), fill: "none", stroke: color, "stroke-width": 3, "stroke-linejoin": "round" }, s);
    pts.forEach(p => { const c = S("circle", { cx: sx(p[0]), cy: sy(p[1] * 100), r: 5, fill: color, stroke: "var(--surface)", "stroke-width": 2 }, s); attachTip(c, () => `<b>${name}</b><br>${fmtInt(p[0])} labels: ${pct1(p[1])}%`); });
  };
  draw(DATA.curve.tfidf, "var(--classical)", "TF-IDF + logistic regression");
  draw(DATA.curve.transformer, "var(--transformer)", "BERT-small fine-tuned");
  T("BERT-small (fine-tuned)", { x: sx(500) + 8, y: sy(+K.lc500_t) - 12, style: "fill:var(--transformer);font-weight:700" }, s);
  T("TF-IDF + logistic regression", { x: sx(500) + 8, y: sy(+K.lc500_c) + 22, style: "fill:var(--classical);font-weight:700" }, s);
  if (n) {
    S("line", { x1: sx(n), x2: sx(n), y1: top, y2: H - bottom, stroke: "var(--text)", "stroke-width": 1.5, "stroke-dasharray": "3 3" }, s);
    [[DATA.curve.transformer, "var(--transformer)"], [DATA.curve.tfidf, "var(--classical)"]].forEach(([c, col]) =>
      S("circle", { cx: sx(n), cy: sy(interp(c, n) * 100), r: 8, fill: "none", stroke: col, "stroke-width": 3 }, s));
  }
}

/* 4 · subsets (negation / contrast): dumbbell and drop views */
function subsetStats() {
  const neg = SENT.filter(r => r.n), con = SENT.filter(r => r.c);
  return M.map((m, i) => ({ m, all: accOn(SENT, i), neg: accOn(neg, i), con: accOn(con, i) })).sort((a, b) => b.all - a.all);
}
function chartSubsets(el, view) {
  const W = widthOf(el), narrow = W < 560, rows = subsetStats();
  const left = narrow ? 112 : 196, right = 16, top = 6, rh = 32, bottom = 36, H = top + rows.length * rh + bottom;
  const s = svgRoot(el, W, H, "Accuracy on negation and contrast sentences");
  const lblx = left - 10;
  if (view === "acc") {
    const sx = lin(60, 95, left, W - right);
    for (let v = 60; v <= 95; v += 5) { S("line", { x1: sx(v), x2: sx(v), y1: top, y2: H - bottom, class: "grid" }, s); T(v + "%", { x: sx(v), y: H - bottom + 18, "text-anchor": "middle", class: "muted-t" }, s); }
    rows.forEach((r, i) => {
      const y = top + i * rh + rh / 2;
      T(narrow ? r.m.short : r.m.name, { x: lblx, y: y + 4, "text-anchor": "end" }, s);
      const xs = [r.all, r.neg, r.con].map(v => sx(v * 100));
      S("line", { x1: Math.min(...xs), x2: Math.max(...xs), y1: y, y2: y, stroke: "var(--muted)", "stroke-width": 2, opacity: .5 }, s);
      [[r.all, "#9a98b4", "all sentences"], [r.neg, "var(--classical)", `negation (n=${K.neg_n})`], [r.con, "var(--recurrent)", `contrast (n=${K.con_n})`]].forEach(([v, c, lab]) => {
        const d = S("circle", { cx: sx(v * 100), cy: y, r: 6, fill: c, stroke: "var(--surface)", "stroke-width": 2 }, s);
        attachTip(d, () => `<b>${esc(r.m.name)}</b><br>${lab}: ${pct1(v)}%`);
      });
    });
    T("Accuracy %", { x: (left + W - right) / 2, y: H - 4, "text-anchor": "middle", class: "muted-t" }, s);
  } else {
    const sx = lin(-12, 3, left, W - right);
    for (let v = -12; v <= 3; v += 3) { S("line", { x1: sx(v), x2: sx(v), y1: top, y2: H - bottom, class: v === 0 ? "axis" : "grid" }, s); T(v, { x: sx(v), y: H - bottom + 18, "text-anchor": "middle", class: "muted-t" }, s); }
    rows.forEach((r, i) => {
      const y = top + i * rh;
      T(narrow ? r.m.short : r.m.name, { x: lblx, y: y + rh / 2 + 4, "text-anchor": "end" }, s);
      [[r.neg - r.all, "var(--classical)", "negation", 6], [r.con - r.all, "var(--recurrent)", "contrast", 17]].forEach(([d, c, lab, off]) => {
        const v = d * 100, x0 = sx(0), x1 = sx(v);
        const bar = S("rect", { x: Math.min(x0, x1), y: y + off - 3, width: Math.max(2, Math.abs(x1 - x0)), height: 9, rx: 3, fill: c }, s);
        T((v > 0 ? "+" : "") + v.toFixed(1), { x: v < 0 ? x1 - 4 : x1 + 4, y: y + off + 5, "text-anchor": v < 0 ? "end" : "start", class: "muted-t", style: "font-size:10.5px" }, s);
        attachTip(bar, () => `<b>${esc(r.m.name)}</b><br>${lab}: ${(v > 0 ? "+" : "") + v.toFixed(1)} points vs overall`);
      });
    });
    T("Change in accuracy compared with all sentences (percentage points; left = worse)", { x: (left + W - right) / 2, y: H - 4, "text-anchor": "middle", class: "muted-t" }, s);
  }
}

/* 5 · dev accuracy per epoch */
function chartDev(el) {
  const W = widthOf(el), left = 46, right = 16, top = 14, bottom = 40, H = Math.round(Math.max(260, W * .4));
  const ms = M.filter(m => m.history.length);
  const sx = lin(1, 4, left + 14, W - right - 14), sy = lin(80, 93, H - bottom, top);
  const s = svgRoot(el, W, H, "Development-set accuracy per epoch");
  for (let v = 80; v <= 92; v += 2) { S("line", { x1: left, x2: W - right, y1: sy(v), y2: sy(v), class: "grid" }, s); T(v, { x: left - 8, y: sy(v) + 4, "text-anchor": "end", class: "muted-t" }, s); }
  [1, 2, 3, 4].forEach(e => T("epoch " + e, { x: sx(e), y: H - bottom + 18, "text-anchor": "middle", class: "muted-t" }, s));
  ms.forEach(m => {
    const pts = m.history.map((a, i) => [sx(i + 1), sy(a * 100)]);
    S("polyline", { points: pts.map(p => p.join(",")).join(" "), fill: "none", stroke: famColor(m), "stroke-width": 3 }, s);
    m.history.forEach((a, i) => { const c = S("circle", { cx: pts[i][0], cy: pts[i][1], r: 5, fill: famColor(m), stroke: "var(--surface)", "stroke-width": 2 }, s); attachTip(c, () => `<b>${esc(m.name)}</b><br>epoch ${i + 1}: dev accuracy ${pct1(a)}%`); });
    const last = pts[pts.length - 1];
    T(m.short, { x: Math.min(last[0] + 8, W - 90), y: last[1] + (m.tag === "ft_bert-small_8000" ? 14 : 4), style: `fill:${famColor(m)};font-weight:700` }, s);
  });
  T("Dev accuracy % (the epoch with the highest value is kept)", { x: (left + W - right) / 2, y: H - 4, "text-anchor": "middle", class: "muted-t" }, s);
}

/* 6 · attention heat-map */
function heatmap(el, tokens, getter, layer, head, selected, onSelect) {
  const n = tokens.length, W = widthOf(el), lab = Math.min(96, Math.max(54, Math.floor(W * .18)));
  const cell = Math.max(14, Math.min(34, Math.floor((W - lab - 8) / n))), side = lab + cell * n + 4, H = lab + cell * n + 4;
  const s = svgRoot(el, side, H, "Attention heat-map");
  s.setAttribute("class", "chart heat"); s.style.maxWidth = side + "px";
  tokens.forEach((t, j) => { const x = lab + j * cell + cell / 2; const tx = T(t, { x, y: lab - 6, transform: `rotate(-55 ${x} ${lab - 6})`, style: "font-size:10.5px" }, s); });
  tokens.forEach((t, i) => {
    T(t, { x: lab - 6, y: lab + i * cell + cell / 2 + 4, "text-anchor": "end", style: `font-size:10.5px;font-weight:${i === selected ? 800 : 400}` }, s);
    for (let j = 0; j < n; j++) {
      const v = getter(layer, head, i, j);
      const r = S("rect", { x: lab + j * cell, y: lab + i * cell, width: cell - 1, height: cell - 1, rx: 3, fill: "var(--violet)", "fill-opacity": Math.min(1, Math.pow(v, .6) * 1.15).toFixed(3) }, s);
      attachTip(r, () => `<b>${esc(tokens[i])}</b> → <b>${esc(tokens[j])}</b><br>attention ${(v * 100).toFixed(0)}%`);
      r.addEventListener("click", () => onSelect && onSelect(i));
    }
  });
}
