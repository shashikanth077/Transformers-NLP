/* Pipeline helpers shared by the page and the Node trainer: split, data preparation, training loop with early stopping, BPE. */
function makeSplit(corpus, seed, coverage) {
  const r = rng(seed), idx = corpus.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const n = idx.length, nt = Math.round(n * 0.15), nv = Math.round(n * 0.15);
  const words = i => new Set([...tok(corpus[i].en, true).map(w => "e:" + w), ...tok(corpus[i].kn, false).map(w => "k:" + w)]);
  const test = [], val = [];
  if (!coverage) { test.push(...idx.slice(0, nt)); val.push(...idx.slice(nt, nt + nv)); }
  else {
    const cnt = new Map(); idx.forEach(i => words(i).forEach(w => cnt.set(w, (cnt.get(w) || 0) + 1)));
    for (const i of idx) {
      if (test.length >= nt && val.length >= nv) break;
      const ws = [...words(i)];
      if (ws.every(w => cnt.get(w) > 1)) { ws.forEach(w => cnt.set(w, cnt.get(w) - 1)); (test.length < nt ? test : val).push(i); }
    }
  }
  const held = new Set([...test, ...val]);
  return { test, val, train: idx.filter(i => !held.has(i)) };
}
function prepare(corpus, split) {
  const tr = split.train.map(i => corpus[i]), Vs = buildVocab(tr.map(p => p.en), true), Vt = buildVocab(tr.map(p => p.kn), false);
  const enc = p => ({ src: [...toIds(Vs, p.en, true), 2], tin: [1, ...toIds(Vt, p.kn, false)], tout: [...toIds(Vt, p.kn, false), 2] });
  return { Vs, Vt, enc, tr, va: split.val.map(i => corpus[i]), te: split.test.map(i => corpus[i]) };
}
const evalLoss = (P, enc, ps) => ps.reduce((s, p) => { const e = enc(p); return s + lossGrad(P, e.src, e.tin, e.tout, false).loss; }, 0) / Math.max(1, ps.length);
const decodeText = (P, prep, p) => { const e = prep.enc(p), g = greedy(P, e.src, 1, 2, 24); return g.ids.map(i => prep.Vt.words[i]); };
function valQuality(P, prep) { const c = [], r = []; let ch = 0; prep.va.forEach(p => { const t = decodeText(P, prep, p); c.push(t); r.push(tok(p.kn, false)); ch += chrf(t.join(" "), tok(p.kn, false).join(" ")); }); return { bleu: bleuStats(c, r).bleu, chrf: ch / Math.max(1, prep.va.length) }; }
const snapshot = P => paramArrays(P).map(([, a]) => Float64Array.from(a));
const restore = (P, snap) => paramArrays(P).forEach(([, a], i) => a.set(snap[i]));
class Trainer {
  constructor(prep, cfg) {
    this.prep = prep; this.cfg = cfg; this.P = initModel(prep.Vs.words.length, prep.Vt.words.length, cfg.d, cfg.H, cfg.seed); this.A = adamInit(this.P);
    this.r = rng(cfg.seed + 5); this.data = prep.tr.map(prep.enc); this.hist = []; this.best = { score: -1, ep: 0, snap: snapshot(this.P) };
  }
  epoch() {
    const { P, A, data, cfg } = this, order = data.map((_, i) => i), r = this.r;
    for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    let tl = 0;
    for (let b = 0; b < order.length; b += cfg.batch) {
      const idxs = order.slice(b, b + cfg.batch); let acc = null;
      for (const i of idxs) {
        const o = lossGrad(P, data[i].src, data[i].tin, data[i].tout, true), ga = gradArrays(o.G); tl += o.loss;
        if (!acc) { acc = new Map(); for (const [k, g] of ga) acc.set(k, Float64Array.from(g, v => v / idxs.length)); } else for (const [k, g] of ga) { const a = acc.get(k); for (let q = 0; q < g.length; q++) a[q] += g[q] / idxs.length; }
      }
      adamStep(P, acc, A, cfg.lr, cfg.clip);
    }
    const ep = this.hist.length + 1, q = valQuality(P, this.prep), h = { ep, train: evalLoss(P, this.prep.enc, this.prep.tr), val: evalLoss(P, this.prep.enc, this.prep.va), valChrf: q.chrf, valBleu: q.bleu };
    this.hist.push(h); if (h.valChrf > this.best.score) this.best = { score: h.valChrf, ep, val: h.val, snap: snapshot(P) };
    return h;
  }
  finish() { restore(this.P, this.best.snap); return this.best; }
}
/* ---- byte-pair encoding on characters (Unicode code points) ---- */
function bpeLearn(wordCounts, nMerges) {
  let seqs = [...wordCounts].map(([w, c]) => ({ s: [...w], c })); const merges = [];
  for (let m = 0; m < nMerges; m++) {
    const pc = new Map(); seqs.forEach(({ s, c }) => { for (let i = 0; i + 1 < s.length; i++) { const k = s[i] + "\u0001" + s[i + 1]; pc.set(k, (pc.get(k) || 0) + c); } });
    if (!pc.size) break; let best = null, bc = 0; for (const [k, c] of pc) if (c > bc || (c === bc && k < best)) { best = k; bc = c; }
    const [a, b] = best.split("\u0001"); merges.push({ a, b, count: bc });
    seqs = seqs.map(({ s, c }) => { const o = []; for (let i = 0; i < s.length; i++) { if (i + 1 < s.length && s[i] === a && s[i + 1] === b) { o.push(a + b); i++; } else o.push(s[i]); } return { s: o, c }; });
  }
  return { merges, seqs };
}
function bpeApply(word, merges) { let s = [...word]; for (const { a, b } of merges) { const o = []; for (let i = 0; i < s.length; i++) { if (i + 1 < s.length && s[i] === a && s[i + 1] === b) { o.push(a + b); i++; } else o.push(s[i]); } s = o; } return s; }
function bpePairCounts(wordCounts, merges) { const seqs = [...wordCounts].map(([w, c]) => ({ s: bpeApply(w, merges), c })), pc = new Map(); seqs.forEach(({ s, c }) => { for (let i = 0; i + 1 < s.length; i++) { const k = s[i] + " + " + s[i + 1]; pc.set(k, (pc.get(k) || 0) + c); } }); return [...pc].sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : 1)); }
