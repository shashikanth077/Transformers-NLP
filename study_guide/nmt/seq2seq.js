/* Tiny GRU encoder-decoder with scaled dot-product (Luong) attention: forward, teacher-forced loss, backpropagation, decoding, BLEU, chrF. */
const sig = x => 1 / (1 + Math.exp(-x));
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = r => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
const zeros = n => new Float64Array(n);
function cat(a, b) { const v = new Float64Array(a.length + b.length); v.set(a); v.set(b, a.length); return v; }
function lin(W, b, v, H, n) { const o = new Float64Array(H); for (let k = 0; k < H; k++) { let s = b ? b[k] : 0; for (let j = 0; j < n; j++) s += W[k * n + j] * v[j]; o[k] = s; } return o; }
function outer(G, da, v, n, H) { for (let k = 0; k < H; k++) for (let j = 0; j < n; j++) G[k * n + j] += da[k] * v[j]; }
function tmul(W, da, n, H, dv) { for (let k = 0; k < H; k++) for (let j = 0; j < n; j++) dv[j] += W[k * n + j] * da[k]; }
function softmax(z) { const m = Math.max(...z), e = Array.from(z, v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return Float64Array.from(e, v => v / s); }
/* ---- GRU cell ---- */
const gruShapes = (D, H) => ({ Wz: H * (D + H), bz: H, Wr: H * (D + H), br: H, Wx: H * D, Uh: H * H, bh: H });
function gruF(g, x, hp, D, H) {
  const n = D + H, v = cat(x, hp), z = lin(g.Wz, g.bz, v, H, n).map(sig), r = lin(g.Wr, g.br, v, H, n).map(sig), rh = r.map((q, k) => q * hp[k]), ax = lin(g.Wx, g.bh, x, H, D);
  for (let k = 0; k < H; k++) { let s = 0; for (let j = 0; j < H; j++) s += g.Uh[k * H + j] * rh[j]; ax[k] += s; }
  const hc = ax.map(Math.tanh), h = z.map((zz, k) => (1 - zz) * hp[k] + zz * hc[k]);
  return { x, hp, v, z, r, rh, hc, h };
}
function gruB(g, Gg, S, dh, D, H) {
  const n = D + H, dz = dh.map((d, k) => d * (S.hc[k] - S.hp[k])), dhc = dh.map((d, k) => d * S.z[k]), dhp = dh.map((d, k) => d * (1 - S.z[k]));
  const dah = dhc.map((d, k) => d * (1 - S.hc[k] * S.hc[k])); outer(Gg.Wx, dah, S.x, D, H); outer(Gg.Uh, dah, S.rh, H, H); for (let k = 0; k < H; k++) Gg.bh[k] += dah[k];
  const drh = zeros(H); tmul(g.Uh, dah, H, H, drh); const dr = drh.map((d, k) => d * S.hp[k]); for (let k = 0; k < H; k++) dhp[k] += drh[k] * S.r[k];
  const daz = dz.map((d, k) => d * S.z[k] * (1 - S.z[k])), dar = dr.map((d, k) => d * S.r[k] * (1 - S.r[k]));
  outer(Gg.Wz, daz, S.v, n, H); outer(Gg.Wr, dar, S.v, n, H); for (let k = 0; k < H; k++) { Gg.bz[k] += daz[k]; Gg.br[k] += dar[k]; }
  const dv = zeros(n); tmul(g.Wz, daz, n, H, dv); tmul(g.Wr, dar, n, H, dv); const dxx = zeros(D); tmul(g.Wx, dah, D, H, dxx);
  const dx = new Float64Array(D); for (let j = 0; j < D; j++) dx[j] = dv[j] + dxx[j]; for (let k = 0; k < H; k++) dhp[k] += dv[D + k];
  return { dx, dhp };
}
/* ---- model ---- */
function initModel(Vs, Vt, d, H, seed) {
  const r = rng(seed), P = { Es: zeros(Vs * d), Et: zeros(Vt * d), enc: {}, dec: {}, Wc: zeros(H * 2 * H), bc: zeros(H), Wo: zeros(Vt * H), bo: zeros(Vt) };
  const fill = (arr, sd) => { for (let i = 0; i < arr.length; i++) arr[i] = gauss(r) * sd; };
  fill(P.Es, 0.3); fill(P.Et, 0.3); fill(P.Wc, 1 / Math.sqrt(2 * H)); fill(P.Wo, 1 / Math.sqrt(H));
  for (const part of ["enc", "dec"]) { const sh = gruShapes(d, H); for (const k in sh) { P[part][k] = zeros(sh[k]); if (/^W|^U/.test(k)) fill(P[part][k], 1 / Math.sqrt(k === "Uh" ? H : k === "Wx" ? d : d + H)); } }
  P.cfg = { Vs, Vt, d, H }; return P;
}
const paramArrays = P => { const o = []; for (const k of ["Es", "Et", "Wc", "bc", "Wo", "bo"]) o.push([k, P[k]]); for (const part of ["enc", "dec"]) for (const k in P[part]) o.push([part + "." + k, P[part][k]]); return o; };
const nParams = P => paramArrays(P).reduce((s, [, a]) => s + a.length, 0);
function zeroGrads(P) { const G = { Es: zeros(P.Es.length), Et: zeros(P.Et.length), Wc: zeros(P.Wc.length), bc: zeros(P.bc.length), Wo: zeros(P.Wo.length), bo: zeros(P.bo.length), enc: {}, dec: {} }; for (const part of ["enc", "dec"]) for (const k in P[part]) G[part][k] = zeros(P[part][k].length); return G; }
function encode(P, src) {
  const { d, H } = P.cfg; let h = zeros(H); const st = [], hs = [];
  for (const id of src) { const S = gruF(P.enc, P.Es.slice(id * d, id * d + d), h, d, H); st.push(S); h = S.h; hs.push(h); }
  return { st, hs, final: h };
}
function attend(P, s, hs) {
  const { H } = P.cfg, sq = Math.sqrt(H), sc = hs.map(h => { let e = 0; for (let k = 0; k < H; k++) e += s[k] * h[k]; return e / sq; }), al = softmax(sc), c = zeros(H);
  hs.forEach((h, i) => { for (let k = 0; k < H; k++) c[k] += al[i] * h[k]; });
  const u = cat(c, s), a = lin(P.Wc, P.bc, u, H, 2 * H).map(Math.tanh), lg = lin(P.Wo, P.bo, a, P.cfg.Vt, H), p = softmax(lg);
  return { sc, al, c, u, a, lg, p, s };
}
/* teacher-forced loss (mean over target tokens) and optional gradients */
function lossGrad(P, src, tin, tout, wantGrad) {
  const { d, H, Vt } = P.cfg, E = encode(P, src); let s = E.final, loss = 0; const T = tin.length, dec = [], att = [];
  for (let t = 0; t < T; t++) { const S = gruF(P.dec, P.Et.slice(tin[t] * d, tin[t] * d + d), s, d, H); s = S.h; const A = attend(P, s, E.hs); dec.push(S); att.push(A); loss -= Math.log(A.p[tout[t]] + 1e-12); }
  loss /= T; if (!wantGrad) return { loss, att };
  const G = zeroGrads(P), dhs = E.hs.map(() => zeros(H)), sq = Math.sqrt(H); let dsNext = zeros(H);
  for (let t = T - 1; t >= 0; t--) {
    const A = att[t], S = dec[t], dl = Float64Array.from(A.p, v => v / T); dl[tout[t]] -= 1 / T;
    outer(G.Wo, dl, A.a, H, Vt); for (let k = 0; k < Vt; k++) G.bo[k] += dl[k];
    const da = zeros(H); tmul(P.Wo, dl, H, Vt, da); const dpre = da.map((v, k) => v * (1 - A.a[k] * A.a[k])); outer(G.Wc, dpre, A.u, 2 * H, H); for (let k = 0; k < H; k++) G.bc[k] += dpre[k];
    const du = zeros(2 * H); tmul(P.Wc, dpre, 2 * H, H, du); const dc = du.slice(0, H), ds = du.slice(H);
    const dal = E.hs.map(h => { let q = 0; for (let k = 0; k < H; k++) q += dc[k] * h[k]; return q; }), mean = dal.reduce((q, v, i) => q + v * A.al[i], 0);
    E.hs.forEach((h, i) => { const de = A.al[i] * (dal[i] - mean); for (let k = 0; k < H; k++) { dhs[i][k] += A.al[i] * dc[k] + de * S.h[k] / sq; ds[k] += de * h[k] / sq; } });
    for (let k = 0; k < H; k++) ds[k] += dsNext[k];
    const b = gruB(P.dec, G.dec, S, ds, d, H); dsNext = b.dhp; for (let j = 0; j < d; j++) G.Et[tin[t] * d + j] += b.dx[j];
  }
  let dh = dsNext; // gradient reaching the encoder's final state through s_0
  for (let i = src.length - 1; i >= 0; i--) { const tot = dhs[i].map((v, k) => v + dh[k]); const b = gruB(P.enc, G.enc, E.st[i], tot, d, H); dh = b.dhp; for (let j = 0; j < d; j++) G.Es[src[i] * d + j] += b.dx[j]; }
  return { loss, att, G };
}
/* ---- optimiser ---- */
function adamInit(P) { const o = { t: 0, m: new Map(), v: new Map() }; for (const [k, a] of paramArrays(P)) { o.m.set(k, zeros(a.length)); o.v.set(k, zeros(a.length)); } return o; }
function gradArrays(G) { const o = new Map(); for (const k of ["Es", "Et", "Wc", "bc", "Wo", "bo"]) o.set(k, G[k]); for (const part of ["enc", "dec"]) for (const k in G[part]) o.set(part + "." + k, G[part][k]); return o; }
function adamStep(P, GA, A, lr, clip) {
  let nrm = 0; for (const g of GA.values()) for (const v of g) nrm += v * v; nrm = Math.sqrt(nrm); const sc = nrm > clip ? clip / nrm : 1; A.t++;
  for (const [k, a] of paramArrays(P)) { const g = GA.get(k), m = A.m.get(k), v = A.v.get(k); for (let i = 0; i < a.length; i++) { const gi = g[i] * sc; m[i] = .9 * m[i] + .1 * gi; v[i] = .999 * v[i] + .001 * gi * gi; a[i] -= lr * (m[i] / (1 - .9 ** A.t)) / (Math.sqrt(v[i] / (1 - .999 ** A.t)) + 1e-8); } }
  return nrm;
}
/* ---- decoding ---- */
function greedy(P, src, BOS, EOS, maxLen) {
  const E = encode(P, src), { d, H } = P.cfg; let s = E.final, y = BOS; const out = [], att = [], lp = [];
  for (let t = 0; t < maxLen; t++) { const S = gruF(P.dec, P.Et.slice(y * d, y * d + d), s, d, H); s = S.h; const A = attend(P, s, E.hs); y = A.p.indexOf(Math.max(...A.p)); lp.push(Math.log(A.p[y])); att.push(A); if (y === EOS) break; out.push(y); }
  return { ids: out, att, logprob: lp.reduce((a, b) => a + b, 0) };
}
function beam(P, src, BOS, EOS, maxLen, width, alpha) {
  const E = encode(P, src), { d, H } = P.cfg; let hyps = [{ ids: [], lp: 0, s: E.final, y: BOS }]; const finished = [];
  for (let t = 0; t < maxLen; t++) {
    const cand = [];
    for (const h of hyps) { const S = gruF(P.dec, P.Et.slice(h.y * d, h.y * d + d), h.s, d, H), A = attend(P, S.h, E.hs); A.p.forEach((p, id) => { if (p > 1e-9) cand.push({ ids: h.ids.concat(id), lp: h.lp + Math.log(p), s: S.h, y: id, done: id === EOS }); }); }
    cand.sort((a, b) => b.lp - a.lp); hyps = [];
    for (const c of cand) { if (c.done) finished.push(c); else hyps.push(c); if (hyps.length >= width) break; }
    if (!hyps.length || finished.length >= width) break;
  }
  const pool = finished.length ? finished : hyps, norm = h => h.lp / Math.pow(h.ids.length, alpha);
  pool.sort((a, b) => norm(b) - norm(a)); return pool.map(h => ({ ids: h.ids.filter(i => i !== EOS), lp: h.lp, score: norm(h) }));
}
/* ---- tokenizer, vocabulary ---- */
const TOKRE = /[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]/gu;
const tok = (s, lower) => (lower ? s.toLowerCase() : s).match(TOKRE) || [];
function buildVocab(sents, lower) { const c = new Map(); sents.forEach(s => tok(s, lower).forEach(w => c.set(w, (c.get(w) || 0) + 1))); const words = ["<pad>", "<s>", "</s>", "<unk>", ...[...c.keys()].sort((a, b) => c.get(b) - c.get(a) || (a < b ? -1 : 1))]; return { words, idx: new Map(words.map((w, i) => [w, i])), counts: c }; }
const toIds = (V, s, lower) => tok(s, lower).map(w => (V.idx.has(w) ? V.idx.get(w) : 3));
/* ---- metrics ---- */
function ngr(t, n) { const m = new Map(); for (let i = 0; i + n <= t.length; i++) { const k = t.slice(i, i + n).join("\u0001"); m.set(k, (m.get(k) || 0) + 1); } return m; }
function bleuStats(cands, refs) {
  const match = [0, 0, 0, 0], tot = [0, 0, 0, 0]; let c = 0, r = 0;
  cands.forEach((cd, i) => { const rf = refs[i]; c += cd.length; r += rf.length; for (let n = 1; n <= 4; n++) { const cg = ngr(cd, n), rg = ngr(rf, n); for (const [g, v] of cg) { match[n - 1] += Math.min(v, rg.get(g) || 0); tot[n - 1] += v; } } });
  const p = match.map((m, i) => (i === 0 ? (tot[0] ? m / tot[0] : 0) : (m + 1) / (tot[i] + 1))), bp = c >= r ? 1 : Math.exp(1 - r / Math.max(c, 1)), lg = p.reduce((s, v) => s + Math.log(Math.max(v, 1e-9)), 0) / 4;
  return { match, tot, p, bp, c, r, bleu: p[0] === 0 ? 0 : 100 * bp * Math.exp(lg) };
}
function chrf(cand, ref, N = 6, beta = 2) {
  const cs = cand.replace(/\s+/g, ""), rs = ref.replace(/\s+/g, ""); let P = 0, R = 0, cnt = 0;
  for (let n = 1; n <= N; n++) { const cg = ngr([...cs], n), rg = ngr([...rs], n); let m = 0, ct = 0, rt = 0; for (const [g, v] of cg) { m += Math.min(v, rg.get(g) || 0); ct += v; } for (const v of rg.values()) rt += v; if (ct && rt) { P += m / ct; R += m / rt; cnt++; } }
  if (!cnt) return 0; P /= cnt; R /= cnt; return P + R === 0 ? 0 : 100 * (1 + beta * beta) * P * R / (beta * beta * P + R);
}
