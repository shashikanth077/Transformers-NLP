/* ===== sequence-model engine: RNN, LSTM, GRU forward + backpropagation through time (BPTT), Adam ===== */
const sig = x => 1 / (1 + Math.exp(-x));
function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const gauss = r => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
const SHAPES = {
  rnn: (D, H) => ({ Wx: H * D, Wh: H * H, b: H, R: H, r: 1 }),
  lstm: (D, H) => ({ Wf: H * (D + H), bf: H, Wi: H * (D + H), bi: H, Wg: H * (D + H), bg: H, Wo: H * (D + H), bo: H, R: H, r: 1 }),
  gru: (D, H) => ({ Wz: H * (D + H), bz: H, Wr: H * (D + H), br: H, Wx: H * D, Uh: H * H, bh: H, R: H, r: 1 }),
};
function initP(kind, D, H, seed) {
  const r = rng(seed), P = {}, sh = SHAPES[kind](D, H);
  for (const k in sh) { P[k] = new Float64Array(sh[k]); if (/^W|^U|^R$/.test(k)) { const fan = /^Wx$/.test(k) ? D : /^R$/.test(k) ? H : (/^Wh$|^Uh$/.test(k) ? H : D + H); for (let i = 0; i < P[k].length; i++) P[k][i] = gauss(r) / Math.sqrt(fan) * (k === "R" ? 1 : 1); } }
  if (kind === "lstm") P.bf.fill(1);           // common practice: start with the forget gate mostly open
  return P;
}
const nParams = (kind, D, H) => Object.values(SHAPES[kind](D, H)).reduce((a, b) => a + b, 0);
const zeros = n => new Float64Array(n);
function cat(x, h) { const v = new Float64Array(x.length + h.length); v.set(x); v.set(h, x.length); return v; }
function lin(W, b, v, H, n) { const o = new Float64Array(H); for (let k = 0; k < H; k++) { let s = b ? b[k] : 0; for (let j = 0; j < n; j++) s += W[k * n + j] * v[j]; o[k] = s; } return o; }

function stepFwd(kind, P, x, hp, cp, D, H) {
  const S = { x, hp, cp };
  if (kind === "rnn") { S.a = lin(P.Wx, P.b, x, H, D); for (let k = 0; k < H; k++) { let s = 0; for (let j = 0; j < H; j++) s += P.Wh[k * H + j] * hp[j]; S.a[k] += s; } S.h = S.a.map(Math.tanh); }
  else if (kind === "lstm") {
    const v = cat(x, hp), n = D + H; S.v = v;
    S.f = lin(P.Wf, P.bf, v, H, n).map(sig); S.i = lin(P.Wi, P.bi, v, H, n).map(sig); S.o = lin(P.Wo, P.bo, v, H, n).map(sig); S.g = lin(P.Wg, P.bg, v, H, n).map(Math.tanh);
    S.c = S.f.map((f, k) => f * cp[k] + S.i[k] * S.g[k]); S.tc = S.c.map(Math.tanh); S.h = S.o.map((o, k) => o * S.tc[k]);
  } else {
    const v = cat(x, hp), n = D + H; S.v = v;
    S.z = lin(P.Wz, P.bz, v, H, n).map(sig); S.r = lin(P.Wr, P.br, v, H, n).map(sig); S.rh = S.r.map((r, k) => r * hp[k]);
    const ax = lin(P.Wx, P.bh, x, H, D); for (let k = 0; k < H; k++) { let s = 0; for (let j = 0; j < H; j++) s += P.Uh[k * H + j] * S.rh[j]; ax[k] += s; }
    S.hc = ax.map(Math.tanh); S.h = S.z.map((z, k) => (1 - z) * hp[k] + z * S.hc[k]);
  }
  return S;
}
function run(kind, P, xs, D, H) {
  let h = zeros(H), c = zeros(H); const steps = [];
  for (const x of xs) { const S = stepFwd(kind, P, x, h, c, D, H); steps.push(S); h = S.h; if (kind === "lstm") c = S.c; }
  let z = P.r[0]; for (let k = 0; k < H; k++) z += P.R[k] * h[k];
  return { steps, logit: z, p: sig(z) };
}
const bce = (p, y) => -(y * Math.log(p + 1e-12) + (1 - y) * Math.log(1 - p + 1e-12));
function outer(Gw, da, v, n, H) { for (let k = 0; k < H; k++) for (let j = 0; j < n; j++) Gw[k * n + j] += da[k] * v[j]; }
function tmul(W, da, n, H, dv) { for (let k = 0; k < H; k++) for (let j = 0; j < n; j++) dv[j] += W[k * n + j] * da[k]; }

function bptt(kind, P, xs, y, D, H) {
  const F = run(kind, P, xs, D, H), T = xs.length, G = {}; for (const k in P) G[k] = new Float64Array(P[k].length);
  const dl = F.p - y; G.r[0] = dl; let dh = new Float64Array(H), dc = new Float64Array(H);
  const hT = F.steps[T - 1].h; for (let k = 0; k < H; k++) { G.R[k] = dl * hT[k]; dh[k] = P.R[k] * dl; }
  const gnorm = new Array(T), n = D + H;
  for (let t = T - 1; t >= 0; t--) {
    const S = F.steps[t]; gnorm[t] = Math.hypot(...dh);
    if (kind === "rnn") {
      const da = Float64Array.from(dh, (d, k) => d * (1 - S.h[k] * S.h[k])); outer(G.Wx, da, S.x, D, H); outer(G.Wh, da, S.hp, H, H); for (let k = 0; k < H; k++) G.b[k] += da[k];
      const nh = new Float64Array(H); tmul(P.Wh, da, H, H, nh); dh = nh;
    } else if (kind === "lstm") {
      const dcT = Float64Array.from(dc, (d, k) => d + dh[k] * S.o[k] * (1 - S.tc[k] * S.tc[k]));
      const dao = Float64Array.from(dh, (d, k) => d * S.tc[k] * S.o[k] * (1 - S.o[k])), dai = Float64Array.from(dcT, (d, k) => d * S.g[k] * S.i[k] * (1 - S.i[k]));
      const daf = Float64Array.from(dcT, (d, k) => d * S.cp[k] * S.f[k] * (1 - S.f[k])), dag = Float64Array.from(dcT, (d, k) => d * S.i[k] * (1 - S.g[k] * S.g[k]));
      outer(G.Wo, dao, S.v, n, H); outer(G.Wi, dai, S.v, n, H); outer(G.Wf, daf, S.v, n, H); outer(G.Wg, dag, S.v, n, H);
      for (let k = 0; k < H; k++) { G.bo[k] += dao[k]; G.bi[k] += dai[k]; G.bf[k] += daf[k]; G.bg[k] += dag[k]; }
      const dv = new Float64Array(n); tmul(P.Wo, dao, n, H, dv); tmul(P.Wi, dai, n, H, dv); tmul(P.Wf, daf, n, H, dv); tmul(P.Wg, dag, n, H, dv);
      dh = dv.slice(D); dc = Float64Array.from(dcT, (d, k) => d * S.f[k]);
    } else {
      const dz = Float64Array.from(dh, (d, k) => d * (S.hc[k] - S.hp[k])), dhc = Float64Array.from(dh, (d, k) => d * S.z[k]), dhp = Float64Array.from(dh, (d, k) => d * (1 - S.z[k]));
      const dah = Float64Array.from(dhc, (d, k) => d * (1 - S.hc[k] * S.hc[k])); outer(G.Wx, dah, S.x, D, H); outer(G.Uh, dah, S.rh, H, H); for (let k = 0; k < H; k++) G.bh[k] += dah[k];
      const drh = new Float64Array(H); tmul(P.Uh, dah, H, H, drh); const dr = Float64Array.from(drh, (d, k) => d * S.hp[k]); for (let k = 0; k < H; k++) dhp[k] += drh[k] * S.r[k];
      const daz = Float64Array.from(dz, (d, k) => d * S.z[k] * (1 - S.z[k])), dar = Float64Array.from(dr, (d, k) => d * S.r[k] * (1 - S.r[k]));
      outer(G.Wz, daz, S.v, n, H); outer(G.Wr, dar, S.v, n, H); for (let k = 0; k < H; k++) { G.bz[k] += daz[k]; G.br[k] += dar[k]; }
      const dv = new Float64Array(n); tmul(P.Wz, daz, n, H, dv); tmul(P.Wr, dar, n, H, dv); for (let k = 0; k < H; k++) dhp[k] += dv[D + k]; dh = dhp;
    }
  }
  return { loss: bce(F.p, y), p: F.p, G, gnorm };
}
function adamInit(P) { const o = { t: 0, m: {}, v: {} }; for (const k in P) { o.m[k] = new Float64Array(P[k].length); o.v[k] = new Float64Array(P[k].length); } return o; }
function adamStep(P, G, A, lr, clip = 1) {
  let nrm = 0; for (const k in G) for (const g of G[k]) nrm += g * g; nrm = Math.sqrt(nrm); const sc = nrm > clip ? clip / nrm : 1; A.t++;
  for (const k in P) for (let i = 0; i < P[k].length; i++) { const g = G[k][i] * sc; A.m[k][i] = .9 * A.m[k][i] + .1 * g; A.v[k][i] = .999 * A.v[k][i] + .001 * g * g; P[k][i] -= lr * (A.m[k][i] / (1 - .9 ** A.t)) / (Math.sqrt(A.v[k][i] / (1 - .999 ** A.t)) + 1e-8); }
  return nrm;
}
/* ---- tasks ---- */
function genA(T, r) { const xs = Array.from({ length: T }, () => Float64Array.of(r() < .5 ? -1 : 1)); return { xs, y: xs[0][0] > 0 ? 1 : 0 }; }
function genB(T, r) {
  const xs = Array.from({ length: T }, () => new Float64Array(2)), pp = Math.floor(r() * T), pol = r() < .5 ? -1 : 1; xs[pp][0] = pol; let neg = false;
  if (pp > 0 && r() < .5) { xs[Math.floor(r() * pp)][1] = 1; neg = true; } return { xs, y: (pol > 0) !== neg ? 1 : 0 };
}
