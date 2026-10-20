// Trains the three small models (hidden size 4) used by the walkthroughs on the negation-sentiment task and writes weights.json
// Run: node study_guide/rnn/train_explorer_weights.js
const fs = require("fs"), path = require("path");
const E = new Function(fs.readFileSync(path.join(__dirname, "engine.js"), "utf8") + "; return {rng,initP,bptt,run,adamInit,adamStep,genB};")();
const { rng, initP, bptt, run, adamInit, adamStep, genB } = E, D = 2, H = 4, out = {};
for (const kind of ["rnn", "lstm", "gru"]) {
  const P = initP(kind, D, H, 1), A = adamInit(P), r = rng(101);
  for (let s = 0; s < 500; s++) {
    const G = {}; for (const k in P) G[k] = new Float64Array(P[k].length);
    for (let b = 0; b < 16; b++) { const q = genB(4 + Math.floor(r() * 5), r), o = bptt(kind, P, q.xs, q.y, D, H); for (const k in G) for (let i = 0; i < G[k].length; i++) G[k][i] += o.G[k][i] / 16; }
    adamStep(P, G, A, 0.03);
  }
  const rt = rng(7); let ok = 0; for (let i = 0; i < 300; i++) { const q = genB(4 + Math.floor(rt() * 5), rt); if ((run(kind, P, q.xs, D, H).p > 0.5) === (q.y === 1)) ok++; }
  out[kind] = { acc: ok / 300, P: Object.fromEntries(Object.entries(P).map(([k, v]) => [k, Array.from(v, x => +x.toFixed(5))])) };
  console.log(kind, "test accuracy", ok / 300);
}
fs.writeFileSync(path.join(__dirname, "weights.json"), JSON.stringify(out));
