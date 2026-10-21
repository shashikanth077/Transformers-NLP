// Trains the toy GRU encoder-decoder (default split, seed 7) with the same procedure the page uses, and writes toy.json.
// Run: node study_guide/nmt/train_toy.js
const fs = require("fs"), path = require("path");
const L = new Function(["corpus.js", "seq2seq.js", "pipeline.js"].map(f => fs.readFileSync(path.join(__dirname, f), "utf8")).join("\n") + "; return {CORPUS,makeSplit,prepare,Trainer,paramArrays,nParams,greedy,tok,bleuStats,evalLoss};")();
const { CORPUS, makeSplit, prepare, Trainer, paramArrays, nParams, greedy, tok, bleuStats } = L;
const SEED = 7, cfg = { d: 16, H: 32, seed: 11, lr: 0.01, batch: 4, clip: 1, epochs: 80 };
const split = makeSplit(CORPUS, SEED, true), prep = prepare(CORPUS, split), T = new Trainer(prep, cfg);
for (let e = 0; e < cfg.epochs; e++) { const h = T.epoch(); if (h.ep % 10 === 0) console.log("epoch", h.ep, "train", h.train.toFixed(3), "val loss", h.val.toFixed(3), "val chrF", h.valChrf.toFixed(1), "val BLEU", h.valBleu.toFixed(1)); }
const best = T.finish();
const bleu = ps => { const c = [], r = []; ps.forEach(p => { const e = prep.enc(p), g = greedy(T.P, e.src, 1, 2, 24); c.push(g.ids.map(i => prep.Vt.words[i])); r.push(tok(p.kn, false)); }); return bleuStats(c, r).bleu; };
console.log("best epoch (by val chrF)", best.ep, "val chrF", best.score.toFixed(1), "params", nParams(T.P), "vocab", prep.Vs.words.length, prep.Vt.words.length, "BLEU train/val/test", bleu(prep.tr).toFixed(1), bleu(prep.va).toFixed(1), bleu(prep.te).toFixed(1));
fs.writeFileSync(path.join(__dirname, "toy.json"), JSON.stringify({ seed: SEED, cfg, bestEpoch: best.ep, hist: T.hist, arrays: paramArrays(T.P).map(([k, a]) => [k, Array.from(a, v => +v.toFixed(5))]) }));
