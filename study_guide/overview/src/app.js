"use strict";
/* Transformers explained: interactive overview. Every section initialises on its own so one failure cannot break the page. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k, d) { try { const v = localStorage.getItem("tov:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("tov:" + k, JSON.stringify(v)); } catch (e) { /* storage unavailable: page still works */ } },
  clear() { try { Object.keys(localStorage).filter(k => k.startsWith("tov:")).forEach(k => localStorage.removeItem(k)); } catch (e) { /* ignore */ } },
};
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const b64bytes = b64 => { const bin = atob(b64); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };

const T = DATA.translations, MT = DATA.mt;
const isSpecial = t => /^<.*>$/.test(t) || /^[a-z]{3}_[A-Z][a-z]{3}$/.test(t);
const disp = t => (t === "</s>" ? "⟨end⟩" : isSpecial(t) ? "⟨" + t.replace(/[<>/]/g, "") + "⟩" : t.replace(/^▁/, "") || "▁");
const tokHTML = (t, cls = "") => `<span class="tok ${isSpecial(t) ? "sp" : ""} ${cls}">${esc(disp(t))}</span>`;
const joinPieces = ps => ps.filter(p => !isSpecial(p)).map(p => (p.startsWith("▁") ? " " : "") + p.replace(/^▁/, "")).join("").trim();
const fmtP = p => (p >= 0.1 ? (p * 100).toFixed(0) : p >= 0.01 ? (p * 100).toFixed(1) : p >= 0.001 ? (p * 100).toFixed(2) : (p * 100).toFixed(3)) + "%";
const bar = (label, frac, val, cls = "pos", lblCls = "") =>
  `<div class="bar wide"><span class="lbl ${lblCls}" title="${esc(label)}">${label}</span><span class="track"><span class="fill ${cls}" style="left:0; width:${Math.max(0, Math.min(1, frac)) * 100}%"></span></span><b>${val}</b></div>`;
const kind = (cls, txt) => `<span class="pill ${cls}">${txt}</span>`;

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
}

/* ---------- shared example sentence (sections 1, 3, 4, 5 stay in sync) ---------- */
let SI = store.get("sent", 0); if (!(SI >= 0 && SI < T.length)) SI = 0;
const sentRenderers = [];
function setSent(i) { SI = i; store.set("sent", i); renderPickers(); sentRenderers.forEach(f => f()); }
function renderPickers() {
  $$("[id^=sentPick]").forEach(box => {
    box.innerHTML = T.map((t, i) => `<button class="btn small ${i === SI ? "on" : ""}" type="button" data-i="${i}" aria-pressed="${i === SI}">${esc(t.src)}</button>`).join("");
    $$("button", box).forEach(b => b.addEventListener("click", () => setSent(+b.dataset.i)));
  });
}

/* ---------- shell: theme, navigation, progress ---------- */
const SECTIONS = [["intro", "Start here"], ["what", "What is a Transformer?"], ["attention", "What does attention do?"], ["training", "Training"], ["inference", "Inference"],
  ["forcing", "Teacher forcing"], ["uses", "What it is used for"], ["rnn", "Why better than RNNs"], ["check", "Quiz and glossary"]];
function setupShell() {
  const root = document.documentElement;
  const effective = () => (root.dataset.theme === "dark" || (root.dataset.theme === "auto" && matchMedia("(prefers-color-scheme: dark)").matches)) ? "dark" : "light";
  root.dataset.theme = store.get("theme", "auto");
  $("#themeBtn").addEventListener("click", () => { const t = effective() === "dark" ? "light" : "dark"; root.dataset.theme = t; store.set("theme", t); });
  $("#resetBtn").addEventListener("click", () => { if (confirm("Reset your progress and quiz score in this browser?")) { store.clear(); location.reload(); } });
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

/* ======================================================= intro */
function initIntro() {
  const J = [["1", "What is it?", "Two stacks of layers that turn one sequence into another.", "what"], ["2", "Attention", "How a word looks at the other words that matter.", "attention"],
    ["3", "Training", "Learning from source and target pairs.", "training"], ["4", "Inference", "Writing the answer one token at a time.", "inference"],
    ["5", "Teacher forcing", "Why training feeds the correct answer.", "forcing"], ["6", "Uses", "Language tasks and the heads behind them.", "uses"],
    ["7", "Beat RNNs", "Parallel and short paths between words.", "rnn"]];
  $("#journey").innerHTML = J.map(j => `<a class="card flat" href="#${j[3]}" style="text-decoration:none;color:inherit"><b>${j[0]}. ${j[1]}</b><br><span class="small muted">${j[2]}</span></a>`).join("");
}

/* ======================================================= 1 what */
const ARCH = {
  emb: ["Embedding + positional encoding", "A token ID alone is just a number. The <b>embedding</b> turns each token into a vector of numbers (here " + MT.d_model + " of them) that can carry meaning. Because the network looks at all words at once, it would not know their order, so a <b>positional encoding</b> is added to tell word 1 from word 2. The encoder and the decoder each have their own embedding."],
  enc: ["Encoder layer", "Reads the source and builds an understanding of it. Every encoder layer is the same design with its <b>own weights</b>.", [["a", "Self-attention: every source word looks at every other source word"], ["n", "Add & normalise (residual connection)"], ["f", "Feed-forward network: processes each position separately"], ["n", "Add & normalise (residual connection)"]]],
  dec: ["Decoder layer", "Writes the target one token at a time. It sees the target written so far <em>and</em> the encoder’s result.", [["a", "Masked self-attention: each position may only look at earlier positions"], ["n", "Add & normalise"], ["x", "Encoder–decoder attention: look back at the source sentence"], ["n", "Add & normalise"], ["f", "Feed-forward network"], ["n", "Add & normalise"]]],
  out: ["Output layer: linear + softmax", "A linear layer turns the decoder’s vector into one score for <b>every token in the vocabulary</b> (" + MT.vocab.toLocaleString("en-US") + " here), and softmax turns the scores into probabilities that sum to 1. The highest-probability token is the prediction."],
};
let archSel = "enc1";
function renderArch() {
  const N = +$("#nLayers").value; $("#nVal").textContent = N;
  const mk = (type, n) => Array.from({ length: n }, (_, i) => n - i).map(i => `<div class="layer ${type}" data-sel="${type}${i}" tabindex="0" role="button" aria-label="${type === "enc" ? "Encoder" : "Decoder"} layer ${i}">${type === "enc" ? "Encoder" : "Decoder"} ${i}</div>`).join("");
  $("#encCol").innerHTML = `<h4>Encoder stack</h4>${mk("enc", N)}<div class="emb" data-sel="embE" tabindex="0" role="button">Embedding + position</div><div class="small muted" style="text-align:center">source tokens in</div>`;
  $("#decCol").innerHTML = `<h4>Decoder stack</h4><div class="layer outl" data-sel="out" tabindex="0" role="button">Linear + softmax</div>${mk("dec", N)}<div class="emb" data-sel="embD" tabindex="0" role="button">Embedding + position</div><div class="small muted" style="text-align:center">target so far in (shifted right)</div>`;
  $$("#encCol [data-sel], #decCol [data-sel]").forEach(n => {
    const go = () => { archSel = n.dataset.sel; paintArch(); };
    n.addEventListener("click", go); n.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });
  if (/^(enc|dec)\d+$/.test(archSel) && +archSel.replace(/\D/g, "") > N) archSel = archSel.slice(0, 3) + "1";
  paintArch();
}
function paintArch() {
  $$("#encCol [data-sel], #decCol [data-sel]").forEach(n => n.classList.toggle("sel", n.dataset.sel === archSel));
  const m = /^(enc|dec)(\d+)$/.exec(archSel); const key = m ? m[1] : archSel.startsWith("emb") ? "emb" : "out";
  const info = ARCH[key]; let h = `<h4 style="margin-top:0">${info[0]}${m ? " " + m[2] : archSel === "embE" ? " (source side)" : archSel === "embD" ? " (target side)" : ""}</h4><p>${info[1]}</p>`;
  if (info[2]) h += `<div class="inside">${info[2].map(r => `<div class="${r[0]}">${r[1]}</div>`).join("")}</div><p class="small muted" style="margin-top:8px">Read the stack bottom to top. ${m[1] === "enc" ? "The top encoder’s output is handed to <b>every</b> decoder layer." : "The “encoder–decoder attention” block is where the decoder consults the encoder’s output."}</p>`;
  $("#archInfo").innerHTML = h;
}
const FLAV = [
  ["Encoder + decoder", "enc-dec", "Reads one sequence, writes another.", "Translation (our NLLB demo), summarisation, speech recognition. Original Transformer, T5, NLLB, Whisper.",
    ["Source text", "Encoder stack", "Decoder stack", "Output text"], "Google Translate-style English→Kannada."],
  ["Encoder only", "enc", "Reads a sequence and understands it. No generation.", "Classification, tagging, search ranking. BERT, DistilBERT (the fine-tuned model of our sentiment project).",
    ["Review text", "Encoder stack", "Classification head", "Positive / negative"], "Is this review positive or negative? Is this SMS a UPI scam?"],
  ["Decoder only", "dec", "Continues a sequence, one token at a time.", "Text generation and chat assistants. The GPT family (we use DistilGPT-2 below) and most modern chatbots.",
    ["Prompt so far", "Decoder stack", "Language-model head", "Next token"], "Type-ahead suggestion: “Thank you very much for your …”."],
];
function initWhat() {
  sentRenderers.push(renderIO);
  $("#nLayers").addEventListener("input", renderArch); renderArch();
  const tabs = $("#flavTabs");
  tabs.innerHTML = FLAV.map((f, i) => `<button class="btn ${i === 0 ? "on" : ""}" type="button" role="tab" data-i="${i}" aria-selected="${i === 0}">${f[0]}</button>`).join("");
  const show = i => {
    $$("button", tabs).forEach((b, j) => { b.classList.toggle("on", i === j); b.setAttribute("aria-selected", i === j); });
    const f = FLAV[i];
    $("#flavInfo").innerHTML = `<h4 style="margin-top:0">${f[0]}: ${f[2]}</h4><div class="flow">${f[4].map((s, j) => `${j ? '<span class="farrow">&rarr;</span>' : ""}<div class="fbox ${j === 0 ? "src" : j === f[4].length - 1 ? "tgt" : "mid"}">${s}</div>`).join("")}</div><p><b>Used for:</b> ${f[3]}</p><p class="small muted">Real-world feel: ${f[5]}</p>`;
  };
  $$("button", tabs).forEach(b => b.addEventListener("click", () => show(+b.dataset.i))); show(0);
  renderIO();
}
function renderIO() {
  const t = T[SI];
  $("#ioSrc").textContent = t.src; $("#ioTgt").textContent = t.greedy;
  $("#ioSrcTok").innerHTML = t.src_tokens.map(p => tokHTML(p)).join("");
  const outToks = t.steps.map(s => s.chosen);
  $("#ioTgtTok").innerHTML = outToks.map(p => tokHTML(p)).join("");
  $("#ioFacts").innerHTML = `${kind("t", "REAL")} Produced by <b>${esc(MT.model.split("/")[1])}</b> (${(MT.params / 1e6).toFixed(0)} M parameters; ${MT.enc_layers} encoder + ${MT.dec_layers} decoder layers; ${MT.heads} attention heads). The input has <b>${t.src_tokens.length}</b> tokens (including language and end markers) and the output <b>${outToks.length}</b> (including the end marker).`;
}

/* ======================================================= 2 attention */
function initAttention() {
  const words = ["The", "trophy", "didn’t", "fit", "in", "the", "suitcase", "because", "it", "was", "too", "big"];
  const draw = big => {
    words[11] = big ? "big" : "small"; const w = words.map(s => s.length * 9 + 22); const xs = []; let x = 10; w.forEach(v => { xs.push(x + v / 2); x += v; });
    const W = x + 10; const iT = 1, iS = 6, iIt = 8; const tgt = big ? iT : iS, oth = big ? iS : iT;
    const arc = (a, b, h, sw, op, clr) => `<path d="M${xs[a]},78 Q${(xs[a] + xs[b]) / 2},${78 - h} ${xs[b]},78" fill="none" stroke="${clr}" stroke-width="${sw}" opacity="${op}" stroke-linecap="round"/>`;
    let s = `<svg viewBox="0 0 ${W + (big ? 0 : 20)} 120" role="img" aria-label="Attention arrows from the word it" style="width:100%;height:auto;max-width:860px"><g class="chart">`;
    s += arc(iIt, oth, 60, 2, .35, "var(--muted)") + arc(iIt, tgt, 75, 6, .9, "var(--violet)");
    s += words.map((t, i) => `<text x="${xs[i]}" y="95" text-anchor="middle" font-size="15" fill="${i === iIt ? "var(--violet)" : "var(--ink)"}" font-weight="${i === iIt || i === tgt ? 800 : 500}">${esc(t)}</text>`).join("");
    s += `</g></svg>`; return s;
  };
  const set = big => {
    $("#itBig").setAttribute("aria-pressed", big); $("#itSmall").setAttribute("aria-pressed", !big);
    $("#itBig").classList.toggle("on", big); $("#itSmall").classList.toggle("on", !big);
    $("#itSvg").innerHTML = draw(big);
    $("#itNote").innerHTML = big ? "Too <b>big</b> to fit: it is the <b>trophy</b> that is too big. “it” must attend most to “trophy”." : "Too <b>small</b> to hold it: it is the <b>suitcase</b> that is too small. “it” must attend most to “suitcase”.";
  };
  $("#itBig").addEventListener("click", () => set(true)); $("#itSmall").addEventListener("click", () => set(false)); set(true);

  /* real DistilBERT attention */
  const A = DATA.attention; let si = 0, layer = 5, head = -1, sel = A[0].it, mode = "words";
  A.forEach(a => { a._u8 = b64bytes(a.att); });
  const [L, H] = A[0].shape;
  const at = (a, l, h, i, j) => {
    const n = a.shape[2]; if (h === -1) { let s = 0; for (let k = 0; k < H; k++) s += a._u8[((l * H + k) * n + i) * n + j]; return s / H / 255; }
    return a._u8[((l * H + h) * n + i) * n + j] / 255;
  };
  $("#attSentPick").innerHTML = A.map((a, i) => `<button class="btn small ${i === 0 ? "on" : ""}" type="button" data-i="${i}">${i === 0 ? "…it was too big" : "…it was too small"}</button>`).join("");
  $("#attLayer").innerHTML = Array.from({ length: L }, (_, i) => `<option value="${i}" ${i === layer ? "selected" : ""}>${i + 1}</option>`).join("");
  $("#attHead").innerHTML = `<option value="-1">average of all ${H}</option>` + Array.from({ length: H }, (_, i) => `<option value="${i}">${i + 1}</option>`).join("");
  const render = () => {
    const a = A[si], n = a.tokens.length;
    $$("#attSentPick button").forEach((b, i) => b.classList.toggle("on", i === si));
    $("#attWords").classList.toggle("on", mode === "words"); $("#attHeat").classList.toggle("on", mode === "heat");
    if (mode === "words") {
      const row = a.tokens.map((_, j) => at(a, layer, head, sel, j)); const mx = Math.max(...row, 1e-6);
      const order = row.map((v, j) => [v, j]).sort((p, q) => q[0] - p[0]).slice(0, 5);
      $("#attView").innerHTML = `<p class="small muted" style="margin:4px 0">Click a word. Showing where <b>“${esc(a.tokens[sel])}”</b> looks (darker = more attention).</p><div class="tokens">` +
        a.tokens.map((t, j) => `<button class="tok att ${j === sel ? "sel" : ""}" type="button" data-j="${j}" style="background:color-mix(in srgb, var(--violet) ${Math.round(row[j] / mx * 70)}%, var(--surface))">${esc(t)}</button>`).join("") +
        `</div><div class="bars" style="margin-top:6px">${order.map(([v, j]) => bar(esc(a.tokens[j]), v, pct(v, 1), "pos")).join("")}</div>`;
      $$("#attView .tok").forEach(b => {
        b.addEventListener("click", () => { sel = +b.dataset.j; render(); });
        attachTip(b, () => `<b>${esc(a.tokens[sel])}</b> → <b>${esc(a.tokens[+b.dataset.j])}</b>: ${pct(row[+b.dataset.j], 1)}`);
      });
    } else {
      let h = `<p class="small muted" style="margin:4px 0">Rows = the word that is looking, columns = the word looked at. Click a row label to select it.</p><div class="scroll"><table class="heat"><tr><th></th>${a.tokens.map(t => `<th class="v">${esc(t)}</th>`).join("")}</tr>`;
      for (let i = 0; i < n; i++) {
        h += `<tr><th class="r ${i === sel ? "selr" : ""}" data-i="${i}">${esc(a.tokens[i])}</th>`;
        for (let j = 0; j < n; j++) { const v = at(a, layer, head, i, j); h += `<td data-i="${i}" data-j="${j}" style="background:color-mix(in srgb, var(--violet) ${Math.round(Math.min(1, v * 1.4) * 90)}%, var(--surface))"></td>`; }
        h += "</tr>";
      }
      $("#attView").innerHTML = h + "</table></div>";
      $$("#attView th.r").forEach(th => th.addEventListener("click", () => { sel = +th.dataset.i; mode = "words"; render(); }));
      $$("#attView td").forEach(td => attachTip(td, () => `<b>${esc(a.tokens[+td.dataset.i])}</b> → <b>${esc(a.tokens[+td.dataset.j])}</b>: ${pct(at(a, layer, head, +td.dataset.i, +td.dataset.j), 1)}`));
    }
  };
  $$("#attSentPick button").forEach(b => b.addEventListener("click", () => { si = +b.dataset.i; sel = A[si].it; render(); }));
  $("#attLayer").addEventListener("change", e => { layer = +e.target.value; render(); });
  $("#attHead").addEventListener("change", e => { head = +e.target.value; render(); });
  $("#attWords").addEventListener("click", () => { mode = "words"; render(); }); $("#attHeat").addEventListener("click", () => { mode = "heat"; render(); });
  const th = A[0].top_heads[0];
  $("#attHint").innerHTML = `Tip: with <b>average of all 12</b> heads at layer 6, attention mostly rests on punctuation and the special [SEP] marker, which is normal for this model. Try <b>layer ${th[2] + 1}, head ${th[3] + 1}</b> instead: there “it” gives ${pct(th[0], 0)} to “trophy” and ${pct(th[1], 0)} to “suitcase”.`;
  render();

  const rows = A.map((a, i) => a.top_heads.slice(0, 4).map(h => `<tr><td>${i === 0 ? "big" : "small"}</td><td class="num">${h[2] + 1}</td><td class="num">${h[3] + 1}</td><td class="num">${pct(h[0], 1)}</td><td class="num">${pct(h[1], 1)}</td></tr>`).join("")).join("");
  $("#headTable").innerHTML = `<thead><tr><th>Sentence ends with</th><th class="num">Layer</th><th class="num">Head</th><th class="num">“it” → trophy</th><th class="num">“it” → suitcase</th></tr></thead><tbody>${rows}</tbody>`;
}

/* ======================================================= 3 training */
const TR_STEPS = [
  ["1. Source in", "Source sequence → embedding + position → encoder stack"],
  ["2. Target in (shifted)", "Start marker + target, without the last token"],
  ["3. Output layer", "Probability of every vocabulary token at each position"],
  ["4. Loss", "Compare with the real next token"],
  ["5. Backpropagate", "Gradients flow to every weight"],
  ["6. Update", "Optimiser nudges the weights"],
];
let trStep = 0;
function renderTraining() {
  const t = T[SI], rows = t.tf.slice(1);               // row 0 only predicts the language marker
  $("#trainFlow").innerHTML = TR_STEPS.map((s, i) => `${i ? '<span class="farrow">&rarr;</span>' : ""}<button class="fbox ${i === trStep ? "on" : ""} ${i < 2 ? "src" : i < 4 ? "mid" : "tgt"}" type="button" data-i="${i}">${s[0]}<small>${s[1]}</small></button>`).join("");
  $$("#trainFlow .fbox").forEach(b => b.addEventListener("click", () => { trStep = +b.dataset.i; renderTraining(); }));
  $("#trLabel").textContent = `Step ${trStep + 1} of ${TR_STEPS.length}`;
  $("#trPrev").disabled = trStep === 0; $("#trNext").disabled = trStep === TR_STEPS.length - 1;
  const maxLoss = Math.max(...rows.map(r => r.loss), 1);
  let h = "";
  if (trStep === 0) {
    h = `<p>The English sentence is split into tokens, each becomes a vector (plus a position code), and the <b>encoder stack</b> turns them into a rich description of the source. This runs once per training example.</p><div class="tokens">${t.src_tokens.map(p => tokHTML(p)).join("")}</div><p class="small muted">Each token above becomes a vector of ${MT.d_model} numbers; the encoder returns ${t.src_tokens.length} improved vectors, one per source token.</p>`;
  } else if (trStep === 1) {
    h = `<p>The human reference translation is given to the decoder, <b>shifted right</b>: a start marker goes in front and the last token is dropped. At every position the decoder must predict the <em>next</em> token of the real answer. Reference: <b class="kn">${esc(t.ref)}</b></p>
      <div class="scroll"><table class="compact"><thead><tr><th>Position</th>${rows.map((_, i) => `<th class="num">${i + 1}</th>`).join("")}</tr></thead><tbody>
      <tr><th>Decoder input</th>${rows.map(r => `<td>${tokHTML(r.input)}</td>`).join("")}</tr>
      <tr><th>Expected next token</th>${rows.map(r => `<td>${tokHTML(r.gold, "good")}</td>`).join("")}</tr></tbody></table></div>`;
  } else if (trStep === 2) {
    h = `<p>At every position the output layer gives a probability for each of the ${MT.vocab.toLocaleString("en-US")} vocabulary tokens. The table shows the model’s top guess and the probability it gave to the <b>correct</b> token. ${kind("t", "REAL")}</p>
      <div class="scroll"><table class="compact"><thead><tr><th>Input</th><th>Top guess</th><th>Correct token</th><th class="num">P(correct)</th></tr></thead><tbody>${rows.map(r => `<tr><td>${tokHTML(r.input)}</td><td>${tokHTML(r.pred, r.pred === r.gold ? "good" : "bad")}</td><td>${tokHTML(r.gold)}</td><td class="num">${fmtP(r.p_gold)}</td></tr>`).join("")}</tbody></table></div>`;
  } else if (trStep === 3) {
    const mean = rows.reduce((s, r) => s + r.loss, 0) / rows.length;
    h = `<p>The <b>loss</b> at one position is −ln P(correct token): small when the model gave the right token a high probability, large when it did not. The loss for the whole example is the average. ${kind("t", "REAL")}</p>
      <div class="bars">${rows.map(r => bar(`${tokHTML(r.gold)}`, r.loss / maxLoss, r.loss.toFixed(2), "neg")).join("")}</div>
      <div class="grid g3" style="margin-top:10px"><div class="stat r"><b>${mean.toFixed(2)}</b><span>mean loss for this example</span></div><div class="stat t"><b>${rows.filter(r => r.pred === r.gold).length}/${rows.length}</b><span>positions where the top guess was right</span></div><div class="stat v"><b>${Math.exp(mean).toFixed(1)}</b><span>perplexity = e<sup>loss</sup></span></div></div>`;
  } else if (trStep === 4) {
    h = `<p><b>Backpropagation</b> works out, for every weight in the whole network (both stacks, all embeddings, the output layer), how much a small change would have changed the loss. That per-weight number is the <b>gradient</b>. Because the encoder’s output was used by the decoder, the gradient reaches the encoder too: the two stacks are trained together as one network.</p>
      <div class="inside" style="max-width:420px"><div class="x">Output layer ← gradient</div><div class="a">Decoder stack ← gradient</div><div class="f">Encoder stack ← gradient (through encoder–decoder attention)</div><div class="n">Embeddings ← gradient</div></div>`;
  } else {
    const pts = Array.from({ length: 40 }, (_, i) => 3.2 * Math.exp(-i / 11) + 0.45 + 0.05 * Math.sin(i * 1.7));
    const W = 420, Hh = 120, path = pts.map((v, i) => `${i ? "L" : "M"}${10 + i * (W - 20) / 39},${Hh - 10 - v / 3.8 * (Hh - 20)}`).join(" ");
    h = `<p>An optimiser (Adam is the usual choice) moves each weight a small step against its gradient. One batch of examples makes one such update; training repeats this over millions of batches, and the loss gradually falls. ${kind("", "ILLUSTRATIVE")}</p>
      <svg viewBox="0 0 ${W} ${Hh}" style="width:100%;max-width:${W}px" role="img" aria-label="Illustrative falling loss curve"><g class="chart"><path d="${path}" fill="none" stroke="var(--recurrent)" stroke-width="3"/><text x="${W - 12}" y="${Hh - 2}" text-anchor="end" font-size="11" class="muted-t">training steps →</text><text x="12" y="14" font-size="11" class="muted-t">loss</text></g></svg>
      <p class="small muted">The curve above is a sketch of the typical shape, not a measurement.</p>`;
  }
  $("#trBody").innerHTML = h;
}
function initTraining() {
  sentRenderers.push(renderTraining);
  $("#trPrev").addEventListener("click", () => { trStep = Math.max(0, trStep - 1); renderTraining(); });
  $("#trNext").addEventListener("click", () => { trStep = Math.min(TR_STEPS.length - 1, trStep + 1); renderTraining(); });
  renderTraining();
}

/* ======================================================= 4 inference */
let infK = 0, infTimer = null;
function infStop() { clearInterval(infTimer); infTimer = null; $("#infAuto").innerHTML = "&#9654; Auto-play"; }
function renderInf() {
  const t = T[SI], K = infK, layer = +$("#infLayer").value;
  $("#infEnc").textContent = 1; $("#infDec").textContent = K;
  const chosen = t.steps.slice(0, K).map(s => s.chosen);
  $("#infLen").textContent = chosen.filter(c => c !== "</s>").length;
  const prefix = K === 0 ? t.steps[0].prefix : t.steps[K - 1].prefix;
  $("#infPrefix").innerHTML = prefix.map((p, i) => tokHTML(p, i === prefix.length - 1 && K > 0 ? "chosen" : "")).join("");
  const done = K > 0 && chosen[K - 1] === "</s>";
  $("#infNext").disabled = K >= t.steps.length;
  if (K === 0) {
    $("#infBars").innerHTML = `<span class="muted small">Press “Predict next token”. The decoder starts with only the language marker (and the start token) and the encoder’s result.</span>`;
    $("#infAtt").innerHTML = ""; $("#infAttNote").textContent = "";
  } else {
    const s = t.steps[K - 1], mx = s.top[0][1];
    $("#infBars").innerHTML = s.top.map(([p, v], i) => bar(tokHTML(p, p === s.chosen && i === 0 ? "chosen" : ""), v / mx, fmtP(v), i === 0 ? "pos" : "c")).join("") +
      `<p class="small muted" style="margin:6px 0 0">Entropy ${s.entropy.toFixed(2)} nats: ${s.entropy < 0.3 ? "very sure" : s.entropy < 1.2 ? "fairly sure" : "several plausible choices"}.</p>`;
    const raw = s.cross[Math.min(layer, s.cross.length - 1)];
    const keep = t.src_tokens.map(p => !isSpecial(p)), rest = raw.reduce((a, v, j) => a + (keep[j] ? v : 0), 0) || 1, parked = 1 - rest;
    const att = raw.map((v, j) => (keep[j] ? v / rest : 0)), am = Math.max(...att, 1e-6);
    $("#infAtt").innerHTML = t.src_tokens.map((p, j) => (keep[j] ? bar(tokHTML(p), att[j] / am, pct(att[j], 0), "acc") : "")).join("");
    $("#infAttNote").innerHTML = `Averaged over the layer’s heads, then re-scaled over the real words. <b>${pct(parked, 0)}</b> of the raw attention rested on the end/language markers, a common “resting place” in trained models. Cross-attention is often a fuzzy, not a clean word-for-word, alignment: change the layer to see it shift.`;
  }
  $("#infOut").textContent = joinPieces(chosen) || "…";
  if (done) { $("#infOut").innerHTML = `${esc(joinPieces(chosen))} <span class="pill t">end token predicted: stop</span>`; infStop(); }
}
function resetInf() { infStop(); infK = 0; renderInf(); }
function initInference() {
  const L = MT.dec_layers;
  $("#infLayer").innerHTML = Array.from({ length: L }, (_, i) => `<option value="${i}" ${i === L - 1 ? "selected" : ""}>${i + 1}</option>`).join("");
  $("#infLayer").addEventListener("change", renderInf);
  $("#infNext").addEventListener("click", () => { infStop(); infK++; renderInf(); });
  $("#infReset").addEventListener("click", resetInf);
  $("#infAuto").addEventListener("click", () => {
    if (infTimer) { infStop(); return; }
    if (infK >= T[SI].steps.length) infK = 0;
    $("#infAuto").innerHTML = "&#10074;&#10074; Pause";
    infTimer = setInterval(() => { if (infK >= T[SI].steps.length) { infStop(); return; } infK++; renderInf(); }, 900);
  });
  sentRenderers.push(() => { resetInf(); renderBranches(); });
  renderInf(); renderBranches();
}
function renderBranches() {
  const t = T[SI];
  $("#branchTable").innerHTML = `<thead><tr><th>First-token choice</th><th>Token</th><th>Translation that follows</th><th class="num">Total log-prob</th></tr></thead><tbody>` +
    t.branches.map((b, i) => `<tr class="${i === 0 ? "hl" : ""}"><td>${["1st (greedy)", "2nd", "3rd"][i]}</td><td>${tokHTML(b.first)}</td><td class="kn">${esc(b.text)}</td><td class="num">${b.logprob.toFixed(2)}</td></tr>`).join("") + `</tbody>`;
}

/* ======================================================= 5 teacher forcing */
let tfBranch = 0;
function renderForcing() {
  const t = T[SI], gold = t.tf.slice(1);
  $("#tfBranchBtns").innerHTML = t.branches.map((b, i) => `<button class="btn small ${i === tfBranch ? "on" : ""}" type="button" data-i="${i}">${["1st", "2nd", "3rd"][i]} choice: ${esc(disp(b.first))}</button>`).join("");
  $$("#tfBranchBtns button").forEach(b => b.addEventListener("click", () => { tfBranch = +b.dataset.i; renderForcing(); }));
  const br = t.branches[tfBranch];
  $("#tfLeft").innerHTML = `<div class="scroll"><table class="compact"><thead><tr><th>Input fed in</th><th>Should predict</th><th>Model’s guess</th></tr></thead><tbody>${gold.map(r => `<tr><td>${tokHTML(r.input)}</td><td>${tokHTML(r.gold)}</td><td>${tokHTML(r.pred, r.pred === r.gold ? "good" : "bad")}</td></tr>`).join("")}</tbody></table></div>`;
  let diverged = false;
  const rr = br.tokens.map((tk, j) => {
    const inp = j === 0 ? "kan_Knda" : br.tokens[j - 1], g = gold[j] ? gold[j].gold : null;
    const ok = g === tk; if (!ok) diverged = true;
    return `<tr><td>${tokHTML(inp)}</td><td>${tokHTML(tk, ok ? "good" : "bad")}</td><td>${g ? tokHTML(g) : "<span class='muted'>–</span>"}</td></tr>`;
  }).join("");
  $("#tfRight").innerHTML = `<div class="scroll"><table class="compact"><thead><tr><th>Input fed in (own output)</th><th>Model writes</th><th>Reference has</th></tr></thead><tbody>${rr}</tbody></table></div>`;
  const matches = br.tokens.filter((tk, j) => gold[j] && gold[j].gold === tk).length;
  $("#tfStats").innerHTML = `<div class="stat t"><b>1</b><span>decoder pass in training (all positions at once)</span></div>
    <div class="stat r"><b>${br.tokens.length}</b><span>decoder passes when writing this branch, one per token</span></div>
    <div class="stat v"><b>${matches}/${gold.length}</b><span>tokens equal to the reference at the same position (free-running branch)</span></div>`;
}
function initForcing() { sentRenderers.push(renderForcing); renderForcing(); }

/* ======================================================= 6 uses */
const TASKS = [
  ["Language modelling", "Predict the next word", "Decoder-only (or the decoder stack)", "Language-model head", "Type-ahead on a phone keyboard: after “Thank you very much for your” it suggests “help”. Also the base ability behind chat assistants."],
  ["Text classification", "Label a whole text", "Encoder only", "Classification head", "Is a movie review positive or negative (our project)? Is an SMS about a bank “KYC update” a scam? One label for the whole text."],
  ["Translation", "Sequence in, sequence out", "Encoder + decoder", "Language-model head on the decoder", "Kannada ↔ English: a bus board, a government form, a WhatsApp message. The first part of this page is exactly this."],
  ["Summarisation", "Long text in, short text out", "Encoder + decoder (or decoder only)", "Language-model head", "Turn a long newspaper report about a state budget into three sentences."],
  ["Question answering", "Question and passage in, answer out", "Encoder (find the answer span) or decoder (write it)", "Span head or language-model head", "A helpline bot answering “What are the timings of the Mysuru Palace?” from the official page."],
  ["Named-entity recognition", "Label every word", "Encoder only", "Per-token classification head", "In “Infosys is headquartered in Bengaluru”, tag Infosys = organisation and Bengaluru = place."],
  ["Speech recognition", "Audio in, text out", "Encoder + decoder", "Language-model head", "Voice typing in Kannada: the encoder reads the sound features, the decoder writes the text."],
];
let taskSel = 0;
function initUses() {
  $("#taskGrid").innerHTML = TASKS.map((t, i) => `<button class="taskcard ${i === 0 ? "on" : ""}" type="button" data-i="${i}"><b>${t[0]}</b><span>${t[1]}</span></button>`).join("");
  const show = i => {
    taskSel = i; $$("#taskGrid .taskcard").forEach((b, j) => b.classList.toggle("on", i === j));
    const t = TASKS[i];
    $("#taskInfo").innerHTML = `<h4 style="margin-top:0">${t[0]}</h4><div class="grid g3"><div><span class="small muted">Architecture</span><br><b>${t[2]}</b></div><div><span class="small muted">Head on top</span><br><b>${t[3]}</b></div><div><span class="small muted">What it does</span><br><b>${t[1]}</b></div></div><p style="margin-bottom:0"><b>Real-world feel:</b> ${t[4]}</p>`;
  };
  $$("#taskGrid .taskcard").forEach(b => b.addEventListener("click", () => show(+b.dataset.i))); show(0);

  /* classification head: real P(positive) of the project's fine-tuned DistilBERT */
  const C = DATA.cls; let ci = 0;
  $("#clsPick").innerHTML = C.map((c, i) => `<button class="btn small ${i === 0 ? "on" : ""}" type="button" data-i="${i}">${esc(c.text.length > 44 ? c.text.slice(0, 42) + "…" : c.text)}</button>`).join("");
  const showC = i => {
    ci = i; $$("#clsPick button").forEach((b, j) => b.classList.toggle("on", i === j));
    const c = C[i], p = c.p;
    $("#clsOut").innerHTML = `<p style="margin:0 0 8px"><b>“${esc(c.text)}”</b> <span class="muted small">(real label: ${c.y ? "positive" : "negative"})</span></p>
      <div class="flow"><div class="fbox src">Text</div><span class="farrow">&rarr;</span><div class="fbox mid">Encoder stack</div><span class="farrow">&rarr;</span><div class="fbox mid">Summary vector ([CLS])</div><span class="farrow">&rarr;</span><div class="fbox tgt">Classification head</div></div>
      <div class="bars">${bar("Positive", p, pct(p, 1), "pos")}${bar("Negative", 1 - p, pct(1 - p, 1), "neg")}</div>
      <p class="small muted" style="margin:6px 0 0">${kind("t", "REAL")} ${p > .5 === !!c.y ? "The model is right here." : "The model is <b>wrong</b> here, which shows these models are good but not perfect."} Source: the DistilBERT fine-tuned in our project on SST-2.</p>`;
  };
  $$("#clsPick button").forEach(b => b.addEventListener("click", () => showC(+b.dataset.i))); showC(0);

  /* language-model head: real DistilGPT-2 next-token logits, re-softmaxed with temperature and top-k */
  const LM = DATA.lm; let li = 0, built = "";
  const dist = () => {
    const temp = +$("#lmTemp").value, k = +$("#lmK").value, top = LM[li].top.slice(0, k);
    const mx = Math.max(...top.map(t => t[1])); const e = top.map(t => Math.exp((t[1] - mx) / temp)); const s = e.reduce((a, b) => a + b, 0);
    return top.map((t, i) => ({ tok: t[0], p: e[i] / s }));
  };
  const renderLM = () => {
    $("#lmTempV").textContent = (+$("#lmTemp").value).toFixed(1); $("#lmKV").textContent = $("#lmK").value;
    $("#lmPick").innerHTML = LM.map((l, i) => `<button class="btn small ${i === li ? "on" : ""}" type="button" data-i="${i}">${esc(l.prompt)}</button>`).join("");
    $$("#lmPick button").forEach(b => b.addEventListener("click", () => { li = +b.dataset.i; built = ""; renderLM(); $("#lmOut").innerHTML = `<b>Prompt:</b> ${esc(LM[li].prompt)} <span class="muted">…</span>`; }));
    const d = dist(), mx = d[0].p;
    $("#lmBars").innerHTML = d.slice(0, 12).map(x => bar(`<span class="tok">${esc(x.tok.replace(/ /g, "␣"))}</span>`, x.p / mx, fmtP(x.p), "t")).join("") + (d.length > 12 ? `<p class="small muted" style="margin:4px 0 0">…and ${d.length - 12} more candidates. The real vocabulary has ${LM[li].vocab.toLocaleString("en-US")} tokens; the 40 most likely hold ${pct(LM[li].mass40, 0)} of the probability. ␣ marks a leading space.</p>` : "");
  };
  const pick = greedy => { const d = dist(); if (greedy) return d[0]; let r = Math.random(), a = 0; for (const x of d) { a += x.p; if (r <= a) return x; } return d[d.length - 1]; };
  $("#lmTemp").addEventListener("input", renderLM); $("#lmK").addEventListener("input", renderLM);
  $("#lmGreedy").addEventListener("click", () => { const x = pick(true); $("#lmOut").innerHTML = `<b>Top word:</b> ${esc(LM[li].prompt)}<b class="mark">${esc(x.tok)}</b> <span class="muted small">(${fmtP(x.p)})</span>`; });
  $("#lmSample").addEventListener("click", () => { const x = pick(false); $("#lmOut").innerHTML = `<b>Sampled:</b> ${esc(LM[li].prompt)}<b class="mark">${esc(x.tok)}</b> <span class="muted small">(${fmtP(x.p)})</span>`; });
  $("#lmSample50").addEventListener("click", () => {
    const c = new Map(); for (let i = 0; i < 50; i++) { const x = pick(false); c.set(x.tok, (c.get(x.tok) || 0) + 1); }
    const rows = [...c.entries()].sort((a, b) => b[1] - a[1]);
    $("#lmOut").innerHTML = `<b>50 samples:</b> ` + rows.map(([t, n]) => `<span class="tok">${esc(t.replace(/ /g, "␣"))}</span> ×${n}`).join(" &nbsp; ");
  });
  renderLM(); $("#lmOut").innerHTML = `<b>Prompt:</b> ${esc(LM[li].prompt)} <span class="muted">…</span>`;
}

/* ======================================================= 7 rnn */
function initRnn() {
  const ws = "The bus to Mysuru left Majestic before I could reach the stand".split(" "); let pair = [];
  const draw = () => {
    $("#pathToks").innerHTML = ws.map((w, i) => `<button class="tok ${pair.includes(i) ? "sel" : ""}" type="button" data-i="${i}">${w}</button>`).join("");
    $$("#pathToks .tok").forEach(b => b.addEventListener("click", () => { const i = +b.dataset.i; pair = pair.length >= 2 ? [i] : pair.includes(i) ? pair : pair.concat(i); draw(); }));
    const k = +$("#kern").value; $("#kernV").textContent = k;
    if (pair.length < 2) { $("#pathBars").innerHTML = ""; $("#pathNote").textContent = pair.length ? "Now click a second word." : "Click a first word."; return; }
    const d = Math.abs(pair[0] - pair[1]), cnn = Math.ceil(d / (k - 1)), mx = Math.max(d, cnn, 1);
    $("#pathBars").innerHTML = bar("RNN: steps", d / mx, d, "r") + bar(`CNN: layers`, cnn / mx, cnn, "c") + bar("Transformer: hops", 1 / mx, 1, "t");
    $("#pathNote").innerHTML = `“${ws[Math.min(...pair)]}” and “${ws[Math.max(...pair)]}” are ${d} word${d > 1 ? "s" : ""} apart. In an RNN the signal passes through every step in between; a CNN needs several stacked layers before its window covers both; in a Transformer one attention layer connects them directly.`;
  };
  $("#kern").addEventListener("input", draw); draw();

  const L = 6;
  const set = () => {
    const n = +$("#nTok").value; $("#nTokV").textContent = n;
    const mx = Math.max(n, L);
    $("#seqBars").innerHTML = bar("RNN steps", n / mx, n, "r") + bar("Transformer passes", L / mx, L, "t") +
      `<p class="small muted" style="margin:6px 0 0">Comparisons inside attention per layer: <b>${(n * n).toLocaleString("en-US")}</b> (n²) vs about <b>${n}</b> for an RNN step chain. Shorter chain, larger square.</p>`;
    $("#seqTrack").innerHTML = `<div class="small">RNN</div><div class="track2" id="trkR">${"<i></i>".repeat(n)}</div><div class="small" style="margin-top:6px">Transformer</div><div class="track2" id="trkT">${"<i></i>".repeat(n)}</div>`;
    $("#seqNote").textContent = "";
  };
  $("#nTok").addEventListener("input", set); set();
  let raf = null;
  $("#seqRun").addEventListener("click", () => {
    cancelAnimationFrame(raf); set();
    const n = +$("#nTok").value, D = Math.min(6000, 1400 + n * 40), Dt = Math.max(250, D * L / n), t0 = performance.now();
    const R = $$("#trkR i"), Tt = $$("#trkT i");
    const frame = now => {
      const e = Math.max(0, now - t0), r = Math.min(n, Math.floor(e / D * n)), tt = e >= Dt ? n : 0;
      R.forEach((c, i) => c.classList.toggle("on", i < r)); if (tt) Tt.forEach(c => c.classList.add("on"));
      $("#seqNote").textContent = `RNN: ${r} of ${n} tokens processed${tt ? " · Transformer: all " + n + " done in " + L + " layer passes (finished first)" : ""}. Real speed-ups depend on hardware; this shows the structure, not a benchmark.`;
      if (e < D) raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
  });
}

/* ======================================================= 8 quiz + glossary */
const QUIZ = [
  ["Which part of a Transformer lets one word use information from other words in the sentence?", ["Attention", "The softmax output layer", "The embedding table", "The optimiser"], 0, "Attention scores say how much each other word matters for the word being processed."],
  ["Why does a Transformer add positional encoding to the embeddings?", ["To make the vectors shorter", "Because it looks at all words at once and would otherwise not know their order", "To remove padding", "To lower the loss"], 1, "Attention itself has no sense of order; position codes supply it."],
  ["In training, what does the decoder receive as input?", ["Only the start token", "The model’s own previous predictions", "The target sequence shifted right (start token in front)", "The source sentence again"], 2, "That is teacher forcing: the correct earlier tokens are fed, and every position predicts its next token at once."],
  ["During inference, how many times does the encoder run for one sentence?", ["Once per generated token", "Once", "Twice", "Never; only the decoder runs"], 1, "The source does not change, so the encoder result is reused at every decoding step."],
  ["What stops the generation loop in inference?", ["A fixed length of 10 tokens", "The model predicting the end token", "The loss reaching zero", "The user pressing a key"], 1, "Usually an end-of-sequence token is predicted (a maximum length is also a safety net)."],
  ["Why is teacher forcing used in training?", ["It makes the model bigger", "Errors do not compound and all positions train in parallel", "It removes the need for a loss", "It is required at inference too"], 1, "A wrong guess cannot poison later positions, and one pass covers the whole target."],
  ["What is the job of a classification head?", ["Generate the next word", "Translate", "Turn the encoder’s output into class probabilities", "Choose the learning rate"], 2, "For example positive vs negative sentiment."],
  ["What is a real drawback of self-attention compared with an RNN?", ["It cannot be parallelised", "It forgets distant words", "Its work grows with the square of the sentence length", "It needs no training data"], 2, "Every word is compared with every other word: n² comparisons."],
];
const GLOSS = [
  ["Token", "A piece of text (a word or part of a word) that the model treats as one unit."],
  ["Embedding", "A vector of numbers that stands for a token and can capture its meaning."],
  ["Positional encoding", "Numbers added to embeddings so the model knows each token’s position."],
  ["Encoder", "A layer (or stack) that reads the input sequence and builds a representation of it."],
  ["Decoder", "A layer (or stack) that produces the output sequence, using the encoder’s result."],
  ["Self-attention", "Each word looks at all words of the same sequence and mixes in the relevant ones."],
  ["Encoder–decoder attention", "The decoder looks at the encoder’s output while writing."],
  ["Masked attention", "Attention that may not look at later positions, so training cannot cheat."],
  ["Multi-head attention", "Several attention computations in parallel, each free to find a different relationship."],
  ["Feed-forward network", "A small network applied to each position separately inside every layer."],
  ["Residual connection", "A shortcut that adds a layer’s input to its output, which makes deep stacks trainable."],
  ["Softmax", "Turns a list of scores into probabilities that add up to 1."],
  ["Loss (cross-entropy)", "How surprised the model is by the correct token; training minimises it."],
  ["Backpropagation", "Computing how each weight should change to lower the loss."],
  ["Teacher forcing", "Training the decoder on the correct previous tokens instead of its own guesses."],
  ["Greedy decoding", "Always taking the most probable next token."],
  ["Temperature", "Sampling control: low makes the model confident, high makes it adventurous."],
  ["Head (task head)", "The last layer(s) that adapt a Transformer to a task, such as classification."],
  ["RNN", "A network that reads a sequence one step at a time, passing a hidden state along."],
];
function initCheck() {
  let qi = 0, score = 0;
  const best = () => store.get("best", null);
  const draw = () => {
    if (qi >= QUIZ.length) {
      const b = Math.max(score, best() ?? 0); store.set("best", b);
      $("#quizProgress").textContent = "Finished"; $("#quizBest").textContent = `Best: ${b}/${QUIZ.length}`;
      $("#quizBody").innerHTML = `<div class="stat t" style="margin:10px 0"><b>${score}/${QUIZ.length}</b><span>${score >= 7 ? "Excellent: you could explain this to someone else." : score >= 5 ? "Good. Revisit the sections of the questions you missed." : "Re-read sections 2 to 5, then try again."}</span></div><button class="btn primary" id="qAgain" type="button">Try again</button>`;
      $("#qAgain").addEventListener("click", () => { qi = 0; score = 0; draw(); }); return;
    }
    const [q, o, a, why] = QUIZ[qi];
    $("#quizProgress").textContent = `Question ${qi + 1} of ${QUIZ.length}`; $("#quizBest").textContent = best() != null ? `Best: ${best()}/${QUIZ.length}` : "";
    $("#quizBody").innerHTML = `<p style="font-weight:700">${q}</p>` + o.map((t, i) => `<button class="quiz-opt" type="button" data-i="${i}">${t}</button>`).join("") + `<div id="qFb"></div>`;
    $$("#quizBody .quiz-opt").forEach(b => b.addEventListener("click", () => {
      if ($("#qFb").innerHTML) return; const i = +b.dataset.i; if (i === a) score++;
      $$("#quizBody .quiz-opt").forEach((x, j) => x.classList.add(j === a ? "right" : j === i ? "wrong" : "x"));
      $("#qFb").innerHTML = `<div class="callout ${i === a ? "tip" : "warn"}"><b>${i === a ? "Correct" : "Not quite"}</b>${why}</div><button class="btn primary" id="qNext" type="button">${qi === QUIZ.length - 1 ? "See result" : "Next question"}</button>`;
      $("#qNext").addEventListener("click", () => { qi++; draw(); });
    }));
  };
  draw();
  const g = () => {
    const q = $("#glossSearch").value.toLowerCase();
    $("#glossList").innerHTML = GLOSS.filter(x => !q || (x[0] + x[1]).toLowerCase().includes(q)).map(x => `<div class="card flat"><b>${x[0]}</b><p class="small muted" style="margin:.3em 0 0">${x[1]}</p></div>`).join("") || `<span class="muted">No match.</span>`;
  };
  $("#glossSearch").addEventListener("input", g); g();
  $("#modelFacts").innerHTML = `Model sizes: NLLB-200 distilled ${(MT.params / 1e6).toFixed(0)} M parameters (${MT.enc_layers}+${MT.dec_layers} layers, d<sub>model</sub> ${MT.d_model}); DistilGPT-2 ${(DATA.lm_model.params / 1e6).toFixed(0)} M parameters. All demos ran locally on a CPU; no network access is needed to view this page.`;
}

/* ======================================================= boot */
(function boot() {
  const run = (name, fn) => { try { return fn(); } catch (e) { console.error("init failed:", name, e); const s = document.getElementById(name); if (s) s.insertAdjacentHTML("afterbegin", `<div class="callout warn"><b>This part failed to load</b>${esc(String(e && e.message || e))}</div>`); } };
  run("intro", setupShell); renderPickers();
  run("intro", initIntro); run("what", initWhat); run("attention", initAttention); run("training", initTraining); run("inference", initInference);
  run("forcing", initForcing); run("uses", initUses); run("rnn", initRnn); run("check", initCheck);
  window.__overviewReady = true;
})();
