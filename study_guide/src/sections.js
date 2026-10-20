"use strict";
/* Interactive sections */

const html = s => { const t = document.createElement("template"); t.innerHTML = s.trim(); return t.content.firstElementChild; };
const labelWord = y => (y === 1 ? "positive" : "negative");
const dot = fam => `<i class="dot ${fam}"></i>`;

/* ===================== 1 · START ===================== */
function initStart() {
  const kn = [
    [K.acc_pt_distilbert_sst2 + "%", "Best accuracy: ready-made DistilBERT (no training by us)", "t"],
    [K.acc_ft_distilbert_8000 + "%", "DistilBERT fine-tuned by us on only 8,000 sentences", "t"],
    [K.best_cls_acc + "%", `Best classical model (TF-IDF + SVM), trained on all ${K.n_train}`, "c"],
    [K.acc_bilstm + "%", "BiLSTM from scratch: no gain over classical", "r"],
    ["+" + K.lc500_gain + " pts", "BERT-small vs TF-IDF with only 500 labels", "v"],
    ["×" + K.latratio, "slower inference: DistilBERT vs TF-IDF", "r"],
    [K.acc_zs_nli_distilroberta + "%", "Zero-shot: no sentiment labels at all", "v"],
    [K.acc_ft_bert_tiny_8000 + "%", "BERT-tiny: too small to beat the classical models", "r"],
  ];
  const box = $("#keyNumbers");
  kn.forEach(([v, d, c]) => box.appendChild(html(`<div class="stat ${c}"><b>${v}</b><span>${d}</span></div>`)));

  const cards = [
    ["What is the best accuracy, and which model?", `${K.acc_pt_distilbert_sst2}%: the ready-made DistilBERT-SST2`],
    ["How many labels did our fine-tuned DistilBERT use, and what did it score?", `8,000 labels → ${K.acc_ft_distilbert_8000}%`],
    ["Best classical model?", `TF-IDF + linear SVM: ${K.best_cls_acc}% (trained on ${K.n_train} sentences)`],
    ["Sizes of train / dev / test?", `${K.n_train} / ${K.n_dev} / ${K.n_test} (test = official SST-2 validation set)`],
    ["Gain of BERT-small over TF-IDF at 500 labels?", `${K.lc500_t}% vs ${K.lc500_c}% = +${K.lc500_gain} points`],
    ["How much slower is DistilBERT than TF-IDF?", `About ${K.latratio}× (${K.lat_ft_distilbert_8000} ms vs ${K.lat_tfidf_svm} ms per sentence)`],
    ["Negation: accuracy loss for bag-of-words vs big Transformers?", "8–9 points vs 2–4 points"],
    ["Parameters of the three fine-tuned models?", `${K.params_ft_bert_tiny_8000} / ${K.params_ft_bert_small_8000} / ${K.params_ft_distilbert_8000}`],
    ["BiLSTM: dev vs test accuracy, and why?", `~91% vs ${K.acc_bilstm}%: dev = short training phrases, test = full sentences`],
    ["Hardware?", "Intel Core i5-8365U, 4 cores, 16 GB RAM, no GPU"],
    ["Statistics used?", "Bootstrap 95% intervals (2,000 resamples) and the exact McNemar test"],
    ["What is the practical recommendation?", "Reuse a public in-domain model; else fine-tune a distilled Transformer; TF-IDF only for extreme speed/size limits"],
  ];
  const fl = $("#flash");
  cards.forEach(([q, a]) => {
    const b = html(`<button class="card flat" type="button" style="text-align:left; min-height:96px" aria-pressed="false"><span class="muted small">Question</span><br><b>${esc(q)}</b></button>`);
    let open = false;
    b.addEventListener("click", () => { open = !open; b.setAttribute("aria-pressed", open); b.innerHTML = open ? `<span class="small" style="color:var(--good)">Answer</span><br><b>${esc(a)}</b>` : `<span class="muted small">Question</span><br><b>${esc(q)}</b>`; });
    fl.appendChild(b);
  });
}

/* ===================== 2 · STORY ===================== */
function initStory() {
  const btns = $("#engineerBtns"), panel = $("#engineerPanel");
  const show = id => {
    const e = ENGINEERS.find(x => x.id === id), m = M[MI[e.tag]];
    $$("button", btns).forEach(b => b.setAttribute("aria-pressed", b.dataset.id === id));
    panel.innerHTML = `<p><b>${e.name}: ${esc(e.idea)}</b> ${resolve(e.text)}</p>
      <div class="grid g4" style="margin:10px 0">
        <div class="stat ${e.color[0]}"><b>${pct1(m.acc)}%</b><span>accuracy (${esc(m.name)})</span></div>
        <div class="stat"><b>${m.train_size ? fmtTime(m.train_s) : "none"}</b><span>training time (${fmtInt(m.train_size)} labels)</span></div>
        <div class="stat"><b>${fmtLat(m.lat_ms)} ms</b><span>per sentence</span></div>
        <div class="stat"><b>${fmtParams(m.params)}</b><span>parameters</span></div></div>
      <p><span class="chip ok">Wins at</span> ${e.wins}</p><p><span class="chip no">Loses at</span> ${e.loses}</p>`;
  };
  ENGINEERS.forEach(e => { const b = html(`<button class="btn" type="button" data-id="${e.id}" aria-pressed="false">${dot(e.color)}${e.name}: ${esc(e.idea)}</button>`); b.addEventListener("click", () => show(e.id)); btns.appendChild(b); });
  show("anna");
  const an = $("#analogies");
  ANALOGIES.forEach(([t, d]) => an.appendChild(html(`<div class="card flat"><b>${esc(t)}</b><p class="muted" style="margin:.3em 0 0">${esc(d)}</p></div>`)));
}

/* ===================== 3 · PIPELINE ===================== */
function initPipeline() {
  const nodes = $("#stepper"), body = $("#stepBody");
  let cur = store.get("step", 0);
  const renderBody = () => {
    const s = PIPE[cur];
    body.innerHTML = `<p class="eyebrow">Step ${cur + 1} of ${PIPE.length} · Paper: ${esc(s.paper)}</p><h3 style="margin-top:0">${esc(s.title)}</h3>
      <div class="callout"><b>In the ReviewRadar story</b>${s.rw}</div>
      <h4>What happens</h4><ul>${s.what.map(w => `<li>${w}</li>`).join("")}</ul>
      <h4>Why</h4><p>${s.why}</p><div data-slot="viz"></div><div data-slot="code"></div>
      <h4>Output</h4><p>${s.out}</p><div class="say">${s.say}</div>
      <h4>Be ready for</h4><div data-slot="ask"></div>`;
    if (s.viz === "split") {
      const d = DATA.dataset, tot = d.train + d.dev + d.test;
      $('[data-slot="viz"]', body).appendChild(html(`<div class="card flat"><div class="timeline" style="height:44px" aria-label="Split sizes">
        <div style="flex:${d.train}; background:var(--classical)">TRAIN ${fmtInt(d.train)}</div><div style="flex:${Math.max(d.dev, tot * .035)}; background:var(--violet)" title="dev"></div><div style="flex:${Math.max(d.test, tot * .02)}; background:var(--recurrent)" title="test"></div></div>
        <div class="legend"><span><i class="dot classical"></i>Train: ${fmtInt(d.train)} short phrases (avg ${d.train_len.toFixed(1)} words, ${(d.train_pos * 100).toFixed(0)}% positive)</span><span><i class="dot" style="background:var(--violet)"></i>Dev: ${fmtInt(d.dev)} (choose the epoch)</span><span><i class="dot recurrent"></i>Test: ${fmtInt(d.test)} full sentences (avg ${d.test_len.toFixed(1)} words, ${(d.test_pos * 100).toFixed(0)}% positive), used once</span></div></div>`));
    }
    if (s.code) $('[data-slot="code"]', body).appendChild(codeBlock(s.code.file, s.code.text));
    const ask = $('[data-slot="ask"]', body);
    s.ask.forEach(([q, a]) => ask.appendChild(html(`<details><summary>${q}</summary><div>${a}</div></details>`)));
    $$(".step-node", nodes).forEach((n, i) => { n.classList.toggle("on", i === cur); n.setAttribute("aria-selected", i === cur); });
    $("#stepPrev").disabled = cur === 0; $("#stepNext").disabled = cur === PIPE.length - 1;
    store.set("step", cur);
  };
  PIPE.forEach((s, i) => {
    const n = html(`<button class="step-node" role="tab" type="button"><b>Step ${i + 1}</b>${esc(s.short)}</button>`);
    n.addEventListener("click", () => { cur = i; renderBody(); });
    nodes.appendChild(n);
  });
  $("#stepPrev").addEventListener("click", () => { if (cur > 0) { cur--; renderBody(); } });
  $("#stepNext").addEventListener("click", () => { if (cur < PIPE.length - 1) { cur++; renderBody(); } });
  renderBody();
}

/* ===================== 4 · THEORY ===================== */
function initTheory() {
  // tokens
  const td = $("#tokenDemo");
  DATA.tokenisation.forEach(t => td.appendChild(html(`<div class="card flat" style="padding:10px 14px"><b>${esc(t.word)}</b> → <span class="tokens" style="display:inline-flex; margin:0">${t.pieces.map(p => `<span class="tok">${esc(p)}</span>`).join("")}</span></div>`)));

  // attention calculator
  const words = ["the", "plot", "was", "not", "good"], base = [0.3, 1.1, 0.4, 2.4, 1.6];
  const sl = $("#attnSliders"), vals = base.slice();
  words.forEach((w, i) => {
    const row = html(`<div><label class="field" for="as${i}">score(good → <b>${w}</b>): <span id="asv${i}">${base[i].toFixed(1)}</span></label><input type="range" id="as${i}" min="-2" max="4" step="0.1" value="${base[i]}"></div>`);
    $("input", row).addEventListener("input", e => { vals[i] = +e.target.value; $("#asv" + i).textContent = vals[i].toFixed(1); drawAttn(); });
    sl.appendChild(row);
  });
  const drawAttn = () => {
    const dk = +$("#attnDk").value, scaled = $("#attnScale").checked; $("#dkVal").textContent = dk;
    const raw = vals.map(v => v * Math.sqrt(dk));                // dot products grow like √d_k
    const used = raw.map(v => (scaled ? v / Math.sqrt(dk) : v));
    const mx = Math.max(...used), ex = used.map(v => Math.exp(v - mx)), tot = ex.reduce((a, b) => a + b, 0), w = ex.map(v => v / tot);
    $("#attnBars").innerHTML = words.map((t, i) => `<div class="bar"><span>${t}</span><span class="track"><span class="fill pos" style="left:0; width:${(w[i] * 100).toFixed(1)}%"></span></span><b>${(w[i] * 100).toFixed(0)}%</b></div>`).join("");
    const top = Math.max(...w);
    $("#attnNote").innerHTML = (!scaled && top > .97) ? `<b style="color:var(--bad)">Saturated:</b> one word takes ${(top * 100).toFixed(0)}% and the others ~0%. Gradients vanish and the model cannot learn nuance. Tick “divide by √d<sub>k</sub>”.`
      : `Raw dot products would be ${raw.map(v => v.toFixed(1)).join(", ")}; scores used in softmax: ${used.map(v => v.toFixed(1)).join(", ")}. The weights add up to 100%.`;
  };
  $("#attnScale").addEventListener("change", drawAttn); $("#attnDk").addEventListener("input", drawAttn); drawAttn();

  // architecture
  const col = $("#archCol"), info = $("#archInfo");
  let loopBox = null, selected = "attn";
  const showArch = id => {
    selected = id; const a = ARCH.find(x => x.id === id);
    $$(".arch-block", col).forEach(b => b.classList.toggle("on", b.dataset.id === id));
    info.innerHTML = `<h4 style="margin-top:0">${esc(a.title)}</h4>${resolve(a.body)}<div class="say">${a.say}</div>`;
  };
  const makeBlock = a => { const b = html(`<button class="arch-block" type="button" data-id="${a.id}">${esc(a.label)}</button>`); b.addEventListener("click", () => showArch(a.id)); return b; };
  const arrow = () => html(`<div class="arch-arrow">&uarr;</div>`);
  // bottom-to-top: build in reverse order so that the input is at the bottom
  const order = [...ARCH].reverse();
  order.forEach((a, i) => {
    if (a.loop) { if (!loopBox) { loopBox = html(`<div class="arch-loop"></div>`); col.appendChild(loopBox); } loopBox.appendChild(makeBlock(a)); if (order[i + 1] && order[i + 1].loop) loopBox.appendChild(arrow()); }
    else { loopBox = null; col.appendChild(makeBlock(a)); }
    if (i < order.length - 1 && !(a.loop && order[i + 1].loop)) col.appendChild(arrow());
  });
  showArch(selected);

  // sequential vs parallel
  const sent = ["the", "plot", "was", "not", "good", ",", "but", "the", "acting", "was", "brilliant"];
  const lr = $("#laneRnn"), lt = $("#laneTr");
  sent.forEach(w => { lr.appendChild(html(`<span class="w">${w}</span>`)); lt.appendChild(html(`<span class="w">${w}</span>`)); });
  let timers = [];
  $("#parRun").addEventListener("click", () => {
    timers.forEach(clearTimeout); timers = [];
    const rw = $$(".w", lr), tw = $$(".w", lt); rw.concat(tw).forEach(x => x.classList.remove("lit"));
    $("#parSteps").textContent = "";
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { rw.concat(tw).forEach(x => x.classList.add("lit")); $("#parSteps").innerHTML = `BiLSTM: <b>${sent.length} sequential steps</b>. Transformer: <b>1 parallel step</b> (${sent.length}×${sent.length} = ${sent.length ** 2} attention scores).`; return; }
    tw.forEach(x => x.classList.add("lit"));
    rw.forEach((x, i) => timers.push(setTimeout(() => { x.classList.add("lit"); $("#parSteps").innerHTML = `BiLSTM: step ${i + 1} of ${sent.length} …`; if (i === sent.length - 1) $("#parSteps").innerHTML = `BiLSTM needed <b>${sent.length} sequential steps</b>. The Transformer needed <b>1 parallel step</b> (${sent.length}×${sent.length} = ${sent.length ** 2} attention scores).`; }, 380 * (i + 1))));
  });
}

/* ===================== 5 · LAB ===================== */
const WALK_TAGS = { 0: "Easy", 4: "Easy", 627: "Negation", 110: "Negation", 398: "Double negation", 481: "Mixed sentiment", 527: "Sarcasm", 183: "Irony · every model fails", 271: "Every model fails", "-1": "Contrast demo (made up)" };
function renderBow(container, text) {
  const r = bowAnalyse(text);
  const maxAbs = Math.max(0.001, ...r.parts.map(p => Math.abs(p.c)));
  const uni = new Map(r.parts.filter(p => !p.bigram).map(p => [p.gram, p.c]));
  const chips = r.tokens.map(t => { const c = uni.get(t); if (c == null) return `<span class="tok" title="not in vocabulary: ignored">${esc(t)}</span>`; const a = Math.min(.85, Math.abs(c) / maxAbs * .85 + .08); const col = c > 0 ? "21,154,106" : "217,84,31"; return `<span class="tok" style="background:rgba(${col},${a.toFixed(2)})" title="${c > 0 ? "+" : ""}${c.toFixed(2)}">${esc(t)}</span>`; }).join("");
  const top = r.parts.slice(0, 10);
  const bars = top.map(p => { const w = Math.abs(p.c) / maxAbs * 50; return `<div class="bar diverge"><span>${esc(p.gram)}${p.bigram ? " <small class='muted'>(pair)</small>" : ""}</span><span class="track"><span class="fill ${p.c > 0 ? "pos" : "neg"}" style="${p.c > 0 ? "left:50%" : `left:${50 - w}%`}; width:${w}%"></span></span><b>${p.c > 0 ? "+" : ""}${p.c.toFixed(2)}</b></div>`; }).join("");
  const pos = r.p > 0.5;
  container.innerHTML = `<div class="row" style="justify-content:space-between"><b>Model decision: <span class="chip ${pos ? "ok" : "no"}">${pos ? "positive" : "negative"}</span> (${(r.p * 100).toFixed(0)}% positive)</b><span class="muted small">evidence = bias ${r.bias > 0 ? "+" : ""}${r.bias.toFixed(2)} + sum of the numbers below = ${r.z > 0 ? "+" : ""}${r.z.toFixed(2)}</span></div>
    <div class="tokens">${chips}</div>
    <div class="bars" style="margin-top:6px">${bars || "<span class='muted'>No known words.</span>"}</div>
    <p class="small muted" style="margin:8px 0 0">Colour intensity = how strongly a word pushes. “(pair)” = two neighbouring words counted together. ${r.unknown.length ? `Ignored (unknown to the model): ${esc(r.unknown.join(", "))}.` : ""}</p>`;
  return r;
}
function initLab() {
  const pick = $("#labPick"), entries = DATA.walk;
  let sel = store.get("labSel", 2); if (sel >= entries.length) sel = 0;
  let layer = 5, head = -1, view = "words", tokSel = null;
  const textOf = e => (e.i >= 0 ? SENT[e.i].s : e.text);
  const render = () => {
    const e = entries[sel], text = textOf(e), row = e.i >= 0 ? SENT[e.i] : null;
    $$("button", pick).forEach((b, i) => b.classList.toggle("on", i === sel));
    const note = row ? (WALK_NOTES[e.i] || "") : "A made-up sentence that shows <b>contrast</b>: negative first half, positive ending. Watch how the models weigh “not good” against “brilliant”.";
    const pp = e.p_pos;
    $("#labHead").innerHTML = `<div class="row" style="justify-content:space-between"><b style="font-size:1.1rem">“${esc(text)}”</b>${row ? `<span class="chip ${row.y ? "ok" : "no"}">Gold label: ${labelWord(row.y)}</span>` : ""}</div>
      <p style="margin:.7em 0 .4em">${note}</p>
      <div class="small muted">DistilBERT-SST2: P(positive) = <b>${(pp * 100).toFixed(1)}%</b></div><div class="meter" aria-hidden="true" style="margin-top:6px"><i style="left:${pp * 100}%"></i></div>`;
    const v = $("#labVerdicts");
    if (row) {
      v.innerHTML = M.map((m, i) => { const ok = isRight(row, i); return `<div class="verdict ${ok ? "ok" : "no"}"><span>${dot(m.family)}${esc(m.short)}</span><b>${labelWord(+row.p[i])} ${ok ? "✓" : "✗"}</b></div>`; }).join("") + `<div class="small muted" style="grid-column:1/-1">${nRight(row)} of ${M.length} models are right on this sentence.</div>`;
    } else v.innerHTML = `<div class="muted small">Verdicts are recorded for the 872 test sentences only. Pick a test sentence on the left to compare all ten models.</div>`;
    const r = renderBow($("#labBow"), text);
    if (row) { const rec = +row.p[MI.tfidf_logreg]; $("#labBow").insertAdjacentHTML("beforeend", `<p class="small ${rec === +(r.p > .5) ? "muted" : ""}" style="margin:6px 0 0">${rec === +(r.p > .5) ? "✓ Matches the prediction recorded in the experiment for TF-IDF + logistic regression." : "⚠ Differs from the recorded prediction (rounding)."}</p>`); }
    renderAtt(e);
  };
  const renderAtt = e => {
    const box = $("#labAtt"), [L, H, n] = e.shape, get = attOf(e), toks = e.tokens;
    if (tokSel == null || tokSel >= n || sel !== renderAtt.last) { const pref = ["good", "entertaining", "heavy", "slight", "excitement", "enjoy", "points", "suggests", "journey", "slow"]; let k = toks.findIndex(t => pref.includes(t)); tokSel = k >= 0 ? k : Math.max(0, n - 3); renderAtt.last = sel; }
    box.innerHTML = `<div class="row"><label class="small" for="lyr">Layer</label><select id="lyr" style="width:auto">${Array.from({ length: L }, (_, i) => `<option value="${i}" ${i === layer ? "selected" : ""}>${i + 1}${i === L - 1 ? " (last)" : ""}</option>`).join("")}</select>
      <label class="small" for="hd">Head</label><select id="hd" style="width:auto"><option value="-1" ${head === -1 ? "selected" : ""}>average of 12</option>${Array.from({ length: H }, (_, i) => `<option value="${i}" ${i === head ? "selected" : ""}>head ${i + 1}</option>`).join("")}</select>
      <button class="btn small ${view === "words" ? "on" : ""}" id="vw" type="button">Words</button><button class="btn small ${view === "heat" ? "on" : ""}" id="vh" type="button">Heat-map</button></div>
      <div id="attView"></div>`;
    $("#lyr", box).addEventListener("change", ev => { layer = +ev.target.value; renderAtt(e); });
    $("#hd", box).addEventListener("change", ev => { head = +ev.target.value; renderAtt(e); });
    $("#vw", box).addEventListener("click", () => { view = "words"; renderAtt(e); });
    $("#vh", box).addEventListener("click", () => { view = "heat"; renderAtt(e); });
    const vbox = $("#attView", box);
    if (view === "words") {
      const rowv = toks.map((_, j) => get(layer, head, tokSel, j));
      vbox.innerHTML = `<p class="small muted" style="margin:10px 0 0">Attention paid by <b>“${esc(toks[tokSel])}”</b> (click another word):</p><div class="tokens" id="attToks"></div>`;
      toks.forEach((t, j) => { const v = rowv[j]; const b = html(`<button type="button" class="tok att ${j === tokSel ? "sel" : ""}" style="background:rgba(98,80,214,${Math.min(.95, Math.pow(v, .6) * 1.1 + .04).toFixed(2)}); color:${v > .35 ? "#fff" : "inherit"}">${esc(t)}${v >= .05 ? `<br><small>${(v * 100).toFixed(0)}%</small>` : ""}</button>`); b.addEventListener("click", () => { tokSel = j; renderAtt(e); }); $("#attToks", vbox).appendChild(b); });
    } else {
      vbox.innerHTML = `<p class="small muted" style="margin:10px 0 0">Row = the word that looks; column = the word it looks at. Darker = more attention. Click a row cell to select that word.</p><div id="heatBox"></div>`;
      heatmap($("#heatBox", vbox), toks, get, layer, head, tokSel, i => { tokSel = i; view = "words"; renderAtt(e); });
    }
  };
  entries.forEach((e, i) => {
    const b = html(`<button type="button" role="option"><span class="tag">${WALK_TAGS[e.i]}</span>${esc(textOf(e))}</button>`);
    b.addEventListener("click", () => { sel = i; store.set("labSel", i); render(); });
    pick.appendChild(b);
  });
  render();
  const own = $("#ownText"), draw = () => renderBow($("#ownResult"), own.value);
  own.addEventListener("input", debounce(draw, 120)); draw();
  addEventListener("resize", debounce(() => { const e = entries[sel]; if (view === "heat") renderAtt(e); }, 250));
}

/* ===================== 6 · ARENA ===================== */
function initArena() {
  const FILTERS = [
    ["all", "All sentences", r => true],
    ["neg", "With negation", r => r.n],
    ["con", "With contrast", r => r.c],
    ["hard", "Hard (≤ 3 models right)", r => nRight(r) <= 3],
    ["allwrong", "Every model wrong", r => nRight(r) === 0],
    ["dis", "Models disagree", r => { const n = nRight(r); return n > 0 && n < M.length; }],
    ["tr", "Fine-tuned DistilBERT right, TF-IDF SVM wrong", r => isRight(r, MI.ft_distilbert_8000) && !isRight(r, MI.tfidf_svm)],
    ["cl", "TF-IDF SVM right, fine-tuned DistilBERT wrong", r => !isRight(r, MI.ft_distilbert_8000) && isRight(r, MI.tfidf_svm)],
  ];
  const fbox = $("#arenaFilters"); let filter = "all", q = "", page = 0; const SIZE = 30;
  FILTERS.forEach(([id, label]) => { const b = html(`<button class="btn small" type="button" data-id="${id}" aria-pressed="${id === filter}">${esc(label)}</button>`); b.addEventListener("click", () => { filter = id; page = 0; draw(); }); fbox.appendChild(b); });
  $("#arenaLegend").innerHTML = M.map((m, i) => `<span><b>${i + 1}</b> ${dot(m.family)}${esc(m.short)}</span>`).join("");
  const draw = () => {
    $$("button", fbox).forEach(b => b.setAttribute("aria-pressed", b.dataset.id === filter));
    const f = FILTERS.find(x => x[0] === filter)[2], ql = q.trim().toLowerCase();
    const rows = SENT.filter(r => f(r) && (!ql || r.s.toLowerCase().includes(ql)));
    $("#arenaCount").textContent = `${fmtInt(rows.length)} sentences`;
    $("#arenaBars").innerHTML = rows.length ? M.map((m, i) => { const a = accOn(rows, i); return `<div class="bar"><span>${dot(m.family)}${esc(m.short)}</span><span class="track"><span class="fill" style="left:0; width:${(a * 100).toFixed(1)}%; background:${FAMCLR[m.family]}"></span></span><b>${(a * 100).toFixed(1)}%</b></div>`; }).join("") : `<span class="muted">No sentences match.</span>`;
    const pages = Math.max(1, Math.ceil(rows.length / SIZE)); page = Math.min(page, pages - 1);
    const tb = $("#arenaTable tbody"); tb.innerHTML = "";
    rows.slice(page * SIZE, page * SIZE + SIZE).forEach(r => {
      const chips = M.map((m, i) => { const ok = isRight(r, i); return `<span class="chip ${ok ? "ok" : "no"}" title="${esc(m.name)}: predicted ${labelWord(+r.p[i])}">${i + 1}${ok ? "✓" : "✗"}</span>`; }).join(" ");
      tb.appendChild(html(`<tr><td>${esc(r.s)}${r.n ? ' <span class="chip fam-classical">neg</span>' : ""}${r.c ? ' <span class="chip fam-recurrent">contrast</span>' : ""}</td><td><span class="chip ${r.y ? "ok" : "no"}" style="${r.y ? "" : "background:var(--surface2); color:var(--text)"}">${labelWord(r.y)}</span></td><td style="line-height:2">${chips}</td></tr>`));
    });
    $("#arenaPage").textContent = `Page ${page + 1} of ${pages}`;
    $("#arenaPrev").disabled = page === 0; $("#arenaNext").disabled = page >= pages - 1;
  };
  $("#arenaSearch").addEventListener("input", debounce(e => { q = e.target.value; page = 0; draw(); }, 150));
  $("#arenaPrev").addEventListener("click", () => { page--; draw(); $("#arena").scrollIntoView(); });
  $("#arenaNext").addEventListener("click", () => { page++; draw(); $("#arena").scrollIntoView(); });
  draw();
}

/* ===================== 7 · RESULTS ===================== */
function initResults() {
  const cAcc = () => chartAccuracy($("#chartAcc")), cCost = () => chartCost($("#chartCost"));
  // learning curve
  const slider = $("#lcSlider"); const nFrom = v => Math.round(500 * Math.pow(16, v / 100));
  const lc = () => {
    const n = nFrom(+slider.value); chartLC($("#chartLC"), n); $("#lcValue").textContent = fmtInt(n);
    const t = interp(DATA.curve.transformer, n), c = interp(DATA.curve.tfidf, n), need = labelsNeeded(DATA.curve.tfidf, t);
    $("#lcReadout").innerHTML = `<div class="stat t"><b>${pct1(t)}%</b><span>BERT-small with ${fmtInt(n)} labels</span></div><div class="stat c"><b>${pct1(c)}%</b><span>TF-IDF with the same labels (+${((t - c) * 100).toFixed(1)} points for BERT-small)</span></div>
      <div class="stat v"><b>${isFinite(need) ? "≈ " + fmtInt(Math.round(need / 10) * 10) : "> 8,000"}</b><span>labels TF-IDF would need to match BERT-small${isFinite(need) && need > n * 1.2 ? ` (${(need / n).toFixed(0)}× more)` : ""}</span></div>`;
  };
  slider.addEventListener("input", lc);
  // subsets
  let subView = "acc"; const tabs = $("#subsetTabs");
  [["acc", "Accuracy on each subset"], ["drop", "Drop vs overall (points)"]].forEach(([id, label]) => { const b = html(`<button class="btn small" type="button" aria-pressed="${id === subView}">${label}</button>`); b.addEventListener("click", () => { subView = id; $$("button", tabs).forEach((x, i) => x.setAttribute("aria-pressed", ["acc", "drop"][i] === id)); chartSubsets($("#chartSub"), subView); }); tabs.appendChild(b); });
  $("#chartSub").insertAdjacentHTML("beforebegin", `<div class="legend"><span><i class="dot" style="background:#9a98b4"></i>All sentences</span><span><i class="dot classical"></i>Negation (n=${K.neg_n})</span><span><i class="dot recurrent"></i>Contrast (n=${K.con_n})</span></div>`);
  // table
  const cols = [["name", "Model", false], ["family", "Family", false], ["train_size", "Train labels", true], ["acc", "Accuracy %", true], ["f1", "F1 %", true], ["train_s", "Train time", true], ["lat_ms", "Latency ms", true], ["params", "Parameters", true]];
  let sortKey = "acc", asc = false;
  const table = $("#resTable");
  const tdraw = () => {
    const rows = [...M].sort((a, b) => { const x = a[sortKey], y = b[sortKey]; return (typeof x === "string" ? x.localeCompare(y) : x - y) * (asc ? 1 : -1); });
    table.innerHTML = `<thead><tr>${cols.map(([k, l, num]) => `<th class="sortable ${num ? "num" : ""}" data-k="${k}" tabindex="0" role="button" aria-label="Sort by ${l}">${l}${sortKey === k ? (asc ? " ▲" : " ▼") : ""}</th>`).join("")}</tr></thead><tbody>${rows.map(m => `<tr><td>${dot(m.family)}${esc(m.name)}</td><td>${FAMNAME[m.family]}</td><td class="num">${m.train_size ? fmtInt(m.train_size) : "0"}</td><td class="num"><b>${pct1(m.acc)}</b></td><td class="num">${pct1(m.f1)}</td><td class="num">${fmtTime(m.train_s)}</td><td class="num">${fmtLat(m.lat_ms)}</td><td class="num">${fmtParams(m.params)}</td></tr>`).join("")}</tbody>`;
    $$("th", table).forEach(th => { const go = () => { const k = th.dataset.k; if (sortKey === k) asc = !asc; else { sortKey = k; asc = k === "name" || k === "family"; } tdraw(); }; th.addEventListener("click", go); th.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } }); });
  };
  // McNemar
  const A = $("#mnA"), B = $("#mnB");
  M.forEach((m, i) => { A.appendChild(new Option(m.name, i)); B.appendChild(new Option(m.name, i)); });
  A.value = MI.ft_distilbert_8000; B.value = MI.tfidf_svm;
  const mn = () => {
    const a = +A.value, b = +B.value, out = $("#mnOut");
    if (a === b) { out.innerHTML = `<span class="muted">Choose two different models.</span>`; return; }
    const r = mcnemar(a, b), sig = r.p < 0.05, better = r.only_a > r.only_b ? M[a].name : M[b].name;
    out.innerHTML = `<div class="grid g3"><div class="stat"><b>${r.only_a}</b><span>sentences only <b>${esc(M[a].short)}</b> gets right</span></div><div class="stat"><b>${r.only_b}</b><span>sentences only <b>${esc(M[b].short)}</b> gets right</span></div><div class="stat ${sig ? "t" : "r"}"><b>${fmtP(r.p)}</b><span>p-value (exact McNemar)</span></div></div>
      <div class="callout ${sig ? "tip" : "warn"}" style="margin-bottom:0"><b>${sig ? "A real difference" : "No evidence of a real difference"}</b>${sig ? `${esc(better)} is better, and a split like ${r.only_a} : ${r.only_b} would be very unlikely if the two models were equally good.` : `A split of ${r.only_a} : ${r.only_b} is quite plausible by chance, so we cannot say these two models differ (accuracies ${pct1(M[a].acc)}% vs ${pct1(M[b].acc)}%).`}</div>`;
  };
  A.addEventListener("change", mn); B.addEventListener("change", mn);
  tdraw(); mn();
  const redraw = () => { cAcc(); cCost(); lc(); chartSubsets($("#chartSub"), subView); chartDev($("#chartDev")); };
  redraw();
  addEventListener("resize", debounce(redraw, 220));
  return redraw;
}

/* ===================== 8 · CHOOSE ===================== */
function initChoose() {
  const latFrom = v => 0.05 * Math.pow(6000, v / 100), latToSlider = ms => Math.log(ms / 0.05) / Math.log(6000) * 100;
  const trFrom = v => Math.pow(600, v / 100), trToSlider = min => Math.log(min) / Math.log(600) * 100;
  const sel = { lab: $("#chLabels"), lat: $("#chLat"), tr: $("#chTrain"), inD: $("#chIn"), rel: $("#chRel") };
  const evalModels = () => {
    const labels = +sel.lab.value, lat = latFrom(+sel.lat.value), trMin = trFrom(+sel.tr.value), inD = sel.inD.checked, rel = sel.rel.checked;
    return M.map(m => {
      let acc = m.acc, why = [], note = "", trainS = m.train_s;
      if (m.tag === "pt_distilbert_sst2") { if (!inD) why.push("needs a public model for your kind of text"); }
      else if (m.tag === "pt_twitter_roberta") { if (!rel) why.push("needs a public model for a related domain"); }
      else if (m.tag === "zs_nli_distilroberta") { /* needs nothing */ }
      else if (m.tag === "tfidf_logreg" && labels >= 500 && labels < 65349) { acc = interp(DATA.curve.tfidf, Math.min(labels, 8000)); trainS = 5; note = labels > 8000 ? "" : "from the learning curve"; if (labels > 8000) why.push("only measured with ≤ 8,000 or all 65,349 labels"); }
      else if (m.tag === "ft_bert-small_8000" && labels >= 500) { acc = interp(DATA.curve.transformer, Math.min(labels, 8000)); trainS = labels < 8000 ? m.train_s * labels / 8000 : m.train_s; note = labels < 8000 ? "learning curve; time scaled" : labels > 8000 ? "measured with 8,000; more data should help" : ""; }
      else if (labels < m.train_size) why.push(`was measured with ${fmtInt(m.train_size)} labels (you have ${fmtInt(labels)})`);
      if (m.train_size && labels === 0) why.push("needs labelled training data");
      if (m.train_size && trainS > trMin * 60) why.push(`training takes ~${fmtTime(trainS)}, longer than your limit`);
      if (m.lat_ms > lat) why.push(`too slow: ${fmtLat(m.lat_ms)} ms > ${fmtLat(lat)} ms`);
      return { m, acc, why, note, ok: why.length === 0 };
    });
  };
  const draw = () => {
    $("#chLatVal").textContent = fmtLat(latFrom(+sel.lat.value)) + " ms";
    const tm = trFrom(+sel.tr.value); $("#chTrainVal").textContent = tm < 60 ? Math.round(tm) + " min" : (tm / 60).toFixed(1) + " h";
    const res = evalModels(), ok = res.filter(r => r.ok).sort((a, b) => b.acc - a.acc), no = res.filter(r => !r.ok);
    const out = $("#chOut");
    if (!ok.length) { out.innerHTML = `<div class="callout warn"><b>No measured model satisfies all constraints.</b>Relax the speed limit, the training limit or collect more labels. The reasons are listed below.</div>`; }
    else {
      const best = ok[0], fast = [...ok].sort((a, b) => a.m.lat_ms - b.m.lat_ms)[0];
      out.innerHTML = `<div class="callout tip"><b>Recommended: ${dot(best.m.family)}${esc(best.m.name)} (${pct1(best.acc)}%)</b>Most accurate among the models that fit your limits${best.note ? ` (${best.note})` : ""}. It takes ${fmtLat(best.m.lat_ms)} ms per sentence, has ${fmtParams(best.m.params)} parameters${best.m.train_size ? ` and training ≈ ${fmtTime(best.m.train_s)}` : " and needs no training"}.${fast.m.tag !== best.m.tag ? `<br><b>If speed matters more:</b> ${esc(fast.m.name)} is ${Math.round(best.m.lat_ms / fast.m.lat_ms)}× faster at ${pct1(fast.acc)}% (${((fast.acc - best.acc) * 100).toFixed(1)} points).` : ""}</div>
        <div class="scroll"><table><thead><tr><th>#</th><th>Model that fits</th><th class="num">Accuracy</th><th class="num">Latency ms</th><th>Notes</th></tr></thead><tbody>${ok.map((r, i) => `<tr><td>${i + 1}</td><td>${dot(r.m.family)}${esc(r.m.name)}</td><td class="num"><b>${pct1(r.acc)}%</b></td><td class="num">${fmtLat(r.m.lat_ms)}</td><td class="muted small">${r.note}</td></tr>`).join("")}</tbody></table></div>`;
    }
    if (no.length) out.insertAdjacentHTML("beforeend", `<details><summary>Excluded models (${no.length}) and why</summary><div><ul>${no.map(r => `<li><b>${esc(r.m.name)}</b>: ${r.why.join("; ")}</li>`).join("")}</ul></div></details>`);
  };
  const apply = s => {
    sel.lab.value = String(s.labels); sel.lat.value = latToSlider(s.lat); sel.tr.value = trToSlider(s.train); sel.inD.checked = s.inDomain; sel.rel.checked = s.related;
    const note = $("#chNote"); note.hidden = !s.lt; if (s.lt) note.innerHTML = "<b>Caveat:</b> our measured models are English. For Lithuanian, repeat the same procedure with LitLat BERT or XLM-R (Ulčar &amp; Robnik-Šikonja, 2021); the ranking logic (reuse → fine-tune → zero-shot → TF-IDF) carries over, the numbers do not.";
    $$("#scenarioBtns button").forEach(b => b.setAttribute("aria-pressed", b.dataset.id === s.id));
    draw();
  };
  SCENARIOS.forEach(s => { const b = html(`<button class="btn" type="button" data-id="${s.id}" aria-pressed="false" title="${esc(s.desc)}">${esc(s.name)}</button>`); b.addEventListener("click", () => apply(s)); $("#scenarioBtns").appendChild(b); });
  Object.values(sel).forEach(e => e.addEventListener("input", () => { $$("#scenarioBtns button").forEach(b => b.setAttribute("aria-pressed", "false")); $("#chNote").hidden = true; draw(); }));
  apply(SCENARIOS[0]);
}

/* ===================== 9 · LITERATURE ===================== */
function initLit() {
  let tag = "All", q = ""; const table = $("#litTable");
  const tags = $("#litTags");
  ["All", "World", "Lithuania"].forEach(t => { const b = html(`<button class="btn small" type="button" aria-pressed="${t === tag}">${t}</button>`); b.addEventListener("click", () => { tag = t; $$("button", tags).forEach(x => x.setAttribute("aria-pressed", x.textContent === t)); draw(); }); tags.appendChild(b); });
  const draw = () => {
    const rows = LIT.filter(r => (tag === "All" || r.tag === tag) && (!q || (r.a + r.m + r.t + r.r).toLowerCase().includes(q)));
    table.innerHTML = `<thead><tr><th>Authors (year)</th><th>Model / approach</th><th>Task, data</th><th>Main result / relevance</th></tr></thead><tbody>${rows.map(r => `<tr><td><b>${esc(r.a)}</b> ${r.tag === "Lithuania" ? '<span class="chip fam-recurrent">LT</span>' : ""}</td><td>${esc(r.m)}</td><td>${esc(r.t)}</td><td>${esc(r.r)}</td></tr>`).join("") || `<tr><td colspan="4" class="muted">No match.</td></tr>`}</tbody>`;
  };
  $("#litSearch").addEventListener("input", debounce(e => { q = e.target.value.trim().toLowerCase(); draw(); }, 120));
  draw();
  const tl = $("#ltTimeline"); tl.style.gridTemplateColumns = "repeat(auto-fit, minmax(210px, 1fr))";
  LT_TIMELINE.forEach(([y, t, d]) => tl.appendChild(html(`<div class="card flat"><div class="eyebrow">${y}</div><b>${t}</b><p class="small muted" style="margin:.4em 0 0">${d}</p></div>`)));
}

/* ===================== 10 · DEFEND ===================== */
function initDefend() {
  const cats = ["All", ...new Set(QA.map(x => x[0]))]; let cat = "All", q = "";
  const pills = $("#qaCats"), list = $("#qaList");
  cats.forEach(c => { const b = html(`<button class="btn small" type="button" aria-pressed="${c === cat}">${c}</button>`); b.addEventListener("click", () => { cat = c; $$("button", pills).forEach(x => x.setAttribute("aria-pressed", x.textContent === c)); draw(); }); pills.appendChild(b); });
  const draw = () => {
    const rows = QA.filter(([c, qq, a]) => (cat === "All" || c === cat) && (!q || (qq + resolve(a)).toLowerCase().includes(q)));
    $("#qaCount").textContent = `${rows.length} questions.`;
    list.innerHTML = rows.map(([c, qq, a]) => `<details><summary><span><span class="qa-cat">${c}</span><br><span class="qa-q">${qq}</span></span></summary><div>${resolve(a)}</div></details>`).join("") || `<p class="muted">No question matches.</p>`;
  };
  $("#qaSearch").addEventListener("input", debounce(e => { q = e.target.value.trim().toLowerCase(); draw(); }, 120)); draw();

  // quiz
  let i = 0, score = 0, answered = false; const body = $("#quizBody");
  const best = () => store.get("quizBest", null);
  const showBest = () => { const b = best(); $("#quizBest").textContent = b == null ? "" : `Best score: ${b}/${QUIZ.length}`; };
  const next = () => {
    if (i >= QUIZ.length) {
      const prev = best(); if (prev == null || score > prev) store.set("quizBest", score); showBest();
      body.innerHTML = `<div class="stat ${score >= QUIZ.length * .8 ? "t" : "r"}"><b>${score} / ${QUIZ.length}</b><span>${score === QUIZ.length ? "Perfect. You are ready." : score >= QUIZ.length * .8 ? "Very good. Review the explanations you missed." : "Re-read the pipeline and theory sections, then try again."}</span></div><p><button class="btn primary" id="quizAgain" type="button">Try again</button></p>`;
      $("#quizAgain").addEventListener("click", () => { i = 0; score = 0; next(); }); $("#quizProgress").textContent = "Finished"; return;
    }
    const qq = QUIZ[i], opts = resolve(qq.o); answered = false;
    $("#quizProgress").textContent = `Question ${i + 1} of ${QUIZ.length} · score ${score}`;
    body.innerHTML = `<p class="qa-q" style="margin-top:12px">${esc(qq.q)}</p><div id="quizOpts">${opts.map((o, k) => `<button class="quiz-opt" type="button" data-k="${k}">${esc(o)}</button>`).join("")}</div><div id="quizFb"></div>`;
    $$(".quiz-opt", body).forEach(b => b.addEventListener("click", () => {
      if (answered) return; answered = true; const k = +b.dataset.k, ok = k === qq.a; if (ok) score++;
      $$(".quiz-opt", body).forEach(x => { const kk = +x.dataset.k; x.classList.toggle("right", kk === qq.a); x.classList.toggle("wrong", kk === k && !ok); x.disabled = true; });
      $("#quizFb").innerHTML = `<div class="callout ${ok ? "tip" : "warn"}"><b>${ok ? "Correct" : "Not quite"}</b>${resolve(qq.e)}</div><button class="btn primary" id="quizNext" type="button">${i === QUIZ.length - 1 ? "See result" : "Next question"}</button>`;
      $("#quizNext").addEventListener("click", () => { i++; next(); }); $("#quizNext").focus();
      $("#quizProgress").textContent = `Question ${i + 1} of ${QUIZ.length} · score ${score}`;
    }));
  };
  showBest(); next();
  const lim = $("#limits"); LIMITS.forEach(([t, d]) => lim.appendChild(html(`<div class="card flat"><b>${esc(t)}</b><p class="small muted" style="margin:.3em 0 0">${esc(d)}</p></div>`)));
}

/* ===================== 11 · PRESENT ===================== */
function initPresent() {
  const total = SLIDES.reduce((a, s) => a + s.s, 0);
  const mmss = s => `${Math.floor(Math.abs(s) / 60)}:${String(Math.round(Math.abs(s)) % 60).padStart(2, "0")}`;
  $("#timeline").innerHTML = SLIDES.map(s => `<div class="sp-${s.who}" style="flex:${s.s}" title="Slide ${s.n}: ${esc(s.t)} (${mmss(s.s)})">${s.n}</div>`).join("");
  let cum = 0; const ends = SLIDES.map(s => (cum += s.s));
  const rows = $("#slideRows");
  rows.innerHTML = SLIDES.map((s, k) => `<div class="slide-row" id="sr${k}"><span class="n">${s.n}</span><span><b>${esc(s.t)}</b> <span class="chip sp-${s.who}" style="color:#fff">${SPEAKERS[s.who]}</span><br><span class="muted small">${esc(s.key)}</span></span><span class="small muted" style="text-align:right">plan ${mmss(s.s)}<br><b id="act${k}"></b></span></div>`).join("");
  let t0 = 0, acc = 0, running = false, timer = null, cur = 0, marks = [];
  const elapsed = () => acc + (running ? (Date.now() - t0) / 1000 : 0);
  const paint = () => {
    const e = elapsed(); $("#clock").textContent = mmss(e); $("#clock").style.color = e > 600 ? "var(--bad)" : e > 540 ? "var(--amber)" : "inherit";
    SLIDES.forEach((s, k) => { const r = $("#sr" + k); r.classList.toggle("now", running && k === cur); r.classList.toggle("done", k < cur); });
    let msg;
    if (!running && e === 0) msg = "Press Start and speak. “Next slide” records your own pace.";
    else if (cur >= SLIDES.length) msg = `Finished in ${mmss(e)}. ${e <= 600 ? (e >= 420 ? "Within the 7–10 minute window." : "A bit short: add depth.") : "Over 10 minutes: shorten."}`;
    else { const s = SLIDES[cur], diff = e - ends[cur]; msg = `Slide ${s.n} · ${SPEAKERS[s.who]} · planned until ${mmss(ends[cur])}. ${diff > 0 ? `You are ${mmss(diff)} over this slide’s plan.` : `${mmss(-diff)} left in the plan for this slide.`}`; }
    $("#clockMsg").textContent = msg;
  };
  $("#clockStart").addEventListener("click", () => {
    if (running) { acc = elapsed(); running = false; clearInterval(timer); $("#clockStart").textContent = "Resume"; }
    else { t0 = Date.now(); running = true; timer = setInterval(paint, 250); $("#clockStart").textContent = "Pause"; }
    paint();
  });
  $("#clockNext").addEventListener("click", () => {
    if (!running || cur >= SLIDES.length) return;
    const e = elapsed(); marks[cur] = e; $("#act" + cur).textContent = `you ${mmss(e - (cur ? marks[cur - 1] : 0))}`; cur++;
    if (cur >= SLIDES.length) { acc = e; running = false; clearInterval(timer); $("#clockStart").textContent = "Start"; }
    paint();
  });
  $("#clockReset").addEventListener("click", () => { running = false; clearInterval(timer); acc = 0; cur = 0; marks = []; $$("[id^=act]").forEach(x => (x.textContent = "")); $("#clockStart").textContent = "Start"; paint(); });
  paint();
  // checklist
  const done = new Set(store.get("check", [])), ul = $("#checklist");
  CHECKLIST.forEach((t, i) => { const li = html(`<li style="margin:6px 0"><label style="display:flex; gap:10px; align-items:flex-start; cursor:pointer"><input type="checkbox" ${done.has(i) ? "checked" : ""} style="margin-top:.4em; width:18px; height:18px; accent-color:var(--violet)"><span>${esc(t)}</span></label></li>`); $("input", li).addEventListener("change", e => { e.target.checked ? done.add(i) : done.delete(i); store.set("check", [...done]); }); ul.appendChild(li); });
}

/* ===================== 12 · FILES ===================== */
function initFiles() {
  const tree = $("#tree"), info = $("#fileInfo"); let cur = "code/04_finetune_transformer.py";
  const show = p => {
    cur = p; const f = FILES.find(x => x.p === p);
    $$("button", tree).forEach(b => (b.style.fontWeight = b.dataset.p === p ? "800" : "400"));
    info.innerHTML = `<code>${esc(p)}</code><p style="margin-top:10px">${f.d}</p>${f.say ? `<div class="say">${esc(f.say)}</div>` : ""}<p class="small muted"><a href="https://github.com/shashikanth077/Transformers-NLP/${f.dir ? "tree" : "blob"}/master/${p.replace(/\/$/, "")}" target="_blank" rel="noopener">Open on GitHub</a></p>`;
  };
  FILES.forEach(f => { const depth = f.p.replace(/\/$/, "").split("/").length - 1; const b = html(`<div style="padding-left:${depth * 18}px"><button type="button" data-p="${esc(f.p)}">${f.dir ? "▸ " : "· "}${esc(f.p.replace(/\/$/, "").split("/").pop())}${f.dir ? "/" : ""}</button></div>`); $("button", b).addEventListener("click", () => show(f.p)); tree.appendChild(b); });
  show(cur);
  const cm = $("#cmds"); CMDS.forEach(([t, c]) => { cm.appendChild(html(`<h4>${esc(t)}</h4>`)); cm.appendChild(codeBlock("terminal", c)); });
  $("#paperMap").innerHTML = `<thead><tr><th>Topic</th><th>Paper</th><th>This guide</th></tr></thead><tbody>${PAPER_MAP.map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

/* ===================== 13 · GLOSSARY ===================== */
function initGlossary() {
  const box = $("#glossList");
  const draw = q => { const rows = GLOSSARY.filter(([t, d]) => !q || (t + d).toLowerCase().includes(q)); box.innerHTML = rows.map(([t, d]) => `<div class="card flat"><b>${esc(t)}</b><p class="muted small" style="margin:.3em 0 0">${esc(d)}</p></div>`).join("") || `<p class="muted">No term matches.</p>`; };
  $("#glossSearch").addEventListener("input", debounce(e => draw(e.target.value.trim().toLowerCase()), 120)); draw("");
}
