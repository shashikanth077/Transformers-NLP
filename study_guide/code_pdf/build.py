"""Builds docs/Transformer_Coding_Walkthrough.pdf: the Transformer encoder/decoder flow explained as code, step by step,
using the real scripts in code/transformer_walkthrough and their real captured outputs in code/results.
Run: python study_guide/code_pdf/build.py   (needs Microsoft Edge for HTML -> PDF)"""
import html, os, re, subprocess, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
CODE_DIR = os.path.join(ROOT, "code", "transformer_walkthrough"); RES = os.path.join(ROOT, "code", "results")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
OUT_PDF = os.path.join(ROOT, "docs", "Transformer_Coding_Walkthrough.pdf")
esc = html.escape


def read_lines(path): return open(path, encoding="utf-8").read().split("\n")


class Script:
    def __init__(self, fname, log):
        self.name = fname; self.lines = read_lines(os.path.join(CODE_DIR, fname)); self.segs = []; cur = None
        for i, ln in enumerate(self.lines, 1):
            m = re.match(r"# -{10,} STEP ([\d\-]+)(?:: (.*))?", ln)
            if m: cur = {"key": m.group(1), "title": m.group(2) or "", "start": i, "lines": []}; self.segs.append(cur)
            if cur is not None: cur["lines"].append((i, ln))
        self.head = [(i, l) for i, l in enumerate(self.lines, 1) if i < (self.segs[0]["start"] if self.segs else 1)]
        self.out = {}; key = None
        for ln in open(os.path.join(RES, log), encoding="utf-8").read().split("\n"):
            m = re.match(r"STEP ([\d\-]+)", ln)
            if m: key = m.group(1); self.out.setdefault(key, [])
            if key and ln.strip(): self.out[key].append(ln)

    def seg(self, key): return next(s for s in self.segs if s["key"] == key)

    def find(self, key, snippet):
        for i, ln in self.seg(key)["lines"]:
            if snippet in ln: return i
        for i, ln in enumerate(self.lines, 1):
            if snippet in ln and i < self.seg(key)["start"]: return i
        raise KeyError(f"{self.name} STEP {key}: {snippet}")


KW = r"\b(def|class|return|import|from|for|in|if|else|elif|with|as|and|or|not|assert|lambda|break|True|False|None|print|range|self|super|zip|len|sum|max|min|all|torch|nn|F|math|re|json|random)\b"


def hl(line):
    out = []; pos = 0
    pat = re.compile(r"(#.*$)|(\"\"\".*?\"\"\"|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')|(\b\d[\d_.e\-]*\b)|" + KW)
    for m in pat.finditer(line):
        out.append(esc(line[pos:m.start()]))
        t = esc(m.group(0))
        if m.group(1): out.append(f'<span class="c">{t}</span>')
        elif m.group(2): out.append(f'<span class="s">{t}</span>')
        elif m.group(3): out.append(f'<span class="n">{t}</span>')
        else: out.append(f'<span class="k">{t}</span>' if m.group(4) not in ("self", "torch", "nn", "F", "math", "re", "json", "random", "print", "range", "zip", "len", "sum", "max", "min", "all", "super") else f'<span class="b">{t}</span>')
        pos = m.end()
    out.append(esc(line[pos:])); return "".join(out)


def code_block(items, title=None):
    rows = "".join(f'<tr><td class="ln">{i}</td><td class="cd">{hl(l) if l.strip() else "&nbsp;"}</td></tr>' for i, l in items if not re.match(r"# -{10,} STEP", l))
    return f'<div class="code">{f"<div class=ct>{esc(title)}</div>" if title else ""}<table>{rows}</table></div>'


def out_block(lines):
    if not lines: return ""
    return '<div class="out"><div class="ct">Output you get (real run)</div><pre>' + esc("\n".join(lines)) + "</pre></div>"


def notes(script, key, items):
    li = "".join(f"<li><b>Line {script.find(key, sn)}:</b> {txt}</li>" for sn, txt in items)
    return f'<div class="notes"><b>Reading the code</b><ul>{li}</ul></div>'


def box(kind, head, body): return f'<div class="box {kind}"><b class="h">{head}</b>{body}</div>'


A = Script("scratch_transformer.py", "scratch_transformer_log.txt")
B = Script("nllb_step_by_step.py", "nllb_step_by_step_log.txt")
P = []


def step_page(script, keys, num, title, what, notes_html, watch=None, pre=None, outkeys=None, partlabel=""):
    items = []
    if script is B and keys == ["1"]: items += [x for x in script.head if x[0] > 6]
    for k in keys: items += script.seg(k)["lines"]
    o = []
    for k in (outkeys or keys): o += script.out.get(k, [])
    cb = code_block(items, f"{script.name}")
    if len(items) > 28: cb = cb.replace('<div class="code">', '<div class="code sm">', 1)
    P.append(f'<div class="page"><div class="eyebrow">{partlabel}</div><h2>{num}. {title}</h2>' + f'<p class="what">{what}</p>' + cb + out_block(o) + notes_html + (pre or "") + (box("wn", "Watch out", watch) if watch else "") + "</div>")


# ------------------------------------------------------------------ cover and overview
P.append("""<div class="page cover"><div class="tag">Coding walkthrough</div><h1>The Transformer in Code:<br>Start to End</h1>
<h2>How a translation Transformer is built, trained and used, one step of code at a time, with the real output of every step.</h2>
<div class="ex"><b>Two programs, both run for this guide</b><br><b>Part A</b> a complete encoder-decoder Transformer written from scratch in PyTorch (176,896 parameters), trained with teacher forcing on an English-Kannada phrasebook.<br><b>Part B</b> the same flow with a real pretrained model, NLLB-200, using Hugging Face Transformers.</div>
<p style="margin-top:10mm;color:#dcdcf6;max-width:150mm">Every code listing is the real file from the repository (with its line numbers), and every output block was copied from an actual run.</p>
<div class="foot">Files: code/transformer_walkthrough/scratch_transformer.py and nllb_step_by_step.py · logs in code/results/ · Python 3.11, PyTorch 2.14, Transformers 5.20</div></div>""")

P.append(f"""<div class="page"><h2>How to run it, and the map of the code</h2>
<h3>Setup and commands</h3>
<div class="code"><div class="ct">terminal</div><table>
<tr><td class="cd"><span class="c"># one-time: create an environment and install the libraries</span></td></tr>
<tr><td class="cd">python -m venv .venv</td></tr><tr><td class="cd">.venv\\Scripts\\activate          <span class="c"># Linux/macOS: source .venv/bin/activate</span></td></tr>
<tr><td class="cd">pip install torch transformers</td></tr><tr><td class="cd">&nbsp;</td></tr>
<tr><td class="cd"><span class="c"># Part A: from scratch (about 50 seconds on a laptop CPU, no GPU needed)</span></td></tr>
<tr><td class="cd">python code/transformer_walkthrough/scratch_transformer.py</td></tr><tr><td class="cd">&nbsp;</td></tr>
<tr><td class="cd"><span class="c"># Part B: real pretrained model (downloads about 2.4 GB the first time)</span></td></tr>
<tr><td class="cd">python code/transformer_walkthrough/nllb_step_by_step.py</td></tr></table></div>
<h3>The pipeline and where each step lives in the code</h3>
<table class="t"><thead><tr><th>#</th><th>Step</th><th>Part A: from scratch</th><th>Part B: Hugging Face</th></tr></thead><tbody>
<tr><td>1</td><td>Collect aligned pairs</td><td><code>json.load(corpus.json)</code></td><td>(the model was already trained on large data)</td></tr>
<tr><td>2</td><td>Tokenize, split, vocabulary</td><td><code>tokenize</code>, <code>build_vocab</code>, <code>encode_src/tgt</code></td><td><code>AutoTokenizer</code></td></tr>
<tr><td>3</td><td>Batches, padding, masks</td><td><code>make_batch</code>, <code>src_mask_of</code>, <code>tgt_mask_of</code></td><td><code>attention_mask</code>, <code>padding=True</code></td></tr>
<tr><td>4</td><td>Embeddings + position</td><td><code>PositionalEncoding</code>, <code>nn.Embedding</code></td><td>inside the model</td></tr>
<tr><td>5-6</td><td>Attention</td><td><code>attention</code>, <code>MultiHeadAttention</code></td><td><code>output_attentions=True</code>; head check by hand</td></tr>
<tr><td>7-8</td><td>Encoder and decoder layers</td><td><code>EncoderLayer</code>, <code>DecoderLayer</code></td><td><code>get_encoder()</code>, <code>get_decoder()</code></td></tr>
<tr><td>9-10</td><td>Whole model, shape trace</td><td><code>Transformer.encode / decode / forward</code></td><td><code>memory = encoder(...)</code></td></tr>
<tr><td>11-12</td><td>Loss and training</td><td><code>run_epoch</code>, <code>F.cross_entropy</code>, <code>AdamW</code></td><td>(pretrained; not trained here)</td></tr>
<tr><td>13</td><td>Decode</td><td><code>translate</code> (greedy loop)</td><td>manual loop, <code>generate()</code>, beam search</td></tr>
<tr><td>14</td><td>Evaluate</td><td><code>corpus_bleu</code></td><td>(see the NMT page for BLEU on NLLB)</td></tr></tbody></table>
<div class="box pl"><b class="h">The data flow in one line</b>text → token ids → (batch, padded) → <b>embeddings + position</b> → <b>encoder layers</b> → memory → <b>decoder layers</b> (masked self-attention, cross-attention to the memory) → logits over the Kannada vocabulary → softmax → loss while training, or argmax and loop while translating.</div>
<div class="box wn"><b class="h">About the numbers</b>Part A trains on just 50 sentence pairs and tests on 10, so its scores show that the code <em>works</em>, not how good a production translator would be. Part B is a real 615-million-parameter model.</div></div>""")

PL = "Part A · Transformer from scratch (PyTorch)"
# ------------------------------------------------------------------ PART A
step_page(A, ["1"], 1, "Collect the aligned sentence pairs",
          "A translation model learns from pairs: an English sentence and its Kannada translation. The pairs live in one JSON list, so index <i>i</i> keeps the source and its translation together; never sort or shuffle the two sides separately.",
          notes(A, "1", [("json.load", "<code>encoding='utf-8'</code> matters: Kannada text must be read as UTF-8 or it becomes garbage."), ("print(\"STEP 1", "Always print a sample pair first: it proves the two sides are aligned.")]),
          "If sources and targets get out of order (for example sorting only one list), the model silently learns nonsense and the loss still goes down.", partlabel=PL)
step_page(A, ["2"], 2, "Tokenize, split and build the vocabulary",
          "Text becomes numbers in three moves. (1) <b>Tokenize</b>: cut a sentence into tokens. (2) <b>Split</b> the pairs into train, validation and test. (3) <b>Build the vocabulary from the training pairs only</b> and map every token to an integer id; unseen words become <code>&lt;unk&gt;</code>. The encoders add <code>&lt;/s&gt;</code> to the source and <code>&lt;s&gt;</code> … <code>&lt;/s&gt;</code> around the target.",
          notes(A, "2", [("def tokenize", "A regular expression: words, with <code>? . , !</code> split off as separate tokens. It works for Kannada because it never relies on letter classes."), ("rng.shuffle(order)", "A seeded shuffle: the split is reproducible."), ("all(count[w] > 1", "Coverage rule: a pair may go to validation/test only if every word still occurs in training (this tiny corpus has no subword tokenizer)."), ("itos = [", "Index 0-3 are reserved: pad, start, end, unknown. The rest are sorted by frequency."), ("encode_tgt = lambda", "Target = <code>&lt;s&gt;</code> + tokens + <code>&lt;/s&gt;</code>. The start token is what the decoder reads first; the end token is what it must learn to predict.")]),
          "Building the vocabulary on all data (including test) leaks information. Forgetting the end token means the model never learns when to stop.", partlabel=PL)
step_page(A, ["3"], 3, "Batches, padding and masks",
          "Sentences differ in length, so a batch is padded with a pad id to one rectangle. Two masks tell attention what it may look at: the <b>source mask</b> hides the padding, and the <b>target mask</b> hides padding <em>and the future</em> (a lower-triangular matrix). The <b>teacher-forcing shift</b> also happens here: the decoder input is the target without its last token, the training target is the target without its first.",
          notes(A, "3", [("pad = lambda", "Pad every sequence to the longest in the batch."), ("return src, tgt[:, :-1]", "The shift: input <code>&lt;s&gt; y1 y2</code>, target <code>y1 y2 &lt;/s&gt;</code>. At position <i>t</i> the decoder sees the words before <i>t</i> and must predict word <i>t</i>."), ("(src != PAD)[:, None, None, :]", "Shape (batch, 1, 1, S) so it broadcasts over heads and query positions. True means <em>may attend</em>."), ("torch.tril", "Lower triangle of ones: row <i>i</i> may look at columns ≤ <i>i</i>. Printed below as the matrix of 1s and 0s.")]),
          "Mask polarity is the classic bug: here True = allowed, and the attention code uses <code>~mask</code> to block. Mixing the two conventions makes the model attend only to padding.", partlabel=PL)
step_page(A, ["4"], 4, "Embeddings and positional encoding",
          "An <code>nn.Embedding</code> turns each token id into a learned vector of <i>d</i> numbers. Attention looks at all words at once, so order must be added by hand: a fixed sine and cosine pattern is added to the embedding of each position. The class below builds that table once and stores it as a buffer (saved with the model, never trained).",
          notes(A, "4", [("angle = pos / (10000 ** (k / d))", "The formula <i>pos</i> / 10000<sup>2<i>k</i>/<i>d</i></sup>: low dimensions oscillate fast, high dimensions slowly."), ("pe[:, 0::2] = torch.sin", "Even columns sine, odd columns cosine."), ("register_buffer", "Part of the module's state but not a parameter: the optimiser never touches it."), ("return x + self.pe", "Added, not concatenated, so the width stays <i>d</i>.")]),
          "The model code multiplies embeddings by √d before adding position (see step 9); without that scale the position signal swamps small embeddings.", partlabel=PL)
step_page(A, ["5"], 5, "Scaled dot-product attention",
          "The heart of the model in four lines: score every query against every key, divide by √<i>d<sub>k</sub></i>, block forbidden positions with −∞, softmax into weights that sum to 1, and take the weighted average of the values. Shapes: queries (B, h, Tq, dk), keys and values (B, h, Tk, dk).",
          notes(A, "5", [("scores = Q @ K.transpose", "<code>Q·Kᵀ</code>: one score per (query position, key position) pair, shape (B, h, Tq, Tk), divided by √dk."), ("masked_fill(~mask", "Blocked positions get −∞, so softmax gives them exactly 0 weight."), ("F.softmax(scores, dim=-1)", "Softmax over the key axis: each query's weights sum to 1."), ("return weights @ V", "Weighted average of the values, shape (B, h, Tq, dk); the weights are returned too so they can be inspected.")]),
          "A row whose keys are <em>all</em> masked gives NaN (softmax of all −∞). The masks here always leave at least one real key per row.", partlabel=PL, outkeys=[])
step_page(A, ["6"], 6, "Multi-head attention",
          "One attention can only learn one kind of relationship, so the model runs <i>h</i> attentions in parallel on slices of the vector. Four linear layers do the work: <code>Wq, Wk, Wv</code> project the input, the result is split into heads, attended, joined, and mixed by <code>Wo</code>. The same class serves self-attention (<code>q_in</code> and <code>kv_in</code> are the same tensor) and cross-attention (queries from the decoder, keys and values from the encoder memory).",
          notes(A, "6", [("assert d % heads", "The width must divide evenly across heads: 64 / 4 = 16 numbers per head."), ("return x.view(B, L, self.h, self.dk).transpose(1, 2)", "Reshape (B, L, d) to (B, L, h, dk), then move the head axis forward: (B, h, L, dk)."), ("Q, K, V = self.split", "Queries come from <code>q_in</code>, keys and values from <code>kv_in</code>: this one line is the difference between self- and cross-attention."), ("self.last_weights", "Stored so the attention maps can be examined after a forward pass."), ("self.Wo(out.transpose", "Heads are joined back to width <i>d</i> and mixed by <code>Wo</code>.")]),
          "<code>.view</code> needs contiguous memory; after <code>transpose</code> use <code>.reshape</code> (as the code does when joining heads).", partlabel=PL, outkeys=[])
items7 = A.seg("7")["lines"]
P.append('<div class="page"><div class="eyebrow">' + PL + '</div><h2>7 and 8. The encoder layer and the decoder layer</h2>' +
         '<p class="what">Each layer is attention plus a small feed-forward network, each wrapped in a skip connection (<code>x + sub_layer(x)</code>) and a LayerNorm. The <b>encoder layer</b> has one self-attention. The <b>decoder layer</b> has three sub-layers: masked self-attention (looks at the target so far), <b>cross-attention</b> (queries from the decoder, keys and values from the encoder memory) and a feed-forward network. LayerNorm comes before each sub-layer (pre-norm), as in the real NLLB model.</p>' +
         code_block(items7, A.name) + notes(A, "7", [("self.drop(self.attn(h, h, src_mask))", "Self-attention: the same normalised tensor <code>h</code> is query, key and value."), ("x = x + self.drop", "The skip connection: the sub-layer output is <em>added</em> to its input, which keeps gradients flowing through deep stacks."), ("self.self_attn(h, h, tgt_mask)", "Masked self-attention: <code>tgt_mask</code> hides future words."), ("self.cross_attn(self.ln2(y), memory, src_mask)", "Cross-attention: query from the decoder state <code>y</code>, keys and values from <code>memory</code>; <code>src_mask</code> hides source padding."), ("nn.Linear(d, ff), nn.ReLU()", "Feed-forward: widen to <code>ff</code>, ReLU, narrow back to <i>d</i>; applied to each position alone.")]) + box("wn", "Watch out", "Cross-attention must receive the <em>encoder's final output</em> (<code>memory</code>), the same tensor in every decoder layer, and the source mask, not the target mask.") + "</div>")
step_page(A, ["9"], 9, "Assemble the whole model",
          "The model class wires the pieces together and exposes two separate methods, because they run at different times: <code>encode</code> runs <b>once</b> per source sentence and returns the memory; <code>decode</code> runs once per training batch, or once per generated token when translating. The output layer reuses (ties) the target embedding matrix, which saves parameters.",
          notes(A, "9", [("class Transformer", "Defaults: width 64, 4 heads, 2 layers, feed-forward 128: small enough to train on a laptop in a minute."), ("nn.init.normal_(emb.weight", "Small embeddings (standard deviation 1/√d). See the box below: skipping this line caused a real bug."), ("self.out.weight = self.tgt_emb.weight", "Weight tying: the output scores use the same table as the input embeddings."), ("def encode", "Source ids → embeddings × √d → position → encoder layers → final LayerNorm = memory."), ("def decode", "Target ids → embeddings × √d → position → decoder layers (given memory) → LayerNorm → logits (B, T, vocabulary)."), ("torch.randn(2, 4, 5, 16)", "A correctness test: our <code>attention</code> against PyTorch's built-in <code>scaled_dot_product_attention</code> on random tensors; the printed difference is rounding noise.")]),
          None, partlabel=PL,
          pre=box("rw", "A real bug we hit while writing this", "The first run printed <b>train loss 62.1</b> in epoch 1, when an untrained 70-word model should start near ln 70 ≈ 4.25. Cause: <code>nn.Embedding</code> starts with standard deviation 1, and with tied weights the output scores became huge. Adding the <code>nn.init.normal_(…, std=d ** -0.5)</code> line fixed it (epoch 1 loss then printed 4.59). <b>Lesson: always check that the first loss is close to ln(vocabulary size).</b>"))
step_page(A, ["10"], 10, "Trace the shapes through one sentence",
          "Before training, push one sentence through and print every tensor shape. This catches most wiring mistakes in a minute. Read the output as the journey of the data: ids → embedding → encoder memory → decoder logits.",
          notes(A, "10", [("model.eval()", "Evaluation mode switches dropout off so the trace is deterministic."), ("torch.no_grad()", "No gradients are stored: faster and lighter for inference."), ("model.encode(src)", "The memory has the same length as the source and width <i>d</i>."), ("last_weights.shape", "Cross-attention maps are (batch, heads, target length, source length): one row of weights over the source for every target position.")]),
          "If a shape is wrong, the error appears here, long before a training run. Check (batch, heads, Tq, Tk) in particular.", partlabel=PL)
step_page(A, ["11-12"], "11 and 12", "The loss with teacher forcing, and the training loop",
          "Because the decoder receives the correct previous words, a single forward pass produces predictions for <b>every</b> target position at once. The loss is the cross-entropy between those predictions and the shifted target, ignoring padding. Then the usual four moves: <code>zero_grad</code>, <code>backward</code>, clip the gradient, <code>step</code>. Validation loss is measured each epoch without learning, and the best checkpoint is kept (early stopping).",
          notes(A, "11-12", [("logits = model(src, tgt_in)", "One call, all positions: this is teacher forcing."), ("F.cross_entropy(logits.reshape", "Flatten (B, T, V) to (B·T, V) and targets to (B·T). <code>ignore_index=PAD</code> removes padded positions from the loss."), ("opt.zero_grad()", "The four-step update: clear old gradients, backpropagate, clip, step."), ("clip_grad_norm_", "Gradient clipping at 1.0 stops rare huge updates."), ("torch.set_grad_enabled(train_mode)", "The same function serves training and validation."), ("if va < best[0]", "Early stopping: remember the weights with the lowest validation loss and restore them afterwards.")]),
          "Call <code>model.train()</code> for training and <code>model.eval()</code> for validation (dropout!). Measure validation loss after every epoch: the training loss keeps falling, the validation loss tells you when the model starts memorising.", partlabel=PL, outkeys=["12"])
step_page(A, ["13"], 13, "Decode: translate a new sentence",
          "At test time there is no target to feed. <code>translate</code> runs the encoder <b>once</b>, then loops: feed the tokens produced so far to the decoder, take the scores for the <b>last</b> position, pick the highest (greedy), append it, stop at the end token. Each turn re-runs the decoder on the longer prefix; production systems cache the keys and values of earlier tokens to avoid that recomputation.",
          notes(A, "13", [("@torch.no_grad()", "Decorator: no gradients inside the function."), ("memory, sm = model.encode(src)", "Encoder outside the loop: the source never changes, so its memory is reused every turn."), ("model.decode(torch.tensor([out])", "The decoder gets <code>&lt;s&gt;</code> plus everything chosen so far."), ("logits[0, -1].argmax()", "Only the last position matters: its scores are for the next word. <code>argmax</code> = greedy."), ("if nxt == EOS: break", "Stop at the end token; <code>max_len</code> is a safety limit.")]),
          "Look at the output honestly: seen sentences are reproduced, most unseen ones are right, but one test sentence (<i>Where is the bus stand?</i>) comes out as a different, plausible phrase. A tiny training set cannot cover every pattern.", partlabel=PL)
step_page(A, ["14"], 14, "Evaluate with BLEU",
          "BLEU counts how many 1- to 4-word sequences of the candidate also occur in the reference (clipped so no word is credited more often than it appears there), multiplies the precisions geometrically, and penalises outputs shorter than the reference. The function below is the whole metric, written from scratch.",
          notes(A, "14", [("def ngrams", "Counts every n-word sequence."), ("min(v, rg.get(g, 0))", "Clipping: a candidate n-gram earns credit at most as many times as it occurs in the reference."), ("(match[i] + 1) / (total[i] + 1)", "Add-one smoothing for n &gt; 1, so one missing 4-gram does not zero the whole score on short sentences."), ("bp = 1.0 if c_len >= r_len", "Brevity penalty: too-short outputs are penalised by e<sup>1 − r/c</sup>."), ("def score", "Translate every sentence of a split, tokenize hypothesis and reference the same way, and score.")]),
          "The train score (100) only shows memorisation; the validation and test scores are the honest ones, and with 10 sentences each they are noisy. Use several thousand sentences and a standard tool (for example sacreBLEU) for real reports.", partlabel=PL)

# ------------------------------------------------------------------ PART B
PLB = "Part B · A real pretrained model with Hugging Face"
step_page(B, ["1"], 1, "Load the tokenizer and the model",
          "Two objects: the <b>tokenizer</b> (text ↔ token ids) and the <b>model</b>. <code>src_lang</code> and <code>tgt_lang</code> tell NLLB which language tags to use. <code>attn_implementation='eager'</code> selects the plain attention code, which can return its weights (the faster fused kernels cannot). <code>torch.set_grad_enabled(False)</code> turns gradients off globally: this program only runs the model.",
          notes(B, "1", [("torch.set_grad_enabled(False)", "Inference only: saves memory and time."), ("AutoTokenizer.from_pretrained", "Downloads and caches the vocabulary of 256,206 word pieces."), ("attn_implementation=\"eager\"", "Needed when you want <code>output_attentions=True</code> to work."), ("model.config", "The configuration object lists the architecture: 12 + 12 layers, width 1024, 16 heads.")]),
          "The first run downloads about 2.4 GB. If memory is short, load with <code>torch_dtype=torch.float16</code> on a GPU.", partlabel=PLB, pre="")
step_page(B, ["2"], 2, "Tokenize: text to ids",
          "The tokenizer cuts the sentence into pieces, adds the language tag <code>eng_Latn</code> at the start and <code>&lt;/s&gt;</code> at the end, and returns two tensors: <code>input_ids</code> and <code>attention_mask</code> (1 = real token, 0 = padding). The leading ▁ marks the start of a word.",
          notes(B, "2", [("tok(SENTENCE, return_tensors=\"pt\")", "<code>return_tensors='pt'</code> gives PyTorch tensors with a batch axis of size 1."), ("convert_ids_to_tokens", "Turns ids back into readable pieces: <i>Mys</i> + <i>uru</i> shows a rare name split in two.")]),
          None, partlabel=PLB)
step_page(B, ["3"], 3, "Run the encoder once",
          "<code>model.get_encoder()</code> returns the encoder stack. One call produces the memory: one vector of 1024 numbers for each of the 10 source pieces, already informed by the whole sentence. With <code>output_attentions=True</code> it also returns one attention map per layer.",
          notes(B, "3", [("get_encoder()(**enc", "<code>**enc</code> passes <code>input_ids</code> and <code>attention_mask</code> as keyword arguments."), ("last_hidden_state", "The final encoder output: the <b>memory</b> of shape (batch, S, 1024)."), ("encoder_out.attentions", "A tuple with one tensor per layer: (batch, heads, S, S).")]),
          None, partlabel=PLB)
step_page(B, ["4"], 4, "The decoder loop, written by hand",
          "This is what <code>generate()</code> does internally. Start with <code>&lt;/s&gt;</code> and the target-language tag; each turn call the decoder with the tokens so far and the encoder memory, take the scores of the <b>last</b> position through <code>lm_head</code>, softmax, pick the most probable piece, append it, and stop at <code>&lt;/s&gt;</code>. The printout shows the three best candidates at each turn, the real probabilities behind the translation.",
          notes(B, "4", [("ids = [start, KAN]", "NLLB forces the output language by putting its tag as the first generated token."), ("get_decoder()(input_ids", "<code>encoder_hidden_states=memory</code> is what the cross-attention reads; <code>encoder_attention_mask</code> hides source padding."), ("model.lm_head(dec.last_hidden_state[:, -1])", "Only the last position's vector is converted to 256,206 scores."), ("torch.softmax(logits.float()", "Scores to probabilities; <code>.float()</code> avoids precision problems in half precision."), ("if int(nxt) == tok.eos_token_id", "Stop at end of sequence.")]),
          "Re-running the whole decoder at every turn is correct but slow; <code>generate()</code> uses a key/value cache so earlier positions are not recomputed.", partlabel=PLB)
step_page(B, ["5"], 5, "The same thing with generate()",
          "<code>generate()</code> wraps the loop, adds the cache, and offers decoding strategies. <code>num_beams=1</code> is greedy and must reproduce the manual loop exactly (the printed <code>True</code> confirms it). <code>num_beams=4</code> is beam search: four partial translations are kept at each turn and the best-scoring complete one is returned.",
          notes(B, "5", [("forced_bos_token_id=KAN", "Same language forcing as in the manual loop."), ("num_beams=1, do_sample=False", "Greedy decoding."), ("num_beams=4", "Beam search; the sum of log-probabilities (length-normalised) ranks the finished hypotheses."), ("== manual", "A good habit: prove your hand-written loop equals the library.")]),
          None, partlabel=PLB)
step_page(B, ["6"], 6, "Inspect the attention",
          "Passing <code>output_attentions=True</code> to the decoder returns two sets of maps per layer: <b>decoder self-attention</b> (T×T, causal) and <b>cross-attention</b> (T×S, target looks at source). The code averages the 16 heads of the last layer and prints, for the pieces of the place name, which source piece gets the most weight and how much weight sits on the two marker tokens.",
          notes(B, "6", [("dec_in = torch.tensor([ids[:-1]])", "Teacher-force the produced output: feed everything except the last piece, as in training."), ("out.cross_attentions[-1][0].mean(0)", "Last layer, first batch item, mean over heads: a (T, S) matrix of percentages."), ("row[1:-1].argmax()", "Ignore the language tag and <code>&lt;/s&gt;</code> when asking which word is most attended.")]),
          "Averaged attention is a rough picture: weight often rests on the end marker, as the printout shows (0.51 to 0.90 on the markers). Do not over-interpret it.", partlabel=PLB)
step_page(B, ["7"], 7, "Verify one attention head by hand",
          "The scaled dot-product maths from Part A, applied to the real model: capture the input the cross-attention module receives (a forward pre-hook), project it with the module's own <code>q_proj</code> and <code>k_proj</code>, take the 64 numbers of head 1, compute <code>Q·Kᵀ/√64</code>, softmax, and compare with the weights the model returned. The difference is exactly zero.",
          notes(B, "7", [("register_forward_pre_hook", "A hook that copies the module's inputs while the model runs; <code>hook.remove()</code> unregisters it."), ("layer.q_proj(captured", "The model's own projection matrices applied to the captured activations."), ("view(-1, layer.num_heads, dk)[:, HEAD]", "Split 1024 numbers into 16 heads of 64 and pick head 0."), ("Q @ K.T / math.sqrt(dk)", "Exactly the formula of step 5 of Part A."), ("(mine - ref).abs().max()", "0.0: the textbook formula and the library agree.")]),
          None, partlabel=PLB)
step_page(B, ["8"], 8, "Batching and padding",
          "To translate many sentences at once, pad them to the same length with <code>padding=True</code>. The <code>attention_mask</code> marks the padding with 0, and the model uses it so padded positions are ignored: this is the source mask of Part A step 3, built for you. <code>batch_decode</code> turns the generated ids back into text.",
          notes(B, "8", [("padding=True", "Pads to the longest sentence in the batch."), ("attention_mask", "The printed zeros mark the two padding slots of the first sentence."), ("tok.batch_decode", "Decodes all outputs at once; <code>skip_special_tokens=True</code> removes tags and end markers.")]),
          "Always pass the attention mask (the <code>**batch</code> pattern does). Without it the padding is treated as real words and results change.", partlabel=PLB)

# ------------------------------------------------------------------ wrap-up pages
P.append("""<div class="page"><div class="eyebrow">Summary</div><h2>Every tensor, from start to end</h2>
<table class="t"><thead><tr><th>Stage</th><th>Variable (Part A)</th><th>Shape</th><th>Meaning</th></tr></thead><tbody>
<tr><td>Tokenized source</td><td><code>src</code></td><td>(B, S)</td><td>integer ids, padded with 0</td></tr>
<tr><td>Decoder input</td><td><code>tgt_in</code></td><td>(B, T)</td><td>&lt;s&gt; y1 … y(T−1), padded</td></tr>
<tr><td>Training target</td><td><code>tgt_out</code></td><td>(B, T)</td><td>y1 … yT &lt;/s&gt; (the input shifted left)</td></tr>
<tr><td>Source mask</td><td><code>src_mask_of(src)</code></td><td>(B, 1, 1, S)</td><td>True for real source tokens</td></tr>
<tr><td>Target mask</td><td><code>tgt_mask_of(tgt_in)</code></td><td>(B, 1, T, T)</td><td>lower triangle AND not padding</td></tr>
<tr><td>Embedded source</td><td><code>src_emb(src)</code></td><td>(B, S, d)</td><td>one vector per token (then × √d + position)</td></tr>
<tr><td>Queries, keys, values</td><td><code>Q, K, V</code></td><td>(B, h, L, d/h)</td><td>after the three linear layers and the head split</td></tr>
<tr><td>Attention scores / weights</td><td><code>scores</code>, <code>weights</code></td><td>(B, h, Tq, Tk)</td><td>weights sum to 1 over Tk</td></tr>
<tr><td>Encoder memory</td><td><code>memory</code></td><td>(B, S, d)</td><td>computed once; keys and values of every cross-attention</td></tr>
<tr><td>Decoder state</td><td><code>y</code></td><td>(B, T, d)</td><td>updated by 3 sub-layers per decoder layer</td></tr>
<tr><td>Logits</td><td><code>logits</code></td><td>(B, T, V)</td><td>one score per vocabulary word, per position</td></tr>
<tr><td>Loss</td><td><code>loss</code></td><td>scalar</td><td>mean cross-entropy over non-padding positions</td></tr></tbody></table>
<h3>Training versus translating, in code terms</h3>
<table class="t"><thead><tr><th></th><th>Training (teacher forcing)</th><th>Translating (the loop)</th></tr></thead><tbody>
<tr><td>Decoder input</td><td>correct target, shifted</td><td>its own previous outputs</td></tr><tr><td>Calls to <code>encode</code></td><td>once per batch</td><td>once per sentence</td></tr>
<tr><td>Calls to <code>decode</code></td><td>once per batch (all positions)</td><td>once per new token</td></tr><tr><td>Dropout, gradients</td><td><code>model.train()</code>, gradients on</td><td><code>model.eval()</code>, <code>no_grad</code></td></tr>
<tr><td>Result</td><td>a loss, then <code>backward()</code> and <code>step()</code></td><td>token ids, then <code>decode</code> back to text</td></tr></tbody></table>
<h3>Debugging checklist (each item was useful while writing this guide)</h3>
<table class="t"><tbody>
<tr><td><b>First loss ≈ ln(vocabulary)</b></td><td>Here ln 70 ≈ 4.25 and epoch 1 gave 4.59. A value like 62 means mis-scaled initialisation (see the bug box in step 9).</td></tr>
<tr><td><b>Print every shape once</b></td><td>Step 10. Wrong head or mask shapes show up immediately.</td></tr>
<tr><td><b>Compare with a reference</b></td><td>Our attention against <code>F.scaled_dot_product_attention</code> (difference 2e-7); our decode loop against <code>generate()</code>; our head maths against the model's weights (difference 0).</td></tr>
<tr><td><b>Overfit a tiny set first</b></td><td>If the model cannot reach near-zero training loss on 50 pairs (ours reached 0.002), something is broken.</td></tr>
<tr><td><b>Mask and shift sanity</b></td><td>Print the causal matrix (step 3). Check <code>tgt_in</code> and <code>tgt_out</code> are offset by exactly one.</td></tr>
<tr><td><b>train() vs eval()</b></td><td>Forgetting <code>eval()</code> leaves dropout on during decoding and gives noisy output.</td></tr>
<tr><td><b>Validation, not training, decides</b></td><td>Training loss fell to 0.002 while validation loss bottomed near 0.72; the best checkpoint is restored.</td></tr></tbody></table></div>""")

HTML_CSS = """
@page { size: A4; margin: 12mm 12mm 13mm 12mm; @bottom-center { content: counter(page); font: 9pt 'Segoe UI', sans-serif; color: #777; } }
@page :first { margin: 0; @bottom-center { content: none; } }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; font: 9.6pt/1.5 'Segoe UI', 'Nirmala UI', sans-serif; color: #1d1d3a; }
.page { page-break-after: always; break-after: page; } .page:last-child { page-break-after: auto; }
.eyebrow { font-size: 8.5pt; letter-spacing: .1em; text-transform: uppercase; color: #6250d6; font-weight: 700; margin-bottom: .5mm; }
h2 { font-size: 15pt; margin: 0 0 2mm; color: #2e2a6b; border-bottom: 2px solid #d9dcef; padding-bottom: 1.2mm; } h3 { font-size: 11pt; margin: 4mm 0 1.2mm; color: #6250d6; }
p { margin: 0 0 2mm; } p.what { font-size: 9.8pt; }
code { font-family: Consolas, 'Cascadia Mono', monospace; font-size: .88em; background: #f1f1f9; padding: 0 .3em; border-radius: 3px; }
.cover { height: 297mm; padding: 28mm 20mm 18mm; background: linear-gradient(160deg, #16163f 0%, #2e2a6b 60%, #3a3590 100%); color: #fff; position: relative; overflow: hidden; }
.cover .tag { letter-spacing: .14em; text-transform: uppercase; font-size: 9.5pt; color: #b9b8e8; margin-bottom: 12mm; } .cover h1 { font-size: 34pt; line-height: 1.12; margin: 0 0 6mm; } .cover h2 { font-size: 14.5pt; font-weight: 400; color: #dcdcf6; border: 0; margin: 0 0 12mm; line-height: 1.5; }
.cover .ex { background: rgba(255,255,255,.1); border-radius: 4mm; padding: 5mm 6mm; font-size: 11pt; max-width: 160mm; line-height: 1.6; } .cover .foot { position: absolute; left: 20mm; right: 20mm; bottom: 16mm; font-size: 9pt; color: #b9b8e8; border-top: 1px solid rgba(255,255,255,.25); padding-top: 4mm; }
.code { background: #f7f7fc; border: 1px solid #d9dcef; border-radius: 2.5mm; margin: 2mm 0 3mm; overflow: hidden; } .ct { background: #2e2a6b; color: #fff; font-size: 8pt; padding: .8mm 3mm; letter-spacing: .04em; }
.code table { border-collapse: collapse; width: 100%; } .code td { padding: 0; vertical-align: top; } td.ln { width: 9mm; text-align: right; padding-right: 2mm; color: #9a9ab5; font: 7pt/1.4 Consolas, monospace; border-right: 1px solid #e0e0ef; user-select: none; background: #f0f0f9; }
.code.sm td.cd { font-size: 6.8pt; } td.cd { padding-left: 2.5mm; font: 7.4pt/1.4 Consolas, 'Cascadia Mono', 'Nirmala UI', monospace; white-space: pre-wrap; word-break: break-word; } .code tr { break-inside: avoid; }
.c { color: #7a7a96; font-style: italic; } .s { color: #0c7a4d; } .n { color: #b3541e; } .k { color: #6250d6; font-weight: 700; } .b { color: #2a78d6; }
.out { background: #14142c; border-radius: 2.5mm; margin: 2mm 0 3mm; overflow: hidden; } .out .ct { background: #159a6a; } .out pre { margin: 0; padding: 2mm 3mm; color: #e8e8fa; font: 7.1pt/1.4 Consolas, 'Cascadia Mono', 'Nirmala UI', monospace; white-space: pre-wrap; word-break: break-word; }
.notes { border-left: 4px solid #6250d6; background: #f3f1fe; border-radius: 0 2.5mm 2.5mm 0; padding: 2mm 3.5mm; margin: 2mm 0; break-inside: avoid; } .notes ul { margin: 1mm 0 0; padding-left: 4.5mm; } .notes li { margin: 0 0 1mm; font-size: 9pt; }
.box { border-radius: 2.5mm; padding: 2.5mm 4mm; margin: 2.5mm 0; break-inside: avoid; } .box.wn { background: #fff3dc; border-left: 4px solid #c27a10; } .box.pl { background: #e6f6ee; border-left: 4px solid #159a6a; } .box.rw { background: #fff1e8; border-left: 4px solid #d9541f; }
.box b.h { display: block; font-size: 7.8pt; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .5mm; } .wn b.h { color: #c27a10; } .pl b.h { color: #159a6a; } .rw b.h { color: #d9541f; }
table.t { border-collapse: collapse; width: 100%; font-size: 8.7pt; margin: 2mm 0 3mm; } table.t th { background: #2e2a6b; color: #fff; text-align: left; padding: 1.3mm 2mm; } table.t td { padding: 1.2mm 2mm; border-bottom: 1px solid #d9dcef; vertical-align: top; } table.t tr { break-inside: avoid; } table.t tr:nth-child(even) td { background: #f8f8fd; }
"""
HTML = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>The Transformer in code</title><style>{HTML_CSS}</style></head><body>{''.join(P)}</body></html>"
html_path = os.path.join(HERE, "walkthrough.html"); open(html_path, "w", encoding="utf-8").write(HTML)
profile = tempfile.mkdtemp(prefix="edge_pdf_"); tmp_pdf = os.path.join(tempfile.gettempdir(), "transformer_code_tmp.pdf")
if os.path.exists(tmp_pdf): os.remove(tmp_pdf)
subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-first-run", "--no-pdf-header-footer", f"--print-to-pdf={tmp_pdf}", "--virtual-time-budget=10000", "file:///" + html_path.replace("\\", "/")], capture_output=True, text=True, timeout=240)
last = -1
for _ in range(60):
    if os.path.exists(tmp_pdf):
        sz = os.path.getsize(tmp_pdf)
        if sz > 0 and sz == last: break
        last = sz
    time.sleep(1)
os.replace(tmp_pdf, OUT_PDF); print("wrote", OUT_PDF, os.path.getsize(OUT_PDF) // 1024, "KB")
