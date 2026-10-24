"""A complete encoder-decoder Transformer written from scratch in PyTorch, trained with teacher forcing on the
English-Kannada phrasebook, then used to translate with greedy decoding. Every step prints what it does.

Run from the repository root:   python code/transformer_walkthrough/scratch_transformer.py
"""
import json, math, random, re
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); random.seed(0)
PAD, BOS, EOS, UNK = 0, 1, 2, 3          # special token ids

# ------------------------------------------------------------------ STEP 1: collect aligned pairs
pairs = json.load(open("study_guide/nmt/corpus.json", encoding="utf-8"))          # [{"en": ..., "kn": ...}, ...]
print("STEP 1  pairs:", len(pairs), "| example:", pairs[0]["en"], "->", pairs[0]["kn"])

# ------------------------------------------------------------------ STEP 2: tokenize, split, build vocabularies
def tokenize(text):
    """Words, with ? . , ! split off as their own tokens (works for Kannada: no letter classes needed)."""
    return re.findall(r"[^\s?.,!]+|[?.,!]", text)

# train / validation / test split (70 / 15 / 15), reproducible; every word of val/test must also occur in train
rng = random.Random(7)
order = list(range(len(pairs))); rng.shuffle(order)
words = lambda i: {("e", w.lower()) for w in tokenize(pairs[i]["en"])} | {("k", w) for w in tokenize(pairs[i]["kn"])}
count = {}
for i in order:
    for w in words(i): count[w] = count.get(w, 0) + 1
n_val = n_test = round(0.15 * len(pairs)); val, test = [], []
for i in order:
    if len(val) >= n_val and len(test) >= n_test: break
    if all(count[w] > 1 for w in words(i)):
        for w in words(i): count[w] -= 1
        (test if len(test) < n_test else val).append(i)
held = set(val) | set(test); train = [i for i in order if i not in held]
print(f"STEP 2  split: train {len(train)} | validation {len(val)} | test {len(test)}")

def build_vocab(sentences, lower):
    counts = {}
    for s in sentences:
        for w in tokenize(s.lower() if lower else s): counts[w] = counts.get(w, 0) + 1
    itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(counts, key=lambda w: (-counts[w], w))
    return itos, {w: i for i, w in enumerate(itos)}          # vocabulary comes from TRAINING data only

src_itos, src_stoi = build_vocab([pairs[i]["en"] for i in train], True)
tgt_itos, tgt_stoi = build_vocab([pairs[i]["kn"] for i in train], False)
encode_src = lambda s: [src_stoi.get(w, UNK) for w in tokenize(s.lower())] + [EOS]               # x_1 ... x_S </s>
encode_tgt = lambda s: [BOS] + [tgt_stoi.get(w, UNK) for w in tokenize(s)] + [EOS]               # <s> y_1 ... y_T </s>
print("        vocab sizes:", len(src_itos), "(English),", len(tgt_itos), "(Kannada)")
ex = pairs[train[0]]
print("        example ids:", encode_src(ex["en"]), "|", encode_tgt(ex["kn"]))

# ------------------------------------------------------------------ STEP 3: batches, padding and masks
def make_batch(idx):
    S = [encode_src(pairs[i]["en"]) for i in idx]; T = [encode_tgt(pairs[i]["kn"]) for i in idx]
    pad = lambda seqs: torch.tensor([s + [PAD] * (max(map(len, seqs)) - len(s)) for s in seqs])
    src, tgt = pad(S), pad(T)
    return src, tgt[:, :-1], tgt[:, 1:]                       # decoder input = tgt without last, target = tgt shifted left

def src_mask_of(src):                                         # (B,1,1,S) True where the key is a real token
    return (src != PAD)[:, None, None, :]

def tgt_mask_of(tgt_in):                                      # (B,1,T,T) real token AND not in the future
    T = tgt_in.size(1)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device))
    return causal[None, None] & (tgt_in != PAD)[:, None, None, :]

src, tgt_in, tgt_out = make_batch(train[:3])
print("STEP 3  src", tuple(src.shape), "tgt_in", tuple(tgt_in.shape), "tgt_out", tuple(tgt_out.shape))
print("        causal mask for T=4:\n", tgt_mask_of(torch.ones(1, 4, dtype=torch.long))[0, 0].int())

# ------------------------------------------------------------------ STEP 4: embeddings + positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=64):
        super().__init__()
        pos = torch.arange(max_len)[:, None]; k = torch.arange(0, d, 2)[None]
        angle = pos / (10000 ** (k / d))                      # pos / 10000^(2k/d)
        pe = torch.zeros(max_len, d); pe[:, 0::2] = torch.sin(angle); pe[:, 1::2] = torch.cos(angle)
        self.register_buffer("pe", pe)                        # fixed, not learned
    def forward(self, x):                                     # x: (B, L, d)
        return x + self.pe[: x.size(1)]

# ------------------------------------------------------------------ STEP 5: scaled dot-product attention
def attention(Q, K, V, mask=None):
    """Q: (B,h,Tq,dk)  K,V: (B,h,Tk,dk)  mask: True = may attend, broadcastable to (B,h,Tq,Tk)."""
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))  # (B,h,Tq,Tk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))     # blocked positions get weight 0 after softmax
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

# ------------------------------------------------------------------ STEP 6: multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.Wq, self.Wk, self.Wv, self.Wo = (nn.Linear(d, d) for _ in range(4))
    def split(self, x):                                       # (B,L,d) -> (B,h,L,dk)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dk).transpose(1, 2)
    def forward(self, q_in, kv_in, mask=None):
        Q, K, V = self.split(self.Wq(q_in)), self.split(self.Wk(kv_in)), self.split(self.Wv(kv_in))
        out, self.last_weights = attention(Q, K, V, mask)     # weights kept so we can inspect them
        B, _, Tq, _ = out.shape
        return self.Wo(out.transpose(1, 2).reshape(B, Tq, -1))   # join heads, mix with Wo

# ------------------------------------------------------------------ STEP 7: encoder layer, STEP 8: decoder layer
class FeedForward(nn.Module):
    def __init__(self, d, ff, drop):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(drop), nn.Linear(ff, d))
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):                                # pre-norm: LayerNorm before each sub-layer
    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.attn, self.ffn = MultiHeadAttention(d, heads), FeedForward(d, ff, drop)
        self.ln1, self.ln2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(drop)
    def forward(self, x, src_mask):
        h = self.ln1(x); x = x + self.drop(self.attn(h, h, src_mask))       # self-attention + skip
        return x + self.drop(self.ffn(self.ln2(x)))                          # feed-forward + skip

class DecoderLayer(nn.Module):
    def __init__(self, d, heads, ff, drop):
        super().__init__()
        self.self_attn, self.cross_attn, self.ffn = MultiHeadAttention(d, heads), MultiHeadAttention(d, heads), FeedForward(d, ff, drop)
        self.ln1, self.ln2, self.ln3, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(drop)
    def forward(self, y, memory, tgt_mask, src_mask):
        h = self.ln1(y); y = y + self.drop(self.self_attn(h, h, tgt_mask))               # masked self-attention
        y = y + self.drop(self.cross_attn(self.ln2(y), memory, src_mask))                # Q: decoder, K,V: encoder memory
        return y + self.drop(self.ffn(self.ln3(y)))

# ------------------------------------------------------------------ STEP 9: the whole model
class Transformer(nn.Module):
    def __init__(self, n_src, n_tgt, d=64, heads=4, layers=2, ff=128, drop=0.1):
        super().__init__()
        self.d = d
        self.src_emb, self.tgt_emb = nn.Embedding(n_src, d, padding_idx=PAD), nn.Embedding(n_tgt, d, padding_idx=PAD)
        self.pos, self.drop = PositionalEncoding(d), nn.Dropout(drop)
        self.enc = nn.ModuleList(EncoderLayer(d, heads, ff, drop) for _ in range(layers))
        self.dec = nn.ModuleList(DecoderLayer(d, heads, ff, drop) for _ in range(layers))
        self.enc_ln, self.dec_ln = nn.LayerNorm(d), nn.LayerNorm(d)
        for emb in (self.src_emb, self.tgt_emb): nn.init.normal_(emb.weight, mean=0.0, std=d ** -0.5)   # small embeddings: start-up loss is about ln(vocab)
        self.out = nn.Linear(d, n_tgt, bias=False); self.out.weight = self.tgt_emb.weight   # tie output and embedding weights
    def encode(self, src):                                    # runs ONCE per sentence
        m = src_mask_of(src); x = self.drop(self.pos(self.src_emb(src) * math.sqrt(self.d)))
        for layer in self.enc: x = layer(x, m)
        return self.enc_ln(x), m                              # memory (B,S,d)
    def decode(self, tgt_in, memory, src_m):                  # runs once per training batch, once per output token at test time
        y = self.drop(self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d)))
        tm = tgt_mask_of(tgt_in)
        for layer in self.dec: y = layer(y, memory, tm, src_m)
        return self.out(self.dec_ln(y))                       # logits (B,T,V)
    def forward(self, src, tgt_in):
        memory, m = self.encode(src); return self.decode(tgt_in, memory, m)

model = Transformer(len(src_itos), len(tgt_itos))
print("STEP 9  parameters:", sum(p.numel() for p in model.parameters()))

# check our attention against PyTorch's built-in implementation
Q, K, V = (torch.randn(2, 4, 5, 16) for _ in range(3))
ours, _ = attention(Q, K, V); ref = F.scaled_dot_product_attention(Q, K, V)
print("        attention check vs torch built-in, max |difference| =", float((ours - ref).abs().max()))

# ------------------------------------------------------------------ STEP 10: trace the shapes for one sentence
model.eval()
with torch.no_grad():
    src, tgt_in, tgt_out = make_batch(train[:1])
    print("STEP 10 shape trace for:", pairs[train[0]]["en"])
    print("        src ids", tuple(src.shape), "-> embedding", tuple(model.src_emb(src).shape))
    memory, sm = model.encode(src); print("        encoder memory", tuple(memory.shape), "| src mask", tuple(sm.shape))
    logits = model.decode(tgt_in, memory, sm)
    print("        decoder input", tuple(tgt_in.shape), "-> logits", tuple(logits.shape), "= (batch, target length, Kannada vocabulary)")
    print("        one attention map: cross-attention", tuple(model.dec[0].cross_attn.last_weights.shape), "= (batch, heads, target length, source length)")

# ------------------------------------------------------------------ STEP 11-12: teacher-forced loss and the training loop
def run_epoch(idx, train_mode, opt=None, bs=8):
    model.train(train_mode); total = n = 0
    idx = idx[:]; random.shuffle(idx) if train_mode else None
    for b in range(0, len(idx), bs):
        src, tgt_in, tgt_out = make_batch(idx[b:b + bs])
        with torch.set_grad_enabled(train_mode):
            logits = model(src, tgt_in)                                          # all positions at once (teacher forcing)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD)
        if train_mode:
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item() * len(idx[b:b + bs]); n += len(idx[b:b + bs])
    return total / n

opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
print("STEP 12 training (teacher forcing):")
best = (1e9, None)
for epoch in range(1, 121):
    tr = run_epoch(train, True, opt); va = run_epoch(val, False)
    if va < best[0]: best = (va, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
    if epoch in (1, 5, 10, 20, 40, 80, 120): print(f"        epoch {epoch:3d}  train loss {tr:.3f}  validation loss {va:.3f}")
model.load_state_dict(best[1]); print(f"        best validation loss {best[0]:.3f} at epoch {best[2]} (checkpoint restored)")

# ------------------------------------------------------------------ STEP 13: greedy decoding
@torch.no_grad()
def translate(sentence, max_len=24):
    model.eval()
    src = torch.tensor([encode_src(sentence)]); memory, sm = model.encode(src)           # encoder: once
    out = [BOS]
    for _ in range(max_len):                                                             # decoder: once per new token
        logits = model.decode(torch.tensor([out]), memory, sm)
        nxt = int(logits[0, -1].argmax())                                                # greedy: most probable token
        if nxt == EOS: break
        out.append(nxt)
    return " ".join(tgt_itos[i] for i in out[1:]).replace(" ?", "?").replace(" .", ".")

print("STEP 13 greedy decoding:")
for i in train[:2] + test[:4]:
    tag = "train" if i in train else "test "
    print(f"        [{tag}] {pairs[i]['en']:38s} -> {translate(pairs[i]['en'])}   (reference: {pairs[i]['kn']})")

# ------------------------------------------------------------------ STEP 14: evaluation (BLEU written from scratch)
def ngrams(t, n):
    c = {}
    for i in range(len(t) - n + 1): c[tuple(t[i:i + n])] = c.get(tuple(t[i:i + n]), 0) + 1
    return c

def corpus_bleu(cands, refs):
    match, total, c_len, r_len = [0] * 4, [0] * 4, 0, 0
    for c, r in zip(cands, refs):
        c_len += len(c); r_len += len(r)
        for n in range(1, 5):
            cg, rg = ngrams(c, n), ngrams(r, n)
            match[n - 1] += sum(min(v, rg.get(g, 0)) for g, v in cg.items()); total[n - 1] += sum(cg.values())
    p = [match[0] / max(total[0], 1)] + [(match[i] + 1) / (total[i] + 1) for i in range(1, 4)]    # +1 smoothing for n > 1
    bp = 1.0 if c_len >= r_len else math.exp(1 - r_len / max(c_len, 1))
    return 100 * bp * math.exp(sum(math.log(max(x, 1e-9)) for x in p) / 4) if p[0] > 0 else 0.0

def score(idx):
    return corpus_bleu([tokenize(translate(pairs[i]["en"])) for i in idx], [tokenize(pairs[i]["kn"]) for i in idx])
print(f"STEP 14 BLEU  train {score(train):.1f} | validation {score(val):.1f} | test {score(test):.1f}   (10-sentence sets: indicative only)")
