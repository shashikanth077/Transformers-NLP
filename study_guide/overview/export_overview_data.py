"""Real model outputs for the interactive 'Transformers explained: overview' page -> data.json

Models (all public, run locally on CPU):
  facebook/nllb-200-distilled-600M  English -> Kannada translation (encoder-decoder Transformer)
  distilgpt2                        next-word probabilities (decoder-only language model)
  distilbert-base-uncased           encoder self-attention (for the 'it' example)

Run:  python study_guide/overview/export_overview_data.py
"""
import base64
import json
import math
import os

import numpy as np
import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data.json")
torch.set_grad_enabled(False)

# ------------------------------------------------------------------ 1. translation (encoder-decoder)
PAIRS = [
    ("Where is the bus stand?", "ಬಸ್ ನಿಲ್ದಾಣ ಎಲ್ಲಿದೆ?"),
    ("Thank you very much for your help.", "ನಿಮ್ಮ ಸಹಾಯಕ್ಕೆ ತುಂಬಾ ಧನ್ಯವಾದಗಳು."),
    ("I am going to Mysuru tomorrow.", "ನಾನು ನಾಳೆ ಮೈಸೂರಿಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ."),
    ("The film was good, but the ending was disappointing.", "ಸಿನಿಮಾ ಚೆನ್ನಾಗಿತ್ತು, ಆದರೆ ಅಂತ್ಯ ನಿರಾಸೆ ಮೂಡಿಸಿತು."),
]
NLLB = "facebook/nllb-200-distilled-600M"
print("loading", NLLB, flush=True)
tok = AutoTokenizer.from_pretrained(NLLB, src_lang="eng_Latn", tgt_lang="kan_Knda")
mt = AutoModelForSeq2SeqLM.from_pretrained(NLLB, attn_implementation="eager").eval()
EOS = tok.eos_token_id
KAN = tok.convert_tokens_to_ids("kan_Knda")
START = mt.config.decoder_start_token_id


def piece(i):
    return tok.convert_ids_to_tokens(int(i)).replace("▁", "▁")


def clean(p):
    """Readable token: SentencePiece word-start marker -> leading space shown as '▁'."""
    return p


def run_decoder(enc_out, dec_ids, attn_mask):
    out = mt.model.decoder(input_ids=torch.tensor([dec_ids]), encoder_hidden_states=enc_out, encoder_attention_mask=attn_mask, output_attentions=True)
    logits = mt.lm_head(out.last_hidden_state)[0]
    cross = torch.stack([a[0] for a in out.cross_attentions])      # layers x heads x tgt x src
    return logits, cross


def step_record(logits_row, cross_row, k=8):
    probs = torch.softmax(logits_row.float(), -1)
    top = torch.topk(probs, k)
    return {"top": [[piece(i), round(float(p), 5)] for p, i in zip(top.values, top.indices)],
            "top_ids": [int(i) for i in top.indices],
            "entropy": round(float(-(probs * torch.log(probs + 1e-12)).sum()), 3),
            "cross": [[round(float(v), 4) for v in layer] for layer in cross_row.mean(1)]}   # layers x src (mean over heads)


def greedy_from(enc_out, mask, prefix, max_new=30):
    ids = list(prefix)
    for _ in range(max_new):
        logits, _ = run_decoder(enc_out, ids, mask)
        nxt = int(logits[-1].argmax())
        ids.append(nxt)
        if nxt == EOS:
            break
    return ids


translations = []
for src, ref in PAIRS:
    enc = tok(src, return_tensors="pt")
    enc_out = mt.model.encoder(**enc).last_hidden_state
    mask = enc["attention_mask"]
    src_tokens = [piece(i) for i in enc["input_ids"][0]]
    ids = [START, KAN]
    steps = []
    for _ in range(40):
        logits, cross = run_decoder(enc_out, ids, mask)
        rec = step_record(logits[-1], cross[:, :, -1, :])
        nxt = int(logits[-1].argmax())
        rec["chosen"] = piece(nxt)
        rec["prefix"] = [piece(i) for i in ids]
        steps.append(rec)
        ids.append(nxt)
        if nxt == EOS:
            break
    greedy_text = tok.decode(ids[2:], skip_special_tokens=True)

    # branches: take each of the top-3 candidates at the FIRST content step, then continue greedily
    branches = []
    for cand_id in steps[0]["top_ids"][:3]:
        full = greedy_from(enc_out, mask, [START, KAN, cand_id])
        lp = 0.0
        for t in range(2, len(full)):
            lg, _ = run_decoder(enc_out, full[:t], mask)
            lp += float(torch.log_softmax(lg[-1].float(), -1)[full[t]])
        branches.append({"first": piece(cand_id), "tokens": [piece(i) for i in full[2:]], "text": tok.decode(full[2:], skip_special_tokens=True), "logprob": round(lp, 2)})

    # teacher forcing: one parallel pass over the human reference translation
    ref_ids = tok(text_target=ref, return_tensors="pt")["input_ids"][0].tolist()      # [kan_Knda, ..., eos]
    dec_in = [START] + ref_ids[:-1]
    logits, _ = run_decoder(enc_out, dec_in, mask)
    lp = torch.log_softmax(logits.float(), -1)
    tf_rows = []
    for t in range(len(dec_in)):
        gold = ref_ids[t]
        top1 = int(lp[t].argmax())
        tf_rows.append({"input": piece(dec_in[t]), "gold": piece(gold), "pred": piece(top1), "p_gold": round(float(lp[t, gold].exp()), 4), "loss": round(float(-lp[t, gold]), 3)})
    translations.append({"src": src, "ref": ref, "src_tokens": src_tokens, "greedy": greedy_text, "steps": steps, "branches": branches,
                         "tf": tf_rows, "tf_loss": round(sum(r["loss"] for r in tf_rows) / len(tf_rows), 3)})
    print(src, "->", greedy_text, "| branches:", [b["text"] for b in branches], "| ref loss", translations[-1]["tf_loss"], flush=True)
n_layers = len(mt.model.decoder.layers)
n_heads = mt.config.decoder_attention_heads
enc_layers, dec_layers, d_model = mt.config.encoder_layers, mt.config.decoder_layers, mt.config.d_model
n_params = sum(p.numel() for p in mt.parameters())
del mt

# ------------------------------------------------------------------ 2. language model head (decoder-only)
PROMPTS = ["In India, people drink a cup of hot", "Thank you very much for your", "The monsoon arrives in Kerala in the month of"]
gtok = AutoTokenizer.from_pretrained("distilgpt2")
lm = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
lm_data = []
for p in PROMPTS:
    ids = gtok(p, return_tensors="pt")
    logits = lm(**ids).logits[0, -1].float()
    probs = torch.softmax(logits, -1)
    top = torch.topk(logits, 40)
    lm_data.append({"prompt": p, "vocab": int(logits.shape[0]),
                    "top": [[gtok.decode([int(i)]), round(float(v), 3), round(float(probs[i]), 5)] for v, i in zip(top.values, top.indices)],
                    "mass40": round(float(probs[top.indices].sum()), 3)})
    print(p, "->", [(t[0], t[2]) for t in lm_data[-1]["top"][:5]], flush=True)
lm_params = sum(p.numel() for p in lm.parameters())
del lm

# ------------------------------------------------------------------ 3. self-attention for the 'it' example (encoder)
SENT = ["the trophy didn't fit in the suitcase because it was too big .", "the trophy didn't fit in the suitcase because it was too small ."]
btok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc_model = AutoModel.from_pretrained("distilbert-base-uncased", attn_implementation="eager").eval()
att_data = []
for s in SENT:
    e = btok(s, return_tensors="pt")
    out = enc_model(**e, output_attentions=True)
    att = torch.stack([a[0] for a in out.attentions]).numpy()          # layers x heads x n x n
    toks = btok.convert_ids_to_tokens(e["input_ids"][0])
    q = np.clip(np.round(att * 255), 0, 255).astype(np.uint8)
    it = toks.index("it"); tr = toks.index("trophy"); sc = toks.index("suitcase")
    heads = sorted(([float(att[l, h, it, tr]), float(att[l, h, it, sc]), l, h] for l in range(att.shape[0]) for h in range(att.shape[1])), key=lambda r: -(r[0] + r[1]))[:6]
    att_data.append({"text": s, "tokens": toks, "shape": list(att.shape), "att": base64.b64encode(q.tobytes()).decode(),
                     "it": it, "trophy": tr, "suitcase": sc, "top_heads": [[round(a, 3), round(b, 3), l, h] for a, b, l, h in heads]})
    print(s, "top heads (it->trophy, it->suitcase, layer, head):", att_data[-1]["top_heads"][:3], flush=True)

data = {"translations": translations, "lm": lm_data, "attention": att_data,
        "mt": {"model": NLLB, "enc_layers": enc_layers, "dec_layers": dec_layers, "d_model": d_model, "heads": n_heads, "params": n_params, "vocab": tok.vocab_size},
        "lm_model": {"name": "distilgpt2", "params": lm_params}}
json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("saved", OUT, round(os.path.getsize(OUT) / 1024), "KB")
