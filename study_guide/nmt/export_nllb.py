"""Real pretrained-model outputs for the NMT pipeline page: NLLB-200 (distilled 600M) on the phrasebook corpus.
Writes nllb.json: tokenisation, beam-4 translations, teacher-forced losses, and one real attention head (Q, K, V, weights) for two sentences.
Run: python study_guide/nmt/export_nllb.py"""
import json, math, os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
corpus = json.load(open(os.path.join(HERE, "corpus.json"), encoding="utf-8"))
torch.set_grad_enabled(False)
NAME = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(NAME, src_lang="eng_Latn", tgt_lang="kan_Knda")
model = AutoModelForSeq2SeqLM.from_pretrained(NAME, attn_implementation="eager").eval()
KAN = tok.convert_tokens_to_ids("kan_Knda"); START = model.config.decoder_start_token_id

pairs = []
for p in corpus:
    enc = tok(p["en"], return_tensors="pt")
    out = model.generate(**enc, forced_bos_token_id=KAN, num_beams=4, max_new_tokens=40)
    text = tok.decode(out[0], skip_special_tokens=True)
    ref_ids = tok(text_target=p["kn"], return_tensors="pt")["input_ids"]
    labels = ref_ids.clone()
    dec_in = torch.cat([torch.tensor([[START]]), ref_ids[:, :-1]], dim=1)
    lg = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=dec_in).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    nll = [float(-lp[t, ref_ids[0, t]]) for t in range(1, ref_ids.shape[1])]       # skip the language marker
    pairs.append({"en": p["en"], "kn": p["kn"], "nllb": text,
                  "src_tok": tok.convert_ids_to_tokens(enc["input_ids"][0]), "ref_tok": tok.convert_ids_to_tokens(ref_ids[0]),
                  "tf_nll": [round(v, 3) for v in nll]})
    print(p["en"], "->", text, flush=True)

# noisy pairs (as found in web-crawled corpora) scored by the same model: dual-cross-entropy style filtering
DIRTY = [("Where is the hospital?", "Where is the hospital?", "untranslated copy"),
         ("Good morning.", "ಶುಭೋದಯ. ಇಂದಿನ ಹವಾಮಾನ ಬಹಳ ಚೆನ್ನಾಗಿದೆ ಮತ್ತು ನಾವು ಇಂದು ಸಂಜೆ ಉದ್ಯಾನವನಕ್ಕೆ ಹೋಗೋಣ.", "extra text on one side"),
         ("Where is the bus stand?", "ಕಾಫಿ ಬೆಲೆ ಎಷ್ಟು?", "misaligned, similar length"),
         ("Please speak slowly.", "ನಾನು ನಾಳೆ ಮೈಸೂರಿಗೆ ಹೋಗುತ್ತೇನೆ.", "misaligned, similar length")]
dirty = []
for en, kn, why in DIRTY:
    enc = tok(en, return_tensors="pt"); ref_ids = tok(text_target=kn, return_tensors="pt")["input_ids"]
    dec_in = torch.cat([torch.tensor([[START]]), ref_ids[:, :-1]], dim=1)
    lg = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=dec_in).logits[0].float(); lp = torch.log_softmax(lg, -1)
    nll = [float(-lp[t, ref_ids[0, t]]) for t in range(1, ref_ids.shape[1])]
    dirty.append({"en": en, "kn": kn, "why": why, "tf_nll": [round(v, 3) for v in nll]}); print("dirty", why, round(sum(nll) / len(nll), 2), flush=True)

# one real attention head (decoder layer L, head 0) for two sentences: Q, K, V and the model's own weights, to verify the maths on the page
L, HEAD = 5, 0
caps = {}
layer = model.model.decoder.layers[L].encoder_attn
def hook(mod, args, kwargs):
    caps["q_in"] = (args[0] if args else kwargs["hidden_states"]).detach()
    caps["kv_in"] = kwargs.get("key_value_states", args[1] if len(args) > 1 else None)
layer.register_forward_pre_hook(hook, with_kwargs=True)
hd = layer.head_dim if hasattr(layer, "head_dim") else 64
showcase = []
for en in ["Where is the bus stand?", "I will go to Mysuru tomorrow."]:
    p = next(x for x in corpus if x["en"] == en)
    enc = tok(en, return_tensors="pt"); ref_ids = tok(text_target=p["kn"], return_tensors="pt")["input_ids"]
    dec_in = torch.cat([torch.tensor([[START]]), ref_ids[:, :-1]], dim=1)
    o = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], decoder_input_ids=dec_in, output_attentions=True)
    q_in, kv = caps["q_in"][0], caps["kv_in"][0]
    q = layer.q_proj(q_in).view(q_in.shape[0], -1, hd)[:, HEAD]; k = layer.k_proj(kv).view(kv.shape[0], -1, hd)[:, HEAD]; v = layer.v_proj(kv).view(kv.shape[0], -1, hd)[:, HEAD]
    sc = (q @ k.T) * hd ** -0.5; w = torch.softmax(sc, -1); ref_w = o.cross_attentions[L][0, HEAD]
    err = float((w - ref_w).abs().max()); print("attention check", en, "max |ours - model| =", err)
    ctx = w @ v
    r = lambda t: [[round(float(x), 4) for x in row] for row in t]
    showcase.append({"en": en, "kn": p["kn"], "src_tok": tok.convert_ids_to_tokens(enc["input_ids"][0]), "dec_tok": tok.convert_ids_to_tokens(dec_in[0]),
                     "tgt_tok": tok.convert_ids_to_tokens(ref_ids[0]), "d_k": hd, "layer": L, "head": HEAD, "check_err": err,
                     "q": r(q), "k": r(k), "v": r(v), "scores": r(sc), "w": r(w), "ctx": r(ctx)})
cfg = model.config
json.dump({"model": NAME, "params": sum(p.numel() for p in model.parameters()), "enc_layers": cfg.encoder_layers, "dec_layers": cfg.decoder_layers, "d_model": cfg.d_model, "heads": cfg.encoder_attention_heads, "vocab": cfg.vocab_size,
           "pairs": pairs, "dirty": dirty, "showcase": showcase}, open(os.path.join(HERE, "nllb.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("saved", os.path.getsize(os.path.join(HERE, "nllb.json")) // 1024, "KB")
