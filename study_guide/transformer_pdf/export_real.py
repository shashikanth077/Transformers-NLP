"""Real tensors from NLLB-200 (distilled 600M) for the worked example of the encoder/decoder flow PDF.
Sentence: "I am going to Mysuru tomorrow."  ->  greedy Kannada output.
Writes real.json: tokens, embeddings (first dims), encoder self-attention (layers 1 and 12, mean over heads),
decoder masked self-attention and cross-attention (layer 12, mean over heads), per-step top-3 probabilities of the greedy decode.
Run: python study_guide/transformer_pdf/export_real.py"""
import json, math, os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_grad_enabled(False)
NAME = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(NAME, src_lang="eng_Latn", tgt_lang="kan_Knda")
model = AutoModelForSeq2SeqLM.from_pretrained(NAME, attn_implementation="eager").eval()
KAN = tok.convert_tokens_to_ids("kan_Knda"); START = model.config.decoder_start_token_id; EOS = tok.eos_token_id
SENT = "I am going to Mysuru tomorrow."
enc = tok(SENT, return_tensors="pt")
src_ids = enc["input_ids"][0].tolist(); src_tok = tok.convert_ids_to_tokens(src_ids)

# greedy decode with probabilities
eo = model.model.encoder(**enc); mem = eo.last_hidden_state
ids = [START, KAN]; steps = []
for _ in range(30):
    out = model.model.decoder(input_ids=torch.tensor([ids]), encoder_hidden_states=mem, encoder_attention_mask=enc["attention_mask"])
    p = torch.softmax(model.lm_head(out.last_hidden_state)[0, -1].float(), -1); top = torch.topk(p, 3)
    nxt = int(p.argmax()); steps.append({"prefix": tok.convert_ids_to_tokens(ids), "top": [[tok.convert_ids_to_tokens(int(i)), round(float(v), 4)] for v, i in zip(top.values, top.indices)], "chosen": tok.convert_ids_to_tokens(nxt)})
    ids.append(nxt)
    if nxt == EOS: break
out_tok = tok.convert_ids_to_tokens(ids[2:]); text = tok.decode(ids[2:], skip_special_tokens=True)

# encoder self-attention (all layers, mean over heads) and the final encoder output norm
eo = model.model.encoder(**enc, output_attentions=True)
r = lambda t: [[round(float(x), 4) for x in row] for row in t]
enc_self = {str(L): r(eo.attentions[L][0].mean(0)) for L in (0, 11)}
# decoder passes on the produced output (teacher-forced on it): masked self-attention and cross-attention
dec_in = torch.tensor([ids[:-1]])
do = model.model.decoder(input_ids=dec_in, encoder_hidden_states=mem, encoder_attention_mask=enc["attention_mask"], output_attentions=True)
dec_self = {str(L): r(do.attentions[L][0].mean(0)) for L in (0, 11)}
cross = {str(L): r(do.cross_attentions[L][0].mean(0)) for L in (0, 11)}
# embeddings: token embedding (already scaled by sqrt(d_model) inside the model), and the input to layer 1 (token + position)
emb_tok = model.model.encoder.embed_tokens(enc["input_ids"])[0]
eo0 = model.model.encoder(**enc, output_hidden_states=True); x0 = eo0.hidden_states[0][0]
e_first = [[round(float(v), 3) for v in emb_tok[i, :4]] for i in range(len(src_ids))]
x_first = [[round(float(v), 3) for v in x0[i, :4]] for i in range(len(src_ids))]
scale = float(math.sqrt(model.config.d_model))
# per-layer norms of the encoder hidden state (shows representations being refined)
eo2 = model.model.encoder(**enc, output_hidden_states=True)
norms = [round(float(h[0].norm(dim=-1).mean()), 1) for h in eo2.hidden_states]
cfg = model.config
json.dump({"model": NAME, "sentence": SENT, "src_tok": src_tok, "src_ids": src_ids, "out_tok": out_tok, "text": text, "dec_in_tok": tok.convert_ids_to_tokens(ids[:-1]),
           "steps": steps, "enc_self": enc_self, "dec_self": dec_self, "cross": cross, "emb_first4": e_first, "x0_first4": x_first, "embed_scale": scale, "enc_norms": norms,
           "cfg": {"enc_layers": cfg.encoder_layers, "dec_layers": cfg.decoder_layers, "d_model": cfg.d_model, "heads": cfg.encoder_attention_heads, "ffn": cfg.encoder_ffn_dim, "vocab": cfg.vocab_size, "params": sum(p.numel() for p in model.parameters())}},
          open(os.path.join(HERE, "real.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("output:", text, "| tokens:", out_tok, "| scale", scale, "| norms", norms)
