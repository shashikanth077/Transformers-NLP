"""The same encoder-decoder flow with a real pretrained model (NLLB-200 distilled 600M) using Hugging Face Transformers:
tokenize -> encoder (once) -> decoder loop by hand -> compare with generate() -> inspect attention -> verify one head by hand.

Run from the repository root:   python code/transformer_walkthrough/nllb_step_by_step.py
"""
import math
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

torch.set_grad_enabled(False)                                    # inference only: no gradients, less memory
NAME = "facebook/nllb-200-distilled-600M"
SENTENCE = "I am going to Mysuru tomorrow."

# ------------------------------------------------------------------ STEP 1: load tokenizer and model
tok = AutoTokenizer.from_pretrained(NAME, src_lang="eng_Latn", tgt_lang="kan_Knda")
model = AutoModelForSeq2SeqLM.from_pretrained(NAME, attn_implementation="eager").eval()   # eager attention so weights can be returned
cfg = model.config
print("STEP 1  loaded:", NAME, "|", cfg.encoder_layers, "encoder +", cfg.decoder_layers, "decoder layers | d_model", cfg.d_model,
      "| heads", cfg.encoder_attention_heads, "| vocabulary", cfg.vocab_size)

# ------------------------------------------------------------------ STEP 2: tokenize (text -> token ids)
enc = tok(SENTENCE, return_tensors="pt")                         # input_ids and attention_mask
print("STEP 2  pieces:", tok.convert_ids_to_tokens(enc["input_ids"][0]))
print("        ids   :", enc["input_ids"][0].tolist(), "| attention_mask:", enc["attention_mask"][0].tolist())

# ------------------------------------------------------------------ STEP 3: run the ENCODER once
encoder_out = model.get_encoder()(**enc, output_attentions=True)
memory = encoder_out.last_hidden_state                           # (batch, source length, d_model)
print("STEP 3  encoder memory:", tuple(memory.shape), "| self-attention maps:", len(encoder_out.attentions), "layers, each", tuple(encoder_out.attentions[0].shape), "= (batch, heads, S, S)")

# ------------------------------------------------------------------ STEP 4: the decoder loop, written by hand (greedy)
KAN = tok.convert_tokens_to_ids("kan_Knda")                      # forces the target language
start = model.config.decoder_start_token_id                      # NLLB starts decoding with </s>
ids = [start, KAN]
print("STEP 4  decoding loop (greedy):")
for turn in range(1, 30):
    dec = model.get_decoder()(input_ids=torch.tensor([ids]), encoder_hidden_states=memory, encoder_attention_mask=enc["attention_mask"])
    logits = model.lm_head(dec.last_hidden_state[:, -1])         # scores for the NEXT token only: (1, vocabulary)
    probs = torch.softmax(logits.float(), dim=-1)[0]
    p, nxt = probs.max(0)
    ids.append(int(nxt))
    top = torch.topk(probs, 3)
    print(f"        turn {turn:2d}: chose {tok.convert_ids_to_tokens(int(nxt))!r:16} p={float(p):.3f}   top-3: {[(tok.convert_ids_to_tokens(int(i)), round(float(v), 3)) for v, i in zip(top.values, top.indices)]}")
    if int(nxt) == tok.eos_token_id: break
manual = tok.decode(ids[2:], skip_special_tokens=True)
print("        manual loop result :", manual)

# ------------------------------------------------------------------ STEP 5: the same thing with generate() (greedy and beam search)
greedy = model.generate(**enc, forced_bos_token_id=KAN, num_beams=1, do_sample=False, max_new_tokens=40)
beam = model.generate(**enc, forced_bos_token_id=KAN, num_beams=4, max_new_tokens=40)
print("STEP 5  generate(greedy):", tok.decode(greedy[0], skip_special_tokens=True), "| identical to the manual loop:", tok.decode(greedy[0], skip_special_tokens=True) == manual)
print("        generate(beam=4) :", tok.decode(beam[0], skip_special_tokens=True))

# ------------------------------------------------------------------ STEP 6: inspect the attention of the decoder
dec_in = torch.tensor([ids[:-1]])                                # what the decoder saw: </s> kan_Knda + its own output
out = model.get_decoder()(input_ids=dec_in, encoder_hidden_states=memory, encoder_attention_mask=enc["attention_mask"], output_attentions=True)
print("STEP 6  decoder self-attention per layer:", tuple(out.attentions[0].shape), "| cross-attention per layer:", tuple(out.cross_attentions[0].shape), "= (batch, heads, T, S)")
cross = out.cross_attentions[-1][0].mean(0)                      # last layer, average over heads: (T, S)
src_pieces = tok.convert_ids_to_tokens(enc["input_ids"][0])
for t in (4, 5, 6):                                              # rows where the decoder writes the pieces of the place name
    row = cross[t]; j = int(row[1:-1].argmax()) + 1              # ignore the language tag and </s>
    print(f"        decoder input {tok.convert_ids_to_tokens(int(dec_in[0, t]))!r:12} -> looks most at source {src_pieces[j]!r} ({float(row[j]):.2f}); weight on the two markers: {float(row[0] + row[-1]):.2f}")

# ------------------------------------------------------------------ STEP 7: verify one attention head by hand (the maths on the page, in code)
L, HEAD = 5, 0                                                   # decoder layer 6, head 1
layer = model.get_decoder().layers[L].encoder_attn               # the cross-attention module
captured = {}
hook = layer.register_forward_pre_hook(lambda m, a, kw: captured.update(q_in=(a[0] if a else kw["hidden_states"]), kv=kw["key_value_states"]), with_kwargs=True)
ref = model.get_decoder()(input_ids=dec_in, encoder_hidden_states=memory, encoder_attention_mask=enc["attention_mask"], output_attentions=True).cross_attentions[L][0, HEAD]
hook.remove()
dk = layer.head_dim
Q = layer.q_proj(captured["q_in"][0]).view(-1, layer.num_heads, dk)[:, HEAD]   # (T, 64)
K = layer.k_proj(captured["kv"][0]).view(-1, layer.num_heads, dk)[:, HEAD]     # (S, 64)
scores = Q @ K.T / math.sqrt(dk)                                  # Q K^T / sqrt(d_k)
mine = torch.softmax(scores, dim=-1)
print(f"STEP 7  Q {tuple(Q.shape)}, K {tuple(K.shape)} -> scores {tuple(scores.shape)} -> softmax; max |ours - model| = {float((mine - ref).abs().max()):.1e}")
print("        row for decoder input", repr(tok.convert_ids_to_tokens(int(dec_in[0, 5]))), "(sums to", round(float(mine[5].sum()), 3), "):", [round(float(v), 3) for v in mine[5]])

# ------------------------------------------------------------------ STEP 8: batching and padding
batch = tok(["Where is the bus stand?", "Thank you very much for your help."], return_tensors="pt", padding=True)
print("STEP 8  batch of 2 sentences padded to the same length:", tuple(batch["input_ids"].shape), "| attention_mask:", batch["attention_mask"].tolist())
res = model.generate(**batch, forced_bos_token_id=KAN, num_beams=4, max_new_tokens=40)
for s, r in zip(["Where is the bus stand?", "Thank you very much for your help."], tok.batch_decode(res, skip_special_tokens=True)): print("        ", s, "->", r)
