import pymupdf, sys, os
d = pymupdf.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "Transformer_Encoder_Decoder_Flow.pdf"))
out = sys.argv[1]; print("pages", len(d))
for i, p in enumerate(d):
    p.get_pixmap(dpi=70).save(os.path.join(out, f"q{i + 1}.png")); print(i + 1, len(p.get_text()))
