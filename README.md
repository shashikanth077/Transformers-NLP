# Transformers and their Application in NLP Tasks

Group project, VILNIUS TECH – Faculty of Fundamental Sciences, *Engineering of Artificial Intelligence* study program.
**Team:** Shashikanth Hosur Ramegowda, Shaeed Khan, Rimsha Tariq

The project explains the Transformer architecture and evaluates it experimentally: **10 sentiment classifiers**
(classical TF-IDF, a BiLSTM, three Transformers fine-tuned by us and three ready-made Transformers) are compared on
the SST-2 dataset on a **CPU-only laptop**, by accuracy, training time, latency, model size, data efficiency and
robustness on negation / contrast.

## Deliverables (`docs/`)

| File | Description |
|---|---|
| `Transformers_in_NLP_Project_Paper.docx` / `.pdf` | Project paper (28 pages, VILNIUS TECH MFT template, English only) |
| `Transformers_in_NLP_Presentation.pptx` | 11-slide presentation for a 7–10 minute talk, with speaker notes |
| `Project_Study_Guide.html` | **Interactive study guide** (open in any browser, works offline): real-world story, step-by-step pipeline, sentence lab with real attention maps, all 872 test sentences for every model, results explorer, Q&A, quiz, rehearsal timer |
| `Transformers_NLP_Kannada_Notes.pdf` | **Study notes in Kannada** (40 pages): NLP and Transformers explained with Karnataka examples, the project's real results, viva Q&A with English answer lines, glossary, quiz, one-page summary |
| `Transformers_Explained_Overview.html` | **Interactive overview of Transformers** (works offline), following the structure of Ketan Doshi's "Transformers Explained Visually, Part 1" article with original text and demos: architecture explorer, real DistilBERT attention maps, step-by-step training and inference with real English→Kannada outputs (NLLB-200), teacher forcing, task heads, RNN comparison, quiz |
| `CNN_Pipeline_Explained.html` | **CNN explained end to end** (offline): image → convolution → ReLU → pooling → flatten → dense → prediction → loss → backpropagation → weight updates on a PIN-code digit reader, with plain-words, real-world and maths boxes; a real CNN runs live in the browser (gradient check, training, draw-your-own digit) |

## Main results (SST-2 test set, 872 sentences)

| Model | Training labels | Accuracy |
|---|---|---|
| DistilBERT-SST2 (ready-made) | – | **91.1%** |
| DistilBERT fine-tuned by us | 8 000 | 88.8% |
| Twitter-RoBERTa (ready-made, other domain) | – | 85.8% |
| BERT-small fine-tuned | 8 000 | 83.7% |
| TF-IDF + Linear SVM | 65 349 | 82.9% |
| BiLSTM (no pre-training) | 65 349 | 82.8% |
| Zero-shot NLI DistilRoBERTa | 0 | 81.8% |
| BERT-tiny fine-tuned | 8 000 | 76.6% |

With only 500 labels BERT-small reaches 75.9% vs 61.8% for TF-IDF (+14.1 p.p.).

## Repository structure

```
code/     experiment (Python): 01_prepare_data.py ... 09_diagrams.py, run_all.sh, requirements.txt
  results/  one JSON per model + summary CSVs (metrics, timings, predictions, statistics)
  figures/  all charts and diagrams
paper/    chapter texts (*.md), build_docx.py (generates the paper from template/MFT_template_2025.docx),
          word_finalize.ps1 (updates the table of contents and exports PDF with MS Word),
          check_docx.py (quality checks of the generated paper)
slides/   build_deck.js (generates the presentation from the results)
docs/     final paper, presentation and the interactive study guide
study_guide/  export_data.py (collects the real results, TF-IDF weights and attention maps) and
              build.py (writes docs/Project_Study_Guide.html from src/ and data.json)
              overview/ export_overview_data.py (real NLLB, DistilGPT-2, DistilBERT outputs) and
              build_overview.py (writes docs/Transformers_Explained_Overview.html)
              kannada/  build_kannada_notes.py (notes_kn.html + Edge print-to-PDF; numbers come from data.json)
```

## Reproduce

```bash
cd code
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
bash run_all.sh                                      # ~2.5–3 h on a 4-core laptop CPU
```

Rebuild the documents (Windows with MS Word / PowerPoint):

```bash
pip install python-docx latex2mathml lxml pandas
python paper/build_docx.py
powershell -File paper/word_finalize.ps1 -Docx docs/Transformers_in_NLP_Project_Paper.docx -Pdf docs/Transformers_in_NLP_Project_Paper.pdf
cd slides && npm install && node build_deck.js
```
