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
| `Transformers_in_NLP_Project_Paper.docx` / `.pdf` | Project paper (29 pages, VILNIUS TECH MFT template) |
| `Transformers_in_NLP_Presentation.pptx` | 11-slide presentation for a 7–10 minute talk, with speaker notes |

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
          word_finalize.ps1 (updates the table of contents and exports PDF with MS Word)
slides/   build_deck.js (generates the presentation from the results)
docs/     final paper and presentation
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
