#* Appendices

#A Appendix A. Source code and reproduction of the results

The complete source code, the results of every model (JSON files with metrics, timings and predictions), the figures, this paper and the presentation are available in the public repository https://github.com/shashikanth077/Transformers-NLP. The repository is organised as follows:

- code/ – the experiment: 01_prepare_data.py … 09_diagrams.py, utils.py, run_all.sh, requirements.txt;
- code/results/ – one JSON file per model and the summary tables (summary_table.csv, statistics.csv, subset_accuracy.csv, learning_curve.csv);
- code/figures/ – all figures used in the paper and the presentation;
- paper/ – the chapter texts and build_docx.py, which generates this document from the VILNIUS TECH template;
- slides/ – build_deck.js, which generates the presentation from the results;
- docs/ – the final paper (DOCX, PDF) and the presentation (PPTX).

To reproduce the experiment, create a Python 3.11 environment, install the CPU build of PyTorch and the packages listed in code/requirements.txt, and run bash run_all.sh in the code directory. On a 4-core laptop the whole pipeline takes about 2.5–3 hours; the dataset and the pre-trained models (about 1.5 GB) are downloaded automatically from the Hugging Face Hub on the first run.
