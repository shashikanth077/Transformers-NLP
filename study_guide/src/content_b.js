/* Static teaching content, part B: Q&A, quiz, limits, literature, slides, files, glossary. */

const QA = [
  // ---------------- theory
  ["Theory", "What is a Transformer, in one sentence?", K => `A neural network architecture (Vaswani et al., 2017) that processes all words of a sentence at once and lets every word look directly at every other word through <b>self-attention</b>, instead of reading word by word like a recurrent network.`],
  ["Theory", "What does self-attention compute?", K => `For every word it computes similarity scores with all words (Q·Kᵀ), turns them into weights with softmax (they sum to 1) and returns the <b>weighted average of the value vectors</b>. Formula: softmax(QKᵀ/√d<sub>k</sub>)·V. So “good” can pull information from “not”.`],
  ["Theory", "Why divide by √d<sub>k</sub>?", K => `Dot products of long vectors grow with their dimension. Without scaling, softmax becomes extremely peaked (one word gets almost all the weight) and its gradients vanish, so training stalls. Dividing by √d<sub>k</sub> keeps the scores in a sensible range. (Try it in the attention calculator.)`],
  ["Theory", "Why do we need positional encoding?", K => `Attention treats a sentence as a <em>set</em>: if you shuffle the words, the scores stay the same. Adding a position vector to each token embedding gives the model word order. The original paper used sine/cosine waves; BERT learns the position vectors.`],
  ["Theory", "What is multi-head attention and why use it?", K => `Several attention operations run in parallel with different learned projections. Each head can specialise (e.g. one links negations to the words they modify, another follows grammar). Our models use 2, 8 and 12 heads.`],
  ["Theory", "What is the difference between BERT, GPT and T5?", K => `<b>BERT</b>: encoder only, trained by masked language modelling, sees both directions, great for understanding and classification (our models). <b>GPT</b>: decoder only, predicts the next word, great for generation and few-shot prompting. <b>T5/BART</b>: encoder–decoder, text-to-text, great for translation and summarisation.`],
  ["Theory", "What are pre-training and fine-tuning?", K => `<b>Pre-training</b>: learn general language from huge unlabelled text (e.g. guess masked words); very expensive, done once. <b>Fine-tuning</b>: continue training on a small labelled task dataset with a small classifier on top; cheap. We only fine-tuned.`],
  ["Theory", "What is DistilBERT?", K => `A smaller student model trained to imitate BERT-base (<b>knowledge distillation</b>): 6 layers instead of 12, 40% smaller, 60% faster, ~97% of BERT’s language-understanding performance (Sanh et al., 2019). It has ${K.params_ft_distilbert_8000} parameters.`],
  ["Theory", "Why are Transformers considered faster to train than LSTMs, yet your Transformers are slower to run?", K => `LSTMs must process words one after another, so they cannot use parallel hardware well; attention processes all positions in parallel (great on GPUs). But our Transformers are also <em>much bigger</em> (4–125 M parameters vs 2 M for the BiLSTM) and we ran on a CPU. Latency: BiLSTM ${K.lat_bilstm} ms vs DistilBERT ${K.lat_ft_distilbert_8000} ms per sentence. Attention also scales as O(n²·d) in sentence length.`],

  // ---------------- method
  ["Method", "Why these ten models?", K => `They are the four options a practitioner really has: cheap classical models (3), a neural network without pre-training (1), Transformers fine-tuned by yourself in three sizes (3), and ready-made Transformers: in-domain, out-of-domain and zero-shot (3). Together they answer both “how much better?” and “at what cost?”.`],
  ["Method", "How did you prevent data leakage?", K => `The test set (official SST-2 validation, ${K.n_test} sentences) was used once per model for the final numbers. Epoch selection used a separate dev set (2,000 sentences cut from training). Hyper-parameters were fixed in advance from the literature, not tuned on the test set. The learning-curve subsets were drawn with a fixed seed.`],
  ["Method", "Why did you fine-tune on only 8,000 sentences instead of all 65,349?", K => `Time. On this laptop, DistilBERT needed ${K.train_h_distil} hours for 8,000 sentences (2 epochs); all 65k would take roughly 20 hours or more (extrapolated). 8,000 is also the realistic scenario of the study: a few thousand labels. The learning curve shows what happens with fewer labels.`],
  ["Method", "Which metrics did you use and why?", K => `Accuracy and macro-averaged precision, recall and F1. The test set is almost balanced (50.9% positive), so accuracy and macro F1 differ by less than 0.2 points. We added cost metrics: training time, latency per sentence (batch 32) and number of parameters.`],
  ["Method", "What is a bootstrap confidence interval?", K => `Resample the 872 test sentences with replacement 2,000 times, compute the accuracy each time, and take the 2.5th and 97.5th percentiles. It shows how much the accuracy could vary with a different sample. Our intervals are about ±2.5 points wide.`],
  ["Method", "What does McNemar’s test do, and what did you find?", K => `It compares two models on the <em>same</em> sentences: b = only A right, c = only B right. If equally good, b ≈ c. Fine-tuned DistilBERT vs TF-IDF SVM: ${K.mc_ft_svm_b} vs ${K.mc_ft_svm_c} sentences, p ${K.mc_ft_svm_p}, so the difference is real. BERT-small vs SVM is <b>not</b> significant. Try any pair in the Results explorer.`],

  // ---------------- results
  ["Results", "What is your main result?", K => `A sufficiently large pre-trained Transformer clearly beats classical and recurrent models: DistilBERT fine-tuned on 8,000 sentences reaches <b>${K.acc_ft_distilbert_8000}%</b> vs <b>${K.best_cls_acc}%</b> for the best classical model trained on ${K.n_train}. The ready-made in-domain DistilBERT reaches <b>${K.acc_pt_distilbert_sst2}%</b>. The price: 20–2,000× slower inference and 60–1,700× more parameters.`],
  ["Results", "Why is BERT-tiny worse than the classical models?", K => `Only 2 layers, 128-dimensional vectors, 4.4 M parameters and 8,000 training sentences: <b>${K.acc_ft_bert_tiny_8000}%</b>. Pre-training does not compensate for too little capacity. Lesson: “Transformer” is not a magic word; model size matters.`],
  ["Results", "Why does the BiLSTM not beat TF-IDF?", K => `Without pre-training it starts from random embeddings and has to learn language only from the training set. Result ${K.acc_bilstm}% vs ${K.best_cls_acc}% for the SVM, and the difference is not significant (p = 1.0). This agrees with Kapočiūtė-Dzikienė et al. (2019): for Lithuanian, Naive Bayes (0.735) beat CNN and LSTM (0.706).`],
  ["Results", "Why is the BiLSTM’s dev accuracy so much higher than its test accuracy?", K => `The dev set is part of the training data, i.e. mostly short phrases, while the test set has complete sentences (19.5 words on average vs 9.4). The model fits the phrase distribution well but generalises less to whole sentences.`],
  ["Results", "How many labels does a Transformer need?", K => `Surprisingly few. With 500 labels BERT-small reaches ${K.lc500_t}% vs ${K.lc500_c}% for TF-IDF (+${K.lc500_gain} points). With 4,000 labels it reaches ${K.lc4000_t}%, matching the best classical model trained on all ${K.n_train} (16× more data). This is the data-efficiency effect of pre-training.`],
  ["Results", "Is zero-shot useful? How can it work without labels?", K => `Yes, as a baseline when you have no labels: <b>${K.acc_zs_nli_distilroberta}%</b>, similar to TF-IDF logistic regression trained on 65k sentences. An NLI model judges whether “The sentiment of this movie review is positive” is entailed by the sentence. It is the most robust on negation (−0.2 points) but the slowest (${K.lat_zs_nli_distilroberta} ms), because every sentence is processed once per candidate label.`],
  ["Results", "What happens with negation and contrast?", K => `Bag-of-words models lose 8–9 points on the ${K.neg_n} negation sentences (they see “good” in “not good”). Large Transformers lose only 2–4 points, zero-shot NLI almost nothing. Small Transformers behave like the classical models (BERT-small −9.3). So the benefit comes from architecture <em>and</em> capacity <em>and</em> pre-training together.`],
  ["Results", "Can you trust the attention heat-map as an explanation?", K => `Not completely. It shows where information flows, and in our example the clause after “but” gets the weight, which is plausible. But Jain &amp; Wallace (2019) showed that attention weights are not necessarily faithful explanations of a prediction. We present it as an illustration, not proof.`],
  ["Results", "Which sentences did every model get wrong?", K => `${K.n_all_wrong} of ${K.n_test} sentences, such as “the lower your expectations, the more you’ll enjoy it” (irony). They need world knowledge or context that single sentences do not contain; some gold labels are simply ambiguous. See the Model arena (“all wrong”).`],
  ["Results", "Which model would you deploy?", K => `It depends on the situation (see “Which model would you choose?”). In-domain public model exists → use it. Otherwise fine-tune a distilled Transformer with a few thousand labels. No labels → zero-shot. Extreme speed/size limits (watch, millions of items per hour) → TF-IDF.`],

  // ---------------- Lithuania
  ["Lithuania", "How is your work related to Lithuanian NLP?", K => `We reviewed Lithuanian research (sentiment of Internet comments, LitLat BERT, BERT/T5 for reviews, Lithuanian Llama2) and our central finding agrees with it: <em>pre-training is the key factor</em>; neural networks without it do not beat classical methods. The same procedure can be repeated for Lithuanian with LitLat BERT or XLM-R.`],
  ["Lithuania", "Why did you use English data?", K => `SST-2 is a standard benchmark with published results for every model family, so we can validate our evaluation (our DistilBERT-SST2 91.1% reproduces the 91.3% reported by its authors), and ready-made models exist for all families. The method is language-independent; Lithuanian is the natural extension.`],

  // ---------------- limits
  ["Limits", "What are the limitations of your study?", K => `One random seed per configuration; one dataset, domain and language; subsets defined by keyword rules; no hyper-parameter tuning; fine-tuning on 8,000 sentences only; timings measured on one laptop that was partly busy; a test set of only ${K.n_test} sentences (±2.5 points). See the list below.`],
  ["Limits", "Are your timing results reliable?", K => `They are <b>upper bounds</b> for training: the laptop was also used for other work during the long runs. Latency is the average over the test set with batch size 32. Relative orders of magnitude (classical ≪ BiLSTM ≪ Transformers) are robust; exact seconds are not.`],
  ["Limits", "Would a GPU change your conclusions?", K => `The accuracy conclusions would not change. Training times of Transformers would shrink by one to two orders of magnitude and inference would be faster, so the cost gap would narrow, but classical models would still be far cheaper per sentence.`],
];

const QUIZ = [
  { q: "What does self-attention compute for each word?", o: ["How often the word occurs in the sentence", "A weighted mix of information from all words, with weights from similarity scores", "The probability of the next word", "A sentiment score for that word"], a: 1,
    e: "Scores Q·Kᵀ → softmax → weights; output = weighted average of the value vectors." },
  { q: "Why is the dot product divided by √d_k?", o: ["To make the sentence shorter", "To keep scores from growing with dimension, which would make softmax saturate and gradients vanish", "To count the heads", "To normalise the labels"], a: 1,
    e: "Large unscaled scores make softmax almost one-hot; its gradients become tiny and training stalls." },
  { q: "Which model had the highest accuracy on the test set?", o: K => ["TF-IDF + SVM", "BiLSTM", `DistilBERT fine-tuned by us (${K.acc_ft_distilbert_8000}%)`, `Ready-made DistilBERT-SST2 (${K.acc_pt_distilbert_sst2}%)`], a: 3,
    e: K => `The ready-made, already fine-tuned DistilBERT-SST2 scored ${K.acc_pt_distilbert_sst2}%; ours, trained on only 8,000 sentences, ${K.acc_ft_distilbert_8000}%.` },
  { q: "How many labelled sentences did BERT-small need to match the best classical model (which used all 65,349)?", o: ["About 500", "About 1,000", "About 4,000", "More than 65,000"], a: 2,
    e: K => `At 4,000 labels BERT-small reached ${K.lc4000_t}% vs ${K.best_cls_acc}%: about 16× less labelled data.` },
  { q: "Why is the official SST-2 validation set used as the test set?", o: ["It is the biggest split", "GLUE keeps the real test labels hidden", "Validation sets are always better", "The authors recommend it for BiLSTMs"], a: 1,
    e: "GLUE does not publish test labels. Common practice: use the validation set as the held-out test set and create your own dev set." },
  { q: "What does McNemar’s test compare?", o: ["The average length of sentences", "Training times of two models", "The sentences where exactly one of two models is right", "The number of parameters"], a: 2,
    e: "b = only A right, c = only B right. If the models are equally good, b ≈ c. The exact binomial test gives the p-value." },
  { q: "The BiLSTM had ~91% dev accuracy but 82.8% test accuracy. Why?", o: ["The test set contained errors", "The dev set consists of short training-distribution phrases while the test set has complete sentences", "The BiLSTM was pre-trained", "Dev sets are always easier by definition"], a: 1,
    e: "Dev is cut from the training data (short phrases, 9.6 words on average); test sentences average 19.5 words." },
  { q: "How does zero-shot classification with an NLI model work?", o: ["It is trained on sentiment labels first", "It checks whether a hypothesis such as “The sentiment of this movie review is positive” is entailed by the sentence", "It counts positive words", "It translates the sentence"], a: 1,
    e: "The label with the highest entailment probability wins. No sentiment labels are used." },
  { q: "What happened on sentences with negation?", o: ["All models did equally well", "Bag-of-words models lost 8–9 points, large Transformers only 2–4", "Transformers did worse than TF-IDF", "BiLSTM lost the most"], a: 1,
    e: "BERT-small and BERT-tiny, however, behaved like classical models: capacity and pre-training both matter." },
  { q: "What is the right way to read an attention heat-map?", o: ["As proof of how the model reasons", "As an illustration of information flow, not a guaranteed explanation", "As the sentiment score", "As the training loss"], a: 1,
    e: "Jain & Wallace (2019): attention weights are not necessarily faithful explanations." },
  { q: "How should the training times be interpreted?", o: ["As exact, hardware-independent values", "As upper bounds: the laptop was partly busy; orders of magnitude are what matter", "As GPU times", "As the time to pre-train the models"], a: 1,
    e: "We did not pre-train anything; we fine-tuned. The laptop was used for other work during the longest runs." },
  { q: "Which statement about BERT-tiny is true?", o: ["It was the best model", "It beat all classical models", "It was the weakest model: too little capacity", "It was pre-trained by us"], a: 2,
    e: K => `${K.acc_ft_bert_tiny_8000}% vs ${K.best_cls_acc}% for TF-IDF SVM. A Transformer is not automatically better.` },
];

const LIMITS = [
  ["One seed per configuration", "Differences below ~1 point could be random variation. We used bootstrap intervals and McNemar tests to judge which differences are reliable."],
  ["One dataset, one domain, one language", "English movie-review sentences (SST-2). Absolute numbers will differ for other domains and for morphologically rich languages such as Lithuanian."],
  ["Keyword-defined subsets", "“Negation” and “contrast” are found by simple word lists. They catch most, not all, such constructions."],
  ["Only 8,000 labels for fine-tuning", "Chosen for CPU time. More data would raise Transformer accuracy further (the learning curve is still rising)."],
  ["Timings on one partly busy laptop", "Training times are upper bounds. Orders of magnitude are reliable, exact seconds are not. No GPU timings were taken."],
  ["No hyper-parameter search", "Learning rates and epochs were fixed from the literature. Tuning might help small models in particular."],
  ["Ready-made models differ in many ways", "DistilBERT-SST2 and Twitter-RoBERTa were trained by others on different data and settings; the comparison shows what you get by downloading, not a controlled ablation."],
  ["Small test set", "872 sentences give ±2.5 point intervals; one point is about 9 sentences. Gold labels of ambiguous sentences are themselves debatable."],
];

const LIT = [
  { a: "Socher et al. (2013)", m: "Recursive neural tensor network", t: "SST sentiment, EN", r: "Created SST; bag-of-words NB ≈ 82%, SVM ≈ 79%, RNTN 85.4% (binary)", tag: "World" },
  { a: "Vaswani et al. (2017)", m: "Transformer (self-attention only)", t: "Machine translation", r: "Better quality than RNN/CNN at lower training cost", tag: "World" },
  { a: "Devlin et al. (2019)", m: "BERT, masked-LM pre-training", t: "GLUE incl. SST-2", r: "SST-2 test 93.5% (base) / 94.9% (large)", tag: "World" },
  { a: "Liu et al. (2019)", m: "RoBERTa", t: "GLUE", r: "Longer training, more data: SST-2 96.4%", tag: "World" },
  { a: "Sanh et al. (2019)", m: "DistilBERT (distillation)", t: "GLUE", r: "40% smaller, 60% faster, ~97% of BERT quality", tag: "World" },
  { a: "Turc et al. (2019)", m: "Compact BERTs (tiny…base)", t: "GLUE, distillation", r: "Pre-training compact models matters", tag: "World" },
  { a: "Yin et al. (2019)", m: "Zero-shot via NLI entailment", t: "Topic, emotion, situation", r: "Classification without task-labelled data", tag: "World" },
  { a: "Brown et al. (2020)", m: "GPT-3, in-context learning", t: "Many NLP tasks", r: "Few-shot learning without fine-tuning", tag: "World" },
  { a: "Zhang et al. (2024)", m: "LLMs (ChatGPT etc.) vs fine-tuned", t: "13 sentiment tasks", r: "Small fine-tuned models still win on complex tasks", tag: "World" },
  { a: "Kapočiūtė-Dzikienė et al. (2019)", m: "NB, SVM vs LSTM, CNN", t: "Lithuanian comments, 3 classes", r: "Traditional NB 0.735 > best deep 0.706 accuracy", tag: "Lithuania" },
  { a: "Ulčar & Robnik-Šikonja (2021)", m: "LitLat BERT (XLM-R based)", t: "Lithuanian, Latvian, English", r: "Trilingual model beats massively multilingual", tag: "Lithuania" },
  { a: "Vileikytė et al. (2024)", m: "Fine-tuned BERT and T5", t: "Lithuanian reviews, multi-domain", r: "High accuracy on clearly polar reviews", tag: "Lithuania" },
  { a: "Nakvosas et al. (2024)", m: "Open Llama2 7B/13B for Lithuanian", t: "Lithuanian language modelling", r: "First open Lithuanian LLMs, 14B+ tokens", tag: "Lithuania" },
];

const LT_TIMELINE = [
  ["2013", "Sentiment of Lithuanian comments", "Kapočiūtė-Dzikienė, Krupavičius &amp; Krilavičius compare machine-learning and lexicon approaches; inflection and non-standard spelling matter."],
  ["2019", "Deep nets vs classical", "Naive Bayes (0.735) beats the best CNN/LSTM (0.706). Neural networks <em>without</em> pre-training do not automatically win."],
  ["2021", "LitLat BERT", "Ulčar &amp; Robnik-Šikonja: a trilingual Lithuanian–Latvian–English model (XLM-R architecture). Corpus and vocabulary size matter."],
  ["2024", "BERT, T5 and Llama2 for Lithuanian", "Vileikytė et al. fine-tune BERT and T5 for reviews; Nakvosas et al. release the first open Lithuanian Llama2 models (7B, 13B)."],
];

const SLIDES = [
  { n: 1, t: "Title", who: "shashi", s: 30, key: "Do Transformers pay off on a laptop CPU? We compared 10 sentiment classifiers on SST-2." },
  { n: 2, t: "Reading word by word vs all at once", who: "rimsha", s: 45, key: "LSTMs are sequential. The Transformer lets every token attend to every other and is trained in parallel." },
  { n: 3, t: "Self-attention", who: "rimsha", s: 60, key: "softmax(QKᵀ/√d_k)·V. Heads, layers, [CLS] → softmax classifier." },
  { n: 4, t: "Literature", who: "shaeed", s: 60, key: "Google Scholar + Web of Science. Lithuanian work shows pre-training is the key. Gap: cost and small data." },
  { n: 5, t: "Experiment design", who: "shaeed", s: 45, key: "SST-2 (65k / 2k / 872), 10 models in four groups, one laptop CPU, criteria incl. CI and McNemar." },
  { n: 6, t: "Implementation", who: "shashi", s: 40, key: "Reproducible pipeline: one generic fine-tuning script; results saved as JSON; report script builds tables." },
  { n: 7, t: "Accuracy", who: "shashi", s: 60, key: "DistilBERT fine-tuned 88.8% vs best classical 82.9% (+5.9); ready-made 91.1%; zero-shot 81.8%." },
  { n: 8, t: "The price", who: "shashi", s: 45, key: "~700× slower inference, hours of CPU training, 67 M parameters. Still only ~40 ms per sentence." },
  { n: 9, t: "Learning curve", who: "rimsha", s: 50, key: "With 500 labels: +14 points. BERT-small with 4,000 labels matches TF-IDF with 65k." },
  { n: 10, t: "Negation and attention", who: "rimsha", s: 45, key: "Bag-of-words lose 8–9 points on negation, big Transformers 2–4. Attention is an illustration, not proof." },
  { n: 11, t: "Conclusions", who: "shaeed", s: 60, key: "Reuse first, fine-tune a distilled model second, TF-IDF when speed/size dominate. Next: LitLat BERT on Lithuanian." },
];
const SPEAKERS = { shashi: "Shashikanth", shaeed: "Shaeed", rimsha: "Rimsha" };

const CHECKLIST = [
  "I can explain the whole project in 60 seconds (question, method, answer).",
  "I know the six key numbers (best accuracy, best classical, 8k vs 65k, +14 at 500 labels, latency ratio, sizes).",
  "I can explain self-attention with the “highlighter” analogy and write softmax(QKᵀ/√d_k)V.",
  "I can draw the nine pipeline steps from memory.",
  "I can explain why the validation set is the test set and what the dev set is for.",
  "I can explain a confidence interval and McNemar’s test in one sentence each.",
  "I can name three limitations before I am asked.",
  "I know who answers which questions in the team.",
  "I rehearsed with the timer and finished between 7:00 and 10:00.",
  "The slides open on the presentation laptop; a PDF backup exists.",
];

const FILES = [
  { p: "README.md", d: "Overview, results table, structure, how to reproduce.", say: "Start here if you only have one minute." },
  { p: "code/", d: "The experiment (Python). Run <code>bash run_all.sh</code> to reproduce everything.", dir: true },
  { p: "code/01_prepare_data.py", d: "Downloads SST-2 and creates train / dev / test CSV files (65,349 / 2,000 / 872).", say: "Official validation = test; dev is carved from train." },
  { p: "code/02_classical_baselines.py", d: "TF-IDF (1–2-grams) + Naive Bayes, logistic regression and linear SVM." },
  { p: "code/03_bilstm.py", d: "BiLSTM with its own tokeniser and vocabulary; max-pooling; 4 epochs; best epoch on dev." },
  { p: "code/04_finetune_transformer.py", d: "Generic fine-tuning for any Hugging Face encoder: <code>--model</code>, <code>--train-size</code>, <code>--epochs</code>, <code>--lr</code>. AdamW + warm-up + linear decay.", say: "The core of the practical work." },
  { p: "code/05_pretrained_transformers.py", d: "Evaluates the three ready-made models (in-domain, out-of-domain, zero-shot) through pipelines." },
  { p: "code/06_learning_curve_classical.py", d: "TF-IDF logistic regression on the same random subsets as BERT-small (500…8,000)." },
  { p: "code/07_attention_visualisation.py", d: "Extracts DistilBERT attention for an example sentence and plots heat-maps." },
  { p: "code/08_report.py", d: "Reads all JSON results; builds tables, figures, subset analysis, bootstrap CIs and McNemar tests." },
  { p: "code/09_diagrams.py", d: "Draws the architecture and pipeline diagrams used in the paper." },
  { p: "code/utils.py", d: "Shared helpers: loading splits, metrics (accuracy, macro P/R/F1, confusion matrix), timer, saving results." },
  { p: "code/run_all.sh", d: "One command that runs every script in the right order (~2.5–3 hours on a 4-core laptop)." },
  { p: "code/results/", d: "One JSON per model (metrics, timings, parameters, predictions) and CSV summaries. <b>This guide is built from these files.</b>", dir: true },
  { p: "code/figures/", d: "All charts and diagrams used in the paper and slides.", dir: true },
  { p: "paper/", d: "Chapter texts (markdown), <code>build_docx.py</code> (builds the paper on the VILNIUS TECH template), <code>check_docx.py</code> (quality checks).", dir: true },
  { p: "slides/build_deck.js", d: "Builds the 11-slide presentation from the results (so slide numbers cannot disagree with the paper)." },
  { p: "docs/", d: "Final paper (DOCX, PDF), presentation (PPTX) and this study guide.", dir: true },
  { p: "study_guide/", d: "<code>export_data.py</code> collects real results (incl. the TF-IDF model weights and DistilBERT attention) → <code>build.py</code> writes this HTML.", dir: true },
];

const CMDS = [
  ["Set up the environment", `cd code
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt`],
  ["Run the whole experiment (~2.5–3 h on a laptop CPU)", `bash run_all.sh`],
  ["Fine-tune one model yourself", `python 04_finetune_transformer.py --model distilbert-base-uncased \\
    --train-size 8000 --epochs 2 --lr 5e-5 --tag ft_distilbert_8000`],
  ["Rebuild tables, figures and statistics", `python 08_report.py`],
  ["Rebuild paper and slides (Windows + MS Office)", `python paper/build_docx.py
powershell -File paper/word_finalize.ps1 -Docx docs/Transformers_in_NLP_Project_Paper.docx -Pdf docs/Transformers_in_NLP_Project_Paper.pdf
cd slides && npm install && node build_deck.js`],
  ["Rebuild this study guide", `python study_guide/export_data.py     # needs the project environment
python study_guide/build.py`],
];

const PAPER_MAP = [
  ["Problem, aim, objectives, RQs", "Introduction; §2.1", "The story; Pipeline step 1"],
  ["Literature search and review", "Chapter 1; Table 2", "Literature and Lithuania"],
  ["Dataset and splits", "§2.2; Table 3", "Pipeline step 3"],
  ["Models compared", "§2.3; Table 4", "Pipeline steps 4–7"],
  ["Metrics, protocol, validity", "§2.4, §2.8", "Pipeline step 8; Limitations"],
  ["Transformer architecture and maths", "§2.6, §2.7", "How Transformers work"],
  ["Overall results", "§3.2; Tables 6–7; Fig. 3", "Results explorer; Model arena"],
  ["Cost (time, speed, size)", "§3.3; Fig. 4", "Results → cost chart"],
  ["Data efficiency", "§3.4; Table 8; Fig. 5", "Results → learning curve"],
  ["Ready-made / zero-shot", "§3.5", "Choose a model"],
  ["Negation, contrast, attention", "§3.6; Table 9; Figs. 6–7", "Sentence lab; Results → subsets"],
  ["Conclusions and recommendations", "Conclusions", "Defence prep"],
];

const GLOSSARY = [
  ["Token", "A piece of text (word or sub-word) that the model treats as one unit, e.g. “un”, “##watch”, “##able”."],
  ["Embedding", "A list of numbers (a vector) that represents a token; similar meanings get similar vectors."],
  ["Positional encoding", "Information about the position of each token, added to its embedding so that word order is known."],
  ["Self-attention", "Each token computes weights over all tokens and collects a weighted mix of their information."],
  ["Query, key, value (Q, K, V)", "Three projections of each token: what it looks for, what it offers for matching, and what it passes on."],
  ["Softmax", "Turns a list of scores into probabilities that sum to 1: softmax(z)ⱼ = e^zⱼ / Σ e^zₖ."],
  ["Multi-head attention", "Several attention operations in parallel, each with its own projections; results are concatenated."],
  ["Residual connection", "Adds a layer’s input to its output, so information and gradients can skip layers."],
  ["Layer normalisation", "Rescales each vector to zero mean and unit variance for stable training."],
  ["Feed-forward network (FFN)", "A small two-layer network applied to every token separately inside each Transformer layer."],
  ["Encoder / decoder", "Encoder reads and understands text (BERT); decoder generates text one token at a time (GPT)."],
  ["BERT", "Bidirectional Encoder Representations from Transformers: a pre-trained encoder (Devlin et al., 2019)."],
  ["DistilBERT", "A smaller BERT trained by distillation: 6 layers, 40% smaller, 60% faster, ~97% of BERT’s quality."],
  ["RoBERTa", "A better-trained BERT variant (more data, longer training, no next-sentence task)."],
  ["GPT", "Generative Pre-trained Transformer: a decoder that predicts the next word; basis of chat assistants."],
  ["[CLS] and [SEP]", "Special tokens: [CLS] at the start gathers a summary of the sentence; [SEP] marks the end."],
  ["Pre-training", "Learning general language from huge unlabelled text (e.g. masked-word prediction)."],
  ["Masked language modelling (MLM)", "Hide ~15% of the words and train the model to predict them from context."],
  ["Fine-tuning", "Continuing training of a pre-trained model on a small labelled task dataset."],
  ["Knowledge distillation", "Training a small “student” model to imitate a large “teacher”."],
  ["Zero-shot classification", "Classifying without task-specific training examples, here via natural language inference."],
  ["Natural language inference (NLI)", "Deciding whether a hypothesis is entailed by, contradicts or is neutral to a premise."],
  ["TF-IDF", "Term frequency × inverse document frequency: weights words by how characteristic they are for a sentence."],
  ["N-gram / bigram", "A sequence of N consecutive words; a bigram is a pair such as “not good”."],
  ["Naive Bayes", "A classifier that multiplies per-word probabilities assuming words are independent."],
  ["Logistic regression", "A linear classifier: P(positive) = σ(w·x + b). Each word adds or subtracts evidence."],
  ["Support vector machine (SVM)", "A linear classifier that maximises the margin between the classes."],
  ["LSTM / BiLSTM", "A recurrent network with gates that keeps a memory; the bidirectional version reads both directions."],
  ["Cross-entropy loss", "The training objective for classification: −log of the probability given to the true class."],
  ["AdamW", "An optimiser (Adam with decoupled weight decay); updates all parameters using running averages of gradients."],
  ["Warm-up and linear decay", "The learning rate rises for the first 10% of steps and then falls linearly to zero."],
  ["Epoch / batch", "One pass over the training data / the group of examples processed together."],
  ["Overfitting", "Fitting the training data (or its distribution) so closely that performance on new data drops."],
  ["Train / dev / test set", "Data to learn from / to make choices (e.g. which epoch) / to report the final, unbiased result."],
  ["Accuracy", "Share of correctly classified sentences."],
  ["Precision, recall, F1", "Precision = right among predicted positives; recall = found among true positives; F1 = their harmonic mean."],
  ["Macro average", "Metric computed per class and averaged, so classes weigh equally."],
  ["Confusion matrix", "Table of true vs predicted classes."],
  ["Bootstrap confidence interval", "Range obtained by resampling the test set many times; shows sampling uncertainty."],
  ["McNemar test", "Paired test of two classifiers using the sentences where exactly one is right."],
  ["p-value", "Probability of a difference at least this large if the models were really equally good."],
  ["Latency", "Time to classify one sentence (we report ms per sentence on the CPU)."],
  ["Parameters", "The learned numbers in a model (weights); 4.4 M for BERT-tiny, 67 M for DistilBERT."],
  ["Learning curve", "Accuracy as a function of the number of training examples."],
  ["SST-2 / GLUE", "Stanford Sentiment Treebank (binary) / a benchmark collection for language understanding."],
  ["Hugging Face Hub", "A repository of pre-trained models and datasets, with the Transformers library to use them."],
  ["LitLat BERT", "A trilingual (Lithuanian, Latvian, English) BERT-type model by Ulčar & Robnik-Šikonja (2021)."],
];
