/* Static teaching content, part A: story, pipeline, theory, scenarios. Functions receive K (formatted key numbers). */

const ENGINEERS = [
  { id: "anna", name: "Anna", idea: "“Count the words.”", tag: "tfidf_svm", color: "classical",
    text: K => `Anna builds a <b>keyword model</b> (TF-IDF + linear SVM). It counts which words and word pairs appear and weighs them. It is fast to build and runs on anything.`,
    wins: "Speed and simplicity: trains in seconds, answers in a few hundredths of a millisecond, tiny model.",
    loses: "It cannot read word order beyond pairs. “not entertaining” contains the happy word “entertaining”. It loses 8–9 points on sentences with negation." },
  { id: "ben", name: "Ben", idea: "“Train a neural network.”", tag: "bilstm", color: "recurrent",
    text: K => `Ben trains a <b>BiLSTM</b> from scratch (a recurrent network that reads the sentence left-to-right and right-to-left). It is a “proper” neural network, and it feels modern.`,
    wins: "It does read the sentence in order, so it handles negation better than word counting.",
    loses: "It starts knowing nothing about language. With 65k labelled sentences it ends up <b>exactly as accurate as Anna’s model</b> (82.8% vs 82.9%) after 100× more training time." },
  { id: "chloe", name: "Chloe", idea: "“Start from a Transformer.”", tag: "ft_distilbert_8000", color: "transformer",
    text: K => `Chloe takes a <b>pre-trained Transformer</b> (DistilBERT) that has already read huge amounts of text, and fine-tunes it on the 8,000 labelled comments. Or she skips training and downloads a model somebody already fine-tuned.`,
    wins: "Accuracy, data efficiency and understanding of context. Even with 8× less labelled data she beats both Anna and Ben.",
    loses: "Cost: the model has 67 million parameters, needs hours of CPU training and ~40 ms per sentence (700× slower than Anna’s)." },
];

const ANALOGIES = [
  ["Bag of words / TF-IDF", "A librarian who counts how often each word occurs and knows which words are rare. She never looks at the order of the words."],
  ["BiLSTM (recurrent network)", "Reading a sentence word by word with a small notebook. By the end of a long sentence you have forgotten the beginning."],
  ["Self-attention", "Reading the whole sentence at once with a highlighter. For each word you highlight the other words that matter for it."],
  ["Pre-training", "A student who has read millions of pages before ever seeing your exam. Expensive, but done once and shared with everybody."],
  ["Fine-tuning", "A short on-the-job course that turns that well-read student into a specialist for your company."],
  ["Zero-shot classification", "Asking the well-read student a question they never practised: “Is this review positive or negative?” and they answer from general knowledge."],
  ["Knowledge distillation (DistilBERT)", "A condensed summary of a thick textbook: 40% smaller, 60% faster, and it keeps about 97% of the knowledge."],
  ["Overfitting / dev ≠ test", "Memorising last year’s practice exam. You score 91% on the practice paper and 83% on the real, different exam (our BiLSTM)."],
  ["Positional encoding", "Page numbers for words. Without them “dog bites man” and “man bites dog” look the same to attention."],
  ["Confidence interval", "A margin of error, like in election polls: with only 872 sentences, 91.1% really means “somewhere between 89 and 93%”."],
];

const ARCH = [
  { id: "in", label: "Input tokens", title: "Input tokens", body: K => `<p>The sentence is cut into sub-word tokens. Special tokens are added: <code>[CLS]</code> at the start (a place where the model gathers a summary of the whole sentence) and <code>[SEP]</code> at the end. We truncate to 64 tokens (this covers 99.99% of training sentences and all test sentences).</p>`, say: "Text is turned into pieces the model knows." },
  { id: "emb", label: "Token embedding + position", title: "Embeddings", body: K => `<p>Each token ID is looked up in an embedding table and becomes a vector of <b>d</b> numbers: 128 in BERT-tiny, 512 in BERT-small, 768 in DistilBERT. A position vector is added so the model knows word order.</p><div class="formula">PE(pos, 2i) = sin(pos / 10000<sup>2i/d</sup>) &nbsp;&nbsp; PE(pos, 2i+1) = cos(pos / 10000<sup>2i/d</sup>)</div><p class="small muted">BERT-type models learn the position vectors instead of using these fixed sine waves, but the idea is the same.</p>`, say: "Meaning plus position." },
  { id: "attn", label: "Multi-head self-attention", title: "Multi-head self-attention", loop: true, body: K => `<p>Every token compares itself with every other token and collects information from the relevant ones. <b>h</b> heads work in parallel (2, 8 and 12 heads in our three fine-tuned models), each with its own Q, K, V projections. Their results are concatenated.</p><div class="formula">Attention(Q,K,V) = softmax(Q·Kᵀ / √d<sub>k</sub>) · V</div><p><b>In our sentence “not good”:</b> the head that links negations to the words they modify lets “good” absorb the meaning of “not”. See it in the Sentence lab.</p>`, say: "Each word looks at all the others and takes what it needs." },
  { id: "norm1", label: "Add & layer normalisation", title: "Residual connection + layer normalisation", loop: true, body: K => `<p><b>Residual (“Add”)</b>: the input of the sub-layer is added back to its output, a shortcut that lets gradients flow through deep stacks. <b>Layer normalisation</b> rescales each vector to mean 0 and variance 1, which stabilises training.</p><div class="formula">H′ = LayerNorm(H + MultiHead(H))</div>`, say: "Shortcuts and rescaling keep deep networks trainable." },
  { id: "ffn", label: "Feed-forward network", title: "Position-wise feed-forward network", loop: true, body: K => `<p>A small two-layer network applied to each token separately (inner size 4×d, GELU activation). Attention <em>moves information between tokens</em>; the feed-forward layer <em>processes it within each token</em>.</p><div class="formula">FFN(x) = GELU(xW₁ + b₁)W₂ + b₂</div>`, say: "Think about what you gathered." },
  { id: "norm2", label: "Add & layer normalisation", title: "Second residual + normalisation", loop: true, body: K => `<p>Same trick again after the feed-forward network. Together, the four blocks inside the dashed frame form <b>one layer</b>. The layer is repeated <b>L times</b>: <b>2</b> in BERT-tiny, <b>4</b> in BERT-small, <b>6</b> in DistilBERT (BERT-base has 12).</p><p>More layers and bigger vectors = more capacity = more parameters (4.4 M → 28.8 M → 67.0 M) = better accuracy but slower and larger. Our results show this trade-off clearly.</p>`, say: "Stack the layer a few times." },
  { id: "cls", label: "[CLS] vector", title: "Contextual vectors and the [CLS] summary", body: K => `<p>After the last layer every token has a <b>contextual vector</b>: “good” in “not good” is now different from “good” in “very good”. For classification we use only the <code>[CLS]</code> vector as a summary of the whole sentence.</p>`, say: "One vector summarises the sentence." },
  { id: "head", label: "Classifier: softmax(W·h + b)", title: "Classification head and loss", body: K => `<p>A new linear layer (the only thing that is added for our task) maps the <code>[CLS]</code> vector to two numbers; <b>softmax</b> turns them into P(negative) and P(positive). Training minimises the <b>cross-entropy</b> between these probabilities and the true labels with AdamW, a learning rate with 10% warm-up then linear decay, and gradient clipping at 1.0.</p><div class="formula">P(y|x) = softmax(W·h<sub>CLS</sub> + b) &nbsp;&nbsp; L = −(1/N) Σ log P(y<sub>i</sub>|x<sub>i</sub>)</div>`, say: "Two numbers, turned into probabilities." },
];

const TOKENS_NOTE = "Real output of the DistilBERT tokenizer.";

const PIPE = [
  { short: "Question", title: "Define the problem and the research questions",
    paper: "Introduction; Chapter 2 §2.1", rw: "ReviewRadar: “Which of the three proposals should we build?”",
    what: ["Task: <b>binary sentiment classification</b>. Given a sentence, predict negative (0) or positive (1).",
           "Problem: how large is the <em>practical</em> advantage of Transformers over classical and recurrent models, and what does it cost on a CPU-only laptop?",
           "Five research questions (RQ1–RQ5), five objectives, a list of methods. See “The story”."],
    why: "A precise question lets you design a <b>fair</b> experiment, and sentiment analysis has published results for every model family, so our numbers can be checked against the literature.",
    out: "Aim, objectives, research questions (paper: Introduction).",
    say: "We did not want to repeat “BERT is great”. We asked what it costs and when it is worth it.",
    ask: [["Why sentiment analysis?", "It is one of the most common real NLP applications, it is a standard benchmark with published results for all model families, and it contains hard phenomena (negation, contrast, irony) that show whether a model really uses context."]] },

  { short: "Literature", title: "Search and analyse the literature",
    paper: "Chapter 1 (Literature review), Table 2", rw: "ReviewRadar: “What do other people already know about this?”",
    what: ["Searched <b>Google Scholar</b> and <b>Web of Science</b> (plus ACL Anthology, arXiv) with a specific query: (transformer OR BERT OR self-attention…) AND (NLP OR text classification OR sentiment analysis).",
           "Screened by title and abstract; ~40 works kept. Added the seminal ones by citation tracking (Transformer, BERT, GPT, RoBERTa, DistilBERT…).",
           "Worldwide and Lithuanian research compared in a table; conclusions list what is missing."],
    why: "The review shows what is known and <b>justifies the experiment</b>: nobody had compared quality, cost and data efficiency together on ordinary hardware.",
    out: "Chapter 1 and the comparison table (13 key works).",
    say: "Most results come from GPUs with all the data. We wanted the view of a student or a small company.",
    ask: [["What is the research gap?", "(1) Results are mostly reported on GPUs with the full training set. (2) Cost (CPU time, latency, size) is rarely reported next to accuracy. (3) The data efficiency of compact pre-trained models is insufficiently studied, and in Lithuanian work deep networks without pre-training did not beat classical methods."]] },

  { short: "Data", title: "Prepare the data: train, dev and test",
    paper: "Chapter 2 §2.2; Table 3", rw: "ReviewRadar: “We keep some comments hidden to test fairly.”", viz: "split",
    what: ["Dataset: <b>Stanford Sentiment Treebank</b>, binary version <b>SST-2</b> from the GLUE benchmark.",
           "GLUE hides the test labels, so the official <b>validation set (872 sentences) is our test set</b>.",
           "We cut <b>2,000 sentences from the training data as a dev set</b> (to choose the best epoch). The remaining <b>65,349</b> sentences are the training set.",
           "Notice the difference: training items are short phrases (9.4 words on average), test items are whole sentences (19.5 words). Models must generalise."],
    why: "The test set must never influence training <em>or</em> any choice we make; otherwise the result is optimistically biased. The dev set exists so that choices (which epoch) never touch the test set.",
    code: { file: "code/01_prepare_data.py", text: `ds = load_dataset("nyu-mll/glue", "sst2")
train_full = ds["train"].to_pandas()[["sentence", "label"]]
test = ds["validation"].to_pandas()[["sentence", "label"]]   # official validation = our TEST set
train, dev = train_test_split(train_full, test_size=2000,
                              random_state=42, stratify=train_full["label"])` },
    out: "data/train.csv, dev.csv, test.csv (not stored in the repo; re-created by the script).",
    say: "Test labels are hidden in GLUE, so we used the official validation set as the test set and carved our own dev set out of the training data.",
    ask: [["Why is the validation set your test set?", "Because the GLUE test labels are hidden. It is common practice. We never used it for training or selection; the epoch was chosen on our own dev set."],
          ["Could the ready-made DistilBERT-SST2 have seen your test set?", "No. It was fine-tuned on the SST-2 <em>training</em> split; our test set is the official validation split. Its authors report 91.3% on that split and we measured 91.1%, which confirms our evaluation is correct."]] },

  { short: "Classical", title: "Baseline 1: word counting (TF-IDF + NB / LR / SVM)",
    paper: "Chapter 2 §2.7.4–2.7.5", rw: "Anna’s keyword model.",
    what: ["Every sentence becomes a long sparse vector of <b>TF-IDF</b> weights for single words and word pairs (73,556 features). A word is important if it is frequent in the sentence but rare overall.",
           "Three simple classifiers on top: <b>Naive Bayes</b>, <b>logistic regression</b>, <b>linear SVM</b>.",
           "Trains in 2–4 seconds on all 65,349 sentences and answers in ~0.05 ms."],
    why: "A strong, cheap baseline. If a fancy model cannot beat it clearly, the fancy model is not worth its cost.",
    code: { file: "code/02_classical_baselines.py", text: `pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(C=4.0, max_iter=2000)),
])
pipe.fit(train["sentence"], train["label"])
pred = pipe.predict(test["sentence"])` },
    out: "results/tfidf_nb.json, tfidf_logreg.json, tfidf_svm.json: ~82–83% accuracy.",
    say: "The simple baselines already reach 82–83%. That is the bar a Transformer has to clear.",
    ask: [["What is TF-IDF?", "Term frequency × inverse document frequency: tf(t,d) damped with 1+log, times log((1+N)/(1+df))+1, then every vector is length-normalised. It rewards words that are frequent in this sentence but rare in the collection."],
          ["Why does it fail on “not good”?", "It sees the words “not” and “good” (and the pair “not good” only if it was frequent in training). It has no real notion of negation scope. Try it in the Sentence lab."]] },

  { short: "BiLSTM", title: "Baseline 2: a recurrent network from scratch",
    paper: "Chapter 2 §2.7.6", rw: "Ben’s neural network.",
    what: ["A <b>bidirectional LSTM</b>: one LSTM reads left→right, another right→left; hidden states are concatenated and <b>max-pooled</b> over time.",
           "Random word embeddings (128-d); <b>no pre-training</b>. 2.0 million parameters. 4 epochs with Adam, best epoch chosen on the dev set.",
           "Result: 82.8%, the same as the keyword models, after 6 minutes of training."],
    why: "Shows what a neural network can do <em>without</em> pre-training, the pre-Transformer standard. It reproduces the Lithuanian finding that deep networks without pre-training do not beat classical methods.",
    code: { file: "code/03_bilstm.py", text: `class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        mask = (x != 0).unsqueeze(-1)
        h = h.masked_fill(~mask, -1e9).max(dim=1).values   # max-pooling over time
        return self.out(self.drop(h))` },
    out: "results/bilstm.json (accuracy 82.8%, dev accuracy up to 91.3%).",
    say: "A neural network is not automatically better. Without pre-training it only matched the keyword models.",
    ask: [["Why is the BiLSTM’s dev accuracy (91%) so much higher than its test accuracy (83%)?", "The dev set is cut from the training data, which consists of short phrases and the same distribution; the test set contains full sentences. The model fits the training distribution well and generalises less to complete sentences."]] },

  { short: "Fine-tune", title: "Fine-tune Transformers (BERT-tiny, BERT-small, DistilBERT)",
    paper: "Chapter 2 §2.6–2.7.3; Chapter 3 §3.1", rw: "Chloe’s approach, step 1: specialise a well-read model.",
    what: ["One generic script: give it any model name from the Hugging Face hub and a number of training sentences.",
           "Adds a classification layer on <code>[CLS]</code>; trains everything with <b>AdamW</b>, 10% warm-up, linear decay, gradient clipping 1.0, batch 32, max 64 tokens.",
           "<b>8,000</b> labelled sentences, 2–3 epochs, best epoch on dev. Sizes: BERT-tiny 4.4 M, BERT-small 28.8 M, DistilBERT 67.0 M parameters.",
           "Why only 8,000? On this CPU DistilBERT already needs ~2.6 hours; all 65k would take about a day."],
    why: "This is the realistic scenario: a few thousand labels, a laptop. Three sizes reveal how quality and cost grow with model size.",
    code: { file: "code/04_finetune_transformer.py", text: `model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=math.ceil(0.1 * steps),
                                        num_training_steps=steps)
for batch in train_dl:
    out = model(**batch)                 # cross-entropy loss computed inside the model
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()` },
    out: "results/ft_bert-tiny_8000.json, ft_bert-small_8000.json, ft_distilbert_8000.json.",
    say: "We did not train a Transformer from scratch. We downloaded pre-trained weights and fine-tuned them. That is how everybody uses Transformers.",
    ask: [["Why is BERT-tiny worse than the classical models?", "It has only 2 layers and 128-dimensional vectors (4.4 M parameters). Pre-training helps, but this capacity is too small for the task. A Transformer is not automatically better: size matters."],
          ["Did you pre-train the models?", "No. Pre-training takes days on many GPUs. We used publicly released pre-trained weights (Google’s compact BERTs, DistilBERT) and only fine-tuned."]] },

  { short: "Ready-made", title: "Ready-made and zero-shot models",
    paper: "Chapter 2 §2.7.7; Chapter 3 §3.5", rw: "Chloe’s approach, step 2: “Maybe somebody already did the work.”",
    what: ["<b>DistilBERT-SST2</b>: already fine-tuned on SST-2 by others (the in-domain best case).",
           "<b>Twitter-RoBERTa</b>: fine-tuned for sentiment on <em>tweets</em> (a different domain, 3 classes; we compare only positive vs negative scores).",
           "<b>Zero-shot NLI</b>: a model trained only to judge entailment. We ask “The sentiment of this movie review is positive/negative.” and choose the more entailed hypothesis. <b>No sentiment label was ever used.</b>"],
    why: "Many teams will download a model instead of training. How good is that, and how much does domain matter?",
    code: { file: "code/05_pretrained_transformers.py", text: `clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
zs = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
out = zs(texts, candidate_labels=["positive", "negative"],
         hypothesis_template="The sentiment of this movie review is {}.")` },
    out: "results/pt_distilbert_sst2.json (91.1%), pt_twitter_roberta.json (85.8%), zs_nli_distilroberta.json (81.8%).",
    say: "In-domain ready-made: best. Wrong domain: still beats classical. Zero-shot with no labels at all: as good as a classical model trained on 65 thousand.",
    ask: [["How does zero-shot classification work?", "An NLI model judges whether a premise entails a hypothesis. We use the sentence as premise and build a hypothesis for each label from a template; the label with the highest entailment probability wins. No labelled training data are needed."]] },

  { short: "Evaluate", title: "Evaluate fairly: metrics, learning curve, statistics, error analysis",
    paper: "Chapter 2 §2.4, §2.7.8; Chapter 3 §3.2–3.6", rw: "ReviewRadar: “Is the difference real, and where does it fail?”",
    what: ["<b>Metrics</b>: accuracy, macro precision/recall/F1 (classes are balanced), training time, latency per sentence (batch 32), parameter count.",
           "<b>Learning curve</b>: BERT-small vs TF-IDF trained on identical random subsets of 500, 1,000, 2,000, 4,000, 8,000 sentences.",
           "<b>Statistics</b>: 95% <b>bootstrap</b> confidence intervals (2,000 resamples) and the exact <b>McNemar test</b> for pairs of models.",
           "<b>Error analysis</b>: accuracy on 184 sentences with negation and 130 with contrast; attention heat-maps; examples where models disagree."],
    why: "With only 872 test sentences, one percentage point = 9 sentences. Statistics tell us which differences are real; subsets tell us <em>why</em> models differ.",
    code: { file: "code/08_report.py", text: `boot = correct[rng.integers(0, n, size=(2000, n))].mean(axis=1)    # bootstrap accuracy
ci = np.percentile(boot, [2.5, 97.5])
p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue                 # exact McNemar test` },
    out: "summary_table.csv, statistics.csv, subset_accuracy.csv, learning_curve.csv, figures/*.png.",
    say: "We did not just report accuracies. We tested whether differences are statistically real, and we looked at where models fail.",
    ask: [["What is a bootstrap confidence interval?", "Resample the test set with replacement 2,000 times, compute the accuracy each time, and take the 2.5th and 97.5th percentiles. It shows how much accuracy could vary with a different sample of sentences."],
          ["What does McNemar’s test tell you?", "It compares two models on the same sentences by counting the cases where only one of them is right (b and c). If the models were equally good, b ≈ c. A small p-value means the split is unlikely to be luck."]] },

  { short: "Deliver", title: "Report, paper, slides and repository",
    paper: "Conclusions; Appendix A", rw: "ReviewRadar: “Tell management what to build.”",
    what: ["<code>08_report.py</code> turns the JSON results into tables, charts and statistics.",
           "<code>paper/build_docx.py</code> builds the paper on the official VILNIUS TECH template; <code>slides/build_deck.py</code> builds the presentation on the VILNIUS TECH template from the same results, so numbers never disagree.",
           "Everything (code, results, paper, slides, this guide) is in the GitHub repository; <code>run_all.sh</code> re-runs the whole experiment (~2.5–3 h)."],
    why: "Reproducibility: anybody can re-run it, and every number in the paper traces back to a result file.",
    out: "docs/ paper (DOCX, PDF) and presentation; conclusions with practical recommendations.",
    say: "Reuse first, fine-tune a distilled model second, and use keyword models only when speed and size dominate.",
    ask: [["What are your practical recommendations?", "(1) Check for a public model already fine-tuned on your kind of text. (2) Otherwise fine-tune a distilled Transformer on a few thousand labels. (3) With no labels, use a zero-shot model as a baseline. (4) Use TF-IDF only when inference must be extremely cheap or the hardware very limited."]] },
];

const SCENARIOS = [
  { id: "cinema", name: "Cinema app (ReviewRadar)", labels: 8000, lat: 100, train: 600, inDomain: true, related: true,
    desc: "8,000 labelled comments, laptop, answers within 100 ms, a model for movie reviews exists." },
  { id: "watch", name: "On-device keyboard", labels: 8000, lat: 1, train: 600, inDomain: false, related: false,
    desc: "Runs on a phone or watch: about 1 ms per sentence, no suitable public model." },
  { id: "startup", name: "Start-up, only 300 labels", labels: 500, lat: 200, train: 600, inDomain: false, related: false,
    desc: "Almost no labelled data, no public in-domain model, 200 ms is fine." },
  { id: "stream", name: "Huge stream (1M comments/hour)", labels: 65349, lat: 3.5, train: 600, inDomain: false, related: false,
    desc: "About 280 sentences per second on one CPU core, so under 3.5 ms each." },
  { id: "lt", name: "Lithuanian news monitoring", labels: 2000, lat: 200, train: 600, inDomain: false, related: false, lt: true,
    desc: "Lithuanian text. Our measured models are English, so read this as a method, not as numbers." },
];

const WALK_NOTES = {
  0: "Easy case: “charming” and “affecting” are clearly positive. All ten models agree. A good example that most sentences are easy; the interesting differences are in the hard minority.",
  4: "A clear negative (“slow … very, very slow”). Everyone is right. Use it to show that on easy text a keyword model is perfectly fine.",
  627: "<b>Negation.</b> “not entertaining”: a word-counting model sees the happy word “entertaining”. Look at how its evidence is split between the words, and at how the Transformers use “not”.",
  110: "<b>Negation with “no”.</b> “scores <em>no</em> points for originality, wit, or intelligence”: all the nouns are positive words, the negation flips them. Classical models are fooled; Transformers read “no points for …”.",
  398: "<b>Double negation.</b> “despite its title … is <em>never</em> heavy-handed”: “heavy-handed” is a negative word that is negated. Only 3 of 10 models get this one right. Hard even for Transformers.",
  481: "<b>Mixed sentiment.</b> “funny <em>but</em> perilously slight”: a positive word, then a critical one. The gold label says <em>positive</em>, which shows how subjective borderline sentences are. Compare how the models weigh the two halves.",
  527: "<b>Sarcasm.</b> “all the excitement of eating oatmeal” uses positive words (“excitement”) ironically. Only 2 of 10 models are right. Irony needs world knowledge.",
  183: "<b>Irony: every model fails.</b> “the lower your expectations, the more you’ll enjoy it” is an insult disguised as a promise. No model, big or small, understands it. A great example for the limitations slide.",
  271: "<b>Every model fails.</b> The gold label is <em>positive</em>, but read alone “as unseemly as its title suggests” sounds like criticism. Some labels depend on the surrounding review, which a single-sentence classifier never sees.",
};
