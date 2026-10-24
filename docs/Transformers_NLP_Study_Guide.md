# Study Guide: Transformers and Their Application in NLP Tasks

Based on the group paper by Shashikanth Hosur Ramegowda and Shaeed Khan (Vilnius Gediminas Technical University, Engineering of AI).

This guide is written so you can learn the paper from beginning to end, understand every concept with a small example, and prepare to present it.

---

## 0. The one-paragraph version

Transformers replaced older neural networks as the standard tool for language tasks. The paper asks a practical question: on an ordinary CPU laptop, with a limited number of labelled examples, how much better are Transformers than classical methods (TF-IDF + linear classifiers) and older neural networks (BiLSTM), and what does that improvement cost in training time, speed and model size? The task is binary sentiment classification of movie-review sentences (SST-2). The answer: a large pre-trained Transformer is clearly the most accurate, pre-training makes it far more data-efficient, but it is 20 to 2,000 times slower at inference and 60 to 1,700 times larger than TF-IDF models. Compact Transformers did no better than classical models.

---

## 1. Your role and where to focus

The paper lists these responsibilities:

| Member | Responsibilities |
|---|---|
| Shashikanth (you) | Problem formulation and experiment design; implementation of the Transformer fine-tuning program; Transformer theory and mathematical methods (Chapter 2); ready-made model evaluation; practical part (Chapter 3); presentation |
| Shaeed Khan | Introduction; literature search and review (Chapter 1); classical and BiLSTM baselines; error analysis; conclusions; references |

**Priority for you:** Chapter 3 and the implementation (Section 2.5 and the code pipeline) in depth. Know the theory (Section 2.6 and 2.7) well enough to explain it, because the presenter will be asked about it. Know the literature review at a high level.

---

## 2. Suggested learning order

1. Read Section 2 of this guide (the problem) until you can state the research question in one sentence.
2. Learn the Transformer building blocks in Section 4, in order. Each one builds on the previous.
3. Learn the baselines and the training maths in Section 5.
4. Learn the experiment setup in Section 6 and the results in Section 7.
5. Practise the explanations in Sections 9 and 10 out loud.
6. Re-read the Glossary before the presentation.

---

## 3. Background: why this research exists

### 3.1 The old way of doing NLP

- **Bag-of-words / TF-IDF + linear classifier.** Each sentence becomes a vector of word counts (weighted). A classifier such as logistic regression or an SVM learns a weight per word. Fast and cheap, but it ignores word order and context.
- **Word embeddings (word2vec, 2013).** Each word gets one fixed vector. "bank" has the same vector in "river bank" and "bank account". This is a problem for ambiguous words.
- **Recurrent networks (RNN, LSTM).** Read the sentence one word at a time, carrying a memory forward. Slow to train (each step waits for the previous one) and weak at linking words that are far apart.

### 3.2 The Transformer idea (2017)

Vaswani et al. removed recurrence completely. Every word looks directly at every other word through **attention**, and all words are processed at the same time. This makes training parallel (fast on GPUs) and captures long-range links directly.

### 3.3 Pre-train, then fine-tune

- **Pre-training:** train a huge Transformer on unlabelled text (billions of words) with a self-supervised task, for example "hide some words and predict them" (BERT) or "predict the next word" (GPT). This learns general language knowledge. Very expensive, done once by big labs.
- **Fine-tuning:** take the pre-trained model, add a small classification layer, and train briefly on your labelled examples. Cheap, and works with few labels.

The paper's central practical point: most people should reuse pre-trained models rather than train from scratch.

---

## 4. The Transformer, step by step (Chapter 2.6)

Each piece below has a simple example. Learn them in this order.

### 4.1 Tokenisation (WordPiece / byte-pair encoding)

Text is split into sub-word pieces. Common words stay whole. Rare words break into known pieces, so the model never meets an unknown word.

> Example: "unwatchable" → "un", "##watch", "##able"

BERT's vocabulary is about 30,000 pieces. A special `[CLS]` token is added at the start of every input and `[SEP]` at the end.

### 4.2 Embeddings

Each token id is looked up in an embedding table and becomes a vector of size d. In the paper: d = 128 (BERT-tiny), 512 (BERT-small), 768 (DistilBERT).

### 4.3 Positional encoding

Self-attention by itself does not know word order. It treats the sentence as a bag of tokens. So a position vector is added to each token vector:

> "dog bites man" and "man bites dog" would otherwise look identical to attention. Adding position vectors tells the model which word came first.

The original paper uses fixed sine and cosine waves (equation 1). BERT-type models learn one position vector per position, up to 512.

### 4.4 Scaled dot-product self-attention (the core idea)

Each token vector is turned into three vectors by learned matrices:
- **Query (Q):** what this token is looking for
- **Key (K):** what this token offers
- **Value (V):** the information it passes on

Attention is:

> Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V

**Worked example (2 tokens, d = 2).** Take "not" and "good", with Q = K = V = X, and vectors:
- "not" = [1, 0]
- "good" = [0, 1]

Step 1, scores Q·Kᵀ:

|  | key: not | key: good |
|---|---|---|
| query: not | 1 | 0 |
| query: good | 0 | 1 |

Step 2, divide by √2 ≈ 1.414: the scores become 0.707 on the diagonal and 0 elsewhere.

Step 3, softmax on each row. For the "not" row: e^0.707 = 2.03 and e^0 = 1, so the weights are 2.03/3.03 = 0.67 and 1/3.03 = 0.33.

Step 4, weighted sum of values:
- New vector for "not" = 0.67·[1, 0] + 0.33·[0, 1] = **[0.67, 0.33]**
- New vector for "good" = 0.33·[1, 0] + 0.67·[0, 1] = **[0.33, 0.67]**

**What this shows:** after attention, each word's vector contains a mix of itself and the other word. In a real model the projection matrices are learned, so the model can learn to make "good" attend strongly to "not" when they sit next to each other. The √d_k division keeps the scores from becoming so large that softmax gives almost all weight to one token and gradients vanish.

**Why this matters for sentiment:** in "not good", the representation of "good" can be changed by "not". A bag-of-words model cannot do that.

### 4.5 Multi-head attention

Instead of one attention, the model runs h attention heads in parallel, each with smaller dimension d_k = d / h. The outputs are joined together.

> Analogy: several readers look at the same sentence with different questions. One checks subject-verb links, another checks negation, another checks adjectives and nouns.

Heads in the paper: BERT-tiny 2, BERT-small 8, DistilBERT 12.

### 4.6 Feed-forward network, residual connections, layer normalisation

After attention, each token passes separately through a two-layer network (equation 6), with GELU activation in BERT. Each sub-layer is wrapped with a **residual connection** (add the input back, so gradients can flow through deep stacks) and **layer normalisation** (keep the numbers in a stable range).

### 4.7 The encoder stack

One encoder layer = multi-head attention + feed-forward, both with residuals and normalisation. Stack L of them:
- BERT-tiny: 2 layers
- BERT-small: 4 layers
- DistilBERT: 6 layers
- BERT-base: 12 layers (the teacher for DistilBERT)

### 4.8 Computational cost (why Transformers are expensive)

- Recurrent layer: about O(n · d²) work, but n steps in sequence.
- Self-attention layer: about O(n² · d) work, but all n positions at once.

For sentences (short n) self-attention is fast and parallel. For very long documents the n² term becomes the bottleneck. In the experiment the maximum length is 64 tokens, which covers 99.99% of training sentences and all test sentences.

---

## 5. Training the models

### 5.1 Pre-training objective: masked language modelling (MLM)

15% of tokens are selected. Most are replaced by `[MASK]`. The model predicts the originals from context on both sides.

> Example: "The movie was really [MASK]." The model should predict "good" or "bad" from the words around it.

The loss (equation 9) is the average negative log-probability of the correct masked words. Pre-training was not done by the team. They used public pre-trained weights, which cost days on many GPUs.

### 5.2 Fine-tuning for classification (equations 10 and 11)

1. Take the final vector of the `[CLS]` token. This is the summary of the sentence.
2. Add a new linear layer with 2 outputs (negative, positive): z = W·h_CLS + b.
3. Softmax turns z into probabilities.
4. Train all parameters with **cross-entropy loss**.

**Cross-entropy example.** If the true label is positive:
- Model says P(positive) = 0.9 → loss = −ln(0.9) = 0.105 (small, good)
- Model says P(positive) = 0.2 → loss = −ln(0.2) = 1.61 (large, bad)

Training pushes the loss down over many examples.

### 5.3 Optimiser and schedule

- **AdamW:** Adam with decoupled weight decay (λ = 0.01). Adam adapts the step size per parameter using running averages of gradients (first moment m) and squared gradients (second moment v).
- **Warm-up:** the learning rate rises from 0 over the first 10% of steps, then decays linearly to 0. Warm-up prevents large damaging updates to pre-trained weights at the start.
- **Gradient clipping at norm 1.0:** stops a single bad batch from making huge updates.

### 5.4 Hyper-parameters (Table 5 of the paper)

| Setting | BERT-tiny | BERT-small | DistilBERT |
|---|---|---|---|
| Layers / hidden size / heads | 2 / 128 / 2 | 4 / 512 / 8 | 6 / 768 / 12 |
| Training examples | 8,000 | 8,000 | 8,000 |
| Epochs (best kept on dev set) | 3 | 3 | 2 |
| Peak learning rate | 2·10⁻⁴ | 1·10⁻⁴ | 5·10⁻⁵ |
| Batch size / max length | 32 / 64 | 32 / 64 | 32 / 64 |
| Warm-up / schedule | 10% / linear | 10% / linear | 10% / linear |
| Weight decay / clipping | 0.01 / 1.0 | 0.01 / 1.0 | 0.01 / 1.0 |

Smaller models get a larger learning rate, which is common practice for compact BERT models.

---

## 6. The classical and recurrent baselines

### 6.1 TF-IDF (term frequency × inverse document frequency)

Each word's weight in a sentence is high if it appears often in that sentence but rarely across all sentences.

**Toy example.** Three sentences: s1 = "good movie", s2 = "bad movie", s3 = "not good movie". N = 3.

| Word | Document frequency (df) | idf = ln(N / df) |
|---|---|---|
| good | 2 (s1, s3) | ln(1.5) = 0.405 |
| movie | 3 | ln(1) = 0 |
| bad | 1 (s2) | ln(3) = 1.099 |
| not | 1 (s3) | ln(3) = 1.099 |

So "movie" gets weight 0 because it is in every sentence and tells us nothing. For s3, the vector is: good = 0.405, not = 1.099.

**Lesson:** a bag-of-words model sees "good" (0.405) and "not" (1.099) as separate features and has to learn from data that "not good" is negative. It cannot combine them by position. The paper uses unigrams and bigrams, so "not good" becomes one feature as well, which helps a little.

### 6.2 Naive Bayes

Applies Bayes' rule with the assumption that words are independent given the class:

> P(positive | sentence) is proportional to P(positive) × product of P(word | positive) over the words.

Simple and fast. Laplace smoothing α = 0.5 stops zero probabilities for unseen words.

### 6.3 Logistic regression

Models P(positive | x) = σ(wᵀx + b) with σ(z) = 1 / (1 + e⁻ᶻ). A weighted sum of features goes through a sigmoid. Trained with log-loss and L2 regularisation (C = 4).

### 6.4 Linear SVM

Finds the straight line (hyperplane) that separates the classes with the largest margin, by minimising hinge loss. C = 0.5. Best classical model in the paper (82.9%).

### 6.5 BiLSTM (bidirectional LSTM)

- An LSTM reads the sentence word by word, with gates (forget, input, output) that decide what to keep in its memory cell.
- "Bidirectional" means one LSTM reads left to right and another reads right to left; the two are combined.
- Max-pooling over time gives one sentence vector, then dropout (p = 0.3) and a linear output.
- Trained from random embeddings, so it has no pre-trained knowledge.

---

## 7. The models compared in the experiment

The paper compares 10 models in 4 families:

| Family | Model | Pre-trained? | Trained by team on | Parameters |
|---|---|---|---|---|
| Classical | TF-IDF + Naive Bayes | no | 65k sentences | 0.1 M* |
| Classical | TF-IDF + Logistic Regression | no | 65k | 74 k* |
| Classical | TF-IDF + Linear SVM | no | 65k | 74 k* |
| Recurrent | BiLSTM, random embeddings | no | 65k | 2.0 M |
| Transformer, fine-tuned | BERT-tiny | yes | 8k | 4.4 M |
| Transformer, fine-tuned | BERT-small | yes | 8k | 28.8 M |
| Transformer, fine-tuned | DistilBERT-base-uncased | yes | 8k | 67.0 M |
| Transformer, ready-made | DistilBERT fine-tuned on SST-2 by its authors | yes | not by us | 67.0 M |
| Transformer, ready-made | Twitter-RoBERTa sentiment (tweets) | yes | not by us | 124.6 M |
| Transformer, zero-shot | NLI DistilRoBERTa | yes | not on sentiment | 82.1 M |

\* For TF-IDF models the number is the count of n-gram weights, not neural parameters.

**The three ways to use a Transformer in the paper:**
1. Fine-tune it yourself (BERT-tiny, BERT-small, DistilBERT).
2. Use a public model already fine-tuned for the task (DistilBERT-SST2) or for another domain (Twitter-RoBERTa).
3. Zero-shot: use a model trained for a different task (natural language inference) and describe the sentiment labels as a hypothesis.

### 7.1 Zero-shot classification explained simply

Natural language inference (NLI) asks: does a premise imply a hypothesis? The paper uses the review as the premise and builds a hypothesis from a template:

> Premise: "the plot was not good, but the acting was brilliant."
> Hypothesis for positive: "The sentiment of this movie review is positive."
> Hypothesis for negative: "The sentiment of this movie review is negative."

The model scores the entailment of each hypothesis, a softmax turns the scores into probabilities, and the highest wins. No sentiment-labelled training data is used.

---

## 8. The data and the experiment design

### 8.1 Dataset: Stanford Sentiment Treebank, binary (SST-2)

- Built by Socher et al. (2013) from 11,855 movie-review sentences from Rotten Tomatoes.
- Every phrase in the parse tree is labelled by annotators.
- SST-2 (binary version, part of the GLUE benchmark) removes neutral sentences.
- Test labels for GLUE are hidden, so the **official validation set (872 sentences)** is used as the test set.
- 2,000 random training examples (stratified by class) are held out as a **dev set** for choosing the best epoch.

| Split | Source | Size | Positive share | Mean length (words) |
|---|---|---|---|---|
| Train | SST-2 train minus dev | 65,349 | 55.8% | 9.4 |
| Dev | 2,000 from SST-2 train | 2,000 | 55.8% | 9.6 |
| Test | SST-2 validation | 872 | 50.9% | 19.5 |

**Important detail:** training sentences are mostly short phrases (under 10 words), while test sentences are complete (about 19 words). The model must generalise from phrases to full sentences. This explains part of the BiLSTM's gap between dev and test accuracy.

### 8.2 The five research questions

| Question | In plain words |
|---|---|
| RQ1 | How much more accurate are Transformers than classical and recurrent models on the same data? |
| RQ2 | What does that improvement cost in training time, inference latency and model size on a CPU? |
| RQ3 | How does the advantage of a pre-trained Transformer change with the number of labelled examples? |
| RQ4 | How do ready-made models (in-domain, out-of-domain, zero-shot) compare with models trained on this data? |
| RQ5 | Are Transformers better on hard sentences with negation and contrast? |

### 8.3 Evaluation criteria

1. **Accuracy, macro precision, macro recall, macro F1** on the 872 test sentences (quality).
2. **Training time** in seconds (cost of building).
3. **Inference latency** in ms per sentence, total prediction time divided by number of sentences, batches of 32 (cost of using).
4. **Number of parameters** (memory and storage).
5. **Accuracy on negation and contrast subsets.** Negation words: not, no, never, n't, nothing, nor, neither. Contrast markers: but, although, though, yet, however.
6. **Data efficiency:** accuracy with 500, 1,000, 2,000, 4,000 and 8,000 labelled training examples, for TF-IDF logistic regression and BERT-small.

### 8.4 Hardware and software

- Laptop: Intel Core i5-8365U (4 cores, 8 threads), 16 GB RAM, **no GPU**, Windows 11.
- Python 3.11, PyTorch 2 (CPU build), Hugging Face Transformers, Datasets, scikit-learn, Matplotlib.
- Fixed random seed 42 for splits, sampling and initialisation.

### 8.5 Experimental protocol (what keeps the comparison fair)

1. Data splits are created once and stored as CSV files. Every model reads the same files.
2. Model selection (best epoch) uses only the dev set.
3. The test set is used once per model for final evaluation.
4. Learning-curve subsets use the same seed for the classical and Transformer model, so both learn from exactly the same examples.
5. Models run one after another on an idle computer, so timings are not distorted by parallel jobs.
6. Hyper-parameters come from the literature and are not tuned on the test set.

### 8.6 Programme structure (Section 2.5 and Fig. 1)

The code is a pipeline of nine scripts that communicate through files. Each stage can run separately, and long jobs can run in the background.

| Script | What it does |
|---|---|
| `utils.py` | Loads splits, computes metrics, times runs, saves results as JSON |
| `01_prepare_data.py` | Downloads SST-2 and writes train, dev and test CSVs |
| `02_classical_baselines.py` | TF-IDF + Naive Bayes, Logistic Regression, Linear SVM |
| `03_bilstm.py` | Tokeniser, vocabulary and BiLSTM in PyTorch |
| `04_finetune_transformer.py` | **The core script.** Loads any pre-trained encoder by name, adds a 2-class head, trains with AdamW and warm-up |
| `05_pretrained_transformers.py` | Evaluates the ready-made models through the pipeline interface |
| `06_learning_curve_classical.py` | Classical baseline on the same subsets as the Transformer learning curve |
| `07_attention_visualisation.py` | Extracts and plots self-attention weights |
| `08_report.py` | Collects all JSON results, creates tables, figures and statistical tests |

Every model writes a JSON file with the same structure: name, family, training size, training time, latency, parameter count, metrics, confusion matrix and **all individual predictions**. Saving individual predictions is what allows McNemar's test later without re-running models.

About 770 lines of code. The whole pipeline starts with one command (`run_all.sh`).

**Two practical problems the team solved (good to mention in the presentation):**
- The newest version of the Datasets library requires full repository names, so SST-2 is loaded as `nyu-mll/glue`.
- Some third-party compact BERT checkpoints lacked tokenizer files for the current library version, so the official Google checkpoints were used.

---

## 9. Evaluation metrics and statistics

### 9.1 Accuracy, precision, recall, F1

Start from the confusion matrix. For the positive class: TP (true positive), FP (false positive), FN (false negative), TN (true negative).

**Worked example (100 test sentences):**

|  | Predicted positive | Predicted negative |
|---|---|---|
| Actually positive | TP = 45 | FN = 5 |
| Actually negative | FP = 10 | TN = 40 |

- Accuracy = (45 + 40) / 100 = **85%**
- Precision (positive) = 45 / (45 + 10) = 81.8%: of the sentences called positive, how many were.
- Recall (positive) = 45 / (45 + 5) = 90%: of the actually positive sentences, how many were found.
- F1 (positive) = 2 · P · R / (P + R) = 85.7%: harmonic mean of precision and recall.
- For the negative class: precision 88.9%, recall 80%, F1 84.2%.
- **Macro F1** = average of the two per-class F1 scores = (85.7 + 84.2) / 2 ≈ **84.9%**.

Macro averaging gives each class equal weight. The test set is almost balanced, so accuracy and macro F1 differ by less than 0.2 percentage points in this paper.

### 9.2 Bootstrap confidence intervals

Question: if we tested on different 872 sentences, how much would accuracy move?

Method: resample the test set **with replacement** 2,000 times (each resample is 872 sentences, some repeated, some missing). Compute accuracy on each resample. The 2.5th and 97.5th percentiles of those 2,000 accuracies form the 95% interval.

Example from Table 7: DistilBERT fine-tuned on 8,000 sentences has accuracy 88.8%, interval [86.6; 90.7]. So a fair estimate lies somewhere between 86.6% and 90.7%.

### 9.3 McNemar's test (paired comparison)

Two models are tested on the same sentences, so compare where they disagree.

- b = sentences only model A gets right
- c = sentences only model B gets right
- Under "no difference", b and c should be equally likely. Exact version: b ~ Binomial(b + c, 0.5).

**Worked example from the paper.** Fine-tuned DistilBERT vs DistilBERT-SST2 (best model): b = 48, c = 28.
- Continuity-corrected chi-square = (|48 − 28| − 1)² / (48 + 28) = 19² / 76 = 4.75.
- p ≈ 0.029, below 0.05, so the difference is statistically significant.

Compare DistilBERT (fine-tuned) with the SVM: it correctly classifies 104 sentences the SVM misses, and the SVM is right in only 53 cases. Result: p < 0.001.

Compare BERT-small with the SVM: 71 vs 64 sentences, p = 0.61. **Not significant.** A difference of 7 sentences is noise here.

---

## 10. The results

### 10.1 Overall accuracy (Table 6)

| Model | Train size | Accuracy % | F1 % | Train time (s) | Latency (ms/sentence) |
|---|---|---|---|---|---|
| DistilBERT-SST2 (off-the-shelf) | 0 | **91.1** | 91.0 | – | 44.6 |
| DistilBERT fine-tuned by us | 8,000 | 88.8 | 88.8 | 9,530 | 39.1 |
| Twitter-RoBERTa (off-the-shelf) | 0 | 85.8 | 85.8 | – | 84.8 |
| BERT-small fine-tuned | 8,000 | 83.7 | 83.7 | 1,581 | 14.1 |
| TF-IDF + Linear SVM | 65,349 | 82.9 | 82.9 | 2 | 0.06 |
| BiLSTM (from scratch) | 65,349 | 82.8 | 82.7 | 380 | 0.45 |
| TF-IDF + Naive Bayes | 65,349 | 82.2 | 82.1 | 2 | 0.05 |
| TF-IDF + Logistic Regression | 65,349 | 81.9 | 81.8 | 4 | 0.05 |
| Zero-shot NLI | 0 | 81.8 | 81.7 | – | 107.0 |
| BERT-tiny fine-tuned | 8,000 | 76.6 | 76.5 | 120 | 1.2 |

**Three groups:**
1. Best: the two DistilBERT models (91.1% and 88.8%).
2. Middle: Twitter-RoBERTa (85.8%) and our BERT-small (83.7%).
3. Classical, BiLSTM and zero-shot cluster between 81.8% and 82.9%.
4. The smallest Transformer, BERT-tiny (76.6%), is the weakest. **A Transformer is not automatically better; its capacity must be sufficient.**

### 10.2 Answer to RQ1

A sufficiently large pre-trained Transformer improves accuracy by about 6 percentage points over the best classical and recurrent models, even when trained on far fewer examples. Compact Transformers only match the classical baselines.

### 10.3 Cost: RQ2 (Fig. 4 and Section 3.3)

- Classical models: about 0.05 ms per sentence, 2 to 4 seconds of training on all 65k sentences.
- BiLSTM: 0.45 ms per sentence, 6 minutes of training (100 to 200 times longer than TF-IDF).
- BERT-tiny: 1.2 ms. BERT-small: 14 ms. DistilBERT: 39 ms. Ready-made larger models: 45 to 107 ms.
- Fine-tuning on the CPU: BERT-tiny 2 minutes, BERT-small 26 minutes, DistilBERT about 2.6 hours.
- DistilBERT speed: about 25 sentences per second.
- Size: TF-IDF about 74 thousand weights; BiLSTM 2 million parameters; Transformers 4.4 to 125 million parameters (17 to 500 MB in 32-bit precision).

**Interpretation:** Transformers are fine for interactive use on a CPU, but for very high-volume streams the TF-IDF models remain hundreds of times cheaper. The timings are upper bounds because the laptop was shared with other tasks.

### 10.4 Data efficiency: RQ3 (Table 9 and Fig. 5)

| Training examples | BERT-small fine-tuned % | TF-IDF + LR % | Difference (p.p.) |
|---|---|---|---|
| 500 | 75.9 | 61.8 | +14.1 |
| 1,000 | 77.6 | 68.1 | +9.5 |
| 2,000 | 80.7 | 71.0 | +9.7 |
| 4,000 | 82.7 | 75.8 | +6.9 |
| 8,000 | 83.7 | 77.5 | +6.2 |

**Key points:**
- The advantage of pre-training is largest when labels are scarce.
- At 500 sentences, TF-IDF has seen most test-set words only rarely. The pre-trained model already knows them.
- BERT-small with 4,000 labels (82.7%) matches the best classical model trained on all 65,349 sentences (82.9%). That is about 16 times less labelled data.
- This matters for smaller languages such as Lithuanian, where labelled data is rare.

### 10.5 Ready-made and zero-shot models: RQ4

- **DistilBERT-SST2 (off-the-shelf): 91.1%.** Best option when a model fine-tuned for the same task and domain already exists. No training needed.
- **Twitter-RoBERTa: 85.8%.** Trained for sentiment, but on tweets. It lost 5.3 points compared with DistilBERT-SST2, yet still beat all classical models. The informal style of tweets differs from film critics.
- **Zero-shot NLI: 81.8%.** Never saw a sentiment label, but matches TF-IDF logistic regression trained on 65k sentences. Costs the most at inference (107 ms) because each sentence is processed once per candidate label.

### 10.6 Negation and contrast: RQ5 (Table 10 and Fig. 6)

| Model | All % | Negation % | Contrast % | Drop on negation (p.p.) |
|---|---|---|---|---|
| DistilBERT-SST2 | 91.1 | 87.0 | 86.2 | −4.1 |
| DistilBERT fine-tuned | 88.8 | 85.3 | 85.4 | −3.4 |
| Twitter-RoBERTa | 85.8 | 83.7 | 84.6 | −2.1 |
| BERT-small | 83.7 | 74.5 | 76.2 | −9.3 |
| TF-IDF + Linear SVM | 82.9 | 74.5 | 77.7 | −8.5 |
| BiLSTM | 82.8 | 78.8 | 77.7 | −4.0 |
| TF-IDF + Naive Bayes | 82.2 | 73.9 | 76.9 | −8.3 |
| TF-IDF + Logistic Regression | 81.9 | 73.9 | 76.9 | −8.0 |
| Zero-shot NLI | 81.8 | 81.5 | 79.2 | −0.2 |
| BERT-tiny | 76.6 | 69.6 | 65.4 | −7.0 |

**Interpretation:**
- Bag-of-words models lose 8 to 9 points on negation. "Not good" contains the positive word "good".
- Large Transformers lose only 2 to 4 points.
- Zero-shot NLI loses almost nothing (−0.2), plausibly because NLI training teaches negation and contradiction directly.
- Compact BERT-small behaves like the classical models (−9.3). **The advantage is not from the architecture alone, but from architecture, capacity and pre-training together.**
- The BiLSTM handles negation better than TF-IDF because it reads in order.

**Attention example (Fig. 7).** DistilBERT-SST2 classifies "the plot was not good, but the acting was brilliant" as positive with probability 1.000. In the last layer much attention goes to the comma and the [SEP] token ("attention sinks"). The words of the second clause attend mainly within that clause. The paper warns (following Jain and Wallace, 2019) that attention pictures show information flow, not a full explanation.

**Concrete examples from the test set:**
- DistilBERT is right, SVM is wrong: "in its best moments, resembles a bad high school production of grease, without benefit of song" (negative). The SVM is misled by "best".
- SVM is right, DistilBERT is wrong: mixed sentiment such as "we root for (clara and paul), even like them, though perhaps it's an emotion closer to pity".

---

## 11. Statistical confidence in the results

| Model | Accuracy % | 95% CI | Only best correct | Only this correct | p-value |
|---|---|---|---|---|---|
| DistilBERT-SST2 | 91.1 | [89.1; 92.9] | – | – | – |
| DistilBERT fine-tuned (8k) | 88.8 | [86.6; 90.7] | 48 | 28 | 0.029 |
| Twitter-RoBERTa | 85.8 | [83.4; 88.1] | 82 | 36 | < 0.001 |
| BERT-small (8k) | 83.7 | [81.1; 85.9] | 90 | 26 | < 0.001 |
| TF-IDF + Linear SVM | 82.9 | [80.4; 85.3] | 106 | 35 | < 0.001 |
| BiLSTM | 82.8 | [80.3; 85.2] | 103 | 31 | < 0.001 |
| TF-IDF + Naive Bayes | 82.2 | [79.6; 84.7] | 115 | 38 | < 0.001 |
| TF-IDF + Logistic Regression | 81.9 | [79.2; 84.4] | 114 | 34 | < 0.001 |
| Zero-shot NLI | 81.8 | [79.1; 84.2] | 116 | 35 | < 0.001 |
| BERT-tiny (8k) | 76.6 | [73.7; 79.2] | 154 | 28 | < 0.001 |

"Only best correct" = sentences the reference model (DistilBERT-SST2) gets right and this model gets wrong. "Only this correct" = the reverse.

**Reading this table:** a difference of about 1 percentage point (roughly 9 sentences) is not reliable on its own. The team ran each configuration only once, with one seed.

---

## 12. Literature review: what the team concluded

### 12.1 The main findings from the literature

- **English:** Transformers are the state of the art. Fine-tuned encoders exceed 90% on SST-2. Classical bag-of-words methods reach about 79 to 83%. Recurrent and CNN networks reach 84 to 88%.
- **Key models:**
  - BERT (2019): masked language modelling, encoder only. 93.5% on SST-2 (base).
  - RoBERTa: same idea, trained longer on more data. 96.4% on SST-2.
  - DistilBERT: distilled, 40% smaller, 60% faster, about 97% of BERT's quality.
  - GPT-3: in-context learning from examples in the prompt, no fine-tuning.
  - Zero-shot NLI (Yin et al., 2019): classify with hypotheses instead of training data.
- **Lithuanian:**
  - Kapočiūtė-Dzikienė et al. (2019): on Lithuanian comments, naive Bayes (about 0.735) beat the best neural model without pre-training (CNN with FastText, about 0.706). Neural models without pre-training do not automatically win with limited data.
  - Ulčar and Robnik-Šikonja (2021): LitLat BERT, a trilingual Lithuanian-Latvian-English model. Models trained on a few related languages beat massively multilingual models on most tasks.
  - Vileikytė et al. (2024): fine-tuned multilingual BERT and T5 for Lithuanian reviews. High accuracy on clearly polar reviews, lower on ambiguous middle ratings.
  - Nakvosas et al. (2024): open Llama2-based Lithuanian LLMs (7B and 13B) from Neurotechnology, Vilnius.

### 12.2 The research gap

1. Almost all results come from GPU hardware with full training sets.
2. Cost (training time, latency, model size) is rarely reported together with accuracy.
3. Data efficiency of compact pre-trained models under a fixed label budget is under-studied.
4. Ready-made models are rarely compared with fine-tuned ones under one protocol.

---

## 13. Limitations (say these before the examiner does)

From the paper's Section 2.8:

- **Internal validity:** one run with one random seed per configuration. Differences below about 1 point may be random.
- **Construct validity:** negation and contrast subsets are defined by keyword rules. They catch most, but not all, such sentences.
- **External validity:** English movie reviews only. Lithuanian results would differ in absolute value, though the relative pattern is expected to hold.
- **Hardware dependence:** timings are for a shared laptop CPU. On a GPU the Transformer training times would be one to two orders of magnitude shorter; classical models gain little.
- **Test set:** the GLUE validation set is used as test. It is public, so the team cannot be sure no model saw it during its own pre-training (a general limitation of this benchmark).

---

## 14. Conclusions and recommendations

**Conclusions:**
- Transformers are the most accurate approach. A DistilBERT fine-tuned on only 8,000 sentences reached 88.8%, significantly above the best classical model trained on all 65,349 (82.9%, p < 0.001).
- The cost is real: 20 to 2,000 times slower at inference and 60 to 1,700 times larger than TF-IDF.
- Pre-training makes them far more data-efficient: with 500 labels BERT-small beat TF-IDF by 14.1 points; with 4,000 it matched a classical model trained on 16 times more data.
- Larger Transformers are much more robust on negation and contrast (2 to 4 point drop versus 8 to 9).
- Compact Transformers (BERT-tiny, BERT-small) are not a free upgrade. They roughly match classical models.

**Recommendations (the paper's practical advice):**
1. First check whether a public model already fine-tuned for the same task and domain exists. It gives the best quality with no training.
2. If not, fine-tune a distilled Transformer such as DistilBERT on a few thousand labelled examples. Feasible on CPU.
3. If there are no labelled data at all, a zero-shot NLI model is a usable baseline, comparable to classical models.
4. Use TF-IDF models only when inference must be extremely cheap or hardware very limited.
5. For Lithuanian: apply the same procedure with LitLat BERT or XLM-RoBERTa. This is the natural next step, together with repeated runs over several seeds and GPU timing.

---

## 15. Glossary

| Term | Plain meaning |
|---|---|
| Attention | A way for each word to take information from other words, weighted by relevance |
| Self-attention | Attention where a sentence attends to itself |
| Query, Key, Value | Three learned views of each token used in attention |
| Multi-head attention | Several attentions in parallel, each looking for different relations |
| Positional encoding | Extra vector added to tell the model word order |
| Encoder | Stack of attention + feed-forward layers that reads the input |
| Residual connection | Adding the input back to a layer's output, to help gradients flow |
| Layer normalisation | Rescaling a vector to stable mean and variance |
| Tokenisation / WordPiece | Splitting words into known sub-word pieces |
| [CLS] token | Special token whose final vector is used as the sentence summary |
| MLM (masked language modelling) | Pre-training by hiding words and predicting them |
| Pre-training | Large unsupervised training on general text |
| Fine-tuning | Short supervised training of a pre-trained model on the task |
| Distillation | Training a small "student" model to imitate a large "teacher" |
| Cross-entropy loss | Penalty for putting low probability on the correct class |
| AdamW | Adaptive optimiser with decoupled weight decay |
| Warm-up | Starting with a tiny learning rate and increasing it |
| Gradient clipping | Capping the size of an update to prevent instability |
| TF-IDF | Word weight: frequent in this sentence, rare across all sentences |
| Naive Bayes | Probabilistic classifier that assumes word independence |
| Logistic regression | Linear model with a sigmoid giving a probability |
| Linear SVM | Linear classifier with maximum margin |
| LSTM / BiLSTM | Recurrent network with memory gates / read in both directions |
| NLI | Natural language inference: does premise imply hypothesis? |
| Zero-shot | Classifying without any labelled examples for the task |
| Accuracy | Share of correct predictions |
| Precision / Recall | Of predicted positives, how many are right / of true positives, how many were found |
| F1 | Harmonic mean of precision and recall |
| Macro average | Average over classes with equal weight |
| Bootstrap CI | Interval estimated by resampling the test set with replacement |
| McNemar's test | Test for a difference between two models on paired predictions |
| Latency | Time to predict one sentence |
| Learning curve | Accuracy as a function of training set size |
| Seed | Fixed starting number for random processes, so results can be repeated |
| Dev set / test set | Dev: used to choose settings. Test: used once for the final score |

---

## 16. Preparing the presentation

### 16.1 A suggested slide flow (about 12 slides)

1. Title and team
2. Motivation: Transformers are everywhere, but what do they cost on a laptop?
3. Research problem and the five research questions
4. Background in one diagram: TF-IDF → BiLSTM → Transformer
5. How a Transformer reads a sentence (tokens → attention → classification) with the "not good" example
6. The dataset (SST-2) and the ten compared models
7. Pipeline and how we kept the comparison fair
8. Main results (Table 6 as a bar chart, Fig. 3)
9. Cost vs accuracy (Fig. 4)
10. Data efficiency (Fig. 5) and the 16x-less-data message
11. Negation and contrast (Fig. 6) and the attention example (Fig. 7)
12. Limitations, recommendations, and the Lithuanian next step

### 16.2 Practice question list (with short answers)

**Q: Why did you use the official validation set as the test set?**
A: GLUE hides its test labels. The validation set is the standard public substitute, and we held out 2,000 training sentences for dev.

**Q: Why is DistilBERT fine-tuned on 8,000 examples compared with a classical model trained on 65,349?**
A: That is the question being asked. The fair statement is that the Transformer needs far fewer labels to beat the classical model. The learning curve shows the same comparison at equal data sizes.

**Q: Your fine-tuned DistilBERT is 88.8%, not above 90%. Why?**
A: It was trained on 8,000 sentences, about 12% of the training data, and for 2 epochs on CPU. The literature's 90%+ numbers use full data and often GPU training. The off-the-shelf DistilBERT (91.1%) matches the published 91.3%, which tells us our evaluation is correct.

**Q: Is the difference between BERT-small and the SVM real?**
A: No. The difference is 71 vs 64 sentences, p = 0.61. We say so explicitly.

**Q: Why only one seed?**
A: Time on a CPU laptop. This is a stated limitation. The bootstrap intervals and McNemar tests are the tools we used to judge reliability.

**Q: What is the attention picture showing?**
A: Where the model's attention goes in the final layer. It shows information flow, not a full explanation. We cite Jain and Wallace (2019) for this caution.

**Q: What does "macro F1" mean and why use it?**
A: F1 averaged over both classes with equal weight. The classes are almost balanced, so it hardly differs from accuracy here.

**Q: What is zero-shot classification and why is it useful?**
A: We turn the labels into hypotheses for an NLI model. It needs no sentiment-labelled data, at the cost of slow inference.

**Q: Why are Transformers slower at inference?**
A: Each sentence passes through many layers, each with attention over all tokens and large matrix multiplications. TF-IDF needs one sparse dot product.

**Q: What would change on a GPU?**
A: Transformer training and inference would be much faster, perhaps one to two orders of magnitude for training. Classical models would hardly change.

---

## 17. Items to check before presenting

Reading the document closely, I found the following points you should verify or prepare for:

1. **Confidence interval wording.** Section 3.2 says the 95% intervals are "about ±2.5 p.p. wide." Table 7 shows the full width is about 3.5 to 4 points, so the half-width is about ±2 points. Say "about ±2 points" on the slide.
2. **Cross-references to subsections.** The text refers to "Section 2.7.7" (zero-shot hypothesis template) and "Section 2.7.8" (statistical tests). The table of contents only numbers two levels, so check that these sub-numbers exist in the final layout.
3. **Table 8 column headers.** "Only best correct" and "Only this correct" are the two McNemar discordant counts. They are easy to misread. Explain them on the slide.
4. **The 90% claim.** The literature review says fine-tuned Transformers exceed 90% on SST-2. Our own fine-tuned models reached 83.7% and 88.8%. This is consistent (the literature uses larger models and more data), but a reviewer may ask about it. Be ready to explain.
5. **Test-set use.** The protocol says the test set is used once per model. The learning curves and error analysis also use the test set for analysis, not for choosing models. Phrase this clearly if asked.
6. **Parameter count consistency.** Table 5 lists BERT-small with 28.8 M parameters, and the text says "29 million". Both are fine; use one consistently on slides.

---

## 18. Quick self-test

Try to answer without looking:

1. What is the difference between pre-training and fine-tuning?
2. Why does self-attention need positional encoding?
3. Compute the attention weights for "not" in the toy example. (Answer: 0.67 and 0.33.)
4. Why is the TF-IDF weight of "movie" zero in the toy example?
5. What does McNemar's test compare, and why use paired predictions?
6. Why does BERT-small with 4,000 examples match the classical model trained on 65,349?
7. Why do bag-of-words models struggle with "not good"?
8. What does the zero-shot model need that the fine-tuned models do not?
9. Why is BERT-tiny the weakest model even though it is a Transformer?
10. State the paper's main conclusion in two sentences.

---

*Prepared as a study aid for the paper. All numbers are taken from the paper's tables and text. Worked examples are simplified illustrations, not outputs of the team's code.*
