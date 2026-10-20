# 3. METHODOLOGY OF THE PRACTICAL PART

## 3.1. Problem formulation

The practical task is binary sentiment classification of English sentences. Given a sentence x = (w₁, w₂, …, wₙ), the goal is to learn a function f(x) → y, where y ∈ {0, 1} denotes negative or positive sentiment. The classifier is learned from a labelled training set D = {(xᵢ, yᵢ)}, i = 1, …, N, and its quality is measured on a separate test set that is never used for training or model selection.

Sentiment classification was chosen for three reasons. First, it is one of the most common real-world applications of NLP (customer reviews, social media monitoring, product feedback). Second, it is a standard benchmark with published results for all model families, so our results can be compared with the literature (Table 2.1). Third, the task contains linguistic phenomena that are difficult for simple models, such as negation ("not good"), contrast ("the plot is weak, but the acting is great") and sarcasm, which makes it a good test of the contextual understanding provided by self-attention.

The experiment is designed to answer the following research questions:

- RQ1. How much more accurate are Transformer models than classical and recurrent models on the same data?
- RQ2. What is the cost of this improvement in training time, inference latency and model size on CPU-only hardware?
- RQ3. How does the advantage of a pre-trained Transformer depend on the number of labelled training examples?
- RQ4. How well do ready-made models (off-the-shelf in-domain, out-of-domain and zero-shot) perform compared with models trained on the task data?
- RQ5. Are Transformers better at linguistically difficult sentences with negation and contrast?

## 3.2. Dataset

The Stanford Sentiment Treebank (SST) was built by Socher et al. (2013) from 11,855 sentences of movie reviews from the website Rotten Tomatoes. Each sentence was parsed into a tree and every phrase was annotated by three human judges. The binary version SST-2, included in the GLUE benchmark (Wang et al., 2019), removes neutral sentences and contains 67,349 training phrases and sentences and 872 validation sentences. Test labels of GLUE are hidden, therefore, following common practice, we used the official validation set as our held-out test set. For model selection (choosing the best epoch) we randomly separated 2,000 examples from the training data as a development set, stratified by class. The resulting splits are shown in Table 3.1.

[[TABLE data]]

The training data are slightly imbalanced towards the positive class (about 56%), while the test set is almost balanced. The training set contains many short phrases (the average length is below 10 words), whereas the test set contains complete sentences of about 19 words on average. This difference between training and test distributions is part of the challenge: models must generalise from phrases to whole sentences.

## 3.3. Compared models (competitors)

Ten models were selected to represent four approaches that a practitioner can realistically choose from. They are listed in Table 3.2.

[[TABLE models]]

The classical models represent the pre-neural approach and serve as strong, cheap baselines. The bidirectional LSTM represents the pre-Transformer neural approach, trained from scratch without pre-training. The three fine-tuned Transformers differ only in size (4.4, 29 and 67 million parameters), which allows us to observe how quality and cost scale with model size. BERT-tiny (2 layers, hidden size 128) and BERT-small (4 layers, hidden size 512) are compact BERT models pre-trained and released by Google together with the study of Turc et al. (2019); DistilBERT is the distilled version of BERT-base (Sanh et al., 2019). The three ready-made models represent the "use a public model as a service" scenario: a DistilBERT model that has already been fine-tuned on SST-2 by its authors (the best case of reuse), a RoBERTa model trained for sentiment on tweets (a different domain, Loureiro et al., 2022) and a zero-shot NLI model (Yin et al., 2019) that has never seen sentiment labels.

## 3.4. Evaluation criteria

Each model is evaluated by the following criteria:

- Accuracy, macro-averaged precision, recall and F1 score on the 872-sentence test set (quality).
- Training time in seconds on the same computer (cost of building the model).
- Inference latency in milliseconds per sentence, measured as total prediction time for the test set divided by the number of sentences, with batches of 32 (cost of using the model).
- Number of trainable parameters (memory footprint and storage).
- Accuracy on two linguistically difficult subsets: sentences containing negation words (not, no, never, n't, nothing, nor, neither) and sentences containing contrast markers (but, although, though, yet, however).
- Data efficiency: accuracy as a function of the number of labelled training examples (500, 1,000, 2,000, 4,000 and 8,000), measured for the TF-IDF logistic regression baseline and for the fine-tuned BERT-small model.

To check whether differences between models are statistically meaningful, 95% bootstrap confidence intervals of accuracy and McNemar's test for paired predictions are computed (Section 3.7.8).

All experiments were run on a laptop with an Intel Core i5-8365U processor (4 cores, 8 threads, 1.6 GHz base frequency), 16 GB of RAM and no GPU, under Windows 11, Python 3.11, PyTorch 2 (CPU build), the Hugging Face Transformers library (Wolf et al., 2020) and scikit-learn. A fixed random seed (42) was used for data splits, sampling and model initialisation.

## 3.5. Design and structure of the developed program

The program is organised as a pipeline of independent Python scripts that communicate through files (Figure 3.1). This design makes every stage reproducible separately and allows long-running training jobs to be executed in the background.

[[FIG pipeline]]

The modules are:

- utils.py – shared functions: loading of data splits, computation of metrics, timing and saving of results in JSON format;
- 01_prepare_data.py – downloads SST-2 through the Hugging Face datasets library and writes the train, dev and test splits to CSV files;
- 02_classical_baselines.py – builds the TF-IDF representation and trains the naive Bayes, logistic regression and linear SVM classifiers;
- 03_bilstm.py – implements the tokeniser, the vocabulary and the BiLSTM network in PyTorch and trains it with early model selection on the dev set;
- 04_finetune_transformer.py – a generic fine-tuning program: given the name of any pre-trained encoder from the Hugging Face hub and the number of training examples, it adds a classification head and trains the whole network with AdamW and a linear warm-up schedule;
- 05_pretrained_transformers.py – evaluates the three ready-made models through the Transformers pipeline interface;
- 06_learning_curve_classical.py – trains the classical baseline on the same subsets as the Transformer learning curve;
- 07_attention_visualisation.py – extracts and plots the self-attention weights of a fine-tuned model for an example sentence;
- 08_report.py – collects all JSON results, produces the tables and figures, the subset (negation and contrast) analysis and the statistical tests.

Every model writes a result file with the same structure (model name, family, training size, training time, latency, number of parameters, metrics, confusion matrix and the list of predictions). Saving the individual predictions makes it possible to compute paired statistics and error analyses afterwards without re-running the models.

## 3.6. The Transformer architecture

This section describes the architecture used by all Transformer models in the experiment. The encoder of the original Transformer (Vaswani et al., 2017), which is the basis of BERT, DistilBERT and RoBERTa, is shown in Figure 3.2. An encoder is a stack of L identical layers; each layer contains a multi-head self-attention sub-layer and a position-wise feed-forward sub-layer, both wrapped with a residual connection and layer normalisation.

[[FIG encoder]]

### 3.6.1. Tokenisation and input embeddings

Text is first split into sub-word units. BERT models use WordPiece (Wu et al., 2016) and RoBERTa uses byte-pair encoding (Sennrich et al., 2016). Frequent words are kept whole, while rare words are split into pieces (for example, "unwatchable" → "un", "##watch", "##able"), so the vocabulary is limited (about 30,000 units for BERT) and no word is ever out of vocabulary. A special [CLS] token is added at the beginning of every input and [SEP] at the end. Each token is mapped to a vector of dimension d (d = 128 for BERT-tiny, 512 for BERT-small and 768 for DistilBERT) by an embedding matrix E.

### 3.6.2. Positional encoding

Self-attention by itself is permutation-invariant: it treats a sentence as a bag of tokens. Information about word order is therefore added to the embeddings. The original Transformer used fixed sinusoidal encodings:

$$ PE(pos, 2i) = sin(pos / 10000^(2i/d)),   PE(pos, 2i+1) = cos(pos / 10000^(2i/d)) || (3.1)

where pos is the position of the token and i is the index of the dimension. BERT-type models instead learn a position embedding vector for each position up to 512. The input of the first layer is the sum of the token embedding and the position embedding: h⁽⁰⁾ₜ = E(wₜ) + P(t).

### 3.6.3. Scaled dot-product attention

The central operation of the Transformer is attention. Every token representation is projected into three vectors: a query q, a key k and a value v. For a sequence of n tokens these are stacked into matrices Q, K, V ∈ ℝⁿˣᵈᵏ, obtained from the input H by learned projections:

$$ Q = H·W_Q,   K = H·W_K,   V = H·W_V || (3.2)

Attention is then computed as:

$$ Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V || (3.3)

The matrix product Q·Kᵀ gives an n × n matrix of similarity scores between every pair of tokens. The softmax function, applied to each row, turns the scores into weights that are positive and sum to one:

$$ softmax(z)ⱼ = exp(zⱼ) / Σₖ exp(zₖ) || (3.4)

The output for each token is a weighted average of the value vectors of all tokens, so a word can directly gather information from any other word in the sentence regardless of distance. Division by √d_k prevents the dot products from becoming very large for high dimensions, which would push the softmax into regions with extremely small gradients. In our task, for example, the representation of the word "good" in "not good" can attend strongly to "not", which allows the model to represent the negated meaning.

### 3.6.4. Multi-head attention

Instead of a single attention function, the Transformer runs h attention "heads" in parallel, each with its own projection matrices of dimension d_k = d / h:

$$ headᵢ = Attention(H·W_Qⁱ, H·W_Kⁱ, H·W_Vⁱ),   MultiHead(H) = Concat(head₁, …, head_h)·W_O || (3.5)

Different heads can specialise in different relations, for example one head may link adjectives with the nouns they modify while another links negation words with their scope (Clark et al., 2019). BERT-tiny uses 2 heads, BERT-small 8 heads and DistilBERT 12 heads per layer.

### 3.6.5. Feed-forward network, residual connections and layer normalisation

After attention, each position is processed independently by a two-layer feed-forward network:

$$ FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂ || (3.6)

where the inner dimension is usually 4d and GELU is the Gaussian error linear unit activation used in BERT (the original Transformer used ReLU). Each sub-layer is wrapped by a residual connection followed by layer normalisation (Ba et al., 2016):

$$ H' = LayerNorm(H + MultiHead(H)),   H'' = LayerNorm(H' + FFN(H')) || (3.7)

$$ LayerNorm(x) = γ ⊙ (x − μ) / √(σ² + ε) + β || (3.8)

where μ and σ² are the mean and variance of the vector x, and γ and β are learned parameters. Residual connections allow gradients to flow through deep stacks, and normalisation stabilises training.

### 3.6.6. Computational complexity

Self-attention compares every token with every other token, so one layer requires O(n²·d) operations for a sequence of length n, compared with O(n·d²) for a recurrent layer. However, all n positions are computed in parallel, whereas a recurrent network needs n sequential steps. For short and medium sequences (n < d), which is the case for sentences, self-attention is both faster and better at modelling long-range dependencies. For very long documents the quadratic cost becomes a limitation, which motivated efficient Transformer variants (Tay et al., 2022). In our experiment the maximum length was set to 64 tokens, which covers more than 99% of SST-2 sentences.

## 3.7. Mathematical methods of training and evaluation

### 3.7.1. Pre-training objective

BERT-type encoders are pre-trained by masked language modelling. A random 15% of the input tokens are selected; most of them are replaced by a [MASK] token, and the model is trained to predict the original tokens from the context. If M is the set of masked positions, the loss is

$$ L_MLM = − Σ_(t∈M) log P(wₜ | x without M) || (3.9)

Because the model sees the context on both sides of the masked word, it learns deep bidirectional representations. DistilBERT is additionally trained to imitate the output distribution of the BERT-base teacher (knowledge distillation). Pre-training is extremely expensive (days on many GPUs or TPUs), therefore in this project we use publicly released pre-trained weights and only perform the fine-tuning step.

### 3.7.2. Fine-tuning for classification

For classification, the final hidden vector of the [CLS] token, h_CLS ∈ ℝᵈ, is used as the representation of the whole sentence. A new classification layer with weights W ∈ ℝ²ˣᵈ and bias b is added on top:

$$ P(y | x) = softmax(W·h_CLS + b) || (3.10)

All parameters (the pre-trained encoder and the new layer) are trained together by minimising the cross-entropy loss over the training set:

$$ L_CE = − (1/N) Σᵢ log P(yᵢ | xᵢ) || (3.11)

### 3.7.3. Optimisation

Parameters are updated with AdamW (Loshchilov & Hutter, 2019), a variant of Adam (Kingma & Ba, 2015) with decoupled weight decay. For gradient gₜ at step t:

$$ mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,   vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²,   θₜ = θₜ₋₁ − ηₜ( m̂ₜ / (√v̂ₜ + ε) + λθₜ₋₁ ) || (3.12)

where m̂ₜ and v̂ₜ are bias-corrected moment estimates, λ = 0.01 is the weight decay and ηₜ is the learning rate. The learning rate follows a linear schedule: it grows from 0 to its peak value during the first 10% of steps (warm-up) and then decreases linearly to 0. Warm-up prevents large, destructive updates of the pre-trained weights at the beginning of training. Gradients are clipped to a maximum norm of 1.0. The hyper-parameters used are shown in Table 3.3; they follow the recommendations of Devlin et al. (2019) and Sun et al. (2019), with a higher learning rate for the smallest models, which is usual for compact BERT variants.

[[TABLE hparams]]

### 3.7.4. TF-IDF representation

For the classical models every sentence is represented as a sparse vector of TF-IDF weights of unigrams and bigrams:

$$ tfidf(t, d) = (1 + log tf(t, d)) · ( log((1 + N) / (1 + df(t))) + 1 ) || (3.13)

where tf(t, d) is the frequency of term t in sentence d (logarithmically damped), N is the number of sentences and df(t) is the number of sentences containing t. Each vector is normalised to unit length. Terms that appear in fewer than two training sentences are ignored. TF-IDF gives high weights to words that are frequent in a sentence but rare in the collection, but it ignores word order beyond bigrams and cannot represent the meaning of unseen words.

### 3.7.5. Classical classifiers

Multinomial naive Bayes applies Bayes' theorem with the assumption that features are conditionally independent given the class: P(y | x) ∝ P(y) Πⱼ P(tⱼ | y)^(xⱼ), with Laplace (additive) smoothing α = 0.5.

Logistic regression models the probability of the positive class as P(y = 1 | x) = σ(wᵀx + b), where σ(z) = 1 / (1 + e⁻ᶻ), and minimises the L2-regularised log-loss (inverse regularisation strength C = 4).

The linear support vector machine finds the hyperplane with the maximum margin by minimising the regularised hinge loss:

$$ min_w  ½‖w‖² + C Σᵢ max(0, 1 − yᵢ(wᵀxᵢ + b)) || (3.14)

with labels coded as yᵢ ∈ {−1, +1} and C = 0.5.

### 3.7.6. Bidirectional LSTM

The recurrent baseline processes the sequence of word embeddings with an LSTM cell (Hochreiter & Schmidhuber, 1997), defined by the gates:

$$ fₜ = σ(W_f[hₜ₋₁, xₜ] + b_f),  iₜ = σ(W_i[hₜ₋₁, xₜ] + b_i),  oₜ = σ(W_o[hₜ₋₁, xₜ] + b_o) || (3.15)

$$ c̃ₜ = tanh(W_c[hₜ₋₁, xₜ] + b_c),   cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ,   hₜ = oₜ ⊙ tanh(cₜ) || (3.16)

The forget, input and output gates (fₜ, iₜ, oₜ) control what information is kept in the memory cell cₜ. A bidirectional LSTM runs one LSTM from left to right and another from right to left and concatenates their hidden states. The sentence representation is obtained by max-pooling over time, followed by dropout (p = 0.3) and a linear output layer. Embeddings (dimension 128) and all weights are initialised randomly, i.e. the model has no pre-trained knowledge. It is trained for 4 epochs with Adam (learning rate 0.001, batch size 64), and the epoch with the best dev accuracy is kept.

### 3.7.7. Zero-shot classification by natural language inference

An NLI model receives a premise p and a hypothesis h and predicts whether p entails, contradicts or is neutral to h. For zero-shot classification (Yin et al., 2019), the sentence is used as the premise and, for each candidate label c, a hypothesis is built from a template, here "The sentiment of this movie review is c." The entailment scores of the candidate labels are normalised with softmax, and the label with the highest entailment probability is predicted:

$$ ŷ = argmax_(c ∈ {positive, negative})  P(entailment | x, h(c)) || (3.17)

No sentiment-labelled data are used, so the method shows how much a Transformer can achieve purely from general language understanding.

### 3.7.8. Evaluation metrics and statistical tests

With TP, TN, FP and FN denoting true positives, true negatives, false positives and false negatives:

$$ Accuracy = (TP + TN) / (TP + TN + FP + FN),   Precision = TP / (TP + FP),   Recall = TP / (TP + FN) || (3.18)

$$ F1 = 2 · Precision · Recall / (Precision + Recall) || (3.19)

Precision, recall and F1 are computed for each class and averaged (macro average), so both classes have equal weight.

The test set contains only 872 sentences, so a difference of one percentage point corresponds to about nine sentences. To quantify uncertainty, 95% confidence intervals of accuracy are estimated with the bootstrap: the test set is resampled with replacement 2,000 times, accuracy is computed for each resample, and the 2.5th and 97.5th percentiles are taken. To test whether two models differ significantly, McNemar's test is applied to their paired predictions. If b is the number of sentences classified correctly only by model A and c the number classified correctly only by model B, then under the null hypothesis of equal accuracy b ~ Binomial(b + c, 0.5). We use the exact binomial version of the test, with significance level 0.05.
