# Literature Review

## Search strategy

The literature was collected from Google Scholar and Web of Science, supplemented with the ACL Anthology (the main open archive of computational linguistics) and arXiv for the most recent preprints. The core search query was:

("transformer" OR "BERT" OR "self-attention" OR "pre-trained language model") AND ("natural language processing" OR "text classification" OR "sentiment analysis")

For the Lithuanian part the query was narrowed with the terms ("Lithuanian" OR "Baltic languages"). The thousands of records returned were sorted by relevance and citation count and screened by title and abstract. Inclusion criteria: (1) the work proposes or evaluates a Transformer-based model or an important predecessor; (2) it reports results on text classification or language understanding, or describes a widely used model; (3) it is peer-reviewed or a widely cited technical report. Seminal works (the Transformer, BERT, GPT, RoBERTa, DistilBERT) were added by backward citation tracking. About 40 publications were kept for analysis; the most important of them are summarised in Table {tab:lit}.

## From word vectors to the Transformer

Early neural approaches represented each word by a static vector, learned with word2vec (Mikolov et al., 2013) or similar methods, and combined these vectors with convolutional or recurrent networks. Such vectors cannot distinguish the meanings of an ambiguous word in different contexts. Bahdanau et al. (2015) introduced the attention mechanism for machine translation, allowing the decoder of a recurrent model to focus on the relevant parts of the source sentence. Contextual representations followed: ELMo (Peters et al., 2018) computed word vectors with a deep bidirectional LSTM language model, and ULMFiT (Howard & Ruder, 2018) showed that a pre-trained language model can be fine-tuned for text classification with very little labelled data. These works established the "pre-train, then fine-tune" paradigm that the Transformer later made dominant.

Vaswani et al. (2017) proposed the Transformer, an encoder–decoder model built entirely from self-attention and feed-forward layers. On English–German and English–French translation it achieved better quality than recurrent and convolutional systems at a fraction of the training cost, because all positions of a sentence are processed in parallel. The architecture is described in detail in Chapter 2.

## Families of pre-trained Transformer models

Three families of models grew out of the original architecture.

Encoder-only models. BERT (Devlin et al., 2019) pre-trains a Transformer encoder with masked language modelling, in which 15% of the tokens are hidden and the model has to predict them from the context on both sides. BERT-base (110 million parameters) and BERT-large (340 million) set new records on the GLUE benchmark (Wang et al., 2019), including SST-2 sentiment classification. RoBERTa (Liu et al., 2019) showed that BERT was under-trained and obtained better results simply by training longer on more data, with larger batches and without the next-sentence prediction task. ALBERT (Lan et al., 2020) reduced the number of parameters by sharing them across layers; ELECTRA (Clark et al., 2020) uses a more sample-efficient "replaced token detection" objective; DeBERTa (He et al., 2021) separates content and position information in attention. XLM-RoBERTa (Conneau et al., 2020) extended the approach to 100 languages, including Lithuanian.

Compact encoders. Because large encoders are expensive at inference time, a line of research focuses on compression. DistilBERT (Sanh et al., 2019) uses knowledge distillation to train a 6-layer student from the 12-layer BERT teacher; it is 40% smaller and 60% faster while retaining about 97% of BERT's language understanding performance. Turc et al. (2019) released a set of 24 compact BERT models (from BERT-tiny with 2 layers to BERT-base) and showed that pre-training compact models directly is important for good results. Tay et al. (2022) survey "efficient Transformers" that reduce the quadratic cost of attention for long sequences.

Decoder-only and encoder–decoder models. GPT (Radford et al., 2018) pre-trains a Transformer decoder as a left-to-right language model. GPT-3 (Brown et al., 2020), with 175 billion parameters, showed that a sufficiently large model can perform new tasks from instructions and a few examples placed in the prompt (in-context learning) without any fine-tuning. Instruction tuning with human feedback made such models usable as assistants. Encoder–decoder models such as T5 (Raffel et al., 2020), which casts every task as text-to-text, and BART (Lewis et al., 2020) are particularly strong for generation tasks such as summarisation and translation.

## Transformers for sentiment analysis and text classification

Sentiment analysis is one of the most studied text classification tasks (Zhang et al., 2018). Its most widely used benchmark is the Stanford Sentiment Treebank (Socher et al., 2013). Classical methods based on bag-of-words or TF-IDF features with linear classifiers (Joachims, 1998) reach about 79–83% accuracy on SST-2, recurrent and convolutional networks reach 84–88%, and fine-tuned Transformers exceed 90%: BERT-base reports 93.5%, RoBERTa-large 96.4% and the largest T5 model 97.5% on the GLUE test set.

Sun et al. (2019) showed that the learning rate, layer-wise learning rates and further in-domain pre-training strongly influence the result of fine-tuning BERT for text classification. Yin et al. (2019) proposed zero-shot classification by reformulating it as natural language inference (NLI): the text is the premise and a sentence such as "This text is about sport" is the hypothesis, so a model trained on NLI data can classify text into arbitrary labels without task-specific examples. Loureiro et al. (2022) trained RoBERTa models on Twitter data and showed that domain-specific pre-training matters for social-media text. Zhang et al. (2024) evaluated large language models such as ChatGPT on sentiment tasks and found that, while zero-shot LLMs are competitive on simple binary sentiment, smaller fine-tuned models still outperform them on tasks that require structured or fine-grained sentiment understanding.

Clark et al. (2019) and Rogers et al. (2020) showed that individual attention heads in BERT specialise in syntactic relations such as direct objects or coreference, whereas Jain and Wallace (2019) warned that attention weights are not automatically explanations of a prediction.

## Research in Lithuania

Lithuanian is a morphologically rich language with free word order and relatively few annotated resources, which makes it a challenging case for NLP. Kapočiūtė-Dzikienė et al. (2013) compared machine learning and lexicon-based approaches for sentiment classification of Lithuanian Internet comments and found that rich inflection and non-normative spelling strongly affect the choice of features. Kapočiūtė-Dzikienė, Damaševičius and Woźniak (2019) later compared traditional classifiers (multinomial naive Bayes, support vector machines) with deep neural networks (LSTM and CNN with word embeddings) on a dataset of Lithuanian comments in a three-class setting (positive, negative, neutral). The best traditional method (naive Bayes, about 0.735 accuracy) outperformed the best deep learning model (CNN with FastText embeddings, about 0.706): neural models without pre-training do not automatically win when data are limited.

The arrival of Transformers changed the situation for Baltic languages. Ulčar and Robnik-Šikonja (2021) trained LitLat BERT, a trilingual Lithuanian–Latvian–English model based on the XLM-RoBERTa architecture, and showed that the size of the training corpus and of the vocabulary strongly determine the quality of BERT models for the Baltic languages; models trained for a few languages outperform massively multilingual models on most tasks. Vileikytė, Lukoševičius and Stankevičius (2024) from Kaunas University of Technology fine-tuned multilingual BERT and T5 models for sentiment analysis of Lithuanian online reviews from several domains, reporting high accuracy on clearly polar reviews and lower accuracy on ambiguous middle ratings. Nakvosas, Daniušis and Mulevičius (2024) released the first open Llama2-based large language models for Lithuanian (7 and 13 billion parameters), pre-trained on more than 14 billion Lithuanian tokens.

## Comparative summary

Table {tab:lit} summarises the most relevant works; it is used later to place our results in context.

[[TABLE lit]]

## Conclusions of the literature review

The review defines the gap addressed by this project:

- Transformer models are the state of the art for text classification; on SST-2 fine-tuned Transformers exceed 90% accuracy, compared with roughly 79–83% for classical models.
- Most reported results come from GPU hardware and the full training set; training time on ordinary hardware, inference latency and model size are rarely reported together with accuracy.
- Lithuanian studies show that without pre-training deep networks do not necessarily outperform classical methods on small datasets (Kapočiūtė-Dzikienė et al., 2019), while Transformer-based work (Ulčar & Robnik-Šikonja, 2021; Vileikytė et al., 2024) indicates that pre-training is the key factor. The data efficiency of compact pre-trained models is insufficiently studied.
- Ready-made public models (off-the-shelf, cross-domain, zero-shot) are increasingly used instead of training, but are seldom compared with a fine-tuned in-domain model under one protocol.

Therefore, the practical part compares classical, recurrent, fine-tuned and ready-made Transformer models under identical CPU-only conditions, measuring quality and cost and the effect of the amount of labelled data.
