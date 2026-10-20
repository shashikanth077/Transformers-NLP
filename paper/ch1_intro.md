#* Introduction

**Relevance of the research.** Natural language processing (NLP) is the branch of artificial intelligence that enables computers to analyse, understand and generate human language. For decades the field relied on hand-crafted features and statistical models such as n-gram language models, naive Bayes classifiers and support vector machines. In the 2010s these were gradually replaced by neural networks: word embeddings such as word2vec (Mikolov et al., 2013) gave words dense vector representations, and recurrent neural networks, in particular long short-term memory networks (Hochreiter & Schmidhuber, 1997), became the standard way to model sentences. Recurrent networks, however, process text one token at a time, which makes training slow and makes it difficult to capture dependencies between words that are far apart.

In 2017 Vaswani et al. proposed the Transformer, an architecture that discards recurrence and relies only on the attention mechanism: every token can directly "look at" every other token, and all positions are processed in parallel. Within two years the Transformer became the foundation of almost every state-of-the-art NLP system. BERT (Devlin et al., 2019) showed that a Transformer encoder pre-trained on unlabelled text can be fine-tuned to beat previous results on eleven language understanding tasks, and the GPT series (Radford et al., 2018; Brown et al., 2020) showed that a Transformer decoder trained to predict the next word becomes a general-purpose text generator. Today Transformers power machine translation, search engines, chat assistants and document analysis systems.

The topic is relevant for three reasons. First, Transformers are now the default tool of NLP practitioners, so engineers need to understand both how they work and when they are worth their cost. Second, the computational price of these models is high (Strubell et al., 2019), so an informed choice between a small classical model, a compact Transformer and a large pre-trained model has a real economic and environmental impact. Third, for smaller languages such as Lithuanian, the question of how well Transformer methods work with limited labelled data is directly connected to the digital sustainability of the language (Ulčar & Robnik-Šikonja, 2021; Nakvosas et al., 2024).

**Research problem.** Published benchmarks usually report the accuracy of the largest models trained on GPU clusters with the full training set. A student, a small company or a public institution, by contrast, often has only an ordinary computer, a limited number of labelled examples and a need for fast predictions. The research problem is therefore: how large is the practical advantage of Transformer models over classical and recurrent approaches for a typical NLP task (sentiment classification), and at what cost in training time, inference latency and model size, when all experiments are carried out on a standard CPU-only laptop?

**Research object.** Transformer-based language models applied to text classification.

**Research aim.** To analyse the Transformer architecture and its applications in NLP and to evaluate experimentally the effectiveness of Transformer-based models for sentiment classification in comparison with classical machine learning and recurrent neural network methods.

**Objectives to achieve the aim:**

1. To review scientific literature on Transformer models and their applications in NLP, in Lithuania and worldwide, and to identify research gaps.
2. To describe the theoretical foundations of the Transformer architecture (self-attention, multi-head attention, positional encoding, pre-training and fine-tuning).
3. To define the experimental methodology: dataset, compared models (competitors), evaluation criteria and mathematical methods.
4. To implement classical, recurrent and Transformer-based classifiers in Python and run a comparative experiment on the SST-2 dataset under equal conditions.
5. To analyse the results in terms of quality, data efficiency, speed and model size, including an error analysis on sentences with negation and contrast, and to formulate practical recommendations.

**Research methods.** Analysis of scientific literature, comparative analysis, computational experiment (training and evaluation of models), statistical evaluation of the results (bootstrap confidence intervals, McNemar's test) and visualisation of attention weights.

**Structure of the work.** The paper consists of an introduction, three chapters, conclusions, a list of references and an appendix. Chapter 1 reviews the literature, Chapter 2 describes the research methodology (problem, dataset, competitors, criteria, program design and mathematical methods), and Chapter 3 presents the implementation and the results of the experiments. The source code is publicly available in the project repository (Appendix A).

**Distribution of work.** The project was carried out by a team of three students. All members took part in formulating the problem, discussing the results and editing the final text; the main responsibilities are shown in Table {tab:team}.

[[TABLE team]]
