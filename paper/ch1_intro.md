# 1. INTRODUCTION

## 1.1. Relevance of the topic

Natural language processing (NLP) is the branch of artificial intelligence that enables computers to analyse, understand and generate human language. For decades the field relied on hand-crafted features and statistical models such as n-gram language models, naive Bayes classifiers and support vector machines. In the 2010s these were gradually replaced by neural networks: word embeddings such as word2vec (Mikolov et al., 2013) gave words dense vector representations, and recurrent neural networks, in particular long short-term memory networks (Hochreiter & Schmidhuber, 1997), became the standard way to model sentences. Recurrent networks, however, process text one token at a time. This makes training slow, because the computation cannot be parallelised over the sequence, and makes it difficult to capture dependencies between words that are far apart.

In 2017 Vaswani et al. proposed the Transformer, an architecture that discards recurrence entirely and relies only on the attention mechanism. Every token in a sentence can directly "look at" every other token, and all positions are processed in parallel. Within two years the Transformer became the foundation of almost every state-of-the-art NLP system. BERT (Devlin et al., 2019) showed that a Transformer encoder pre-trained on unlabelled text can be fine-tuned to beat previous results on eleven language understanding tasks, and the GPT series (Radford et al., 2018; Brown et al., 2020) showed that a Transformer decoder trained to predict the next word becomes a general-purpose text generator able to solve tasks from a few examples. Today Transformers power machine translation, search engines, chat assistants, code generation tools and document analysis systems, and the same architecture has spread to computer vision, speech and biology.

The topic is relevant for three reasons. First, Transformers are now the default tool of NLP practitioners, so engineers need to understand both how they work and when they are worth their cost. Second, the computational price of these models is high: training and even running them requires far more memory and energy than classical methods (Strubell et al., 2019), so an informed choice between a small classical model, a compact Transformer and a large pre-trained model has a real economic and environmental impact. Third, for smaller languages such as Lithuanian, the question of how well Transformer methods transfer with limited labelled data is directly connected to the digital sustainability of the language (Ulčar & Robnik-Šikonja, 2021; Nakvosas et al., 2024).

## 1.2. Problem statement

Published benchmarks usually report the accuracy of the largest models, trained on GPU clusters with the full training set. A student, a small company or a public institution, by contrast, often has only an ordinary computer, a limited number of labelled examples and a need for fast predictions. The research problem of this project is therefore formulated as follows: how large is the practical advantage of Transformer models over classical and recurrent approaches for a typical NLP task (sentiment classification), and at what cost in training time, inference latency and model size, when all experiments are carried out on a standard CPU-only laptop?

## 1.3. Aim and objectives

The aim of the project is to analyse the Transformer architecture and its applications in NLP and to evaluate experimentally the effectiveness of Transformer-based models for sentiment classification in comparison with classical machine learning and recurrent neural network methods.

To reach the aim, the following objectives were set:

- to review the theoretical foundations of the Transformer architecture (self-attention, multi-head attention, positional encoding, pre-training and fine-tuning);
- to review scientific literature on Transformer models and their applications in NLP, in Lithuania and worldwide, and to identify gaps;
- to define the experimental methodology: dataset, compared models (competitors), evaluation criteria and the mathematical methods used;
- to implement the classical, recurrent and Transformer-based classifiers in Python and to run a comparative experiment on the SST-2 dataset under equal conditions;
- to analyse the results in terms of quality, data efficiency, speed and model size, including an error analysis on sentences with negation and contrast;
- to formulate conclusions and practical recommendations for choosing a model.

## 1.4. Research object and methods

The research object is Transformer-based language models applied to text classification. The research methods are analysis of scientific literature, comparative analysis, computational experiment (training and evaluation of models), statistical evaluation of the results and visualisation of attention weights.

## 1.5. Structure of the paper

The paper consists of an introduction, four main chapters, general conclusions and a list of references. Chapter 2 reviews the literature. Chapter 3 describes the methodology of the practical part: the problem formulation, the dataset, the compared models, the evaluation criteria, the design of the developed program and the mathematical methods. Chapter 4 presents the implementation and the results of the experiments. Chapter 5 summarises the conclusions. The source code of the program is provided together with the paper.
