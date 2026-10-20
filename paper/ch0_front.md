#* ABSTRACT

This project paper studies the Transformer architecture and its application to natural language processing (NLP) tasks. The theoretical part explains how self-attention, multi-head attention, positional encoding and large-scale pre-training replaced recurrent neural networks as the dominant approach in NLP, and reviews the main families of Transformer models (encoder-only BERT-type models, decoder-only GPT-type models and encoder–decoder models such as T5 and BART). The literature review covers international research and research carried out in Lithuania, including work on Lithuanian sentiment analysis, the trilingual LitLat BERT model and the first open Lithuanian large language models. It concludes that most published comparisons are done on GPU servers with large models, while evidence on what Transformers offer under realistic resource limits (a laptop CPU, a few thousand labelled examples) is scarce.

The practical part answers this question experimentally on binary sentiment classification of movie-review sentences (Stanford Sentiment Treebank, SST-2). Ten models from four families were implemented in Python (PyTorch, Hugging Face Transformers, scikit-learn) and compared under one protocol: three classical TF-IDF models, a bidirectional LSTM trained from scratch, three Transformer encoders that we fine-tuned ourselves on a CPU (BERT-tiny, BERT-small and DistilBERT), and three ready-made Transformer models used without any training (an in-domain DistilBERT, an out-of-domain Twitter RoBERTa and a zero-shot natural language inference model). Models were evaluated by accuracy, macro F1, training time, inference latency and size, as well as on linguistically difficult subsets with negation and contrast. A learning-curve experiment measured how much labelled data each approach needs.

Keywords: Transformer, self-attention, BERT, DistilBERT, fine-tuning, transfer learning, sentiment analysis, natural language processing.

#* TEAM AND DISTRIBUTION OF WORK

The project was carried out by a team of three students. The work was divided as shown in Table 0, while all members took part in the formulation of the problem, the discussion of the results and the editing of the final text.

[[TABLE team]]
