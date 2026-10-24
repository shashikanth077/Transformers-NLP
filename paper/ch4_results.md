# Practical Part: Implementation and Results

## Implementation of the program

The program described in Section 2.5 was implemented in Python 3.11 with PyTorch 2 (CPU build), Hugging Face Transformers and Datasets, scikit-learn and Matplotlib; it consists of about 770 lines of code in nine scripts and is published in the project repository (https://github.com/shashikanth077/Transformers-NLP). The data preparation script downloads SST-2 from the Hugging Face Hub and creates the three splits of Table {tab:data}. The classical baselines are scikit-learn pipelines (TF-IDF vectoriser followed by the classifier). The BiLSTM is a custom PyTorch module with its own tokeniser and vocabulary of the words occurring at least twice in the training data. The core of the work is the generic fine-tuning script: it loads any pre-trained encoder and its tokeniser by name, adds a two-class classification head, pads every batch dynamically to its longest sentence, and runs the training loop with AdamW, linear warm-up and decay, gradient clipping and selection of the best epoch on the dev set. The ready-made models are evaluated through the Transformers pipeline interface, and the zero-shot model through the zero-shot-classification pipeline with the hypothesis template of Section 2.7.7.

During development two practical problems had to be solved. The newest version of the Datasets library requires full repository names, so SST-2 is loaded as nyu-mll/glue. The compact BERT models of Turc et al. (2019) published by third parties on the Hub lacked tokenizer files compatible with the current library version, therefore the official Google checkpoints (google/bert_uncased_L-2_H-128_A-2 and L-4_H-512_A-8) were used. The whole pipeline is started by one command (run_all.sh) and executes all models one after another; every result is saved as a JSON file, from which the report script generates all tables and figures of this chapter.

## Overall comparison of the models

The results of all ten models on the 872 test sentences are given in Table {tab:results} and Fig. {fig:accuracy}. Because the classes are almost balanced, accuracy and macro F1 differ by less than 0.2 percentage points (p.p.), so accuracy is used in the discussion.

[[TABLE results]]

[[FIG accuracy]]

Three groups can be distinguished. The best results were obtained by the DistilBERT models: the publicly available model that its authors fine-tuned on the full SST-2 training set reached 91.1%, and the DistilBERT that we fine-tuned ourselves on only 8,000 sentences reached 88.8%. The out-of-domain Twitter-RoBERTa (85.8%) and our fine-tuned BERT-small (83.7%) form the middle group. The classical models, the BiLSTM and the zero-shot model are clustered in a narrow band between 81.8% and 82.9%. The smallest Transformer, BERT-tiny with only 4.4 million parameters, was the weakest model (76.6%), which shows that a Transformer is not automatically better: its capacity must be sufficient for the task.

Table {tab:stats} shows how reliable these differences are. With 872 test sentences the 95% bootstrap confidence intervals are about ±2.5 p.p. wide. All models are significantly worse than the best one (McNemar p < 0.05), including our fine-tuned DistilBERT (p = 0.029), which was trained on eight times less data. Pairwise tests against the best classical model (TF-IDF + linear SVM) show that our fine-tuned DistilBERT is significantly better: it correctly classifies 104 sentences that the SVM misses, while the opposite happens for only 53 sentences (p < 0.001). In contrast, the differences between BERT-small and the SVM (71 vs 64 sentences, p = 0.61), between the BiLSTM and the SVM (p = 1.0) and between the zero-shot model and logistic regression (p = 1.0) are not statistically significant. The answer to RQ1 is therefore: a sufficiently large pre-trained Transformer improves accuracy by about 6 p.p. over the best classical and recurrent models, even when trained on far fewer labelled examples, whereas compact Transformers only match the classical baselines.

[[TABLE stats]]

The BiLSTM trained from scratch reached 82.8%, practically identical to the TF-IDF models, although its training took 100–200 times longer. This agrees with the finding of Kapočiūtė-Dzikienė et al. (2019) for Lithuanian that neural networks without pre-training do not outperform well-tuned classical methods. The BiLSTM also showed clear overfitting to the training distribution: its dev accuracy (91.3%, measured on short phrases from the training distribution) was much higher than its test accuracy on complete sentences.

## Cost: training time, inference latency and model size

Fig. {fig:latency} relates accuracy to inference latency (RQ2). The classical models classify a sentence in about 0.05 ms and train in 2–4 seconds on all 65 thousand sentences. The BiLSTM needs 0.45 ms per sentence and 6 minutes of training. The fine-tuned Transformers are slower by two to three orders of magnitude: BERT-tiny needs 1.2 ms, BERT-small 14 ms and DistilBERT 39 ms per sentence, and the larger ready-made models 45–107 ms. The zero-shot model is the slowest because it must process every sentence once for each candidate label.

[[FIG latency]]

Fine-tuning on the laptop CPU took 2 minutes for BERT-tiny, 26 minutes for BERT-small and about 2.6 hours for DistilBERT (8,000 sentences, 2–3 epochs). These times should be read as upper bounds: the laptop was also used for other tasks while the longest runs were executing, which slowed them down. Even so, the conclusion is clear: on a CPU, a compact or distilled Transformer can be fine-tuned within hours, and its inference speed (about 25 sentences per second for DistilBERT) is sufficient for most interactive applications, but not for very high-volume streams, where TF-IDF models remain hundreds of times cheaper. In terms of size, the TF-IDF models have about 74 thousand weights (one per n-gram), the BiLSTM 2 million parameters, and the Transformers 4.4–125 million parameters (17–500 MB of memory in 32-bit precision).

## Data efficiency of pre-training

The learning-curve experiment (RQ3) is summarised in Table {tab:lc} and Fig. {fig:lc}. Both models were trained on exactly the same random subsets.

[[TABLE lc]]

[[FIG lc]]

The advantage of pre-training is largest when labelled data are scarce. With only 500 labelled sentences BERT-small reaches 75.9%, while the TF-IDF model reaches only 61.8%, a difference of 14.1 p.p.; the TF-IDF model simply has not seen most of the words of the test sentences, whereas the pre-trained model already knows them. The difference decreases as the data grow (6.9 p.p. at 4,000 and 6.2 p.p. at 8,000 sentences). BERT-small trained on 4,000 sentences (82.7%) already matches the best classical model trained on all 65,349 sentences (82.9%), i.e. it achieves the same quality with about 16 times less labelled data, and DistilBERT trained on 8,000 sentences exceeds it by 5.9 p.p. This result is directly relevant for less-resourced languages such as Lithuanian, where large labelled datasets rarely exist.

## Ready-made and zero-shot models

The ready-made models answer RQ4. When a public model has been fine-tuned on the same domain and task, it is the best option: DistilBERT-SST2 achieved the highest accuracy without any training on our side. A model trained for the same task in a different domain (Twitter-RoBERTa) lost 5.3 p.p. compared with it, although it still beat all classical models; the informal style of tweets differs from the language of film critics. The zero-shot NLI model, which has never seen a sentiment label, reached 81.8% – the same level as TF-IDF logistic regression trained on 65 thousand labelled sentences. This illustrates the general language understanding acquired during pre-training: the task can be specified by a natural-language hypothesis instead of training data, at the price of the highest inference cost.

## Error analysis: negation and contrast

To answer RQ5, accuracy was measured on 184 test sentences containing negation and 130 sentences containing contrast markers (Table {tab:subsets}, Fig. {fig:subsets}).

[[TABLE subsets]]

[[FIG subsets]]

The bag-of-words models lose 8–9 p.p. on sentences with negation, because a sentence such as "not good" contains the positive word "good". The large Transformers are much more robust: DistilBERT-SST2 loses 4.1 p.p., our fine-tuned DistilBERT 3.4 p.p. and Twitter-RoBERTa 2.1 p.p. The zero-shot NLI model is the most robust of all (−0.2 p.p.), which is plausible because NLI training specifically teaches a model to handle negation and contradiction. The compact Transformers, however, behave like the classical models: BERT-small loses 9.3 p.p. on negation, and BERT-tiny 7.0 p.p. Interestingly, the BiLSTM, which reads the sentence in order, handles negation better (−4.0 p.p.) than TF-IDF. The advantage of Transformers on difficult sentences is therefore not a property of the architecture alone but of the combination of architecture, model capacity and pre-training.

Fig. {fig:attention} shows the self-attention weights of DistilBERT-SST2 for the sentence "the plot was not good, but the acting was brilliant". The model classifies the sentence as positive with a probability of 1.000, i.e. it gives the clause after "but" the decisive weight, which agrees with how people read such sentences. In the last layer much of the attention of all words goes to the comma that separates the two clauses and to the [SEP] token; such "attention sinks" on punctuation and separator tokens were also observed in BERT by Clark et al. (2019). The words of the second clause ("acting", "was", "brilliant") spread their remaining attention mainly over the other words of the same clause. As noted by Jain and Wallace (2019), such pictures illustrate the information flow but should not be treated as a complete explanation of the decision.

[[FIG attention]]

Examples from the test set show the typical behaviour. The fine-tuned DistilBERT correctly classified 104 sentences that the SVM misclassified, for example "in its best moments, resembles a bad high school production of grease, without benefit of song" (negative; the SVM is misled by "best") and "the subtle strength of elling is that it never loses touch with the reality of the grim situation" (positive; "grim" and "never loses" mislead the bag-of-words model). The SVM was right in 53 cases where DistilBERT failed, for example in sentences with mixed sentiment such as "we root for (clara and paul), even like them, though perhaps it's an emotion closer to pity".

## Comparison with previous research

Our results are consistent with the literature summarised in Table {tab:lit}. The classical models (81.9–82.9%) match the 79–83% reported for bag-of-words models on SST-2 (Socher et al., 2013), and the off-the-shelf DistilBERT (91.1%) is close to the 91.3% reported by Sanh et al. (2019) on the same validation set, which confirms that our evaluation procedure is correct. Our fine-tuned DistilBERT stays 2.3 p.p. below it because it was trained on 12% of the training data. Compared with the BERT-base result of 93.5% (Devlin et al., 2019), our models are smaller and trained with fewer data, which explains the gap. The new element of this study is the joint view of quality, cost and data efficiency on a CPU-only computer, which shows when the advantage of Transformers is worth its price.
