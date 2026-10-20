"""Kannada study notes, chapters 6-12."""
from kn_diagrams import pipeline_kn, split_bar_kn
from kn_content_a import box, fig

IMG = "../../code/figures/"

FAM_KN = {"classical": "ಸಾಂಪ್ರದಾಯಿಕ (ಪದ ಎಣಿಕೆ)", "recurrent": "ಪುನರಾವರ್ತಿತ ನೆಟ್‌ವರ್ಕ್", "ft": "ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್: ನಾವು ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿದ್ದು", "ready": "ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್: ಬೇರೆಯವರು ಮಾಡಿದ್ದು (ready-made)", "zero": "ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್: zero-shot"}
MODEL_KN = {
    "tfidf_nb": ("TF-IDF + Naive Bayes", "classical", "ಪದಗಳ ಸಂಭಾವ್ಯತೆ ಗುಣಿಸುವ ಸರಳ ಮಾಡೆಲ್"),
    "tfidf_logreg": ("TF-IDF + Logistic Regression", "classical", "ಪ್ರತಿ ಪದ ಸಾಕ್ಷ್ಯ ಕೂಡಿಸುತ್ತದೆ"),
    "tfidf_svm": ("TF-IDF + Linear SVM", "classical", "ಎರಡು ವರ್ಗಗಳ ನಡುವೆ ಗೆರೆ ಎಳೆಯುತ್ತದೆ"),
    "bilstm": ("BiLSTM", "recurrent", "ಎರಡೂ ದಿಕ್ಕಿನಲ್ಲಿ ಒಂದೊಂದೇ ಪದ ಓದುತ್ತದೆ, ಪೂರ್ವ-ತರಬೇತಿ ಇಲ್ಲ"),
    "ft_bert-tiny_8000": ("BERT-tiny", "ft", "2 ಪದರ, ಅತಿ ಚಿಕ್ಕ"),
    "ft_bert-small_8000": ("BERT-small", "ft", "4 ಪದರ, ಮಧ್ಯಮ"),
    "ft_distilbert_8000": ("DistilBERT", "ft", "6 ಪದರ, BERT ನ ಸಾರಾಂಶ ರೂಪ"),
    "pt_distilbert_sst2": ("DistilBERT-SST2", "ready", "ಇದೇ ಕ್ಷೇತ್ರಕ್ಕೆ ಈಗಾಗಲೇ ತರಬೇತಿ ಪಡೆದದ್ದು"),
    "pt_twitter_roberta": ("Twitter-RoBERTa", "ready", "ಟ್ವೀಟ್‌ಗಳಲ್ಲಿ ತರಬೇತಿ: ಬೇರೆ ಕ್ಷೇತ್ರ"),
    "zs_nli_distilroberta": ("Zero-shot NLI DistilRoBERTa", "zero", "ಭಾವನೆಯ ಒಂದೇ ಲೇಬಲ್ ಕೂಡ ನೋಡಿಲ್ಲ"),
}


def ch6(N):
    rows = ""
    for tag, (en, fam, desc) in MODEL_KN.items():
        lab = N["labels"][tag]
        rows += f'<tr><td class="en"><b>{en}</b></td><td>{FAM_KN[fam]}</td><td class="num">{lab}</td><td class="num">{N["params"][tag]}</td><td>{desc}</td></tr>'
    return f"""
<h1 class="ch" id="c6">ಅಧ್ಯಾಯ 6 · ನಮ್ಮ ಪ್ರಯೋಗ ಹೇಗೆ ನಡೆಯಿತು<small>THE EXPERIMENT, STEP BY STEP</small></h1>
<p>ಪ್ರಾಜೆಕ್ಟ್ ಒಂಬತ್ತು ಹೆಜ್ಜೆಗಳಲ್ಲಿ ನಡೆಯಿತು. ಪ್ರತಿ ಹೆಜ್ಜೆಯ ಫಲಿತಾಂಶ ಒಂದು ಫೈಲ್‌ನಲ್ಲಿ ಉಳಿಯುತ್ತದೆ, ಮುಂದಿನ ಹೆಜ್ಜೆ ಅದನ್ನು ಬಳಸುತ್ತದೆ. ಹಾಗಾಗಿ ಯಾರು ಬೇಕಾದರೂ ಇದೇ ಪ್ರಯೋಗವನ್ನು ಮತ್ತೆ ಓಡಿಸಬಹುದು (<span class="en">reproducible</span>).</p>
{fig(pipeline_kn(), "ಚಿತ್ರ 6.1: ಪ್ರಾಜೆಕ್ಟ್‌ನ ಒಂಬತ್ತು ಹೆಜ್ಜೆಗಳು.")}
<table><thead><tr><th style="width:5%">#</th><th style="width:22%">ಹೆಜ್ಜೆ</th><th>ಏನು ಮಾಡಿದೆವು</th><th style="width:28%">ಏಕೆ</th></tr></thead><tbody>
<tr><td>1</td><td><b>ಪ್ರಶ್ನೆ ರೂಪಿಸುವುದು</b></td><td>ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ ಕಾರ್ಯ; 5 ಸಂಶೋಧನೆ ಪ್ರಶ್ನೆಗಳು</td><td>ಸ್ಪಷ್ಟ ಪ್ರಶ್ನೆ ಇದ್ದರೆ ಮಾತ್ರ ನ್ಯಾಯವಾದ ಪ್ರಯೋಗ ಮಾಡಬಹುದು</td></tr>
<tr><td>2</td><td><b>ಸಾಹಿತ್ಯ ಸಮೀಕ್ಷೆ</b></td><td>Google Scholar ಮತ್ತು Web of Science ನಲ್ಲಿ ನಿರ್ದಿಷ್ಟ ಕ್ವೆರಿ ಹಾಕಿ ಸುಮಾರು 40 ಲೇಖನ ಆರಿಸಿ ವಿಶ್ಲೇಷಿಸಿದೆವು; ಲಿಥುವೇನಿಯಾ ಸಂಶೋಧನೆಯೂ</td><td>ಏನು ಈಗಾಗಲೇ ಗೊತ್ತಿದೆ, ಏನು ಬಾಕಿ ಇದೆ (gap) ಎಂದು ತಿಳಿಯಲು</td></tr>
<tr><td>3</td><td><b>ಡೇಟಾ ವಿಭಜನೆ</b></td><td>SST-2 ಡೇಟಾವನ್ನು train / dev / test ಎಂದು ವಿಭಾಗಿಸಿದೆವು (ಕೆಳಗೆ ನೋಡಿ)</td><td>ಪರೀಕ್ಷೆಯ ಪ್ರಶ್ನೆ ಮೊದಲೇ ನೋಡಬಾರದು</td></tr>
<tr><td>4</td><td><b>ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್‌ಗಳು</b></td><td>TF-IDF + NB / LR / SVM</td><td>ಬಲವಾದ, ಅಗ್ಗದ ಹೋಲಿಕೆ ಮಾನದಂಡ (baseline)</td></tr>
<tr><td>5</td><td><b>BiLSTM</b></td><td>ಪೂರ್ವ-ತರಬೇತಿ ಇಲ್ಲದ ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್</td><td>ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಬರುವ ಮೊದಲಿನ ಮಾನದಂಡ</td></tr>
<tr><td>6</td><td><b>ಫೈನ್-ಟ್ಯೂನಿಂಗ್</b></td><td>BERT-tiny, BERT-small, DistilBERT ಅನ್ನು 8,000 ವಾಕ್ಯಗಳಲ್ಲಿ</td><td>ನಿಜ ಜೀವನದ ಸನ್ನಿವೇಶ: ಕೆಲವು ಸಾವಿರ ಲೇಬಲ್, ಒಂದು ಲ್ಯಾಪ್ಟಾಪ್</td></tr>
<tr><td>7</td><td><b>Ready-made ಮತ್ತು zero-shot</b></td><td>ತರಬೇತಿ ಕೊಡದೆ ಮೂರು ಮಾಡೆಲ್‌ಗಳ ಪರೀಕ್ಷೆ</td><td>ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ ಬಳಸಿದರೆ ಎಷ್ಟು ಒಳ್ಳೆಯದು?</td></tr>
<tr><td>8</td><td><b>ಮೌಲ್ಯಮಾಪನ</b></td><td>ನಿಖರತೆ, F1, ಸಮಯ, ವೇಗ, ಗಾತ್ರ, learning curve, ವಿಶ್ವಾಸ ಅಂತರ, McNemar ಪರೀಕ್ಷೆ, ನಿರಾಕರಣೆ/ವೈದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆ</td><td>ವ್ಯತ್ಯಾಸ ನಿಜವೋ ಅದೃಷ್ಟವೋ ಎಂದು ತಿಳಿಯಲು</td></tr>
<tr><td>9</td><td><b>ವರದಿ</b></td><td>ಟೇಬಲ್, ಚಿತ್ರ, ಪೇಪರ್, ಸ್ಲೈಡ್, GitHub ರೆಪೊಸಿಟರಿ</td><td>ಪ್ರತಿ ಸಂಖ್ಯೆಯೂ ಫೈಲ್‌ನಿಂದ ಬರುವಂತೆ: ತಪ್ಪಿಗೆ ಅವಕಾಶವಿಲ್ಲ</td></tr></tbody></table>

<h2>ಡೇಟಾ: ಪಠ್ಯಪುಸ್ತಕ, ಮಾದರಿ ಪ್ರಶ್ನೆಪತ್ರಿಕೆ, ನಿಜವಾದ ಪರೀಕ್ಷೆ</h2>
{fig(split_bar_kn(N["n_train_i"], N["n_dev_i"], N["n_test_i"]), "ಚಿತ್ರ 6.2: ಡೇಟಾ ವಿಭಜನೆ: train / dev / test.")}
<p>ಡೇಟಾ <b>SST-2</b> (Stanford Sentiment Treebank): ಸಿನಿಮಾ ವಿಮರ್ಶೆಗಳ ವಾಕ್ಯಗಳು, ಧನಾತ್ಮಕ ಅಥವಾ ಋಣಾತ್ಮಕ. GLUE ಯ ನಿಜವಾದ ಟೆಸ್ಟ್ ಉತ್ತರಗಳು ಮುಚ್ಚಿಡಲ್ಪಟ್ಟಿರುವುದರಿಂದ ಅವರ <b>validation set ಅನ್ನೇ ನಮ್ಮ test set</b> ಮಾಡಿಕೊಂಡೆವು (ಇದು ಸಾಮಾನ್ಯ ಪದ್ಧತಿ). ತರಬೇತಿ ಡೇಟಾದಿಂದ 2,000 ವಾಕ್ಯ ತೆಗೆದು ನಮ್ಮದೇ <b>dev set</b> ಮಾಡಿದೆವು.</p>
{box("kn", "ನಮ್ಮ ಊರಿನ ಉದಾಹರಣೆ: ಬೋರ್ಡ್ ಪರೀಕ್ಷೆ", "ಪಠ್ಯಪುಸ್ತಕದಿಂದ ಓದಿ ಕಲಿಯುವುದು (train). ಶಾಲೆಯ ಮಾದರಿ ಪ್ರಶ್ನೆಪತ್ರಿಕೆಯಲ್ಲಿ ಅಭ್ಯಾಸ ಮಾಡಿ ಯಾವ ವಿಧಾನ ಉತ್ತಮ ಎಂದು ನೋಡುವುದು (dev). ನಿಜವಾದ SSLC ಬೋರ್ಡ್ ಪರೀಕ್ಷೆ ಒಮ್ಮೆ ಮಾತ್ರ, ಅದರ ಪ್ರಶ್ನೆಪತ್ರಿಕೆ ಮೊದಲೇ ನೋಡಿದರೆ ಅದು ಪರೀಕ್ಷೆಯೇ ಅಲ್ಲ (test). ಮಾಡೆಲ್ ಅನ್ನು ಟೆಸ್ಟ್ ಡೇಟಾದಲ್ಲಿ ತರಬೇತಿ ಮಾಡಿದರೆ ಫಲಿತಾಂಶ ಸುಳ್ಳು ಒಳ್ಳೆಯದಾಗಿ ಕಾಣುತ್ತದೆ.")}
{box("note", "ಟೆಸ್ಟ್ ಮತ್ತು ತರಬೇತಿ ವಾಕ್ಯಗಳು ಬೇರೆ ರೀತಿ", f"ತರಬೇತಿಯಲ್ಲಿ ಚಿಕ್ಕ ಪದಗುಚ್ಛಗಳು (ಸರಾಸರಿ {N['train_len']} ಪದ), ಟೆಸ್ಟ್‌ನಲ್ಲಿ ಪೂರ್ತಿ ವಾಕ್ಯಗಳು (ಸರಾಸರಿ {N['test_len']} ಪದ). ಅಂದರೆ ಮಾಡೆಲ್‌ಗಳು ಚಿಕ್ಕದರಿಂದ ಕಲಿತು ದೊಡ್ಡದಕ್ಕೆ ಅನ್ವಯಿಸಬೇಕು: ಇದೇ ಸವಾಲಿನ ಭಾಗ.")}

<h2>ನಮ್ಮ ಹತ್ತು ಮಾಡೆಲ್‌ಗಳು</h2>
<table><thead><tr><th>ಮಾಡೆಲ್</th><th>ಕುಟುಂಬ</th><th class="num">ತರಬೇತಿ ಲೇಬಲ್</th><th class="num">ಪ್ಯಾರಾಮೀಟರ್</th><th>ಒಂದು ಸಾಲಿನ ವಿವರಣೆ</th></tr></thead><tbody>{rows}</tbody></table>
<p class="small">ಪ್ಯಾರಾಮೀಟರ್ ಅಂದರೆ ಮಾಡೆಲ್ ಕಲಿತ ಸಂಖ್ಯೆಗಳು. TF-IDF ಮಾಡೆಲ್‌ಗಳಲ್ಲಿ ಇದು ಪದ/ಪದ-ಜೋಡಿಗಳ ತೂಕಗಳ ಸಂಖ್ಯೆ.</p>
{box("kn", "ನಮ್ಮ ಊರಿನ ಉದಾಹರಣೆ: ಪ್ಯಾರಾಮೀಟರ್ = ರೇಡಿಯೋ ತಿರುಗಣೆ ಗುಂಡಿಗಳು", f"ಹಳೆಯ ರೇಡಿಯೋದಲ್ಲಿ ಸ್ಟೇಷನ್ ಹುಡುಕಲು ಎರಡು ಗುಂಡಿ ತಿರುಗಿಸಿದರೆ ಸಾಕು. ಪ್ರತಿ ಗುಂಡಿಯೂ ಒಂದು ಪ್ಯಾರಾಮೀಟರ್. BERT-tiny ಗೆ {N['params']['ft_bert-tiny_8000']} ಗುಂಡಿಗಳು, DistilBERT ಗೆ {N['params']['ft_distilbert_8000']} ಗುಂಡಿಗಳು! ತರಬೇತಿ ಅಂದರೆ ಆ ಎಲ್ಲಾ ಗುಂಡಿಗಳನ್ನು ಸರಿಯಾದ ಸ್ಥಾನಕ್ಕೆ ತಿರುಗಿಸುವುದು.")}

<h2>ತರಬೇತಿ ಹೇಗೆ ನಡೆಯುತ್ತದೆ? (ಸರಳ ತಾಂತ್ರಿಕ ಟಿಪ್ಪಣಿ)</h2>
<ul>
<li><b>ನಷ್ಟ (loss):</b> ಮಾಡೆಲ್ ಎಷ್ಟು ತಪ್ಪು ಮಾಡಿದೆ ಎಂಬ ಅಳತೆ (cross-entropy). ಇದನ್ನು ಕಡಿಮೆ ಮಾಡುವುದೇ ತರಬೇತಿ.</li>
<li><b>ಕಲಿಕೆ ದರ (learning rate):</b> ಒಂದು ಬಾರಿ ಗುಂಡಿಗಳನ್ನು ಎಷ್ಟು ತಿರುಗಿಸುವುದು ಎಂಬ ಹೆಜ್ಜೆ ಗಾತ್ರ.</li>
<li><b>AdamW:</b> ತಿರುಗಿಸುವ ವಿಧಾನ (optimizer). <b>Warm-up:</b> ಮೊದಲ 10% ಹೆಜ್ಜೆ ನಿಧಾನ, ನಂತರ ಕ್ರಮೇಣ ಕಡಿಮೆ.</li>
<li><b>ಎಪಾಕ್ (epoch):</b> ಎಲ್ಲಾ ತರಬೇತಿ ವಾಕ್ಯಗಳನ್ನು ಒಮ್ಮೆ ನೋಡುವುದು. ನಾವು 2–3 ಎಪಾಕ್ ಮಾಡಿ, dev ನಲ್ಲಿ ಉತ್ತಮ ಎಪಾಕ್ ಆರಿಸಿದೆವು.</li></ul>
{box("kn", "ನಮ್ಮ ಊರಿನ ಉದಾಹರಣೆ: ಚಾಮುಂಡಿ ಬೆಟ್ಟ ಇಳಿಯುವುದು", "ಮಂಜು ಮುಸುಕಿದಾಗ ಚಾಮುಂಡಿ ಬೆಟ್ಟದ ಮೆಟ್ಟಿಲು ಇಳಿಯುವಾಗ ಕಾಲಿನ ಕೆಳಗಿನ ಇಳಿಜಾರು ನೋಡಿ ಹೆಜ್ಜೆ ಇಡುತ್ತೇವೆ. ದೊಡ್ಡ ಹೆಜ್ಜೆ ಹಾಕಿದರೆ ಜಾರಿ ಬೀಳುತ್ತೇವೆ, ತುಂಬಾ ಚಿಕ್ಕ ಹೆಜ್ಜೆಯಾದರೆ ಕೆಳಗೆ ತಲುಪುವುದೇ ಇಲ್ಲ. ಇದೇ <b>ಕಲಿಕೆ ದರ</b>. ಮೆಟ್ಟಿಲು ಇಳಿಯುವ ಜಾಗ = ನಷ್ಟ ಕಡಿಮೆಯಾಗುವ ದಿಕ್ಕು.")}
<h3>ಅಳತೆಗಳು (metrics)</h3>
<table><thead><tr><th>ಅಳತೆ</th><th>ಸರಳ ಅರ್ಥ</th></tr></thead><tbody>
<tr><td><b>ನಿಖರತೆ (accuracy)</b></td><td>100 ವಿಮರ್ಶೆಗಳಲ್ಲಿ ಎಷ್ಟು ಸರಿಯಾಗಿ ಹೇಳಿತು. ನಮ್ಮಲ್ಲಿ ಒಳ್ಳೆಯ/ಕೆಟ್ಟ ವಿಮರ್ಶೆಗಳು ಬಹುತೇಕ ಸಮ, ಹಾಗಾಗಿ ನಿಖರತೆ ಮತ್ತು F1 ಒಂದೇ ರೀತಿ</td></tr>
<tr><td><b>F1, precision, recall</b></td><td>ಪ್ರತಿ ವರ್ಗಕ್ಕೆ ಎಷ್ಟು ಸರಿ ಎಂದು ಸಮತೋಲನವಾಗಿ ನೋಡುವ ಅಳತೆಗಳು</td></tr>
<tr><td><b>ತರಬೇತಿ ಸಮಯ</b></td><td>ಮಾಡೆಲ್ ತಯಾರಿಸಲು ಬೇಕಾದ ಸಮಯ</td></tr>
<tr><td><b>ಪ್ರತಿ ವಾಕ್ಯದ ಸಮಯ (latency)</b></td><td>ಒಂದು ವಿಮರ್ಶೆಗೆ ಉತ್ತರ ಕೊಡಲು ಬೇಕಾದ ಸಮಯ, ಹೋಟೆಲ್‌ನಲ್ಲಿ ದೋಸೆ ಬರುವ ಸಮಯದಂತೆ (ms = ಮಿಲಿಸೆಕೆಂಡ್)</td></tr>
<tr><td><b>ಗಾತ್ರ (parameters)</b></td><td>ಮಾಡೆಲ್ ಎಷ್ಟು ದೊಡ್ಡದು</td></tr></tbody></table>
"""


def ch7(N):
    a = N["acc"]
    rows = ""
    for tag in N["rank"]:
        en, fam, _ = MODEL_KN[tag]
        hi = " class='hl'" if tag in ("pt_distilbert_sst2",) else ""
        lo, up = N["ci"][tag]
        rows += f'<tr{hi}><td class="en"><b>{en}</b></td><td class="num"><b>{a[tag]}</b></td><td class="num">{lo}–{up}</td><td class="num">{N["train"][tag]}</td><td class="num">{N["lat"][tag]}</td><td class="num">{N["params"][tag]}</td></tr>'
    sub_rows = ""
    for tag, allv, neg, con, dn, dc in N["sub"]:
        sub_rows += f'<tr><td class="en"><b>{MODEL_KN[tag][0]}</b></td><td class="num">{allv}</td><td class="num">{neg}</td><td class="num">{con}</td><td class="num">{dn}</td><td class="num">{dc}</td></tr>'
    mc = N["mc"]
    return f"""
<h1 class="ch" id="c7">ಅಧ್ಯಾಯ 7 · ಫಲಿತಾಂಶಗಳು: ಸಂಖ್ಯೆಗಳು ಏನು ಹೇಳುತ್ತವೆ?<small>RESULTS</small></h1>
<div class="grid4">
<div class="stat"><b>{a['pt_distilbert_sst2']}%</b><span>ಅತ್ಯುತ್ತಮ: ready-made DistilBERT (ನಾವು ತರಬೇತಿ ಕೊಟ್ಟಿಲ್ಲ)</span></div>
<div class="stat"><b>{a['ft_distilbert_8000']}%</b><span>ನಾವು ಕೇವಲ 8,000 ವಾಕ್ಯಗಳಲ್ಲಿ ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿದ DistilBERT</span></div>
<div class="stat b"><b>{a['tfidf_svm']}%</b><span>ಅತ್ಯುತ್ತಮ ಪದ ಎಣಿಕೆ (SVM), ಎಲ್ಲಾ {N['n_train']} ವಾಕ್ಯಗಳಲ್ಲಿ ಕಲಿತದ್ದು</span></div>
<div class="stat o"><b>{a['bilstm']}%</b><span>BiLSTM: ಪದ ಎಣಿಕೆಗಿಂತ ಹೆಚ್ಚೇನಿಲ್ಲ</span></div></div>
<h2>7.1 ಎಲ್ಲಾ ಹತ್ತು ಮಾಡೆಲ್‌ಗಳ ಸ್ಪರ್ಧೆ (RQ1)</h2>
<table><thead><tr><th>ಮಾಡೆಲ್</th><th class="num">ನಿಖರತೆ %</th><th class="num">95% ವಿಶ್ವಾಸ ಅಂತರ</th><th class="num">ತರಬೇತಿ ಸಮಯ</th><th class="num">ms / ವಾಕ್ಯ</th><th class="num">ಪ್ಯಾರಾಮೀಟರ್</th></tr></thead><tbody>{rows}</tbody></table>
{fig(f'<img src="{IMG}fig_accuracy.png" alt="Accuracy of all models">', f"ಚಿತ್ರ 7.1: ಎಲ್ಲಾ ಮಾಡೆಲ್‌ಗಳ ನಿಖರತೆ (ನೀಲಿ = ಪದ ಎಣಿಕೆ, ಕೇಸರಿ = BiLSTM, ಹಸಿರು = ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್). ಇದು ನಮ್ಮ ಪ್ರಯೋಗದ ನಿಜವಾದ ಚಿತ್ರ.")}
<ul>
<li><b>ಕಾವ್ಯ (ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್) ಗೆದ್ದಳು.</b> ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿದ DistilBERT {a['ft_distilbert_8000']}% ಮತ್ತು ರಮೇಶನ SVM {a['tfidf_svm']}%: ಅಂದರೆ <b>+{N['gain']} ಅಂಕ</b>, ಅದೂ 8 ಪಟ್ಟು ಕಡಿಮೆ ಲೇಬಲ್ ಡೇಟಾದಿಂದ.</li>
<li><b>ಇದು ನಿಜವಾದ ವ್ಯತ್ಯಾಸವೇ?</b> McNemar ಪರೀಕ್ಷೆ: DistilBERT ಮಾತ್ರ ಸರಿಯಾಗಿ ಹೇಳಿದ ವಾಕ್ಯಗಳು <b>{mc['ft_svm'][0]}</b>, SVM ಮಾತ್ರ ಸರಿಯಾಗಿ ಹೇಳಿದ ವಾಕ್ಯಗಳು <b>{mc['ft_svm'][1]}</b>, p {mc['ft_svm'][2]}. ಹೌದು, ಇದು ಅದೃಷ್ಟವಲ್ಲ.</li>
<li><b>ಸುಮಾ (BiLSTM) = ರಮೇಶ.</b> {a['bilstm']}% ಮತ್ತು {a['tfidf_svm']}%: ವ್ಯತ್ಯಾಸ ಗಣನೀಯವಲ್ಲ (McNemar: {mc['lstm_svm'][0]} vs {mc['lstm_svm'][1]}, p = {mc['lstm_svm'][2]}).</li>
<li><b>BERT-tiny ಅತ್ಯಂತ ಕೆಳಗೆ ({a['ft_bert-tiny_8000']}%).</b> ಪೂರ್ವ-ತರಬೇತಿ ಇದ್ದರೂ ಮಾಡೆಲ್ ತುಂಬಾ ಚಿಕ್ಕದು (2 ಪದರ). ಐದನೇ ತರಗತಿಯ ಮಗುವಿಗೆ ಇಂಜಿನಿಯರಿಂಗ್ ಪ್ರಶ್ನೆಪತ್ರಿಕೆ ಕೊಟ್ಟಂತೆ. <b>ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಎಂದರೆ ಯಾವಾಗಲೂ ಉತ್ತಮ ಎಂದಲ್ಲ; ಗಾತ್ರ ಮುಖ್ಯ.</b></li>
<li><b>Ready-made DistilBERT-SST2: {a['pt_distilbert_sst2']}%</b> ತರಬೇತಿ ಇಲ್ಲದೆ! ಇದನ್ನು ಬೇರೆಯವರು ಇದೇ ಕ್ಷೇತ್ರದಲ್ಲಿ ತರಬೇತಿ ಮಾಡಿದ್ದಾರೆ.</li>
<li><b>Twitter-RoBERTa: {a['pt_twitter_roberta']}%.</b> ಟ್ವೀಟ್‌ಗಳಲ್ಲಿ ತರಬೇತಿ ಪಡೆದದ್ದು, ಸಿನಿಮಾ ವಿಮರ್ಶೆಗೆ ಬೇರೆ ಕ್ಷೇತ್ರ: ಕ್ರಿಕೆಟ್ ಕಾಮೆಂಟರಿ ಕಲಿತವನು ಸಿನಿಮಾ ವಿಮರ್ಶೆ ಬರೆದಂತೆ; ಇನ್ನೂ ಸಾಂಪ್ರದಾಯಿಕ ಮಾಡೆಲ್‌ಗಳಿಗಿಂತ ಉತ್ತಮ, ಆದರೆ ಇದೇ-ಕ್ಷೇತ್ರದ ಮಾಡೆಲ್‌ಗಿಂತ ಕಡಿಮೆ.</li>
<li><b>Zero-shot: {a['zs_nli_distilroberta']}%</b>, ಒಂದೇ ಒಂದು ಭಾವನೆ-ಲೇಬಲ್ ನೋಡದೆ! ಇದು {N['n_train']} ವಾಕ್ಯಗಳಲ್ಲಿ ಕಲಿತ logistic regression ನಷ್ಟೇ ಒಳ್ಳೆಯದು.</li></ul>
{box("exam", "ವಿಶ್ವಾಸ ಅಂತರ (confidence interval) ಅಂದರೆ?", f"ಕೇವಲ {N['n_test']} ವಾಕ್ಯಗಳಲ್ಲಿ ಪರೀಕ್ಷಿಸಿದ್ದರಿಂದ “91.1%” ಎಂದರೆ ನಿಖರವಾಗಿ 91.1% ಅಲ್ಲ, ಸುಮಾರು 89 ರಿಂದ 93% ನಡುವೆ ಎಂದರ್ಥ. ಚುನಾವಣಾ ಸಮೀಕ್ಷೆಯಲ್ಲಿ “±3% ತಪ್ಪಿಗೆ ಅವಕಾಶ” ಎನ್ನುವಂತೆ. ನಾವು <b>bootstrap</b> ವಿಧಾನದಲ್ಲಿ (ಟೆಸ್ಟ್ ವಾಕ್ಯಗಳನ್ನು 2,000 ಬಾರಿ ಮರು-ಆರಿಸಿ ಲೆಕ್ಕ ಹಾಕಿ) ಇದನ್ನು ಕಂಡುಹಿಡಿದೆವು. ಎರಡು ಅಂತರಗಳು ಒಂದರ ಮೇಲೊಂದು ಬಂದರೆ ವ್ಯತ್ಯಾಸ ಅದೃಷ್ಟವಾಗಿರಬಹುದು.")}
{box("kn", "McNemar ಪರೀಕ್ಷೆ: ಇಬ್ಬರ ಉತ್ತರಪತ್ರಿಕೆ ಹೋಲಿಕೆ", f"ಇಬ್ಬರು ವಿದ್ಯಾರ್ಥಿಗಳು ಒಂದೇ ಪ್ರಶ್ನೆಪತ್ರಿಕೆ ಬರೆದರು. ಇಬ್ಬರಿಗೂ ಸರಿಯಾದ ಅಥವಾ ಇಬ್ಬರಿಗೂ ತಪ್ಪಾದ ಪ್ರಶ್ನೆಗಳನ್ನು ಬಿಟ್ಟುಬಿಡಿ. ಉಳಿದದ್ದರಲ್ಲಿ “ಒಬ್ಬನೇ ಸರಿ ಬರೆದ” ಪ್ರಶ್ನೆಗಳನ್ನು ಎಣಿಸಿ: ರಾಮ ಮಾತ್ರ ಸರಿ ಬರೆದದ್ದು {mc['ft_svm'][0]}, ಲಕ್ಷ್ಮಿ ಮಾತ್ರ ಸರಿ ಬರೆದದ್ದು {mc['ft_svm'][1]} (ಇದು ನಮ್ಮ DistilBERT vs SVM ಸಂಖ್ಯೆಗಳು). ಇಬ್ಬರೂ ಸಮಾನ ಬುದ್ಧಿವಂತರಾಗಿದ್ದರೆ ಈ ಎರಡು ಸಂಖ್ಯೆ ಹೆಚ್ಚುಕಡಿಮೆ ಸಮ ಇರುತ್ತಿತ್ತು. ಇಷ್ಟು ವ್ಯತ್ಯಾಸ ಆಕಸ್ಮಿಕ ಅಲ್ಲ ಎಂದು p-ಮೌಲ್ಯ ಹೇಳುತ್ತದೆ.")}

<h2>7.2 ಬೆಲೆ ಎಷ್ಟು? ಸಮಯ, ವೇಗ, ಗಾತ್ರ (RQ2)</h2>
{fig(f'<img src="{IMG}fig_accuracy_latency.png" alt="Accuracy vs latency">', "ಚಿತ್ರ 7.2: ನಿಖರತೆ vs ಪ್ರತಿ ವಾಕ್ಯಕ್ಕೆ ಬೇಕಾದ ಸಮಯ (ಬಲಕ್ಕೆ ಹೋದಷ್ಟೂ ನಿಧಾನ, ಪ್ರತಿ ಗೆರೆ 10 ಪಟ್ಟು). ಗುಳ್ಳೆಯ ಗಾತ್ರ = ಪ್ಯಾರಾಮೀಟರ್‌ಗಳ ಸಂಖ್ಯೆ.")}
<div class="grid4">
<div class="stat o"><b>≈ {N['latratio']}×</b><span>DistilBERT ({N['lat']['ft_distilbert_8000']} ms) vs SVM ({N['lat']['tfidf_svm']} ms)</span></div>
<div class="stat"><b>{N['train']['ft_distilbert_8000']}</b><span>DistilBERT ಫೈನ್-ಟ್ಯೂನಿಂಗ್ (8,000 ವಾಕ್ಯ, CPU)</span></div>
<div class="stat b"><b>{N['train']['tfidf_svm']}</b><span>SVM ತರಬೇತಿ (ಎಲ್ಲಾ {N['n_train']} ವಾಕ್ಯ)</span></div>
<div class="stat v"><b>{N['params']['ft_distilbert_8000']}</b><span>DistilBERT ಪ್ಯಾರಾಮೀಟರ್</span></div></div>
{box("kn", "ನಮ್ಮ ಊರಿನ ಉದಾಹರಣೆ: ಸ್ಕೂಟರ್ vs ಟ್ರಕ್", "ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್ ಬೆಂಗಳೂರು ಟ್ರಾಫಿಕ್‌ನಲ್ಲಿ ಸ್ಕೂಟರ್‌ನಂತೆ: ವೇಗ, ಚಿಕ್ಕದು, ಎಲ್ಲಿ ಬೇಕಾದರೂ ನುಗ್ಗುತ್ತದೆ. ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಟ್ರಕ್‌ನಂತೆ: ನಿಧಾನ ಮತ್ತು ದೊಡ್ಡದು, ಆದರೆ ಹೆಚ್ಚು ಭಾರ (ಹೆಚ್ಚು ಅರ್ಥ) ಹೊರುತ್ತದೆ. ಪ್ರತಿ ವಿಮರ್ಶೆಗೆ ಸುಮಾರು 40 ms ಅಂದರೆ ಸೆಕೆಂಡಿಗೆ ಸುಮಾರು 25 ವಿಮರ್ಶೆ: ಬಹುತೇಕ ಆಪ್‌ಗಳಿಗೆ ಅದು ಸಾಕು. ಆದರೆ ಸೆಕೆಂಡಿಗೆ ನೂರಾರು ವಿಮರ್ಶೆ ಬಂದರೆ ಸ್ಕೂಟರ್ (TF-IDF) ಬೇಕಾಗಬಹುದು.")}
{box("warn", "ಎಚ್ಚರ: ಸಮಯದ ಅಳತೆ", "ತರಬೇತಿ ಸಮಯ ಮಾಪನದ ಸಮಯದಲ್ಲಿ ಲ್ಯಾಪ್ಟಾಪ್ ಬೇರೆ ಕೆಲಸಕ್ಕೂ ಬಳಕೆಯಾಗಿತ್ತು, ಹಾಗಾಗಿ ಅವು “ಇಷ್ಟಕ್ಕಿಂತ ಹೆಚ್ಚಲ್ಲ” (upper bound) ಸಮಯಗಳು. ಪಟ್ಟು-ಪ್ರಮಾಣ (ಸ್ಕೂಟರ್ ≪ ಟ್ರಕ್) ನಂಬಬಹುದು; ನಿಖರ ಸೆಕೆಂಡುಗಳನ್ನಲ್ಲ.")}

<h2>7.3 ಎಷ್ಟು ಲೇಬಲ್ ಬೇಕು? ಕಲಿಕೆ ವಕ್ರರೇಖೆ (RQ3)</h2>
{fig(f'<img src="{IMG}fig_learning_curve.png" alt="Learning curves">', "ಚಿತ್ರ 7.3: ಲೇಬಲ್ ಸಂಖ್ಯೆ ಹೆಚ್ಚಿದಂತೆ ನಿಖರತೆ (ಹಸಿರು = BERT-small, ನೀಲಿ = TF-IDF). ಚುಕ್ಕಿ ಗೆರೆ = ಎಲ್ಲಾ ಡೇಟಾದಲ್ಲಿ ಕಲಿತ ಅತ್ಯುತ್ತಮ ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್.")}
<div class="keep"><table><thead><tr><th class="num">ಲೇಬಲ್ ಮಾಡಿದ ವಾಕ್ಯಗಳು</th><th class="num">BERT-small %</th><th class="num">TF-IDF %</th><th class="num">ವ್ಯತ್ಯಾಸ (ಅಂಕ)</th></tr></thead><tbody>{N['lc_rows']}</tbody></table></div>
<ul>
<li>ಕೇವಲ <b>500 ಲೇಬಲ್</b>ನಲ್ಲಿ BERT-small <b>{N['lc500_t']}%</b>, TF-IDF <b>{N['lc500_c']}%</b>: <b>+{N['lc500_gain']} ಅಂಕ</b>.</li>
<li><b>4,000 ಲೇಬಲ್‌ನಲ್ಲಿ</b> BERT-small <b>{N['lc4000_t']}%</b>, ಇದು ಎಲ್ಲಾ {N['n_train']} ವಾಕ್ಯಗಳಲ್ಲಿ ಕಲಿತ ಅತ್ಯುತ್ತಮ ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್‌ಗೆ ({a['tfidf_svm']}%) ಸಮ: ಅಂದರೆ ಸುಮಾರು <b>16 ಪಟ್ಟು ಕಡಿಮೆ ಡೇಟಾ</b>.</li></ul>
{box("kn", "ನಮ್ಮ ಊರಿನ ಉದಾಹರಣೆ: ಟ್ಯೂಷನ್", "ಈಗಾಗಲೇ ಬಹಳ ಓದಿರುವ ವಿದ್ಯಾರ್ಥಿ ಒಂದು ವಾರದ ಟ್ಯೂಷನ್‌ನಲ್ಲೇ ಪರೀಕ್ಷೆ ಪಾಸು ಮಾಡುತ್ತಾನೆ (ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್). ಏನೂ ಓದದವನಿಗೆ ವರ್ಷವಿಡೀ ಟ್ಯೂಷನ್ ಬೇಕು (ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್, ಶೂನ್ಯದಿಂದ ಕಲಿಯುತ್ತದೆ). <b>ಲೇಬಲ್ ಮಾಡಿದ ಡೇಟಾ ಕಡಿಮೆ ಇರುವ ಕನ್ನಡದಂತಹ ಭಾಷೆಗಳಿಗೆ ಇದು ಬಹಳ ಮುಖ್ಯ ಸುದ್ದಿ.</b>")}

<h2>7.4 ನಿರಾಕರಣೆ ಮತ್ತು ವೈದೃಶ್ಯ (RQ5)</h2>
<p>ಟೆಸ್ಟ್ ವಾಕ್ಯಗಳಲ್ಲಿ ನಿರಾಕರಣೆ ಪದಗಳಿರುವ (not, no, never, n’t…) <b>{N['neg_n']}</b> ವಾಕ್ಯಗಳು ಮತ್ತು ವೈದೃಶ್ಯ ಪದಗಳಿರುವ (but, although, though, yet, however) <b>{N['con_n']}</b> ವಾಕ್ಯಗಳನ್ನು ಬೇರೆ ಮಾಡಿ ಪರೀಕ್ಷಿಸಿದೆವು.</p>
<table><thead><tr><th>ಮಾಡೆಲ್</th><th class="num">ಎಲ್ಲಾ %</th><th class="num">ನಿರಾಕರಣೆ %</th><th class="num">ವೈದೃಶ್ಯ %</th><th class="num">ನಿರಾಕರಣೆಯಲ್ಲಿ ಇಳಿಕೆ</th><th class="num">ವೈದೃಶ್ಯದಲ್ಲಿ ಇಳಿಕೆ</th></tr></thead><tbody>{sub_rows}</tbody></table>
{fig(f'<img src="{IMG}fig_subsets.png" alt="Negation and contrast">', "ಚಿತ್ರ 7.4: ಎಲ್ಲಾ ವಾಕ್ಯಗಳು, ನಿರಾಕರಣೆ ವಾಕ್ಯಗಳು ಮತ್ತು ವೈದೃಶ್ಯ ವಾಕ್ಯಗಳಲ್ಲಿ ಪ್ರತಿ ಮಾಡೆಲ್‌ನ ನಿಖರತೆ.")}
<ul>
<li>ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್‌ಗಳು ನಿರಾಕರಣೆ ವಾಕ್ಯಗಳಲ್ಲಿ ಸುಮಾರು <b>8–8.5 ಅಂಕ</b> ಕಳೆದುಕೊಳ್ಳುತ್ತವೆ: “ಚೆನ್ನಾಗಿಲ್ಲ” ನಲ್ಲಿ “ಚೆನ್ನಾಗಿ” ಕಂಡಂತೆ.</li>
<li>ದೊಡ್ಡ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್‌ಗಳು ಕೇವಲ <b>2–4 ಅಂಕ</b>; zero-shot ಬಹುತೇಕ ಏನೂ ಕಳೆದುಕೊಳ್ಳುವುದಿಲ್ಲ.</li>
<li>ಆದರೆ <b>ಚಿಕ್ಕ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್‌ಗಳು (BERT-small, BERT-tiny) ಪದ ಎಣಿಕೆಯಂತೆಯೇ</b> ಸೋಲುತ್ತವೆ. ಅಂದರೆ ಲಾಭ ಕೇವಲ “ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್” ಹೆಸರಿನಿಂದ ಬರುವುದಿಲ್ಲ; ವಿನ್ಯಾಸ + ಗಾತ್ರ + ಪೂರ್ವ-ತರಬೇತಿ ಮೂರೂ ಬೇಕು.</li></ul>
{box("kn", "ಕನ್ನಡದ ದೃಷ್ಟಿಯಿಂದ", "ಕನ್ನಡದಲ್ಲಿ ನಿರಾಕರಣೆ ಪ್ರತ್ಯಯವಾಗಿ ಪದಕ್ಕೆ ಅಂಟಿಕೊಳ್ಳುತ್ತದೆ: “ಚೆನ್ನಾಗಿದೆ” → “ಚೆನ್ನಾಗಿಲ್ಲ”, “ಬಂದಿತು” → “ಬರಲಿಲ್ಲ”. ಆದ್ದರಿಂದ ಕನ್ನಡಕ್ಕೆ ಉತ್ತಮ ಉಪಪದ ಟೋಕನ್‌ಗಳಿರುವ, ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದ ಬಹುಭಾಷಾ ಮಾಡೆಲ್ ಇನ್ನಷ್ಟು ಮುಖ್ಯ (ಅಧ್ಯಾಯ 5).")}

<h2>7.5 ಎಲ್ಲಾ ಮಾಡೆಲ್‌ಗಳೂ ಸೋತ ವಾಕ್ಯಗಳು</h2>
<p>{N['n_test']} ವಾಕ್ಯಗಳಲ್ಲಿ <b>{N['allwrong']}</b> ವಾಕ್ಯಗಳನ್ನು ಹತ್ತರಲ್ಲಿ ಯಾವ ಮಾಡೆಲ್ಲೂ ಸರಿಯಾಗಿ ಹೇಳಲಿಲ್ಲ. ಉದಾಹರಣೆ: <span class="en"><b>“the lower your expectations, the more you’ll enjoy it”</b></span> (ಲೇಬಲ್: ಋಣಾತ್ಮಕ). ಇದು ವ್ಯಂಗ್ಯ: “ಅಪೇಕ್ಷೆ ಕಡಿಮೆ ಇದ್ದಷ್ಟೂ ಸಿನಿಮಾ ಹೆಚ್ಚು ಇಷ್ಟವಾಗುತ್ತದೆ” ಎಂದರೆ ಸಿನಿಮಾ ಚೆನ್ನಾಗಿಲ್ಲ ಎಂದು. ವ್ಯಂಗ್ಯ ಮತ್ತು ಸಂದರ್ಭ ಗೊತ್ತಿರಬೇಕಾದ ವಾಕ್ಯಗಳು ಇನ್ನೂ ದೊಡ್ಡ ಮಾಡೆಲ್‌ಗಳಿಗೂ ಕಷ್ಟ.</p>

<h2>7.6 ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಎಲ್ಲಿ ನೋಡಿತು? (ಅಟೆನ್ಷನ್ ಚಿತ್ರ)</h2>
{fig(f'<img src="{IMG}attention_heatmap.png" alt="Attention heatmap" style="max-height:105mm">', "ಚಿತ್ರ 7.5: ನಿಜವಾದ DistilBERT ಅಟೆನ್ಷನ್ ಚಿತ್ರ: ಸಾಲು = ನೋಡುವ ಪದ, ಕಂಬಸಾಲು = ನೋಡಲ್ಪಡುವ ಪದ; ಗಾಢ ಬಣ್ಣ = ಹೆಚ್ಚು ಗಮನ.")}
{box("warn", "ಎಚ್ಚರ", "ಅಟೆನ್ಷನ್ ಚಿತ್ರ ಮಾಹಿತಿ ಹೇಗೆ ಹರಿಯುತ್ತದೆ ಎಂದು ತೋರಿಸುವ <b>ಚಿತ್ರಣ</b>, ಮಾಡೆಲ್ ಹೇಗೆ “ಯೋಚಿಸುತ್ತದೆ” ಎಂಬುದರ ಪೂರ್ತಿ <b>ಸಾಕ್ಷ್ಯವಲ್ಲ</b> (Jain ಮತ್ತು Wallace, 2019). ವೈವಾದಲ್ಲಿ ಹೀಗೇ ಹೇಳಿ.")}

<h2>7.7 ನಮ್ಮ ಅಧ್ಯಯನದ ಮಿತಿಗಳು</h2>
<ul>
<li>ಪ್ರತಿ ಮಾಡೆಲ್ ಅನ್ನು <b>ಒಂದು ಬಾರಿ ಮಾತ್ರ</b> (ಒಂದು random seed) ತರಬೇತಿ ಮಾಡಿದ್ದೇವೆ: 1 ಅಂಕಕ್ಕಿಂತ ಚಿಕ್ಕ ವ್ಯತ್ಯಾಸ ಅದೃಷ್ಟವಾಗಿರಬಹುದು.</li>
<li>ಒಂದೇ ಡೇಟಾಸೆಟ್, ಒಂದೇ ಕ್ಷೇತ್ರ (ಸಿನಿಮಾ ವಿಮರ್ಶೆ), ಒಂದೇ ಭಾಷೆ (ಇಂಗ್ಲಿಷ್).</li>
<li>“ನಿರಾಕರಣೆ/ವೈದೃಶ್ಯ” ವಾಕ್ಯಗಳನ್ನು ಸರಳ ಪದ-ಪಟ್ಟಿಯಿಂದ ಗುರುತಿಸಿದ್ದೇವೆ.</li>
<li>ಫೈನ್-ಟ್ಯೂನಿಂಗ್‌ಗೆ ಕೇವಲ 8,000 ವಾಕ್ಯ (CPU ಸಮಯದ ಕಾರಣ); ಹೈಪರ್‌ಪ್ಯಾರಾಮೀಟರ್ ಹುಡುಕಾಟ ಮಾಡಿಲ್ಲ; ಸಮಯ ಮಾಪನ ಒಂದೇ ಲ್ಯಾಪ್ಟಾಪ್‌ನಲ್ಲಿ.</li>
<li>ಟೆಸ್ಟ್ ಸೆಟ್ ಚಿಕ್ಕದು ({N['n_test']} ವಾಕ್ಯ): ±2.5 ಅಂಕ ಅಂತರ; 1 ಅಂಕ ≈ 9 ವಾಕ್ಯ.</li></ul>
"""


def ch8(N):
    a = N["acc"]
    return f"""
<h1 class="ch" id="c8">ಅಧ್ಯಾಯ 8 · ಯಾವ ಮಾಡೆಲ್ ಯಾವಾಗ?<small>WHICH MODEL WHEN</small></h1>
<p>ರಿವ್ಯೂ ರೇಡಾರ್‌ನ ಮ್ಯಾನೇಜರ್ ಕೇಳುತ್ತಾರೆ: “ಹಾಗಾದರೆ ನಾವು ಯಾವುದನ್ನು ಬಳಸಬೇಕು?” ಉತ್ತರ ನಿಮ್ಮ ಪರಿಸ್ಥಿತಿಯನ್ನು ಅವಲಂಬಿಸಿದೆ. ನಮ್ಮ ಅಳತೆಗಳ ಆಧಾರದ ಮೇಲೆ ಒಂದು ಕ್ರಮ (ಮೇಲಿನಿಂದ ಕೆಳಗೆ ಯೋಚಿಸಿ):</p>
<table><thead><tr><th style="width:5%">#</th><th style="width:30%">ನಿಮ್ಮ ಪರಿಸ್ಥಿತಿ</th><th>ಶಿಫಾರಸು</th><th style="width:24%">ನಮ್ಮ ಸಂಖ್ಯೆ</th></tr></thead><tbody>
<tr class="hl"><td>1</td><td>ನಿಮ್ಮದೇ ಕ್ಷೇತ್ರಕ್ಕೆ (ಉದಾ: ಸಿನಿಮಾ ವಿಮರ್ಶೆ) ಈಗಾಗಲೇ ತರಬೇತಿ ಪಡೆದ ಮಾಡೆಲ್ ಲಭ್ಯವಿದೆ</td><td><b>ಮೊದಲು ಅದನ್ನೇ ಬಳಸಿ.</b> ತರಬೇತಿಯ ಅಗತ್ಯವಿಲ್ಲ.</td><td>DistilBERT-SST2: {a['pt_distilbert_sst2']}%</td></tr>
<tr><td>2</td><td>ಕೆಲವು ಸಾವಿರ ಲೇಬಲ್ (ಉದಾ: 8,000) ಮತ್ತು ಲ್ಯಾಪ್ಟಾಪ್ ಇದೆ</td><td>DistilBERT ಯಂತಹ ಚಿಕ್ಕ (distilled) ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಅನ್ನು ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿ. GPU ಬೇಕಿಲ್ಲ.</td><td>{a['ft_distilbert_8000']}%, ತರಬೇತಿ ≈ {N['train']['ft_distilbert_8000']}</td></tr>
<tr><td>3</td><td>ಲೇಬಲ್ ಬಹುತೇಕ ಇಲ್ಲ (300–500)</td><td>Zero-shot ಮಾಡೆಲ್ ಅಥವಾ ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದ ಚಿಕ್ಕ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್. ಪದ ಎಣಿಕೆ ಇಲ್ಲಿ ಬಹಳ ಕಷ್ಟ.</td><td>Zero-shot {a['zs_nli_distilroberta']}%; 500 ಲೇಬಲ್‌ಗೆ BERT-small {N['lc500_t']}% vs TF-IDF {N['lc500_c']}%</td></tr>
<tr><td>4</td><td>ಸೆಕೆಂಡಿಗೆ ನೂರಾರು ವಿಮರ್ಶೆ; ಸ್ಮಾರ್ಟ್‌ವಾಚ್/ಫೋನ್ ಒಳಗೆ ಓಡಬೇಕು; ವೇಗ ಬಹಳ ಮುಖ್ಯ</td><td><b>ಪದ ಎಣಿಕೆ (TF-IDF)</b>: ಸ್ಕೂಟರ್ ಬಳಸಿ.</td><td>{a['tfidf_svm']}%, {N['lat']['tfidf_svm']} ms / ವಾಕ್ಯ</td></tr>
<tr><td>5</td><td>ಕನ್ನಡ (ಅಥವಾ ಇನ್ನಾವುದೇ ಇಂಗ್ಲಿಷ್ ಅಲ್ಲದ) ಪಠ್ಯ</td><td>ಅದೇ ಕ್ರಮ, ಆದರೆ ಬಹುಭಾಷಾ ಮಾಡೆಲ್ (XLM-R, MuRIL, IndicBERT, mBERT) ಬಳಸಿ ಮತ್ತು ಕನ್ನಡ ಲೇಬಲ್ ಡೇಟಾ ಸಂಗ್ರಹಿಸಿ.</td><td>ಇನ್ನೂ ಪ್ರಯೋಗಿಸಿಲ್ಲ (ಭವಿಷ್ಯದ ಕೆಲಸ)</td></tr></tbody></table>
<h2>ಕರ್ನಾಟಕದ ಕೆಲವು ಉದಾಹರಣೆ ಸನ್ನಿವೇಶಗಳು</h2>
<table><thead><tr><th>ಸನ್ನಿವೇಶ (ಕಾಲ್ಪನಿಕ)</th><th>ಯಾವ ಕ್ರಮ?</th><th>ಕಾರಣ</th></tr></thead><tbody>
<tr><td>ನಮ್ಮ ಮೆಟ್ರೋ ಪ್ರಯಾಣಿಕರ ದೂರುಗಳನ್ನು ವಿಭಾಗಕ್ಕೆ ವಿಂಗಡಿಸುವುದು</td><td>ಕ್ರಮ 2 ಅಥವಾ 5: ಕೆಲವು ಸಾವಿರ ಲೇಬಲ್ ಮಾಡಿ ಬಹುಭಾಷಾ ಮಾಡೆಲ್ ಫೈನ್-ಟ್ಯೂನ್</td><td>ಕನ್ನಡ ಮತ್ತು ಇಂಗ್ಲಿಷ್ ಮಿಶ್ರ ಪಠ್ಯ; ನಿಖರತೆ ಮುಖ್ಯ; ವೇಗ ಅಷ್ಟು ಕಷ್ಟವಲ್ಲ</td></tr>
<tr><td>ದರ್ಶಿನಿ ರೆಸ್ಟೋರೆಂಟ್ ಸರಣಿಯ ಸಾವಿರಾರು ರಿವ್ಯೂ ಪರಿಶೀಲನೆ</td><td>ಕ್ರಮ 1: ಹೋಟೆಲ್ ರಿವ್ಯೂಗೆ ಈಗಾಗಲೇ ತರಬೇತಿ ಪಡೆದ ಮಾಡೆಲ್ ಇದ್ದರೆ ಬಳಸಿ</td><td>ಲೇಬಲ್ ಮಾಡುವ ಶ್ರಮ ಉಳಿಯುತ್ತದೆ</td></tr>
<tr><td>ಕೆಎಸ್ಆರ್ಟಿಸಿ ಬಸ್ ಬುಕಿಂಗ್ ಚಾಟ್‌ಬಾಟ್ (ಲಕ್ಷಾಂತರ ಸಂದೇಶ)</td><td>ಕ್ರಮ 4 ಮತ್ತು 2: ಮೊದಲ ಹಂತಕ್ಕೆ ವೇಗವಾದ ಚಿಕ್ಕ ಮಾಡೆಲ್, ಕಠಿಣ ಪ್ರಶ್ನೆಗಳಿಗೆ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್</td><td>ಸಂದೇಶ ಸಂಖ್ಯೆ ದೊಡ್ಡದು, ವೆಚ್ಚ ಮುಖ್ಯ</td></tr>
<tr><td>ಹೊಸ ಚಿತ್ರದ ಮೊದಲ ದಿನದ ವಿಮರ್ಶೆ ತ್ವರಿತ ವಿಶ್ಲೇಷಣೆ, ಲೇಬಲ್ ಇಲ್ಲ</td><td>ಕ್ರಮ 3: Zero-shot</td><td>ಇಂದೇ ಬೇಕು, ತರಬೇತಿಗೆ ಸಮಯವಿಲ್ಲ</td></tr></tbody></table>
{box("exam", "ವೈವಾಗೆ: ನಿಮ್ಮ ಶಿಫಾರಸು ಒಂದೇ ವಾಕ್ಯದಲ್ಲಿ", "<b>Reuse first, fine-tune a distilled model second, use TF-IDF only when speed and size dominate.</b> (ಮೊದಲು ಈಗಾಗಲೇ ಇರುವುದನ್ನು ಬಳಸಿ, ಎರಡನೆಯದಾಗಿ ಚಿಕ್ಕ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿ, ವೇಗ/ಗಾತ್ರ ಮುಖ್ಯವಾದಾಗ ಮಾತ್ರ ಪದ ಎಣಿಕೆ.)")}
"""


QA = [
    ("What is a Transformer?", "ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಒಂದು ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್ ವಿನ್ಯಾಸ (2017). ವಾಕ್ಯದ ಎಲ್ಲಾ ಪದಗಳನ್ನು ಒಮ್ಮೆಲೇ ಸಂಸ್ಕರಿಸುತ್ತದೆ ಮತ್ತು ಪ್ರತಿ ಪದವೂ ಉಳಿದ ಎಲ್ಲಾ ಪದಗಳನ್ನು ನೇರವಾಗಿ ನೋಡುತ್ತದೆ (self-attention). ಎಲ್‌ಎಸ್‌ಟಿಎಂ ನಂತೆ ಒಂದೊಂದೇ ಪದ ಓದುವುದಿಲ್ಲ.",
     "A neural network architecture that processes all words at once and lets every word attend directly to every other word through self-attention, instead of reading word by word like an LSTM."),
    ("What does self-attention compute?", "ಪ್ರತಿ ಪದಕ್ಕೆ ಉಳಿದ ಪದಗಳೊಂದಿಗೆ ಹೊಂದಾಣಿಕೆ ಅಂಕ (Q·Kᵀ) ಲೆಕ್ಕ ಹಾಕಿ, softmax ನಿಂದ ಶೇಕಡಾವಾರು ಮಾಡಿ, ಆ ತೂಕದಲ್ಲಿ ಎಲ್ಲಾ ಪದಗಳ ಮಾಹಿತಿಯನ್ನು (V) ಬೆರೆಸುತ್ತದೆ. ಸೂತ್ರ: softmax(QKᵀ/√d_k)·V. ಮದುವೆ ಮನೆಯಲ್ಲಿ ನಿಮ್ಮ ಹೆಸರು ಕೇಳಿದ ಕಡೆ ಕಿವಿ ತಿರುಗುವಂತೆ.",
     "For each word it scores its relevance to all other words, turns the scores into weights with softmax, and returns the weighted mix of their value vectors: softmax(QKᵀ/√d_k)·V."),
    ("Why do we divide by √d_k?", "ವೆಕ್ಟರ್ ದೊಡ್ಡದಾದಂತೆ ಗುಣಲಬ್ಧ (dot product) ಸಂಖ್ಯೆಗಳು ದೊಡ್ಡದಾಗುತ್ತವೆ. ಭಾಗಿಸದಿದ್ದರೆ softmax ನಲ್ಲಿ ಒಂದೇ ಪದ ~100% ಪಡೆದು ಉಳಿದವು ~0% ಆಗುತ್ತವೆ, ಗ್ರೇಡಿಯಂಟ್ ಬಹಳ ಚಿಕ್ಕದಾಗಿ ಕಲಿಕೆ ನಿಲ್ಲುತ್ತದೆ.",
     "Dot products grow with dimension; without scaling, softmax saturates (one word gets ~100%) and gradients vanish. Dividing by √d_k keeps scores in a useful range."),
    ("Why do we need positional encoding?", "ಅಟೆನ್ಷನ್ ವಾಕ್ಯವನ್ನು ಪದಗಳ ಒಂದು ಗುಂಪಿನಂತೆ ನೋಡುತ್ತದೆ: ಪದಗಳನ್ನು ಅದಲು ಬದಲು ಮಾಡಿದರೂ ಅಂಕ ಅದೇ. ಆದರೆ “ರಾಮ ರಾವಣನನ್ನು ಕೊಂದ” ಮತ್ತು “ರಾವಣ ರಾಮನನ್ನು ಕೊಂದ” ಬೇರೆ ಅರ್ಥ. ಹಾಗಾಗಿ ಪ್ರತಿ ಪದಕ್ಕೆ ಸ್ಥಾನ ಸಂಖ್ಯೆ ಸೇರಿಸುತ್ತೇವೆ.",
     "Attention treats a sentence as a set; adding position vectors gives the model word order."),
    ("What are pre-training and fine-tuning?", "ಪೂರ್ವ-ತರಬೇತಿ: ಬೃಹತ್ ಪಠ್ಯದಲ್ಲಿ ಖಾಲಿ ಜಾಗ ಊಹಿಸುವ ಮೂಲಕ ಭಾಷೆ ಕಲಿಯುವುದು (ದುಬಾರಿ, ಬೇರೆಯವರು ಮಾಡಿದ್ದು). ಫೈನ್-ಟ್ಯೂನಿಂಗ್: ನಮ್ಮ ಚಿಕ್ಕ ಲೇಬಲ್ ಡೇಟಾದಲ್ಲಿ ಆ ಮಾಡೆಲ್‌ಗೆ ಚಿಕ್ಕ ತರಬೇತಿ ಕೊಡುವುದು (ಅಗ್ಗ, ನಾವು ಮಾಡಿದ್ದು). SSLC-PUC ಓದಿದ ಮೇಲೆ ಕಂಪನಿ ತರಬೇತಿ ಕೊಟ್ಟಂತೆ.",
     "Pre-training learns general language from huge unlabelled text (expensive, done by others). Fine-tuning adapts it with a small labelled set (cheap, what we did)."),
    ("Why is the validation set your test set?", "GLUE ನ ನಿಜವಾದ ಟೆಸ್ಟ್ ಲೇಬಲ್‌ಗಳು ಮುಚ್ಚಿಡಲ್ಪಟ್ಟಿವೆ; ಹಾಗಾಗಿ ಅದರ validation set ಅನ್ನು ಟೆಸ್ಟ್ ಆಗಿ ಬಳಸುವುದು ಸಾಮಾನ್ಯ ಪದ್ಧತಿ. ನಾವು ತರಬೇತಿ ಡೇಟಾದಿಂದ ನಮ್ಮದೇ dev set ಮಾಡಿ ಎಪಾಕ್ ಆರಿಸಿದೆವು; ಟೆಸ್ಟ್ ಅನ್ನು ಪ್ರತಿ ಮಾಡೆಲ್‌ಗೆ ಒಮ್ಮೆ ಮಾತ್ರ ಬಳಸಿದೆವು.",
     "GLUE hides its test labels, so using the official validation set as the held-out test set is common practice. We chose epochs on our own dev set and used the test set once per model."),
    ("Why did you fine-tune on only 8,000 sentences?", "ಸಮಯ ಕಾರಣ. ಈ ಲ್ಯಾಪ್ಟಾಪ್‌ನಲ್ಲಿ DistilBERT ಗೆ 8,000 ವಾಕ್ಯಕ್ಕೇ ಸುಮಾರು 2.6 ಗಂಟೆ ಆಯಿತು; ಎಲ್ಲಾ 65 ಸಾವಿರಕ್ಕೆ ಸುಮಾರು 20 ಗಂಟೆಗಿಂತ ಹೆಚ್ಚು (ಅಂದಾಜು). ಜೊತೆಗೆ “ಕೆಲವು ಸಾವಿರ ಲೇಬಲ್” ಎಂಬುದೇ ನಮ್ಮ ನಿಜ ಜೀವನದ ಸನ್ನಿವೇಶ.",
     "Time: DistilBERT already needed about 2.6 hours for 8,000 sentences on this CPU. And a few thousand labels is the realistic scenario we wanted to study."),
    ("What is your main result?", "ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿದ DistilBERT 88.8% vs ಅತ್ಯುತ್ತಮ ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್ 82.9% (ಅದು ಎಲ್ಲಾ 65,349 ವಾಕ್ಯಗಳಲ್ಲಿ ಕಲಿತದ್ದು). ಇದು ಸಂಖ್ಯಾಶಾಸ್ತ್ರೀಯವಾಗಿ ನಿಜ (p < 0.001). ಬೆಲೆ: ಸುಮಾರು 700 ಪಟ್ಟು ನಿಧಾನ ಉತ್ತರ ಮತ್ತು ಹೆಚ್ಚು ಗಾತ್ರ. Ready-made DistilBERT 91.1%.",
     "DistilBERT fine-tuned on only 8,000 sentences reaches 88.8% versus 82.9% for the best classical model trained on all 65,349 (significant, p < 0.001), at roughly 700 times the inference time. The ready-made in-domain model reaches 91.1%."),
    ("Why is BERT-tiny worse than the classical models?", "ಅದಕ್ಕೆ ಕೇವಲ 2 ಪದರ ಮತ್ತು 4.4 ಮಿಲಿಯನ್ ಪ್ಯಾರಾಮೀಟರ್. ಪೂರ್ವ-ತರಬೇತಿ ಇದ್ದರೂ ಸಾಮರ್ಥ್ಯ ಸಾಲದು: 76.6%. “ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್” ಎಂಬ ಹೆಸರಷ್ಟೇ ಸಾಲದು; ಗಾತ್ರ ಮುಖ್ಯ.",
     "It has only 2 layers and 4.4 million parameters; pre-training cannot compensate for too little capacity. A Transformer is not automatically better; size matters."),
    ("Why does the BiLSTM not beat TF-IDF? Why is its dev accuracy higher than test?", "ಪೂರ್ವ-ತರಬೇತಿ ಇಲ್ಲದೆ ಅದು ಭಾಷೆಯನ್ನು ಮೊದಲಿನಿಂದ ಕಲಿಯಬೇಕು: 82.8% vs SVM 82.9%, ವ್ಯತ್ಯಾಸ ಗಣನೀಯವಲ್ಲ (p = 1.0). Dev ನಲ್ಲಿ ~91% ಬಂದದ್ದು ಏಕೆಂದರೆ dev ತರಬೇತಿ ಡೇಟಾದ ಚಿಕ್ಕ ಪದಗುಚ್ಛಗಳು, ಟೆಸ್ಟ್‌ನಲ್ಲಿ ಪೂರ್ತಿ ವಾಕ್ಯಗಳು.",
     "Without pre-training it learns language from scratch: 82.8% vs 82.9%, not significant (p = 1.0). Its dev accuracy is higher because dev contains short training-distribution phrases, while test has complete sentences."),
    ("How many labels does a Transformer need?", "ಬಹಳ ಕಡಿಮೆ. 500 ಲೇಬಲ್‌ನಲ್ಲಿ BERT-small 75.9% vs TF-IDF 61.8% (+14.1 ಅಂಕ). 4,000 ಲೇಬಲ್‌ನಲ್ಲಿ 82.7%, ಅಂದರೆ ಎಲ್ಲಾ ಡೇಟಾದಲ್ಲಿ ಕಲಿತ ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್‌ಗೆ ಸಮ: ಸುಮಾರು 16 ಪಟ್ಟು ಕಡಿಮೆ ಡೇಟಾ.",
     "Very few. With 500 labels BERT-small gets 75.9% versus 61.8% for TF-IDF (+14.1 points); with 4,000 it matches the classical model trained on 16 times more data."),
    ("What is zero-shot classification and how does it work here?", "ಲೇಬಲ್ ಮಾಡಿದ ಭಾವನೆ ಡೇಟಾ ಇಲ್ಲದೆ ವರ್ಗೀಕರಣ. ಒಂದು NLI ಮಾಡೆಲ್ “ಒಂದು ವಾಕ್ಯ ಇನ್ನೊಂದು ಹೇಳಿಕೆಯನ್ನು ಸೂಚಿಸುತ್ತದೆಯೇ?” ಎಂದು ಹೇಳುತ್ತದೆ. ನಾವು “The sentiment of this movie review is positive/negative.” ಹೇಳಿಕೆ ಮಾಡಿ ಯಾವುದು ಹೆಚ್ಚು ಸೂಚಿತ ಎಂದು ಆರಿಸಿದೆವು: 81.8%.",
     "An NLI model judges whether the sentence entails a hypothesis such as “The sentiment of this movie review is positive”; the label with the highest entailment probability wins. No sentiment labels are used (81.8%)."),
    ("What happened on negation and contrast sentences?", "ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್‌ಗಳು ನಿರಾಕರಣೆ ವಾಕ್ಯಗಳಲ್ಲಿ ~8–8.5 ಅಂಕ ಕಳೆದುಕೊಂಡವು, ದೊಡ್ಡ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್‌ಗಳು ಕೇವಲ 2–4, zero-shot ~0.2. ಚಿಕ್ಕ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್‌ಗಳು ಪದ ಎಣಿಕೆಯಂತೆಯೇ ಸೋತವು.",
     "Bag-of-words models lost about 8 to 8.5 points on negation sentences, large Transformers only 2 to 4, zero-shot almost nothing; small Transformers behaved like the classical models."),
    ("Is the attention heat-map an explanation?", "ಪೂರ್ತಿ ಅಲ್ಲ. ಇದು ಮಾಹಿತಿ ಹರಿವಿನ ಚಿತ್ರಣ. Jain & Wallace (2019) ಪ್ರಕಾರ ಅಟೆನ್ಷನ್ ತೂಕಗಳು ಮಾಡೆಲ್‌ನ ನಿರ್ಧಾರದ ನಿಷ್ಠಾವಂತ ವಿವರಣೆ ಎಂದೇನಿಲ್ಲ.",
     "Not completely. It illustrates information flow, but attention weights are not necessarily a faithful explanation of the prediction (Jain and Wallace, 2019)."),
    ("What are a confidence interval and McNemar’s test?", "ವಿಶ್ವಾಸ ಅಂತರ: ಟೆಸ್ಟ್ ಸೆಟ್ ಅನ್ನು 2,000 ಬಾರಿ ಮರು-ಆರಿಸಿ ನಿಖರತೆ ಲೆಕ್ಕ ಹಾಕಿ 2.5 ಮತ್ತು 97.5 ಶತಮಾನವನ್ನು ತೆಗೆದುಕೊಳ್ಳುವುದು (ಸುಮಾರು ±2.5 ಅಂಕ). McNemar: ಒಂದೇ ವಾಕ್ಯಗಳಲ್ಲಿ “ಒಬ್ಬನೇ ಸರಿ” ಎಂಬ ಸಂಖ್ಯೆಗಳನ್ನು ಹೋಲಿಸಿ ವ್ಯತ್ಯಾಸ ಅದೃಷ್ಟವೋ ಎಂದು ನೋಡುವುದು.",
     "A bootstrap confidence interval resamples the test set 2,000 times and takes the 2.5th and 97.5th percentiles. McNemar’s test compares the sentences where exactly one of two models is right to see if the difference could be chance."),
    ("What are the limitations?", "ಒಂದು random seed; ಒಂದೇ ಡೇಟಾಸೆಟ್/ಭಾಷೆ; ಪದ-ಪಟ್ಟಿಯಿಂದ ಗುರುತಿಸಿದ ವಾಕ್ಯ ಗುಂಪುಗಳು; ಹೈಪರ್‌ಪ್ಯಾರಾಮೀಟರ್ ಹುಡುಕಾಟ ಇಲ್ಲ; 8,000 ಲೇಬಲ್ ಮಾತ್ರ; ಒಂದೇ (ಸ್ವಲ್ಪ ಬ್ಯುಸಿ) ಲ್ಯಾಪ್ಟಾಪ್‌ನ ಸಮಯ ಮಾಪನ; ಚಿಕ್ಕ ಟೆಸ್ಟ್ ಸೆಟ್ (872).",
     "One seed per configuration, one dataset and language, keyword-defined subsets, no hyper-parameter search, only 8,000 labels, timings from one partly busy laptop, and a small test set of 872 sentences."),
    ("How does your work relate to Lithuanian (or Kannada) NLP?", "ಲಿಥುವೇನಿಯನ್ ಸಂಶೋಧನೆಯ ಪಾಠವೂ ನಮ್ಮ ಫಲಿತಾಂಶವೂ ಒಂದೇ: ಪೂರ್ವ-ತರಬೇತಿ ಮುಖ್ಯ; ಅದಿಲ್ಲದ ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್ ಸರಳ ವಿಧಾನಗಳನ್ನು ಮೀರಿಸುವುದಿಲ್ಲ. ಕನ್ನಡ ಅಥವಾ ಲಿಥುವೇನಿಯನ್‌ಗೆ ಇದೇ ವಿಧಾನವನ್ನು LitLat BERT/XLM-R/MuRIL ಬಳಸಿ ವಿಸ್ತರಿಸುವುದು ನಮ್ಮ ಮುಂದಿನ ಹೆಜ್ಜೆ.",
     "The Lithuanian studies and our results agree: pre-training is the key factor. The same procedure can be repeated for Kannada or Lithuanian with multilingual models such as XLM-R, MuRIL or LitLat BERT; that is our future work."),
    ("Which model would you deploy?", "ಪರಿಸ್ಥಿತಿ ಅವಲಂಬಿತ: ಇದೇ ಕ್ಷೇತ್ರದ ಮಾಡೆಲ್ ಇದ್ದರೆ ಅದನ್ನು ಬಳಸಿ; ಇಲ್ಲದಿದ್ದರೆ ಕೆಲವು ಸಾವಿರ ಲೇಬಲ್‌ನಲ್ಲಿ DistilBERT ಫೈನ್-ಟ್ಯೂನ್; ಲೇಬಲ್ ಇಲ್ಲದಿದ್ದರೆ zero-shot; ಅತಿ ವೇಗ/ಚಿಕ್ಕ ಗಾತ್ರ ಬೇಕಾದರೆ TF-IDF.",
     "It depends: reuse an in-domain public model if one exists; otherwise fine-tune DistilBERT on a few thousand labels; with no labels use zero-shot; use TF-IDF only for extreme speed or size limits."),
]


def ch9(N):
    items = ""
    for i, (q, kn, en) in enumerate(QA, 1):
        items += f'<div class="qa"><div class="q">{i}. <span class="en">{q}</span></div><p>{kn}</p><div class="e"><b>English answer (30 sec):</b> {en}</div></div>'
    return f"""
<h1 class="ch" id="c9">ಅಧ್ಯಾಯ 9 · ವೈವಾ ಪ್ರಶ್ನೆ–ಉತ್ತರ<small>LIKELY EXAMINER QUESTIONS</small></h1>
<p>ಪ್ರತಿ ಪ್ರಶ್ನೆಗೆ ಮೊದಲು <b>ಕನ್ನಡದಲ್ಲಿ</b> ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ, ನಂತರ ನೇರಳೆ ಪೆಟ್ಟಿಗೆಯಲ್ಲಿರುವ <b>ಇಂಗ್ಲಿಷ್ ಉತ್ತರ</b> ಜೋರಾಗಿ ಹೇಳಿ ಅಭ್ಯಾಸ ಮಾಡಿ. ಪರೀಕ್ಷಕರಿಗೆ ನೀವು ಹೇಳುವುದು ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ.</p>
{items}
"""


GLOSS = [
    ("Natural Language Processing (NLP)", "ಸ್ವಾಭಾವಿಕ ಭಾಷಾ ಸಂಸ್ಕರಣೆ", "ಮಾನವ ಭಾಷೆಯನ್ನು ಕಂಪ್ಯೂಟರ್ ಓದಿ ಅರ್ಥ ಮಾಡಿಕೊಳ್ಳುವ ವಿಜ್ಞಾನ"),
    ("Transformer", "ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್", "ಎಲ್ಲಾ ಪದಗಳನ್ನು ಒಮ್ಮೆಲೇ ನೋಡಿ ಅಟೆನ್ಷನ್‌ನಿಂದ ಕಲಿಯುವ ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್"),
    ("Sentiment analysis", "ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ", "ವಾಕ್ಯ ಧನಾತ್ಮಕವೋ ಋಣಾತ್ಮಕವೋ ಎಂದು ಹೇಳುವುದು"),
    ("Token", "ಟೋಕನ್ (ಪದದ ತುಣುಕು)", "ಮಾಡೆಲ್ ಒಂದು ಘಟಕವಾಗಿ ನೋಡುವ ಪದ ಅಥವಾ ಪದದ ಭಾಗ"),
    ("Embedding", "ಎಂಬೆಡ್ಡಿಂಗ್", "ಪ್ರತಿ ಟೋಕನ್‌ಗೆ ನೀಡಿದ ಸಂಖ್ಯೆಗಳ ಪಟ್ಟಿ: ನಕ್ಷೆಯಲ್ಲಿನ ವಿಳಾಸದಂತೆ"),
    ("Positional encoding", "ಸ್ಥಾನ ಸಂಕೇತ", "ಪದ ಎಷ್ಟನೇ ಸ್ಥಾನದಲ್ಲಿದೆ ಎಂಬ ಮಾಹಿತಿ"),
    ("Self-attention", "ಸ್ವ-ಗಮನ (ಅಟೆನ್ಷನ್)", "ಪ್ರತಿ ಪದ ಉಳಿದ ಪದಗಳಿಗೆ ತೂಕ ಕೊಟ್ಟು ಮಾಹಿತಿ ಸಂಗ್ರಹಿಸುವುದು"),
    ("Query, Key, Value", "ಪ್ರಶ್ನೆ, ಕೀ, ಮೌಲ್ಯ", "ಲೈಬಲ್ ಹುಡುಕಾಟ: ಹುಡುಕುವ ವಿಷಯ, ಪುಸ್ತಕದ ಲೇಬಲ್, ಪುಸ್ತಕದ ಒಳಗಿನ ವಿಷಯ"),
    ("Softmax", "ಸಾಫ್ಟ್‌ಮ್ಯಾಕ್ಸ್", "ಅಂಕಗಳನ್ನು ಒಟ್ಟು 100% ಆಗುವ ಶೇಕಡಾವಾರು ಮಾಡುವುದು"),
    ("Multi-head attention", "ಬಹು-ಹೆಡ್ ಅಟೆನ್ಷನ್", "ಹಲವು ಕೋನಗಳಿಂದ ಒಂದೇ ಸಮಯ ನೋಡುವ ಹಲವು ಅಟೆನ್ಷನ್‌ಗಳು"),
    ("Residual connection", "ಶಾರ್ಟ್‌ಕಟ್ ಸಂಪರ್ಕ", "ಹಿಂದಿನ ಮಾಹಿತಿಯನ್ನು ಮತ್ತೆ ಸೇರಿಸುವ ದಾರಿ"),
    ("Layer normalisation", "ಲೇಯರ್ ಸಮತೋಲನ", "ಸಂಖ್ಯೆಗಳನ್ನು ಸ್ಥಿರ ಮಟ್ಟಕ್ಕೆ ತರುವುದು"),
    ("Feed-forward network", "ಫೀಡ್-ಫಾರ್ವರ್ಡ್ ನೆಟ್‌ವರ್ಕ್", "ಪ್ರತಿ ಪದಕ್ಕೆ ಪ್ರತ್ಯೇಕವಾಗಿ ಆಲೋಚಿಸುವ ಚಿಕ್ಕ ನೆಟ್‌ವರ್ಕ್"),
    ("[CLS] token", "[CLS] ಟೋಕನ್", "ವಾಕ್ಯದ ಸಾರಾಂಶ ಹಿಡಿಯುವ ವಿಶೇಷ ಟೋಕನ್"),
    ("BERT", "ಬರ್ಟ್", "ಗೂಗಲ್‌ನ ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದ ಎನ್‌ಕೋಡರ್ ಮಾಡೆಲ್ (2019)"),
    ("DistilBERT", "ಡಿಸ್ಟಿಲ್‌ಬರ್ಟ್", "BERT ನ 40% ಚಿಕ್ಕ, 60% ವೇಗದ ಆವೃತ್ತಿ, ~97% ಗುಣಮಟ್ಟ"),
    ("GPT", "ಜಿಪಿಟಿ", "ಮುಂದಿನ ಪದ ಊಹಿಸಿ ಬರೆಯುವ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ (ChatGPT ಯ ಆಧಾರ)"),
    ("Pre-training", "ಪೂರ್ವ-ತರಬೇತಿ", "ಬೃಹತ್ ಪಠ್ಯದಲ್ಲಿ ಸಾಮಾನ್ಯ ಭಾಷೆ ಕಲಿಯುವುದು"),
    ("Masked language modelling", "ಖಾಲಿ ಜಾಗ ತುಂಬುವ ಆಟ", "15% ಪದ ಮರೆಮಾಡಿ ಊಹಿಸುವುದು"),
    ("Fine-tuning", "ಫೈನ್-ಟ್ಯೂನಿಂಗ್", "ನಮ್ಮ ಕೆಲಸಕ್ಕಾಗಿ ಚಿಕ್ಕ ಹೆಚ್ಚುವರಿ ತರಬೇತಿ"),
    ("Knowledge distillation", "ಜ್ಞಾನ ಸಾರ ತೆಗೆಯುವುದು", "ದೊಡ್ಡ ಮಾಡೆಲ್‌ನ ಜ್ಞಾನ ಚಿಕ್ಕದಕ್ಕೆ ಕಲಿಸುವುದು"),
    ("Zero-shot", "ಜೀರೋ-ಶಾಟ್", "ಉದಾಹರಣೆ ತೋರಿಸದೆ ಪ್ರಶ್ನೆ ಕೇಳಿ ಉತ್ತರ ಪಡೆಯುವುದು"),
    ("Natural language inference (NLI)", "ಸೂಚನೆ ತೀರ್ಮಾನ", "ಒಂದು ಹೇಳಿಕೆ ಇನ್ನೊಂದನ್ನು ಸೂಚಿಸುತ್ತದೆಯೇ ಎಂದು ನೋಡುವುದು"),
    ("TF-IDF", "ಟಿಎಫ್-ಐಡಿಎಫ್ (ಪದ ತೂಕ)", "ಅಪರೂಪದ ಪದಕ್ಕೆ ಹೆಚ್ಚು ತೂಕ ಕೊಡುವ ಪದ ಎಣಿಕೆ"),
    ("Naive Bayes / Logistic Regression / SVM", "ಸಾಂಪ್ರದಾಯಿಕ ವರ್ಗೀಕರಣಕಾರರು", "ಪದ ತೂಕಗಳಿಂದ ಉತ್ತರ ಕೊಡುವ ಸರಳ ಮಾಡೆಲ್‌ಗಳು"),
    ("LSTM / BiLSTM", "ಎಲ್‌ಎಸ್‌ಟಿಎಂ", "ಒಂದೊಂದೇ ಪದ ಓದುವ ನೆನಪಿನ ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್"),
    ("Loss / cross-entropy", "ನಷ್ಟ", "ಮಾಡೆಲ್ ಎಷ್ಟು ತಪ್ಪಿದೆ ಎಂಬ ಅಳತೆ"),
    ("Learning rate", "ಕಲಿಕೆ ದರ", "ಪ್ರತಿ ಬಾರಿ ಎಷ್ಟು ತಿರುಗಿಸುವುದು ಎಂಬ ಹೆಜ್ಜೆ ಗಾತ್ರ"),
    ("Epoch", "ಎಪಾಕ್", "ಎಲ್ಲಾ ತರಬೇತಿ ಡೇಟಾವನ್ನು ಒಮ್ಮೆ ನೋಡುವುದು"),
    ("Overfitting", "ಬಾಯಿಪಾಠ", "ತರಬೇತಿಯಲ್ಲಿ ಬಹಳ ಚೆನ್ನಾಗಿ ಆದರೆ ಹೊಸ ಡೇಟಾದಲ್ಲಿ ಕಳಪೆ"),
    ("Train / dev / test set", "ತರಬೇತಿ / ಅಭ್ಯಾಸ / ಪರೀಕ್ಷೆ ಡೇಟಾ", "ಪಠ್ಯಪುಸ್ತಕ / ಮಾದರಿ ಪ್ರಶ್ನೆಪತ್ರಿಕೆ / ನಿಜವಾದ ಪರೀಕ್ಷೆ"),
    ("Accuracy", "ನಿಖರತೆ", "ಸರಿಯಾದ ಉತ್ತರಗಳ ಶೇಕಡಾ"),
    ("Precision, recall, F1", "ಪ್ರೆಸಿಶನ್, ರಿಕಾಲ್, ಎಫ್1", "ವರ್ಗವಾರು ಸರಿ-ತಪ್ಪುಗಳ ಸಮತೋಲಿತ ಅಳತೆಗಳು"),
    ("Confusion matrix", "ಗೊಂದಲ ಕೋಷ್ಟಕ", "ನಿಜವಾದದ್ದು vs ಊಹಿಸಿದ್ದರ ಕೋಷ್ಟಕ"),
    ("Confidence interval (bootstrap)", "ವಿಶ್ವಾಸ ಅಂತರ", "ನಿಖರತೆ ಎಷ್ಟು ಅಲುಗಾಡಬಹುದು ಎಂಬ ಮಿತಿ"),
    ("McNemar test", "ಮೆಕ್‌ನೆಮಾರ್ ಪರೀಕ್ಷೆ", "ಎರಡು ಮಾಡೆಲ್‌ಗಳ ವ್ಯತ್ಯಾಸ ನಿಜವೋ ಎಂದು ನೋಡುವ ಪರೀಕ್ಷೆ"),
    ("p-value", "ಪಿ-ಮೌಲ್ಯ", "ಅದೃಷ್ಟದಿಂದಲೇ ಇಷ್ಟು ವ್ಯತ್ಯಾಸ ಬರುವ ಸಂಭಾವ್ಯತೆ; ಚಿಕ್ಕದಿದ್ದರೆ ವ್ಯತ್ಯಾಸ ನಿಜ"),
    ("Latency", "ಪ್ರತಿಕ್ರಿಯೆ ಸಮಯ", "ಒಂದು ವಾಕ್ಯಕ್ಕೆ ಉತ್ತರ ಕೊಡಲು ಬೇಕಾದ ಸಮಯ"),
    ("Parameters", "ಪ್ಯಾರಾಮೀಟರ್‌ಗಳು", "ಮಾಡೆಲ್ ಕಲಿತ ಸಂಖ್ಯೆಗಳು: ರೇಡಿಯೋದ ಗುಂಡಿಗಳಂತೆ"),
    ("Learning curve", "ಕಲಿಕೆ ವಕ್ರರೇಖೆ", "ಲೇಬಲ್ ಸಂಖ್ಯೆ ಹೆಚ್ಚಿದಂತೆ ನಿಖರತೆ ಹೇಗೆ ಬದಲಾಗುತ್ತದೆ"),
    ("SST-2 / GLUE", "ಎಸ್‌ಎಸ್‌ಟಿ-2 / ಗ್ಲೂ", "ನಮ್ಮ ಡೇಟಾಸೆಟ್ / ಭಾಷಾ ಅರ್ಥಗ್ರಹಣ ಪರೀಕ್ಷೆಗಳ ಸಂಗ್ರಹ"),
    ("Hugging Face Hub", "ಹಗ್ಗಿಂಗ್ ಫೇಸ್", "ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದ ಮಾಡೆಲ್‌ಗಳ ಸಾರ್ವಜನಿಕ ಭಂಡಾರ"),
    ("Sub-word tokenisation", "ಉಪಪದ ಟೋಕನೀಕರಣ", "ಪದವನ್ನು ಅರ್ಥಪೂರ್ಣ ತುಣುಕುಗಳಾಗಿ ಒಡೆಯುವುದು: ಕನ್ನಡಕ್ಕೆ ಬಹಳ ಉಪಯುಕ್ತ"),
]


def ch10(N):
    rows = "".join(f'<tr><td class="en"><b>{e}</b></td><td>{k}</td><td>{d}</td></tr>' for e, k, d in GLOSS)
    return f"""
<h1 class="ch" id="c10">ಅಧ್ಯಾಯ 10 · ಪದಕೋಶ<small>GLOSSARY: ENGLISH · KANNADA · SIMPLE MEANING</small></h1>
<p>ಪ್ರೆಸೆಂಟೇಶನ್ ಮತ್ತು ವೈವಾದಲ್ಲಿ ಇಂಗ್ಲಿಷ್ ಪದಗಳೇ ಬೇಕು. ಎಡಗಡೆಯ ಇಂಗ್ಲಿಷ್ ಪದವನ್ನು ಮುಚ್ಚಿ, ಕನ್ನಡ ವಿವರಣೆ ನೋಡಿ ಇಂಗ್ಲಿಷ್ ಪದ ನೆನಪಿಸಿಕೊಳ್ಳುವುದು ಒಳ್ಳೆಯ ಅಭ್ಯಾಸ.</p>
<table><thead><tr><th style="width:30%">English term</th><th style="width:25%">ಕನ್ನಡ</th><th>ಸರಳ ವಿವರಣೆ</th></tr></thead><tbody>{rows}</tbody></table>
"""


MCQ = [
    ("ಸೆಲ್ಫ್-ಅಟೆನ್ಷನ್ ಪ್ರತಿ ಪದಕ್ಕೆ ಏನನ್ನು ಲೆಕ್ಕ ಹಾಕುತ್ತದೆ?", ["ಪದ ಎಷ್ಟು ಸಲ ಬಂದಿದೆ ಎಂದು", "ಎಲ್ಲಾ ಪದಗಳ ಮಾಹಿತಿಯ ತೂಕದ ಮಿಶ್ರಣ", "ಮುಂದಿನ ಪದದ ಸಂಭಾವ್ಯತೆ", "ಪದದ ಭಾವನೆ ಅಂಕ"], 1, "ಹೊಂದಾಣಿಕೆ ಅಂಕ → softmax ತೂಕ → ಮಾಹಿತಿಯ ಮಿಶ್ರಣ."),
    ("ಅಟೆನ್ಷನ್‌ನಲ್ಲಿ √d_k ನಿಂದ ಏಕೆ ಭಾಗಿಸುತ್ತೇವೆ?", ["ವಾಕ್ಯ ಚಿಕ್ಕದಾಗಲು", "softmax ಒಂದೇ ಪದಕ್ಕೆ 100% ಕೊಡದಂತೆ ಮತ್ತು ಗ್ರೇಡಿಯಂಟ್ ಉಳಿಯುವಂತೆ", "ಹೆಡ್ಸ್ ಎಣಿಸಲು", "ಲೇಬಲ್ ಸರಿಪಡಿಸಲು"], 1, "ದೊಡ್ಡ ಸಂಖ್ಯೆಗಳು softmax ಅನ್ನು ‘ಒಂದೇ ವಿಜೇತ’ ಮಾಡುತ್ತವೆ."),
    ("ನಮ್ಮ ಪ್ರಯೋಗದಲ್ಲಿ ಅತಿ ಹೆಚ್ಚು ನಿಖರತೆ ಯಾವ ಮಾಡೆಲ್‌ಗೆ?", ["TF-IDF + SVM", "BiLSTM", "ನಾವು ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿದ DistilBERT", "Ready-made DistilBERT-SST2"], 3, "ready-made DistilBERT-SST2: {pt}%; ನಮ್ಮ DistilBERT: {ft}%."),
    ("BERT-small ಎಲ್ಲಾ ಡೇಟಾದಲ್ಲಿ ಕಲಿತ ಅತ್ಯುತ್ತಮ ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್‌ಗೆ ಸಮವಾಗಲು ಎಷ್ಟು ಲೇಬಲ್ ಬೇಕಾಯಿತು?", ["ಸುಮಾರು 500", "ಸುಮಾರು 1,000", "ಸುಮಾರು 4,000", "65,000 ಕ್ಕಿಂತ ಹೆಚ್ಚು"], 2, "4,000 ಲೇಬಲ್‌ನಲ್ಲಿ {lc4}% vs {svm}%: ಸುಮಾರು 16 ಪಟ್ಟು ಕಡಿಮೆ."),
    ("SST-2 validation set ಅನ್ನು test set ಆಗಿ ಏಕೆ ಬಳಸಿದೆವು?", ["ಅದು ಅತಿ ದೊಡ್ಡದು", "GLUE ನ ನಿಜವಾದ ಟೆಸ್ಟ್ ಲೇಬಲ್‌ಗಳು ಮುಚ್ಚಿಡಲ್ಪಟ್ಟಿವೆ", "validation ಯಾವಾಗಲೂ ಉತ್ತಮ", "BiLSTM ಗೆ ಅದು ಬೇಕು"], 1, "ಇದು ಸಾಮಾನ್ಯ ಪದ್ಧತಿ; ಜೊತೆಗೆ ನಮ್ಮದೇ dev set ಮಾಡಿದೆವು."),
    ("McNemar ಪರೀಕ್ಷೆ ಏನನ್ನು ಹೋಲಿಸುತ್ತದೆ?", ["ವಾಕ್ಯಗಳ ಸರಾಸರಿ ಉದ್ದ", "ತರಬೇತಿ ಸಮಯ", "ಇಬ್ಬರಲ್ಲಿ ಒಬ್ಬರು ಮಾತ್ರ ಸರಿ ಹೇಳಿದ ವಾಕ್ಯಗಳು", "ಪ್ಯಾರಾಮೀಟರ್ ಸಂಖ್ಯೆ"], 2, "b ಮತ್ತು c ಸಂಖ್ಯೆ ಸಮವಾಗಿದ್ದರೆ ಮಾಡೆಲ್‌ಗಳು ಸಮಾನ."),
    ("BiLSTM ಗೆ dev ನಲ್ಲಿ ~91% ಆದರೆ test ನಲ್ಲಿ {bi}% ಏಕೆ?", ["ಟೆಸ್ಟ್‌ನಲ್ಲಿ ತಪ್ಪುಗಳಿದ್ದವು", "dev ಚಿಕ್ಕ ತರಬೇತಿ-ಪದಗುಚ್ಛಗಳು, test ಪೂರ್ತಿ ವಾಕ್ಯಗಳು", "BiLSTM ಪೂರ್ವ-ತರಬೇತಿ ಪಡೆದಿತ್ತು", "dev ಯಾವಾಗಲೂ ಸುಲಭ"], 1, "dev ಸರಾಸರಿ 9.6 ಪದ; test 19.5 ಪದ."),
    ("NLI ಮೂಲಕ zero-shot ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ?", ["ಮೊದಲು ಭಾವನೆ ಲೇಬಲ್‌ಗಳಲ್ಲಿ ತರಬೇತಿ", "“The sentiment of this movie review is positive” ಎಂಬ ಹೇಳಿಕೆ ವಾಕ್ಯದಿಂದ ಸೂಚಿತವೇ ಎಂದು ನೋಡುತ್ತದೆ", "ಧನಾತ್ಮಕ ಪದ ಎಣಿಸುತ್ತದೆ", "ವಾಕ್ಯ ಅನುವಾದ ಮಾಡುತ್ತದೆ"], 1, "ಯಾವುದೇ ಭಾವನೆ ಲೇಬಲ್ ಬಳಸುವುದಿಲ್ಲ."),
    ("ನಿರಾಕರಣೆ ವಾಕ್ಯಗಳಲ್ಲಿ ಏನಾಯಿತು?", ["ಎಲ್ಲಾ ಮಾಡೆಲ್‌ಗಳು ಸಮ", "ಪದ ಎಣಿಕೆ ಸುಮಾರು 8 ಅಂಕ ಕಳೆದುಕೊಂಡವು; ದೊಡ್ಡ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ 2–4 ಅಂಕ ಮಾತ್ರ", "ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಹೆಚ್ಚು ಕಳೆದುಕೊಂಡವು", "BiLSTM ಅತಿ ಹೆಚ್ಚು ಕಳೆದುಕೊಂಡಿತು"], 1, "ಆದರೆ BERT-small/BERT-tiny ಪದ ಎಣಿಕೆಯಂತೆ ಸೋತವು."),
    ("ಅಟೆನ್ಷನ್ ಹೀಟ್-ಮ್ಯಾಪ್ ಅನ್ನು ಹೇಗೆ ಓದಬೇಕು?", ["ಮಾಡೆಲ್ ಯೋಚಿಸುವ ಪೂರ್ತಿ ಸಾಕ್ಷ್ಯವಾಗಿ", "ಮಾಹಿತಿ ಹರಿವಿನ ಚಿತ್ರಣವಾಗಿ, ಪೂರ್ಣ ವಿವರಣೆಯಾಗಿ ಅಲ್ಲ", "ಭಾವನೆ ಅಂಕವಾಗಿ", "ನಷ್ಟ ಮೌಲ್ಯವಾಗಿ"], 1, "Jain & Wallace (2019)."),
    ("ತರಬೇತಿ ಸಮಯಗಳನ್ನು ಹೇಗೆ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಬೇಕು?", ["GPU ಸಮಯ", "ಲ್ಯಾಪ್ಟಾಪ್ ಬ್ಯುಸಿ ಇದ್ದ ಕಾರಣ ‘ಇಷ್ಟಕ್ಕಿಂತ ಹೆಚ್ಚಲ್ಲ’ (upper bound); ಪಟ್ಟು-ಪ್ರಮಾಣ ಮುಖ್ಯ", "ಪೂರ್ವ-ತರಬೇತಿ ಸಮಯ", "ನಿಖರ ಸೆಕೆಂಡ್‌ಗಳು"], 1, "ನಾವು ಪೂರ್ವ-ತರಬೇತಿ ಮಾಡಿಲ್ಲ; ಫೈನ್-ಟ್ಯೂನ್ ಮಾತ್ರ."),
    ("BERT-tiny ಬಗ್ಗೆ ಸರಿಯಾದ ಹೇಳಿಕೆ ಯಾವುದು?", ["ಅತ್ಯುತ್ತಮ ಮಾಡೆಲ್", "ಎಲ್ಲಾ ಪದ ಎಣಿಕೆ ಮಾಡೆಲ್‌ಗಳನ್ನು ಮೀರಿಸಿತು", "ಅತ್ಯಂತ ದುರ್ಬಲ: ಸಾಮರ್ಥ್ಯ ಸಾಲದು", "ನಾವು ಪೂರ್ವ-ತರಬೇತಿ ಮಾಡಿದೆವು"], 2, "{tiny}% vs {svm}% (SVM)."),
]


def ch11(N):
    a = N["acc"]
    ft, svm, pt = a["ft_distilbert_8000"], a["tfidf_svm"], a["pt_distilbert_sst2"]
    fmt = dict(pt=a["pt_distilbert_sst2"], ft=a["ft_distilbert_8000"], lc4=N["lc4000_t"], svm=a["tfidf_svm"], bi=a["bilstm"], tiny=a["ft_bert-tiny_8000"])
    qs, ans = "", ""
    for i, (q, opts, k, why) in enumerate(MCQ, 1):
        o = "".join(f'<div class="o">({"ಅಆಇಈ"[j]}) {t}</div>' for j, t in enumerate(opts))
        qs += f'<div class="mcq"><b>{i}. {q.format(**fmt)}</b>{o}</div>'
        ans += f'<tr><td>{i}</td><td><b>({"ಅಆಇಈ"[k]})</b></td><td>{why.format(**fmt)}</td></tr>'
    return f"""
<h1 class="ch" id="c11">ಅಧ್ಯಾಯ 11 · ಸ್ವಯಂ ಪರೀಕ್ಷೆ ಮತ್ತು ಭಾಷಣ ಅಭ್ಯಾಸ<small>SELF-TEST AND SPEECH PRACTICE</small></h1>
<h2>11.1 ಹನ್ನೆರಡು ಬಹು-ಆಯ್ಕೆ ಪ್ರಶ್ನೆಗಳು</h2>
<p>ಮೊದಲು ಉತ್ತರ ನೋಡದೆ ಬರೆಯಿರಿ. ಉತ್ತರಗಳು ಮುಂದಿನ ಪುಟದಲ್ಲಿ.</p>
{qs}
<h2>ಉತ್ತರಗಳು</h2>
<table><thead><tr><th style="width:6%">#</th><th style="width:10%">ಉತ್ತರ</th><th>ವಿವರಣೆ</th></tr></thead><tbody>{ans}</tbody></table>
<div class="keep"><h2>11.2 ಒಂದು ನಿಮಿಷದ ಭಾಷಣ: ಮೊದಲು ಕನ್ನಡದಲ್ಲಿ, ನಂತರ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ಹೇಳಿ ಅಭ್ಯಾಸ ಮಾಡಿ</h2>
<div class="say"><b>ಕನ್ನಡದಲ್ಲಿ:</b> “ಎಲ್ಲರೂ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಅತ್ಯುತ್ತಮ ಎನ್ನುತ್ತಾರೆ. ಆದರೆ ಸಾಮಾನ್ಯ ಲ್ಯಾಪ್ಟಾಪ್ ಮತ್ತು ಕೆಲವು ಸಾವಿರ ಲೇಬಲ್ ಮಾತ್ರ ಇದ್ದಾಗ ಅದು ಎಷ್ಟು ಉತ್ತಮ, ಮತ್ತು ಅದಕ್ಕೆ ಯಾವ ಬೆಲೆ? ಇದನ್ನು ಕಂಡುಹಿಡಿಯಲು ನಾವು ಹತ್ತು ಮಾಡೆಲ್‌ಗಳನ್ನು ಒಂದೇ ಕಾರ್ಯದಲ್ಲಿ, ಒಂದೇ ಟೆಸ್ಟ್ ಡೇಟಾದಲ್ಲಿ ಹೋಲಿಸಿದೆವು. ಫಲಿತಾಂಶ: ಕೇವಲ 8,000 ವಾಕ್ಯಗಳಲ್ಲಿ ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿದ DistilBERT {ft}% ತಲುಪಿತು; ಎಲ್ಲಾ 65 ಸಾವಿರ ವಾಕ್ಯಗಳಲ್ಲಿ ಕಲಿತ ಅತ್ಯುತ್ತಮ ಸಾಂಪ್ರದಾಯಿಕ ಮಾಡೆಲ್ {svm}%. ಬೆಲೆ: ಸುಮಾರು 700 ಪಟ್ಟು ನಿಧಾನ ಉತ್ತರ. ತರಬೇತಿ ಇಲ್ಲದ ready-made ಮಾಡೆಲ್ {pt}%. 500 ಲೇಬಲ್ ಮಾತ್ರ ಇದ್ದರೂ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ 14 ಅಂಕ ಮುಂದಿದೆ. ನಮ್ಮ ಶಿಫಾರಸು: ಮೊದಲು ಇರುವ ಮಾಡೆಲ್ ಬಳಸಿ, ನಂತರ ಚಿಕ್ಕ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿ, ವೇಗವೇ ಮುಖ್ಯವಾದರೆ ಮಾತ್ರ ಪದ ಎಣಿಕೆ.”</div>
<div class="say"><b>In English:</b> <span class="en">“Everyone says Transformers are the best. But how much better are they when you only have an ordinary laptop and a few thousand labelled examples, and what do they cost? We compared ten models on the same task and the same test data. A DistilBERT fine-tuned on only 8,000 sentences reaches {ft}%, versus {svm}% for the best classical model trained on all 65 thousand sentences, at about 700 times the inference time. A ready-made in-domain model reaches {pt}% without any training, and with only 500 labels the Transformer is 14 points ahead. Our recommendation: reuse first, fine-tune a distilled model second, and use TF-IDF only when speed dominates.”</span></div></div>
"""


def ch12(N):
    a = N["acc"]
    return f"""
<h1 class="ch" id="c12">ಅಧ್ಯಾಯ 12 · ಒಂದು ಪುಟದ ಸಾರಾಂಶ<small>ONE-PAGE REVISION SHEET</small></h1>
<div class="compact">
<div class="grid4">
<div class="stat"><b>{a['pt_distilbert_sst2']}%</b><span>Ready-made DistilBERT</span></div>
<div class="stat"><b>{a['ft_distilbert_8000']}%</b><span>DistilBERT, 8,000 ಲೇಬಲ್</span></div>
<div class="stat b"><b>{a['tfidf_svm']}%</b><span>ಅತ್ಯುತ್ತಮ TF-IDF (65,349 ಲೇಬಲ್)</span></div>
<div class="stat o"><b>{a['bilstm']}%</b><span>BiLSTM</span></div>
<div class="stat v"><b>+{N['lc500_gain']}</b><span>500 ಲೇಬಲ್‌ನಲ್ಲಿ BERT-small ಮುನ್ನಡೆ (ಅಂಕ)</span></div>
<div class="stat o"><b>≈ {N['latratio']}×</b><span>ನಿಧಾನ ಉತ್ತರ: DistilBERT vs SVM</span></div>
<div class="stat"><b>{a['zs_nli_distilroberta']}%</b><span>Zero-shot (ಲೇಬಲ್ ಇಲ್ಲದೆ)</span></div>
<div class="stat o"><b>{a['ft_bert-tiny_8000']}%</b><span>BERT-tiny: ಚಿಕ್ಕದು ಸಾಲದು</span></div></div>
<div class="cols">
<div class="card"><b>ಮುಖ್ಯ ಐಡಿಯಾಗಳು</b><ul class="small" style="margin:1mm 0 0">
<li><b>ಟೋಕನ್ → ಎಂಬೆಡ್ಡಿಂಗ್ + ಸ್ಥಾನ</b>: ಪದ ಸಂಖ್ಯೆಯಾಗುತ್ತದೆ</li>
<li><b>ಅಟೆನ್ಷನ್</b>: softmax(QKᵀ/√d<sub>k</sub>)V: ಮದುವೆ ಮನೆಯಲ್ಲಿ ಕಿವಿ ತಿರುಗುವಂತೆ</li>
<li><b>ಹೆಡ್ಸ್ + ಲೇಯರ್ಸ್</b>: ಹಲವು ಕೋನ, ಹಲವು ಹಂತ</li>
<li><b>ಸಮಾನಾಂತರ</b>: ಎಲ್ಲಾ ಪದಗಳು ಒಮ್ಮೆಲೇ; ವೆಚ್ಚ O(n²)</li>
<li><b>ಪೂರ್ವ-ತರಬೇತಿ + ಫೈನ್-ಟ್ಯೂನಿಂಗ್</b>: SSLC ನಂತರ ಕಂಪನಿ ತರಬೇತಿ</li></ul></div>
<div class="card"><b>ನಮ್ಮ ಪ್ರಯೋಗ</b><ul class="small" style="margin:1mm 0 0">
<li>SST-2: ಸಿನಿಮಾ ವಿಮರ್ಶೆ, {N['n_train']} / 2,000 / {N['n_test']}</li>
<li>10 ಮಾಡೆಲ್, 4 ಕುಟುಂಬ, ಒಂದೇ ಲ್ಯಾಪ್ಟಾಪ್ CPU</li>
<li>ಅಳತೆ: ನಿಖರತೆ, F1, ಸಮಯ, ವೇಗ, ಗಾತ್ರ</li>
<li>ವಿಶ್ವಾಸ ಅಂತರ (bootstrap) + McNemar ಪರೀಕ್ಷೆ</li>
<li>ನಿರಾಕರಣೆ {N['neg_n']} / ವೈದೃಶ್ಯ {N['con_n']} ವಾಕ್ಯಗಳ ವಿಶ್ಲೇಷಣೆ</li></ul></div></div>
{box("exam", "ಐದು ಉತ್ತರಗಳು (ಐದು ಸಂಶೋಧನೆ ಪ್ರಶ್ನೆಗಳಿಗೆ)", f'''<ol class="small">
<li><b>RQ1:</b> ಸಾಕಷ್ಟು ದೊಡ್ಡ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ ಸ್ಪಷ್ಟವಾಗಿ ಉತ್ತಮ (+{N['gain']} ಅಂಕ, ನಿಜವಾದ ವ್ಯತ್ಯಾಸ); ಚಿಕ್ಕದು (BERT-tiny) ಸಾಲದು.</li>
<li><b>RQ2:</b> ಬೆಲೆ: ≈ {N['latratio']}× ನಿಧಾನ, ಗಂಟೆಗಳ CPU ತರಬೇತಿ, 67 M ಪ್ಯಾರಾಮೀಟರ್.</li>
<li><b>RQ3:</b> ಲೇಬಲ್ ಕಡಿಮೆಯಾದಷ್ಟೂ ಲಾಭ ಹೆಚ್ಚು: 500 ಲೇಬಲ್‌ನಲ್ಲಿ +{N['lc500_gain']}; 4,000 ಲೇಬಲ್ ≈ 65 ಸಾವಿರದ TF-IDF.</li>
<li><b>RQ4:</b> ಇದೇ-ಕ್ಷೇತ್ರದ ready-made ಅತ್ಯುತ್ತಮ ({a['pt_distilbert_sst2']}%); ಬೇರೆ ಕ್ಷೇತ್ರ {a['pt_twitter_roberta']}%; zero-shot {a['zs_nli_distilroberta']}%.</li>
<li><b>RQ5:</b> ನಿರಾಕರಣೆಯಲ್ಲಿ ಪದ ಎಣಿಕೆ ~8 ಅಂಕ ಕಳೆದುಕೊಂಡರೆ ದೊಡ್ಡ ಟ್ರಾನ್ಸ್ಫಾರ್ಮರ್ 2–4.</li></ol>''')}
{box("note", "ವೈವಾದ ಮೊದಲು ಈ ಹತ್ತು ಸಾಲು ನೆನಪಿಡಿ", f'''<ol class="small">
<li>ಕಾರ್ಯ: ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ (binary), ಡೇಟಾ SST-2.</li>
<li>Test = GLUE validation set ({N['n_test']}); dev = ತರಬೇತಿಯಿಂದ 2,000.</li>
<li>ಫೈನ್-ಟ್ಯೂನ್ ಮಾಡಿದ್ದೇವೆ; ಪೂರ್ವ-ತರಬೇತಿ ಮಾಡಿಲ್ಲ.</li>
<li>ಒಂದು seed, ಒಂದು ಡೇಟಾಸೆಟ್, ಇಂಗ್ಲಿಷ್: ಮಿತಿಗಳು.</li>
<li>ಸಮಯ ಅಳತೆಗಳು “ಇಷ್ಟಕ್ಕಿಂತ ಹೆಚ್ಚಲ್ಲ”.</li>
<li>ಅಟೆನ್ಷನ್ ಚಿತ್ರ = ಚಿತ್ರಣ, ಪೂರ್ಣ ವಿವರಣೆಯಲ್ಲ.</li>
<li>ಶಿಫಾರಸು: reuse → fine-tune distilled → TF-IDF only if speed dominates.</li>
<li>ಕನ್ನಡ/ಲಿಥುವೇನಿಯನ್ = ಭವಿಷ್ಯದ ಕೆಲಸ (XLM-R, MuRIL, LitLat BERT).</li>
<li>ಕೋಡ್ + ಫಲಿತಾಂಶ GitHub ನಲ್ಲಿದೆ: shashikanth077/Transformers-NLP.</li>
<li>ಗೊತ್ತಿಲ್ಲದ್ದನ್ನು ಒಪ್ಪಿಕೊಳ್ಳಿ: “ಅದನ್ನು ನಾವು ಪರೀಕ್ಷಿಸಿಲ್ಲ, ಅದೇ ಮುಂದಿನ ಹೆಜ್ಜೆ.”</li></ol>''')}
</div>
"""
