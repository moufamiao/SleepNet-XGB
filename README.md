# SleepNet-XGB
### SleepNet-XGB: Single-Channel Sleep Staging via Deep Feature Extraction and Bayesian-Optimized XGBoost
## Abstract
![SleepNet-XGB](network.png)
Automatic sleep staging is critical for diagnosing sleep disorders and analyzing sleep quality. However, existing methods often struggle with temporal modeling, class imbalance, and generalization across datasets. In this paper, we propose \textbf{SleepNet-XGB}, a hybrid framework that integrates deep feature extraction with gradient-boosted decision trees. Specifically, we design a CNN-LSTM architecture to capture both local and temporal EEG patterns, followed by an XGBoost classifier to enhance stage discrimination. Furthermore, we apply Bayesian optimization (via Optuna) to fine-tune tree-based parameters automatically. We evaluate our model on three public datasets (Sleep-EDF-20, Sleep-EDF-78, and SHHS) using 20-fold cross-validation. Results show that SleepNet-XGB consistently outperforms existing state-of-the-art methods in terms of accuracy, macro-F1, and Cohen’s $\kappa$. The proposed design achieves a strong balance between deep modeling and classical robustness, offering a scalable and effective solution for practical sleep staging.
## Requirmenets:
- Python 3.11
- Pytorch=='1.8'
- Numpy
- Sklearn
- mne=='0.20.7'

## Prepare datasets

We used three public datasets in this study:
Sleep-EDF-20(https://physionet.org/content/sleep-edfx/1.0.0/)
Sleep-EDF-78(https://physionet.org/content/sleep-edfx/1.0.0/)
SHHS dataset(https://sleepdata.org/datasets/shhs)

