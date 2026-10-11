# Criticality vs. criticality + gait statistics for gait-disorder detection

## Motivation
The paper "Detecting Human Gait Metabolism Disorders Based on A Novel Criticality Analysis Methodology" (2023) applies criticality analysis to human gait to detect metabolism disorders. Its method is only partly described and its data is not stated, so this experiment uses a standard stand-in for criticality features and a public dataset.

## Question
Can a tuned gradient-boosting (or random-forest) classifier on criticality plus standard gait-variability features beat a criticality-only classifier at separating healthy from pathological gait in stride-interval recordings?

## Dataset
PhysioNet *Gait in Neurodegenerative Disease* (gaitndd): 64 stride-interval recordings (healthy controls vs. ALS, Huntington's and Parkinson's patients), downloaded over HTTP from physionet.org. This is **not** the paper's dataset (a stand-in), and the baseline is not the paper's method.

## Models
- **Baseline:** logistic regression on criticality-style features (DFA exponent, Hurst exponent, sample entropy, lag-1 autocorrelation, power-law tail exponent of large fluctuations).
- **Improved:** the same features plus gait statistics (mean, CV, swing/stance/double-support ratios, left-right asymmetry), with a LightGBM or random-forest model tuned by inner cross-validation on the training split only.

Both use identical subject-level stratified 70/30 splits, preprocessing and evaluation. Main metric: test ROC-AUC; secondary: balanced accuracy and F1. A single seed-42 split with model seeds 42-44 is the headline; 20 repeated splits are reported as supplementary.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the PhysioNet gaitndd dataset, a criticality-only logistic regression reached ROC-AUC 0.77 on a 19-subject test split, versus 0.85 for a tuned tree model adding gait statistics; the gain was small and held in only 70% of 20 repeated splits.

| Metric | Value |
|---|---|
| baseline_score | 0.7667 |
| baseline_score_std | 0.04714 |
| baseline_balanced_accuracy | 0.7238 |
| baseline_f1 | 0.7843 |
| baseline_repeated_auc_mean | 0.7843 |
| baseline_repeated_auc_std | 0.07914 |
| baseline_repeated_balacc_mean | 0.7068 |
| baseline_repeated_f1_mean | 0.7751 |
| improved_score | 0.8524 |
| improved_score_std | 0.0526 |
| improved_balanced_accuracy | 0.8167 |
| improved_f1 | 0.8746 |
| improved_repeated_auc_mean | 0.8629 |
| improved_repeated_auc_std | 0.07248 |
| improved_repeated_balacc_mean | 0.7232 |
| improved_repeated_f1_mean | 0.8706 |
| improvement | 0.08571 |
| improvement_repeated_auc | 0.07857 |
| repeated_fraction_splits_improved_better | 0.7 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset (neurodegenerative gait, not metabolic disorders); pathological = ALS/Huntington/Parkinson pooled. Only 63 subjects, so test sets are ~19 subjects and scores are noisy; the repeated-split metrics are supplementary and use model seed 42 only. Baseline LR is deterministic so its seed std reflects only inner-CV fold shuffling. Model families chosen on primary split by seed: ['RandomForestClassifier', 'RandomForestClassifier', 'LGBMClassifier']. Criticality features are simple estimators computed on the mean of left/right stride intervals after crude outlier removal.

## Reference

This project is motivated by the following paper:

> S Eltanani, TV olde Scheper, M Muñoz-Balbontin (2023). *Detecting Human Gait Metabolism Disorders Based on A Novel Criticality Analysis Methodology*. preprints.org. https://www.preprints.org/frontend/manuscript/82494eed87acf0bc587908c81077c9d6/download_pub

Paper author: Prof. Arantza Aldea, Oxford Brookes University (aaldea@brookes.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
