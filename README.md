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

## Reference

This project is motivated by the following paper:

> S Eltanani, TV olde Scheper, M Muñoz-Balbontin (2023). *Detecting Human Gait Metabolism Disorders Based on A Novel Criticality Analysis Methodology*. preprints.org. https://www.preprints.org/frontend/manuscript/82494eed87acf0bc587908c81077c9d6/download_pub

Paper author: Prof. Arantza Aldea, Oxford Brookes University (aaldea@brookes.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
