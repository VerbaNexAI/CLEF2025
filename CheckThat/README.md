
# CLEF2025 CheckThat Lab - Subtask 4a: Scientific Web Discourse Detection

![GitHub last commit](https://img.shields.io/github/last-commit/mervinso/CLEF2025_Task4a)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X9OuZ5tziJ7UIxwPv5jZsDEGHcagN9fC?usp=sharing)
<!-- ![GitHub license](https://img.shields.io/github/license/mervinso/CLEF2025_Task4a) -->

## 🏁 Overview

This repository details our submission for the CLEF2025 CheckThat Lab Task 4a: Scientific Web Discourse Detection. The primary goal is to accurately identify scientific discourse in Twitter data using a multilabel classification approach, aiming to maximize the macro-averaged F1-score. Provides code, data structure, and reproducible experiments.

Target audience: researchers, data scientists, and practitioners interested in natural language processing (NLP), social media analysis, and machine learning competitions.

---

## 📌 Objective

This notebook and codebase describe the **experimental approach** taken to develop a multi-label classification system for identifying scientific discourse in Twitter data. The model is built on top of `microsoft/deberta-v3-base` and optimized through a multi-phase strategy aimed at maximizing macro-averaged F1-score.

---

## 📦 Requirements

To run this project, you need to install the necessary Python packages. These are listed in the `requirements.txt` file. Install all dependencies with:

```bash
pip install -r requirements.txt
```
---

## 🚚 Installation & Data Preparation

Clone this repository:
```bash
git clone https://github.com/mervinso/CLEF2025_Task4a.git
cd CLEF2025_Task4a
```

Download the dataset:
- Obtain the files `ct_train.tsv` and `ct_test.tsv` as provided by the organizers.
- Place them in the data/ directory:
```bash
CLEF2025_Task4a/
  └── data/
      ├── ct_train.tsv
      └── ct_test.tsv
```

---

## 📋 General Notes

- **Dataset:** 1229 tweets (train), 137 (dev), 240 (test)
- **Task:** Multilabel classification (cat1, cat2, cat3)
- **Target metric:** macro-averaged F1-score
- **Submission format:** predictions.csv with columns [index, cat1_pred, cat2_pred, cat3_pred]

---
## 🔬 Dataset

- `ct_train.tsv` – training set
- `ct_dev.tsv` – development set
- `ct_test.tsv` – test set for leaderboard submission
- Format: each tweet labeled across three binary categories (`cat1`, `cat2`, `cat3`)

---

## 🚀 How to Reproduce (in Colab)

1. Open `CLEF2025-SubTask4a-SciDiscourse.ipynb`[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1X9OuZ5tziJ7UIxwPv5jZsDEGHcagN9fC?usp=sharing) in Google Colab.
2. Clone the official CLEF2025 CheckThat repository and extract the folder `task4/subtask_4a`.
3. Copy `ct_train.tsv` and `ct_test.tsv` into the `/data/` folder inside your working directory.
4. Execute the notebook sequentially through all six phases:
    - Baseline → Threshold Tuning → Fine-Tuning → Class Weights → Ensemble → Final Prediction.
5. The output file `predictions.csv` will be saved under `/predictions/` and is ready to be submitted to the leaderboard.
  
---

## 📂 Project Structure

```
clef2025_task4a/
├── data/
│ ├── ct_dev.tsv
│ ├── ct_test.tsv
│ └── ct_train.tsv
├── models/
│   └── final_model/
├── predictions/
│   └── predictions.csv
├── notebooks/
│   └── CLEF2025_SubTask4a_SciDiscourse.ipynb
└── requirements.txt
├── README.md
```
---

## ⚙️ Phases Overview

| Phase | Description                          | Output                        |
|-------|--------------------------------------|-------------------------------|
| 1     | Baseline training (DeBERTa-v3-base)  | `cv_preds`                    |
| 2     | Threshold tuning (PR curve)          | `thresholds.json`             |
| 3     | Fine-tuning (lr, epochs search)      | `best_macro_f1`, config       |
| 4     | Training with class weights          | `macro_f1_class_weights`      |
| 5     | Ensemble of models (soft voting)     | `macro_f1_ensemble`           |
| 6     | Final training + test prediction     | `predictions.csv`             |

---

## 🧪 Results Summary

| Model           | Macro F1 | Cat1 F1 | Cat2 F1 | Cat3 F1 | Notes                          |
|------------------|----------|---------|---------|---------|--------------------------------|
| Baseline         | 0.8021   | 0.79xx  | 0.76xx  | 0.83xx  | lr=2e-5, 10 epochs              |
| Fine-tuned       | 0.8143   | 0.81xx  | 0.78xx  | 0.84xx  | lr=2e-5, 12 epochs              |
| Class Weights    | 0.8195   | 0.82xx  | 0.79xx  | 0.85xx  | weights applied per class      |
| **Ensemble**     | **0.8274** | 0.83xx  | 0.80xx  | 0.85xx  | Averaged predictions (FT + CW) |

> Thresholds tuned per class via `precision_recall_curve` to optimize F1 individually.

---

## 💡 Thresholds Used

```json
{
  "cat1": 0.4607,
  "cat2": 0.6438,
  "cat3": 0.7325
}
```
---

## 🔗 References

- CLEF2025 CheckThat: [Official Website](https://checkthat.gitlab.io/clef2025/)
- GitLab Repo: [Official Repository](https://gitlab.com/checkthat_lab/clef2025-checkthat-lab)
- Codalab: [Competition Link](https://codalab.lisn.upsaclay.fr/competitions/22359)

---

## 📄 License
This project is licensed under the MIT License. See LICENSE for details.

---

## 📌 Credits

- Developed by: UTB - CEDNAV
- For the CLEF2025 CheckThat Lab challenge
- Contact: sosam@utb.edu.co


