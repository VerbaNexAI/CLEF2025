# 🧠 SapBERT-based Hybrid Re-ranking for BioNNE-L 2025-1

This repository contains the codebase for the **VerbaNex AI Lab**'s submission to the **BioNNE-L 2025-1 Challenge (Subtask 1: English)**, which achieved **1st place in Accuracy@1 (0.70)**. The system implements a biomedical entity linking (BEL) pipeline using **SapBERT** enhanced by a hybrid re-ranking strategy that combines **cosine**, **Jaccard**, and **Levenshtein** similarities.

---

## 📘 Introduction

The **BioNNE-L 2025-1** challenge focuses on mapping English biomedical entity mentions to **UMLS concepts** (e.g., disorders, chemicals, anatomy).

Our system:
- Generates **SapBERT embeddings** for candidate concepts.
- Applies a **hybrid re-ranking mechanism**.
- Outputs top-k predictions with competitive evaluation metrics.

📄 Refer to our paper:  
**_Hybrid Re-ranking for Biomedical Entity Linking using SapBERT Embeddings: A High-Performance System for BioNNE-L 2025-1_**, CLEF 2025 Working Notes.

---

## ⚙️ Installation

### ✅ Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended: A100)
- Jupyter Notebook
- Git

### 📦 Setup Steps

1. **Clone the repository:**
```bash
    git clone https://github.com/VerbaNexAI/CLEF2025.git  
    cd CLEF2025/BioASQ_Task4
```
2. **Create a virtual environment:**
```bash
    python -m venv venv  
    source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install dependencies:**
```bash
    pip install -r requirements.txt
```
Packages include:  
`jupyter`, `transformers`, `torch`, `pandas`, `scikit-learn`, `tqdm`, `python-Levenshtein`, `datasets`

---

## 🚀 Usage

### 📒 Notebooks

- **`notebook/find_best_model.ipynb`**  
  Compares SapBERT, PubMedBERT, BioBERT, and a baseline model using:
  - Accuracy@1
  - Accuracy@5
  - MRR (Mean Reciprocal Rank)

- **`notebook/reranking_sapbert.ipynb`**  
  Executes the BEL pipeline using **SapBERT + hybrid re-ranking**.  
  Tasks:
  - Download and preprocess dataset
  - Generate embeddings
  - Apply cosine, Jaccard, and Levenshtein re-ranking
  - Save evaluation metrics to `data/results/`

### ▶️ Running the Pipeline

1. Launch **Jupyter Notebook** and navigate to the `notebook/` directory.
2. Run `reranking_sapbert.ipynb` to build the full pipeline.
3. Run `find_best_model.ipynb` to evaluate and compare performance across models.

---

## 📊 Results

### Development Set
- **Accuracy@1**: `0.718`
- **Accuracy@5**: `0.802`
- **MRR**: `0.750`

### BioNNE-L 2025-1 Competition
- 🥇 **1st** in Accuracy@1 (`0.70`)
- 🏅 **4th** in Accuracy@5 (`0.80`)
- 🥈 **2nd** in MRR (`0.74`)

➡️ Reproduce dev results by running `find_best_model.ipynb`.

---

## 🙏 Acknowledgments

- **VerbaNex AI Lab** – Team collaboration and development  
- **Universidad Tecnológica de Bolívar** – Institutional support  
- **Hugging Face Datasets** – Access to BioNNE-L and SapBERT models

---

## 📬 Contact

**Daniel Peña Gnecco**  
📧 [dgnecco@utb.edu.co](mailto:dgnecco@utb.edu.co)  
🔗 [LinkedIn](https://www.linkedin.com/in/danielarturopeña) | [GitHub](https://github.com/Danp06)
