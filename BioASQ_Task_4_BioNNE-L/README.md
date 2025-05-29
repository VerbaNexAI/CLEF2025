# SapBERT-based Hybrid Re-ranking for BioNNE-L 2025-1

This repository contains the code for the **VerbaNex AI Lab**'s submission to the **BioNNE-L 2025-1 challenge** (Subtask 1: English), achieving **first place in Accuracy@1 (0.70)**. The project implements a biomedical entity linking (BEL) system using SapBERT with a hybrid re-ranking strategy (cosine, Jaccard, Levenshtein similarities) via two Jupyter notebooks.

## Introduction

The BioNNE-L 2025-1 challenge advances BEL by mapping English biomedical text mentions to UMLS concepts (disorders, chemicals, anatomy). Our system, implemented in two notebooks, downloads the BioNNE-L dataset, generates SapBERT embeddings, applies re-ranking, and saves metrics, achieving an Accuracy@1 of 0.718 on the development set.

See our paper: *SapBERT-based Hybrid Re-ranking for Biomedical Entity Linking in BioNNE-L 2025-1* (CLEF 2025 Working Notes).

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended: A100)
- Jupyter Notebook
- Git

### Steps
1. Clone the repository:
```bash
git clone https://github.com/your-username/bionne-l-2025.git
cd bionne-l-2025
```
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: 'venv\Scripts\activate'
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
   Includes 'jupyter', 'transformers', 'torch', 'pandas', 'scikit-learn', 'tqdm', 'python-levenshtein', 'datasets'.

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Usage

### Notebooks
- **'notebook/find_best_model.ipynb'**: Compares models (SapBERT, PubMedBERT, BioBERT, baseline) and computes metrics (Accuracy@1, Accuracy@5, MRR).
- **'notebook/reranking_sapbert.ipynb'**: Executes the BEL pipeline, downloading the dataset to 'data/bionne_l_dataset/', generating embeddings, re-ranking, and saving metrics to 'results/'.

### Running the Pipeline
1. Open Jupyter Notebook and navigate to 'notebook/'.
2. Run 'reranking_sapbert.ipynb' to:
   - Download the dataset.
   - Process data, generate embeddings, and apply re-ranking.
   - Save metrics to 'data/results/'.
3. Run 'find_best_model.ipynb' to compare model performance.

## Results
- **Development Set**: Accuracy@1: 0.718, Accuracy@5: 0.802, MRR: 0.750.
- **Competition**: 1st in Accuracy@1 (0.70), 4th in Accuracy@5 (0.80), 2nd in MRR (0.74).

Run 'find_best_model.ipynb' to reproduce results.

## Acknowledgments
- **VerbaNex AI Lab** for collaboration.
- **Universidad Tecnológica de Bolívar** for support.
- **Hugging Face** for the BioNNE-L dataset.

## Contact
- Daniel Peña Gnecco: [dgnecco@utb.edu.co](mailto:dgnecco@utb.edu.co)
```