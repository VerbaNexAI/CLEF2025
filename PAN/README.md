# PAN 2025: Human-AI Collaborative Text Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains our solution for **PAN 2025 Subtask 2: Human-AI Collaborative Text Classification**. The task focuses on categorizing documents that have been co-authored by humans and Large Language Models (LLMs) into six distinct categories based on the nature of human and machine contributions.

## Problem Statement

### Subtask 2: Human-AI Collaborative Text Classification

The goal is to classify texts into six distinct categories:

| Label | Category | Description |
|-------|----------|-------------|
| **0** | **Fully human-written** | Document entirely authored by a human without any AI assistance |
| **1** | **Human-initiated, then machine-continued** | Human starts writing, AI model completes the text |
| **2** | **Human-written, then machine-polished** | Initially written by human, later refined/edited by AI |
| **3** | **Machine-written, then machine-humanized** | AI generates text, later modified to obscure machine origin |
| **4** | **Machine-written, then human-edited** | AI-generated content subsequently edited/refined by human |
| **5** | **Deeply-mixed text** | Interwoven sections by both humans and AI without clear separation |

Accurately distinguishing between these categories enhances our understanding of human-AI collaboration and helps mitigate risks associated with synthetic text.

## Dataset Characteristics

- **Multi-domain documents**: Academic papers, journalism, social media content
- **Multiple AI models**: GPT-4, Claude, PaLM generated samples
- **Collaborative texts**: Annotation layers for human/machine contributions
- **Multilingual support**: English, Spanish, German
- **Class distribution**: Highly imbalanced with varying sample sizes per category

## Technical Approach

### Model Architecture
- **Base Model**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Task**: Multi-class sequence classification (6 classes)
- **Fine-tuning**: Task-specific fine-tuning with class-weighted loss

### Key Features
1. **Advanced Data Preprocessing**
   - Intelligent class balancing (undersampling/oversampling)
   - Text augmentation for minority classes
   - Stratified train/validation/test splits

2. **Class Imbalance Handling**
   - Computed class weights using sklearn's `balanced` strategy
   - Custom weighted trainer with CrossEntropyLoss
   - Targeted data augmentation for underrepresented classes

3. **Text Augmentation Techniques**
   - Random word deletion (10% of words)
   - Word position swapping
   - Strategic word duplication

4. **Robust Training Setup**
   - Gradient checkpointing for memory efficiency
   - Mixed precision training (FP16)
   - Early stopping with F1-weighted metric
   - Comprehensive evaluation metrics

## Repository Structure

```
├── data/
│   └── dat_train_v2.csv              # Training dataset
├── pan2025_notebook.ipynb            # Main training notebook
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- 16GB RAM minimum

### Setup Environment

```bash
# Clone repository
git clone https://github.com/your-username/pan2025-collaborative-text-classification.git
cd pan2025-collaborative-text-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
pip install scikit-learn pandas numpy
pip install tqdm jupyter ipywidgets
```

## Usage

### Quick Start

1. **Prepare your data**: Place your training dataset as `data/dat_train_v2.csv`

2. **Run the notebook**: Open and execute `pan2025_notebook.ipynb` cell by cell

3. **Training process**:
   ```bash
   # The notebook includes 17 organized cells:
   # Cells 1-3:   Environment setup and data exploration
   # Cells 4-7:   Data preprocessing and balancing
   # Cells 8-12:  Model configuration and dataset preparation
   # Cells 13-17: Training, evaluation, and model saving
   ```

### Training Configuration

The model uses the following default hyperparameters:

```python
# Training Arguments
num_train_epochs = 3
per_device_train_batch_size = 64
per_device_eval_batch_size = 96
learning_rate = 2e-5
warmup_steps = 500
weight_decay = 0.01
```

### Expected Training Time
- **Preprocessing**: 5-10 minutes
- **Training**: 2-4 hours (depending on hardware)
- **Evaluation**: 10-15 minutes

## Model Performance

### Evaluation Metrics
- **Primary metric**: F1-Score (weighted)
- **Additional metrics**: Accuracy, Precision, Recall
- **Per-class analysis**: Individual F1 scores for each category
- **Confusion matrix**: Detailed classification results

### Class Distribution After Balancing
The preprocessing pipeline handles severe class imbalance:
- **Majority classes** (0, 1, 2): Undersampled to 80,000 samples
- **Minority classes** (3, 4, 5): Oversampled to 10,000+ samples
- **Data augmentation**: Applied to boost minority class diversity

## Key Innovations

1. **Intelligent Balancing Strategy**: Different approaches for majority vs. minority classes
2. **Targeted Augmentation**: Focuses on underrepresented categories
3. **Weighted Loss Function**: Penalizes misclassification of minority classes more heavily
4. **Comprehensive Evaluation**: Multiple metrics and per-class analysis
5. **Memory Optimization**: Gradient checkpointing and mixed precision training

## Files Description

- `pan2025_notebook.ipynb`: Complete training pipeline with detailed documentation
- `dataset_balanceado_roberta.csv`: Preprocessed and balanced training data
- `RoBERTa_IA_Final/`: Directory containing the final trained model and tokenizer
- `model_info.json`: Metadata about the trained model and label mappings

## Reproducibility

To ensure reproducible results:
- Fixed random seeds (42) throughout the pipeline
- Deterministic train/validation/test splits
- Stratified sampling maintains class proportions
- Detailed logging of all preprocessing steps

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (NVIDIA GTX 1080 or equivalent)
- **RAM**: 16GB system memory
- **Storage**: 10GB free space

### Recommended Setup
- **GPU**: NVIDIA RTX 3080/4080 or better (12GB+ VRAM)
- **RAM**: 32GB system memory
- **Storage**: SSD with 20GB+ free space

## Contributing

We welcome contributions to improve the model performance or extend the approach:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pan2025-collaborative-classification,
  title={Human-AI Collaborative Text Classification for PAN 2025},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-username/pan2025-collaborative-text-classification}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PAN 2025 Organizers**: For providing the dataset and challenge framework
- **Hugging Face**: For the excellent Transformers library
- **RoBERTa Authors**: For the robust pre-trained model
- **scikit-learn**: For comprehensive machine learning utilities

## Contact

For questions, issues, or collaboration opportunities:

- **Email**: your.email@domain.com
- **GitHub Issues**: [Open an issue](https://github.com/your-username/pan2025-collaborative-text-classification/issues)
- **PAN 2025 Forum**: [Competition discussion](https://pan.webis.de/)

---

**Competition**: PAN 2025 - Authorship Analysis  
**Task**: Subtask 2 - Human-AI Collaborative Text Classification  
**Model**: RoBERTa-base fine-tuned for 6-class classification  
**Framework**: PyTorch + Hugging Face Transformers
