# BioASQ 13B Challenge Project

This project implements a system for biomedical question answering and information retrieval, participating in the BioASQ 13B challenge. The project is divided into two main phases: Phase A (Document Retrieval) and Phase B (Question Answering).

## Project Structure

```
BioASQ 13B/
├── PhaseA/                    # Document Retrieval Phase
│   ├── services/             # External service integrations
│   ├── data/                 # Data storage and results
│   ├── processing/           # Core processing modules
│   ├── main.py              # Main execution script
│   ├── analyze_metrics.py   # Metrics analysis and visualization
│   └── requirements.txt     # Phase A dependencies
│
└── PhaseB/                   # Question Answering Phase
    ├── baseline_QA.py       # Baseline QA implementation
    └── test_data/          # Test datasets
```

## Phase A: Document Retrieval

Phase A focuses on retrieving relevant biomedical documents for given queries. It includes:
- Query processing and keyword extraction
- PubMed document retrieval
- Result re-ranking using various methods (BM25, TF-IDF, PubMedBERT)
- Performance evaluation and metrics analysis

### Key Features
- Integration with PubMed API
- Multiple re-ranking strategies
- Comprehensive metrics analysis
- Visualization of results

## Phase B: Question Answering

Phase B implements question answering capabilities for biomedical questions, including:
- Baseline QA system implementation
- Processing of BioASQ test datasets
- Answer generation and evaluation

## Installation

1. Ensure you have Python 3.8 or higher installed
2. Install dependencies for both phases:

```bash
# Install Phase A dependencies
cd PhaseA
pip install -r requirements.txt

# Install Phase B dependencies
cd ../PhaseB
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the PhaseA directory with the following variables:

```
PUBMED_API_KEY=<your_pubmed_api_key>
GEMINI_API_KEY=<your_gemini_api_key>
TYPE_EVALUATION=TRAIN  # Change to TEST for test data
RERANKER_TYPE=BM25     # Options: TF-IDF, BM25, PubMedBERT
RERANKER_QUERY=KEYWORDS # Options: BODY, BODY + KEYWORDS, KEYWORDS
```

## Usage

### Phase A: Document Retrieval

1. Run the main processing script:
```bash
cd PhaseA
python main.py
```

2. Analyze metrics and generate visualizations:
```bash
python analyze_metrics.py
```

### Phase B: Question Answering

1. Run the baseline QA system:
```bash
cd PhaseB
python baseline_QA.py
```

## Requirements

- Python 3.8 or higher
- Internet connection for API access
- Valid API keys for PubMed and Gemini (for Phase A)
- Sufficient disk space for data storage and processing

## Data

- Training and test datasets are provided in the respective data directories
- Results and metrics are stored in the `data/result_data` directory
- Generated reports and visualizations are saved in `data/result_data/reports`

## Notes

- Ensure all required API keys are properly configured before running the system
- The system requires significant computational resources for processing large datasets
- Regular backups of important data are recommended 