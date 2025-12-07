# DermaOps - Dermatology Image Classification System

An industry-grade MLOps pipeline for skin lesion classification using the HAM10000 dataset.

## Project Structure

```
DermaOps/
├── .github/workflows/      # CI/CD Pipeline
├── data/
│   ├── raw/                # HAM10000 dataset
│   └── processed/          # Resized images
├── models/                 # Saved .h5 or .pt models
├── notebooks/              # For EDA and Experiments
├── src/
│   ├── api/                # FastAPI code
│   ├── pipelines/          # Prefect flows
│   └── ui/                 # Streamlit app
├── tests/                  # DeepChecks & Unit tests
├── Dockerfile
├── requirements.txt
└── report.md               # The final report file
```

## Setup

### Prerequisites
- Python 3.10+
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DermaOps
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the API
```bash
uvicorn src.api.main:app --reload
```

### Running the Streamlit UI
```bash
streamlit run src/ui/app.py
```

### Running with Docker
```bash
docker build -t dermaops .
docker run -p 8000:8000 -p 8501:8501 dermaops
```

## License
MIT
