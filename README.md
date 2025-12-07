# DermaOps: AI-Powered Skin Cancer Detection

![DermaOps Banner](https://img.shields.io/badge/DermaOps-AI%20Dermatology-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**DermaOps** is an "Industry Grade" end-to-end machine learning system designed to assist dermatologists in the early detection of skin cancer. It leverages deep learning (ResNet50) to classify skin lesions into 7 diagnostic categories with high sensitivity for Melanoma.

The system is built with a microservices architecture, featuring a **FastAPI** backend for inference, a **Streamlit** frontend for user interaction, and **Prefect** for robust MLOps pipeline orchestration.

---

## 🚀 Features

- **Deep Learning Model**: Fine-tuned ResNet50 achieving **~76% Accuracy** and **0.75 F1-Score** on the HAM10000 dataset.
- **Real-Time Inference**: Optimized inference engine delivering predictions in **<50ms**.
- **MLOps Pipeline**: Automated data ingestion, preprocessing, training, and evaluation pipelines orchestrated by **Prefect 3.x**.
- **Resilience**: Built-in retry logic, error handling, and Discord notifications for pipeline monitoring.
- **Interactive UI**: User-friendly **Streamlit** dashboard for image upload, visualization, and Grad-CAM explainability.
- **Containerized**: Fully Dockerized application for consistent deployment across environments.
- **REST API**: Robust **FastAPI** endpoints for model serving and health checks.

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+, PyTorch, torchvision, pandas, numpy, scikit-learn
- **Orchestration**: Prefect
- **Serving**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Infrastructure**: Docker, Docker Compose
- **Monitoring**: Discord Webhooks (Alerts)

---

## 🏁 How to Run

### Prerequisites
- Docker & Docker Compose installed on your machine.

### Quick Start (Recommended)

You can launch the entire stack (API + UI) with a single command:

```bash
# 1. Clone the repository
git clone https://github.com/Mibrahim2003/Skin-Cancer-Detection.git
cd Skin-Cancer-Detection

# 2. Build and Launch Services
docker-compose up --build
```

Once running, access the services at:
- **Web App (UI):** [http://localhost:8501](http://localhost:8501)
- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prefect Dashboard:** [http://localhost:4200](http://localhost:4200) (if enabled)

### Manual Setup (Local Development)

If you prefer running without Docker:

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Pipeline (Optional - to retrain model)
python -m src.pipelines.orchestration

# 4. Start API Server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &

# 5. Start Streamlit UI
streamlit run src/ui/app.py --server.port 8501
```

---

## 📂 Project Structure

```
DermaOps/
├── data/               # Dataset (Raw & Processed)
├── models/             # Trained PyTorch models
├── notebooks/          # Jupyter notebooks for experimentation
├── reports/            # Generated metrics and figures
├── src/
│   ├── api/            # FastAPI backend code
│   ├── models/         # Model architecture & training scripts
│   ├── pipelines/      # Prefect orchestration flows
│   ├── ui/             # Streamlit frontend application
│   └── utils/          # Helper functions (alerts, logging)
├── tests/              # Unit tests
├── Dockerfile.api      # API Container definition
├── Dockerfile.ui       # UI Container definition
├── docker-compose.yml  # Service orchestration
└── report.md           # Detailed project report
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 72.85% |
| **Weighted F1** | 0.7486 |
| **Melanoma Recall** | 54.95% |

For a detailed analysis of the model's performance, methodology, and architecture, please refer to the [Project Report](report.md).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Developed by the DermaOps Team - December 2025*
