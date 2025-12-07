# DermaOps Setup & Installation Guide

Welcome to **DermaOps**! This guide will walk you through setting up the project locally.

**Good News:** The processed dataset and the trained model are included in this repository (via Git LFS), so you can run the application immediately without needing to retrain!

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
2.  **Git**: [Download Git](https://git-scm.com/downloads)
3.  **Git LFS**: Required to download the model file. [Download Git LFS](https://git-lfs.com/)
4.  **Docker Desktop** (Optional, for containerized deployment): [Download Docker](https://www.docker.com/products/docker-desktop/)

---

## 🚀 Step-by-Step Installation

### 1. Clone the Repository

Open your terminal or command prompt and run:

```bash
# Install Git LFS first if you haven't
git lfs install

# Clone the repo
git clone https://github.com/Mibrahim2003/Skin-Cancer-Detection.git
cd Skin-Cancer-Detection

# Pull the large model file
git lfs pull
```

### 2. Set Up a Virtual Environment

It is best practice to use a virtual environment to manage dependencies.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

---

## 📦 Data & Model Setup

The repository comes pre-packaged with:
-   **Processed Data**: `data/processed/` (Train/Val/Test splits ready for use)
-   **Trained Model**: `models/best_model_resnet.pth` (ResNet50 fine-tuned)

**You do NOT need to run the training pipeline to start the app.**

### (Optional) Retraining from Scratch

Only run this if you want to reproduce the training process or download the raw HAM10000 dataset from Kaggle:

```bash
# This will download raw data, preprocess it, and retrain the model
python -m src.pipelines.orchestration
```

---

## 🖥️ Running the Application

You are ready to launch the app immediately.

### Method 1: Using Docker (Easiest)

If you have Docker installed, you can spin up the entire stack (API + UI) with one command:

```bash
docker-compose up --build
```

- **UI**: http://localhost:8501
- **API**: http://localhost:8000

### Method 2: Running Locally (No Docker)

You will need two terminal windows.

**Terminal 1: Start the API Backend**
```bash
# Make sure your venv is activated
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Start the Streamlit UI**
```bash
# Make sure your venv is activated
streamlit run src/ui/app.py --server.port 8501
```

---

## 🧪 Verifying the Installation

1.  Open your browser to [http://localhost:8501](http://localhost:8501).
2.  You should see the **DermaOps** dashboard.
3.  Go to the `data/processed/test` folder (created during the pipeline run) and pick an image.
4.  Upload it to the dashboard and click **Analyze**.
5.  If you see a prediction and confidence score, congratulations! Your system is fully operational.

---

## ❓ Troubleshooting

**Q: I get a "CUDA out of memory" error during training.**
A: Try reducing the batch size. You can do this by modifying `src/pipelines/training_pipeline.py` or running:
```bash
python -m src.pipelines.orchestration --batch-size 16
```

**Q: The download fails.**
A: The pipeline uses `kagglehub`. Ensure you have internet access. If it persists, you may need to set up your Kaggle API credentials manually in `~/.kaggle/kaggle.json`.

**Q: `ModuleNotFoundError`?**
A: Ensure you activated your virtual environment (`source venv/bin/activate`) before running commands.
