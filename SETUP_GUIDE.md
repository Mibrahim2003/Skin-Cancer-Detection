# DermaOps Setup & Installation Guide

Welcome to **DermaOps**! This guide will walk you through setting up the project locally from scratch. Since large files like datasets and trained models are not stored in the Git repository, you will need to generate them using the provided pipelines.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
2.  **Git**: [Download Git](https://git-scm.com/downloads)
3.  **Docker Desktop** (Optional, for containerized deployment): [Download Docker](https://www.docker.com/products/docker-desktop/)
4.  **NVIDIA GPU Drivers** (Optional, but highly recommended for training): If you have an NVIDIA GPU, ensure you have the latest drivers and CUDA toolkit installed.

---

## 🚀 Step-by-Step Installation

### 1. Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/Mibrahim2003/Skin-Cancer-Detection.git
cd Skin-Cancer-Detection
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

Since the dataset (HAM10000) and the trained model (`best_model_resnet.pth`) are too large for GitHub, you need to run the automated pipeline to download the data and train the model locally.

### Option A: Run the Full Automated Pipeline (Recommended)

We use **Prefect** to orchestrate the entire process. This single command will:
1.  Download the HAM10000 dataset from Kaggle.
2.  Preprocess the images (resize, normalize, split).
3.  Train the ResNet50 model (this may take 30-60 mins depending on your GPU).
4.  Evaluate the model and save the best checkpoint.

Run the following command:

```bash
python -m src.pipelines.orchestration
```

*Note: The first time you run this, it will download ~3GB of data. Ensure you have a stable internet connection.*

### Option B: Manual Steps (Advanced)

If you prefer to run steps individually:

1.  **Download Data**:
    ```bash
    python -m src.pipelines.orchestration --stage ingest
    ```
2.  **Preprocess Data**:
    ```bash
    python -m src.pipelines.orchestration --stage preprocess
    ```
3.  **Train Model**:
    ```bash
    python -m src.pipelines.orchestration --stage train
    ```

---

## 🖥️ Running the Application

Once the pipeline completes and you see `models/best_model_resnet.pth`, you are ready to launch the app.

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
