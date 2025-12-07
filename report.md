# DermaOps Project Report

## Overview
This document contains the project report for DermaOps - a dermatology image classification system using the HAM10000 dataset for skin cancer detection.

---

## 1. Data Quality Observations

### Class Imbalance

- **Observation**: The dataset is highly imbalanced. The class `nv` (Melanocytic nevi) dominates with **6,705 images (66.95%)**, while classes like `df` (Dermatofibroma) have only **115 images (1.15%)**. The imbalance ratio is **58.3:1** between the most and least common classes.

- **Impact**: This severe imbalance means a standard accuracy metric will be misleading (the model could predict "nv" for everything and still get ~67% accuracy). Additionally, rare but critical classes like melanoma (`mel` with 1,113 images, 11.11%) could be underrepresented in predictions.

- **Mitigation Strategy**: I have implemented **Stratified Splitting** in the preprocessing pipeline to ensure the train/validation/test sets all maintain the same class proportions. I will also use **Class Weights** during training to penalize misclassification of minority classes more heavily.

### Missing Data

- **Observation**: The metadata check revealed **57 missing values** in the dataset, all in the `age` column (0.57% of records).

- **Resolution**: The missing age values do not affect our image classification task since we are training on image data only. The core features (`image_id`, `dx` diagnosis label, and image files) have **zero missing values**. No data was dropped as a result.

### Class Distribution Summary

| Diagnosis | Full Name | Count | Percentage |
|-----------|-----------|-------|------------|
| nv | Melanocytic Nevi | 6,705 | 66.95% |
| mel | Melanoma | 1,113 | 11.11% |
| bkl | Benign Keratosis | 1,099 | 10.97% |
| bcc | Basal Cell Carcinoma | 514 | 5.13% |
| akiec | Actinic Keratoses | 327 | 3.27% |
| vasc | Vascular Lesions | 142 | 1.42% |
| df | Dermatofibroma | 115 | 1.15% |
| **Total** | | **10,015** | **100%** |

### Dataset Split (Stratified)

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 8,011 | 80% |
| Validation | 1,002 | 10% |
| Test | 1,002 | 10% |

*Class proportions are maintained across all splits.*

![Class Distribution](reports/figures/class_distribution.png)

---

## 2. ML Experimentation & Model Comparison

### Experiment A: Baseline Model (Random Forest)

A Random Forest classifier was trained as a baseline to establish a performance floor for comparison.

| Metric | Value |
|--------|-------|
| Validation Accuracy | 69.36% |
| F1 Score (Weighted) | 0.61 |
| F1 Score (Macro) | 0.24 |
| Training Time | 7.85 seconds |

**Observations**: The baseline model heavily favors the majority class (`nv` with 98.36% recall) while completely failing on minority classes (`df` with 0% recall, `mel` with 0.9% recall). The low macro F1 score (0.24) indicates poor performance across all classes.

### Experiment B: ResNet50 with Transfer Learning

#### Phase 1: Feature Extraction (Frozen Backbone)

- **Approach**: Pretrained ResNet50 with frozen convolutional layers, only training the classifier head.
- **Best F1 Score**: 0.7314 (Epoch 12)
- **Observation**: Using a frozen Feature Extractor (Linear Probing) limited the model to ~73% F1 score. This is because the features learned from ImageNet (generic objects like cats, dogs, cars) were not specific enough for dermatoscopy images.

#### Phase 2: Fine-Tuning (Unfreezing layer4)

- **Approach**: Unfreezing the final convolutional block (`layer4`) and fine-tuning with a lower learning rate (1e-5).
- **Best F1 Score**: 0.7784 (Epoch 30)
- **Improvement**: +4.70% F1 score over Phase 1

**Improvement Strategy**: By unfreezing the final convolutional block (`layer4`) and fine-tuning with a lower learning rate (1×10⁻⁵), the model was able to adapt its feature maps to specific skin textures, boosting the F1-score from 73.14% to **77.84%**.

### Model Comparison Summary

| Model | Val Accuracy | F1 Score (Weighted) | Improvement |
|-------|--------------|---------------------|-------------|
| Random Forest (Baseline) | 69.36% | 0.61 | - |
| ResNet50 (Frozen) | 70.46% | 0.73 | +19.7% F1 |
| ResNet50 (Fine-tuned) | **76.35%** | **0.78** | **+27.9% F1** |

### Key Findings

1. **Transfer Learning Works**: The pretrained ResNet50 significantly outperforms the traditional ML baseline, validating the use of deep learning for medical image classification.

2. **Fine-Tuning is Essential**: Simply using ImageNet features (frozen backbone) is insufficient for domain-specific tasks like dermatoscopy. Fine-tuning the last convolutional block improved F1 by 4.7%.

3. **Class Weights are Critical**: Using inverse-frequency class weights in the loss function helped the model pay attention to minority classes like `df` (Dermatofibroma) which has only 92 training samples.

### Training Curves

![Loss Curve](reports/figures/loss_curve.png)
![Accuracy Curve](reports/figures/accuracy_curve.png)
![F1 Score Curve](reports/figures/f1_curve.png)

---

## 3. Overfitting & Underfitting Patterns

### Analysis of Learning Curves

**Phase 1 (Feature Extraction):** The model quickly learned high-level features, stabilizing at a Validation F1-score of **0.7314**. The gap between Train and Val loss remained consistent, suggesting the model had not yet overfitted but was limited by the frozen weights (Underfitting relative to potential). The training loss plateaued around ~1.00, indicating the model couldn't extract more information without adapting its feature maps.

**Phase 2 (Fine-Tuning):** By unfreezing the final layers (`layer4`), we observed an immediate drop in Training Loss (from ~1.00 down to ~0.64 by epoch 30). The Validation F1-score improved from **0.7314 to 0.7784** (+4.70%), proving that the model successfully adapted to the domain-specific features of dermatoscopy without significant overfitting. The validation loss continued decreasing throughout fine-tuning (0.76 → 0.60), confirming healthy generalization.

### Key Indicators

| Phase | Train Loss (Final) | Val Loss (Final) | Val F1 (Best) | Status |
|-------|-------------------|------------------|---------------|--------|
| Feature Extraction | 1.004 | 0.762 | 0.7314 | Underfitting (limited capacity) |
| Fine-Tuning | 0.642 | 0.601 | 0.7784 | Good fit (balanced) |

---

## 4. Model Comparison & Selection

We compared three versions of the model:

| Model Version | Accuracy | F1-Score (Weighted) | F1-Score (Macro) | Observations |
|---------------|----------|---------------------|------------------|--------------|
| Baseline (Random Forest) | 69.36% | 0.61 | 0.24 | High bias towards majority class ('nv'). Failed to detect Melanoma (0.9% recall). |
| ResNet50 (Frozen) | 70.46% | 0.73 | ~0.50 | Good generalization, but limited by ImageNet features. |
| ResNet50 (Fine-Tuned) | **76.35%** | **0.78** | ~0.59 | **Best Model.** Significant improvement in minority class detection. |

### Best-Performing Model Selection

The **Fine-Tuned ResNet50** is selected for deployment. The fine-tuning process yielded a **~5% improvement in F1-score** over the frozen model, which is critical for reducing false negatives in a medical context.

### Test Set Validation

The final model was evaluated on the held-out test set (1,002 images never seen during training):

| Metric | Test Set Value |
|--------|----------------|
| **Test Accuracy** | 72.85% |
| **F1 Score (Weighted)** | 0.7486 |
| **F1 Score (Macro)** | 0.5891 |

### Per-Class Test Performance

| Class | Full Name | Precision | Recall | F1-Score | Support |
|-------|-----------|-----------|--------|----------|---------|
| akiec | Actinic Keratoses | 0.4314 | 0.6667 | 0.5238 | 33 |
| bcc | Basal Cell Carcinoma | 0.5588 | 0.7451 | 0.6387 | 51 |
| bkl | Benign Keratosis | 0.5182 | 0.6455 | 0.5749 | 110 |
| df | Dermatofibroma | 0.3333 | **0.8333** | 0.4762 | 12 |
| mel | **Melanoma** | 0.4178 | **0.5495** | 0.4747 | 111 |
| nv | Melanocytic Nevi | **0.9503** | 0.7690 | **0.8501** | 671 |
| vasc | Vascular Lesions | 0.4444 | **0.8571** | 0.5854 | 14 |

### Critical Improvements Over Baseline

| Class | Baseline Recall | Fine-Tuned Recall | Improvement |
|-------|-----------------|-------------------|-------------|
| mel (Melanoma) | 0.9% | **54.95%** | +54× |
| df (Dermatofibroma) | 0.0% | **83.33%** | ∞ |
| vasc (Vascular) | 7.1% | **85.71%** | +12× |

The fine-tuned model dramatically improved detection of rare but clinically important conditions, especially Melanoma which is critical for early cancer detection.

### Visual Proof

![Confusion Matrix](reports/figures/confusion_matrix_test.png)

The confusion matrix above shows the model's predictions vs. true labels on the test set. The diagonal elements represent correct classifications, while off-diagonal elements show misclassifications. The normalized view (right) shows recall percentages per class.

### Model Explainability (Grad-CAM)

![Explainability Samples](reports/figures/explainability_samples.png)

Grad-CAM visualizations show where the model focuses when classifying Melanoma samples. The heatmaps highlight that the model attends to lesion boundaries and texture patterns, which are clinically relevant features for diagnosis.

---

## 5. Orchestration & Reliability (Prefect Pipeline)

### Pipeline Architecture

The DermaOps ML pipeline is orchestrated using **Prefect 3.x**, providing automated execution, failure recovery, and monitoring.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DermaOps-ML-Pipeline                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ ingest_data   │ →  │ preprocess_   │ →  │ train_model   │
│   (Task)      │    │    data       │    │   (Task)      │
│               │    │   (Task)      │    │               │
│ retries: 3    │    │ retries: 1    │    │ retries: 1    │
│ backoff: exp  │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
                                                  │
                    ┌─────────────────────────────┤
                    ▼                             ▼
           ┌───────────────┐             ┌───────────────┐
           │ evaluate_     │             │ send_         │
           │    model      │ ─────────→  │ notification  │
           │   (Task)      │             │   (Task)      │
           │ retries: 1    │             │               │
           └───────────────┘             └───────────────┘
```

### Failure Handling & Retry Strategy

| Task | Retries | Backoff | Rationale |
|------|---------|---------|-----------|
| `ingest_data` | 3 | [30s, 60s, 120s] | Network failures during Kaggle download are transient |
| `preprocess_data` | 1 | 60s | Disk/memory issues; retry once before failing |
| `train_model` | 1 | 60s | GPU OOM or CUDA errors may recover after memory release |
| `evaluate_model` | 1 | - | Quick task, one retry sufficient |

### Smart Checkpointing

The pipeline implements **idempotent execution** - it skips already-completed stages:

```python
# Example: Ingestion task checks for existing data
if not force_download and data_path.exists():
    if verify_dataset(target_dir):
        return {"status": "skipped", "message": "Data already present"}
```

**Benefits:**
- ✅ Re-running the pipeline skips completed stages
- ✅ Failures resume from the failed stage, not the beginning
- ✅ `--force-*` flags allow explicit re-execution when needed

### Notification System

Upon pipeline completion (success or failure), the pipeline:

1. **Creates a Prefect Artifact** (visible in Prefect UI)
2. **Saves a local markdown report** to `reports/pipeline_notification.md`
3. **Logs detailed summary** to console

**Success Notification Example:**
```markdown
# ✅ DermaOps Pipeline Completed Successfully

## Training Results
- Best F1 Score: 0.7784
- Training Duration: 45.2 minutes

## Test Set Performance
- Accuracy: 0.7285
- F1 Score (Weighted): 0.7486
```

### CLI Usage

```bash
# Run full pipeline (skips existing data/models)
python -m src.pipelines.orchestration

# Force retraining from scratch
python -m src.pipelines.orchestration --force-retrain

# Run specific stages
python -m src.pipelines.orchestration --stage ingest
python -m src.pipelines.orchestration --stage preprocess
python -m src.pipelines.orchestration --stage train
python -m src.pipelines.orchestration --stage evaluate

# Customize training
python -m src.pipelines.orchestration --epochs 20 --finetune-epochs 10 --batch-size 64
```

### Tested Scenarios

| Scenario | Result |
|----------|--------|
| Full pipeline with existing data | ✅ Skipped download, ran preprocessing |
| Evaluation only | ✅ Loaded model, generated metrics |
| Network disconnect during download | ✅ Retried 3x with exponential backoff |
| Force retrain | ✅ Overwrote existing model |

---

*Report content will continue as the project progresses.*
