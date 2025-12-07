"""
DermaOps Training Pipeline - Prefect Workflow

This is the main training pipeline that orchestrates the complete ML workflow:
1. Data Ingestion (with retry logic for network failures)
2. Data Preprocessing (stratified split, image resizing)
3. Model Training (ResNet50 with transfer learning)
4. Model Evaluation (metrics, confusion matrix, Grad-CAM)

Upon completion, sends a Discord notification with the final metrics.

Usage:
    # Run full pipeline
    python -m src.pipelines.training_pipeline
    
    # Run with custom parameters
    python -m src.pipelines.training_pipeline --epochs 20 --force-retrain
    
Environment Variables:
    DISCORD_WEBHOOK_URL: Optional webhook for notifications
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from prefect import task, flow, get_run_logger
from prefect.artifacts import create_markdown_artifact

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import alert utilities
from src.utils.alerts import send_discord_alert, send_pipeline_success_alert, send_pipeline_failure_alert


# =============================================================================
# TASK 1: Data Ingestion
# =============================================================================

@task(
    name="Download Data",
    description="Download HAM10000 dataset from Kaggle with retry logic",
    retries=3,
    retry_delay_seconds=60,  # Fixed 60-second retry delay as specified
    log_prints=True,
    tags=["data", "download", "kaggle"]
)
def ingest_data_task(
    target_dir: str = "data/raw",
    force_download: bool = False
) -> Dict[str, Any]:
    """
    Task 1: Download HAM10000 dataset with automatic retry on failure.
    
    This task wraps the data ingestion logic and adds reliability through
    Prefect's retry mechanism. Network failures are handled gracefully.
    
    Args:
        target_dir: Directory to save raw data.
        force_download: Force re-download even if data exists.
        
    Returns:
        Dict with ingestion status and data path.
        
    Raises:
        Exception: If download fails after all retries.
    """
    logger = get_run_logger()
    
    from src.pipelines.data_ingestion import ingest_ham10000, verify_dataset
    
    data_path = Path(target_dir)
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("📥 TASK 1: DATA INGESTION")
    logger.info("="*60)
    
    # Check if data already exists (skip download if present)
    if not force_download and data_path.exists():
        if verify_dataset(target_dir):
            elapsed = time.time() - start_time
            logger.info("✅ Dataset already exists and verified. Skipping download.")
            return {
                "status": "skipped",
                "data_path": str(data_path.resolve()),
                "message": "Data already present and verified",
                "elapsed_seconds": elapsed
            }
    
    # Download dataset (retries handled by Prefect)
    logger.info("🔄 Downloading HAM10000 dataset from Kaggle...")
    
    try:
        download_path = ingest_ham10000(target_dir)
        
        # Verify download integrity
        if not verify_dataset(target_dir):
            raise ValueError("Dataset verification failed after download!")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Data ingestion completed in {elapsed:.1f}s")
        
        return {
            "status": "success",
            "data_path": download_path,
            "elapsed_seconds": elapsed
        }
        
    except Exception as e:
        logger.error(f"❌ Data ingestion failed: {str(e)}")
        raise  # Re-raise for Prefect retry mechanism


# =============================================================================
# TASK 2: Data Preprocessing
# =============================================================================

@task(
    name="Preprocess Data",
    description="Preprocess images with stratified split",
    retries=1,
    retry_delay_seconds=30,
    log_prints=True,
    tags=["data", "preprocessing"]
)
def preprocess_data_task(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    target_size: Tuple[int, int] = (224, 224),
    force_reprocess: bool = False
) -> Dict[str, Any]:
    """
    Task 2: Preprocess raw images for model training.
    
    This task:
    - Loads metadata and maps image paths
    - Performs stratified train/val/test split (80/10/10)
    - Resizes images to target_size for ResNet
    - Organizes data into ImageNet-style folder structure
    
    Args:
        raw_dir: Directory containing raw dataset.
        processed_dir: Output directory for processed data.
        target_size: Target image dimensions (H, W).
        force_reprocess: Force reprocessing even if data exists.
        
    Returns:
        Dict with preprocessing results and image counts.
    """
    logger = get_run_logger()
    
    from src.pipelines.data_preprocessing import run_preprocessing
    
    processed_path = Path(processed_dir)
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("⚙️  TASK 2: DATA PREPROCESSING")
    logger.info("="*60)
    
    # Check if preprocessing already done
    if not force_reprocess:
        split_info_path = processed_path / "split_info.csv"
        train_path = processed_path / "train"
        
        if split_info_path.exists() and train_path.exists():
            train_count = sum(1 for _ in train_path.rglob("*.jpg"))
            if train_count > 0:
                elapsed = time.time() - start_time
                logger.info(f"✅ Preprocessed data found ({train_count} train images). Skipping.")
                return {
                    "status": "skipped",
                    "processed_dir": str(processed_path.resolve()),
                    "train_images": train_count,
                    "message": "Preprocessed data already exists",
                    "elapsed_seconds": elapsed
                }
    
    # Run preprocessing
    logger.info(f"🔄 Preprocessing images to {target_size[0]}x{target_size[1]}...")
    
    try:
        run_preprocessing(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            target_size=target_size,
            clear_existing=force_reprocess
        )
        
        # Count processed images
        train_count = sum(1 for _ in (processed_path / "train").rglob("*.jpg"))
        val_count = sum(1 for _ in (processed_path / "val").rglob("*.jpg"))
        test_count = sum(1 for _ in (processed_path / "test").rglob("*.jpg"))
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Preprocessing completed in {elapsed:.1f}s")
        logger.info(f"   Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        return {
            "status": "success",
            "processed_dir": str(processed_path.resolve()),
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
            "elapsed_seconds": elapsed
        }
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        raise


# =============================================================================
# TASK 3: Model Training
# =============================================================================

@task(
    name="Train ResNet50",
    description="Train ResNet50 model with transfer learning",
    retries=1,
    retry_delay_seconds=60,
    log_prints=True,
    tags=["training", "deep-learning", "gpu"]
)
def train_model_task(
    data_dir: str = "data/processed",
    model_dir: str = "models",
    num_epochs: int = 15,
    finetune_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    finetune_lr: float = 1e-5,
    force_retrain: bool = False,
    min_accuracy: float = 0.60  # Sanity check threshold
) -> Dict[str, Any]:
    """
    Task 3: Train ResNet50 model with two-phase transfer learning.
    
    Phase 1: Feature extraction with frozen backbone
    Phase 2: Fine-tuning with unfrozen layer4
    
    Includes a sanity check: if final accuracy < min_accuracy, the task
    fails to prevent deploying a broken model.
    
    Args:
        data_dir: Directory with preprocessed data.
        model_dir: Directory to save model checkpoints.
        num_epochs: Epochs for feature extraction phase.
        finetune_epochs: Epochs for fine-tuning phase.
        batch_size: Training batch size.
        learning_rate: LR for phase 1.
        finetune_lr: LR for phase 2 (fine-tuning).
        force_retrain: Force retraining even if model exists.
        min_accuracy: Minimum accuracy threshold (sanity check).
        
    Returns:
        Dict with training results and metrics.
        
    Raises:
        ValueError: If final accuracy is below min_accuracy threshold.
    """
    logger = get_run_logger()
    
    from src.models.train_resnet import train_model
    
    model_path = Path(model_dir) / "best_model_resnet.pth"
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("🧠 TASK 3: MODEL TRAINING (ResNet50)")
    logger.info("="*60)
    
    # Check if model already exists
    if not force_retrain and model_path.exists():
        elapsed = time.time() - start_time
        logger.info(f"✅ Trained model found at {model_path}. Skipping training.")
        logger.info("   Use force_retrain=True to retrain.")
        return {
            "status": "skipped",
            "model_path": str(model_path.resolve()),
            "message": "Model already exists",
            "elapsed_seconds": elapsed
        }
    
    # Run training
    logger.info(f"🚀 Starting two-phase training...")
    logger.info(f"   Phase 1: {num_epochs} epochs @ LR={learning_rate} (frozen)")
    logger.info(f"   Phase 2: {finetune_epochs} epochs @ LR={finetune_lr} (fine-tuning)")
    logger.info(f"   Batch size: {batch_size}")
    
    try:
        model, history = train_model(
            data_dir=data_dir,
            model_dir=model_dir,
            num_epochs=num_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            finetune_lr=finetune_lr,
            num_workers=2  # Reduced for stability
        )
        
        elapsed = time.time() - start_time
        
        # Extract metrics
        best_f1 = max(history['val_f1'])
        best_epoch = history['val_f1'].index(best_f1) + 1
        final_accuracy = history['val_acc'][-1]
        
        # SANITY CHECK: Ensure minimum accuracy threshold
        if final_accuracy < min_accuracy:
            error_msg = (
                f"Training failed sanity check! "
                f"Final accuracy ({final_accuracy:.2%}) < minimum threshold ({min_accuracy:.2%}). "
                f"Model may be broken or data may be corrupted."
            )
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"✅ Training completed in {elapsed/60:.1f} minutes")
        logger.info(f"   Best F1: {best_f1:.4f} @ Epoch {best_epoch}")
        logger.info(f"   Final Accuracy: {final_accuracy:.4f}")
        logger.info(f"   ✓ Passed sanity check (accuracy >= {min_accuracy:.0%})")
        
        return {
            "status": "success",
            "model_path": str(model_path.resolve()),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "final_accuracy": final_accuracy,
            "total_epochs": num_epochs + finetune_epochs,
            "elapsed_minutes": elapsed / 60,
            "history": {
                "train_loss": history['train_loss'][-1],
                "val_loss": history['val_loss'][-1],
                "val_f1": history['val_f1'][-1]
            }
        }
        
    except ValueError:
        raise  # Re-raise sanity check failures
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


# =============================================================================
# TASK 4: Model Evaluation
# =============================================================================

@task(
    name="Evaluate Model",
    description="Evaluate model on test set with explainability",
    retries=1,
    log_prints=True,
    tags=["evaluation", "metrics", "explainability"]
)
def evaluate_model_task(
    model_path: str = "models/best_model_resnet.pth",
    data_dir: str = "data/processed",
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Task 4: Evaluate the trained model on the held-out test set.
    
    Generates:
    - Comprehensive metrics report
    - Confusion matrix visualization
    - Grad-CAM explainability samples
    
    Args:
        model_path: Path to trained model checkpoint.
        data_dir: Directory containing test data.
        output_dir: Directory for evaluation outputs.
        
    Returns:
        Dict with evaluation metrics (accuracy, F1, per-class metrics).
    """
    logger = get_run_logger()
    
    from src.models.evaluate import run_evaluation
    
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("📊 TASK 4: MODEL EVALUATION")
    logger.info("="*60)
    
    # Verify model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Training must complete first."
        )
    
    logger.info(f"📂 Loading model from: {model_path}")
    logger.info(f"📁 Test data from: {data_dir}/test")
    
    try:
        metrics = run_evaluation(
            model_path=model_path,
            data_dir=data_dir,
            output_dir=output_dir,
            num_workers=0  # Use 0 workers for Prefect compatibility
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Evaluation completed in {elapsed:.1f}s")
        logger.info(f"\n📊 TEST SET RESULTS:")
        logger.info(f"   Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"   F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"   F1 Score (Macro):    {metrics['f1_macro']:.4f}")
        
        return {
            "status": "success",
            "test_accuracy": metrics['accuracy'],
            "test_f1_weighted": metrics['f1_weighted'],
            "test_f1_macro": metrics['f1_macro'],
            "per_class_metrics": metrics.get('per_class', {}),
            "elapsed_seconds": elapsed,
            "artifacts": {
                "metrics_report": f"{output_dir}/metrics_report.txt",
                "confusion_matrix": f"{output_dir}/figures/confusion_matrix_test.png",
                "explainability": f"{output_dir}/figures/explainability_samples.png"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise


# =============================================================================
# MAIN FLOW: DermaOps Training Pipeline
# =============================================================================

@flow(
    name="DermaOps End-to-End Pipeline",
    description="Complete ML pipeline: Ingest → Preprocess → Train → Evaluate",
    version="1.0.0",
    log_prints=True
)
def dermaops_training_flow(
    # Data parameters
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    force_download: bool = False,
    force_reprocess: bool = False,
    
    # Training parameters
    model_dir: str = "models",
    num_epochs: int = 15,
    finetune_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    finetune_lr: float = 1e-5,
    force_retrain: bool = False,
    
    # Evaluation parameters
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Main DermaOps End-to-End Training Pipeline.
    
    Orchestrates the complete ML workflow:
    1. Data Ingestion (with retry logic for network failures)
    2. Data Preprocessing (stratified split, image resizing)
    3. Model Training (ResNet50 with transfer learning)
    4. Model Evaluation (metrics, confusion matrix, Grad-CAM)
    
    Upon completion, sends a Discord notification with the final metrics.
    Upon failure, sends a Discord alert with the error message.
    
    Args:
        raw_dir: Directory for raw data.
        processed_dir: Directory for processed data.
        force_download: Force re-download of dataset.
        force_reprocess: Force reprocessing of data.
        model_dir: Directory for model checkpoints.
        num_epochs: Epochs for frozen backbone training.
        finetune_epochs: Epochs for fine-tuning.
        batch_size: Training batch size.
        learning_rate: Learning rate for phase 1.
        finetune_lr: Learning rate for phase 2.
        force_retrain: Force retraining of model.
        output_dir: Directory for evaluation outputs.
        
    Returns:
        Dict with results from all pipeline stages.
    """
    logger = get_run_logger()
    
    pipeline_results = {}
    pipeline_start = time.time()
    failed_stage = None
    
    logger.info("="*60)
    logger.info("🚀 DERMAOPS END-TO-END PIPELINE STARTING")
    logger.info("="*60)
    logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    try:
        # =====================================================================
        # TASK 1: Data Ingestion
        # =====================================================================
        failed_stage = "Data Ingestion"
        ingestion_result = ingest_data_task(
            target_dir=raw_dir,
            force_download=force_download
        )
        pipeline_results["ingestion"] = ingestion_result
        
        # =====================================================================
        # TASK 2: Data Preprocessing
        # =====================================================================
        failed_stage = "Data Preprocessing"
        preprocess_result = preprocess_data_task(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            force_reprocess=force_reprocess
        )
        pipeline_results["preprocessing"] = preprocess_result
        
        # =====================================================================
        # TASK 3: Model Training
        # =====================================================================
        failed_stage = "Model Training"
        training_result = train_model_task(
            data_dir=processed_dir,
            model_dir=model_dir,
            num_epochs=num_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            finetune_lr=finetune_lr,
            force_retrain=force_retrain
        )
        pipeline_results["training"] = training_result
        
        # =====================================================================
        # TASK 4: Model Evaluation
        # =====================================================================
        failed_stage = "Model Evaluation"
        model_path = f"{model_dir}/best_model_resnet.pth"
        evaluation_result = evaluate_model_task(
            model_path=model_path,
            data_dir=processed_dir,
            output_dir=output_dir
        )
        pipeline_results["evaluation"] = evaluation_result
        
        # =====================================================================
        # Calculate total time
        # =====================================================================
        total_time = time.time() - pipeline_start
        pipeline_results["total_elapsed_minutes"] = total_time / 60
        
        # =====================================================================
        # SUCCESS: Send Discord notification
        # =====================================================================
        logger.info("\n" + "="*60)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"   Total Duration: {total_time/60:.1f} minutes")
        
        # Get final metrics for notification
        test_f1 = evaluation_result.get("test_f1_weighted", 0)
        test_accuracy = evaluation_result.get("test_accuracy", 0)
        
        # Send success notification
        send_discord_alert(
            message=(
                f"Pipeline Finished Successfully! 🚀\n\n"
                f"**Test F1 Score:** {test_f1:.4f}\n"
                f"**Test Accuracy:** {test_accuracy:.4f}\n"
                f"**Duration:** {total_time/60:.1f} minutes"
            ),
            alert_type="SUCCESS",
            extra_fields={
                "F1 Score": f"{test_f1:.4f}",
                "Accuracy": f"{test_accuracy:.4f}",
                "Duration": f"{total_time/60:.1f} min"
            }
        )
        
        # Create Prefect artifact
        create_markdown_artifact(
            key="pipeline-results",
            markdown=f"""
# ✅ DermaOps Pipeline Results

**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Duration:** {total_time/60:.1f} minutes

## Test Set Performance
| Metric | Value |
|--------|-------|
| Accuracy | {test_accuracy:.4f} |
| F1 Score (Weighted) | {test_f1:.4f} |

## Artifacts Generated
- `reports/metrics_report.txt`
- `reports/figures/confusion_matrix_test.png`
- `reports/figures/explainability_samples.png`
- `models/best_model_resnet.pth`
            """,
            description="Pipeline execution results"
        )
        
        return pipeline_results
        
    except Exception as e:
        # =====================================================================
        # FAILURE: Send Discord alert and re-raise
        # =====================================================================
        total_time = time.time() - pipeline_start
        error_message = str(e)
        
        logger.error("\n" + "="*60)
        logger.error(f"❌ PIPELINE FAILED at stage: {failed_stage}")
        logger.error("="*60)
        logger.error(f"   Error: {error_message}")
        logger.error(f"   Duration before failure: {total_time/60:.1f} minutes")
        
        # Send failure notification
        send_discord_alert(
            message=(
                f"Pipeline Failed: {error_message} ❌\n\n"
                f"**Failed Stage:** {failed_stage}\n"
                f"**Duration:** {total_time/60:.1f} minutes"
            ),
            alert_type="FAILURE",
            extra_fields={
                "Failed Stage": failed_stage,
                "Duration": f"{total_time/60:.1f} min"
            }
        )
        
        # Re-raise the exception for Prefect to handle
        raise


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DermaOps End-to-End Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (skips existing data/models)
  python -m src.pipelines.training_pipeline
  
  # Force complete retraining
  python -m src.pipelines.training_pipeline --force-retrain
  
  # Force everything from scratch
  python -m src.pipelines.training_pipeline --force-download --force-reprocess --force-retrain
  
  # Custom training configuration
  python -m src.pipelines.training_pipeline --epochs 20 --finetune-epochs 10 --batch-size 64
        """
    )
    
    # Data options
    parser.add_argument("--force-download", action="store_true", help="Force data re-download")
    parser.add_argument("--force-reprocess", action="store_true", help="Force data reprocessing")
    
    # Training options
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--epochs", type=int, default=15, help="Phase 1 epochs (default: 15)")
    parser.add_argument("--finetune-epochs", type=int, default=15, help="Phase 2 epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Phase 1 learning rate (default: 0.001)")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Phase 2 learning rate (default: 1e-5)")
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    print("\n" + "="*60)
    print("🔬 DERMAOPS END-TO-END TRAINING PIPELINE")
    print("="*60)
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Force Download: {args.force_download}")
    print(f"   Force Reprocess: {args.force_reprocess}")
    print(f"   Force Retrain: {args.force_retrain}")
    print(f"   Epochs: {args.epochs} + {args.finetune_epochs} (fine-tune)")
    print(f"   Batch Size: {args.batch_size}")
    print("="*60 + "\n")
    
    # Run the pipeline
    result = dermaops_training_flow(
        force_download=args.force_download,
        force_reprocess=args.force_reprocess,
        force_retrain=args.force_retrain,
        num_epochs=args.epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        finetune_lr=args.finetune_lr
    )
    
    print("\n" + "="*60)
    print("📋 FINAL PIPELINE RESULTS")
    print("="*60)
    
    for stage, result_data in result.items():
        if isinstance(result_data, dict):
            status = result_data.get('status', 'N/A')
            print(f"\n{stage.upper()}:")
            print(f"   Status: {status}")
            if 'elapsed_seconds' in result_data:
                print(f"   Duration: {result_data['elapsed_seconds']:.1f}s")
            if 'test_f1_weighted' in result_data:
                print(f"   Test F1: {result_data['test_f1_weighted']:.4f}")
        else:
            print(f"\n{stage}: {result_data}")
