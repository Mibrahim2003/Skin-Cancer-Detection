"""
Prefect Orchestration Pipeline for DermaOps.

This module orchestrates the complete ML pipeline:
Data Ingestion → Preprocessing → Training → Evaluation

Features:
- Automatic retries for network failures (data download)
- Graceful error handling with detailed logging
- Notifications on pipeline completion/failure
- Checkpointing to skip completed steps
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.states import Completed, Failed

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# TASK: Data Ingestion
# =============================================================================

@task(
    name="ingest_data",
    description="Download HAM10000 dataset from Kaggle",
    retries=3,
    retry_delay_seconds=[30, 60, 120],  # Exponential backoff
    log_prints=True,
    tags=["data", "download"]
)
def ingest_data_task(
    target_dir: str = "data/raw",
    force_download: bool = False
) -> Dict[str, str]:
    """
    Download HAM10000 dataset with retry logic for network failures.
    
    Args:
        target_dir: Directory to save raw data.
        force_download: Force re-download even if data exists.
        
    Returns:
        Dict with status and data path.
    """
    logger = get_run_logger()
    
    from src.pipelines.data_ingestion import ingest_ham10000, verify_dataset
    
    data_path = Path(target_dir)
    
    # Check if data already exists
    if not force_download and data_path.exists():
        if verify_dataset(target_dir):
            logger.info("✅ Dataset already exists and verified. Skipping download.")
            return {
                "status": "skipped",
                "data_path": str(data_path.resolve()),
                "message": "Data already present and verified"
            }
    
    # Download dataset
    logger.info("🔄 Starting data download from Kaggle...")
    start_time = time.time()
    
    try:
        download_path = ingest_ham10000(target_dir)
        elapsed = time.time() - start_time
        
        # Verify download
        if verify_dataset(target_dir):
            logger.info(f"✅ Data ingestion completed in {elapsed:.1f}s")
            return {
                "status": "success",
                "data_path": download_path,
                "elapsed_seconds": elapsed
            }
        else:
            raise ValueError("Dataset verification failed after download")
            
    except Exception as e:
        logger.error(f"❌ Data ingestion failed: {str(e)}")
        raise


# =============================================================================
# TASK: Data Preprocessing
# =============================================================================

@task(
    name="preprocess_data",
    description="Preprocess images: resize, stratified split, organize",
    retries=1,
    log_prints=True,
    tags=["data", "preprocessing"]
)
def preprocess_data_task(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    target_size: tuple = (224, 224),
    force_reprocess: bool = False
) -> Dict[str, any]:
    """
    Preprocess the HAM10000 dataset.
    
    Args:
        raw_dir: Directory with raw data.
        processed_dir: Output directory for processed data.
        target_size: Target image dimensions.
        force_reprocess: Force reprocessing even if data exists.
        
    Returns:
        Dict with preprocessing results.
    """
    logger = get_run_logger()
    
    from src.pipelines.data_preprocessing import run_preprocessing
    
    processed_path = Path(processed_dir)
    
    # Check if preprocessing already done
    if not force_reprocess:
        split_info_path = processed_path / "split_info.csv"
        train_path = processed_path / "train"
        
        if split_info_path.exists() and train_path.exists():
            # Count existing images
            train_count = sum(1 for _ in train_path.rglob("*.jpg"))
            if train_count > 0:
                logger.info(f"✅ Preprocessed data found ({train_count} train images). Skipping.")
                return {
                    "status": "skipped",
                    "processed_dir": str(processed_path.resolve()),
                    "train_images": train_count,
                    "message": "Preprocessed data already exists"
                }
    
    # Run preprocessing
    logger.info("🔄 Starting data preprocessing...")
    start_time = time.time()
    
    try:
        run_preprocessing(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            target_size=target_size,
            clear_existing=force_reprocess
        )
        elapsed = time.time() - start_time
        
        # Count results
        train_count = sum(1 for _ in (processed_path / "train").rglob("*.jpg"))
        val_count = sum(1 for _ in (processed_path / "val").rglob("*.jpg"))
        test_count = sum(1 for _ in (processed_path / "test").rglob("*.jpg"))
        
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
# TASK: Model Training
# =============================================================================

@task(
    name="train_model",
    description="Train ResNet50 model with transfer learning",
    retries=1,
    retry_delay_seconds=60,
    log_prints=True,
    tags=["training", "deep-learning"]
)
def train_model_task(
    data_dir: str = "data/processed",
    model_dir: str = "models",
    num_epochs: int = 15,
    finetune_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    finetune_lr: float = 1e-5,
    force_retrain: bool = False
) -> Dict[str, any]:
    """
    Train the ResNet50 model with two-phase training.
    
    Args:
        data_dir: Directory with preprocessed data.
        model_dir: Directory to save model checkpoints.
        num_epochs: Epochs for feature extraction phase.
        finetune_epochs: Epochs for fine-tuning phase.
        batch_size: Training batch size.
        learning_rate: LR for phase 1.
        finetune_lr: LR for phase 2 (fine-tuning).
        force_retrain: Force retraining even if model exists.
        
    Returns:
        Dict with training results.
    """
    logger = get_run_logger()
    
    from src.models.train_resnet import train_model
    
    model_path = Path(model_dir) / "best_model_resnet.pth"
    
    # Check if model already exists
    if not force_retrain and model_path.exists():
        logger.info(f"✅ Trained model found at {model_path}. Skipping training.")
        return {
            "status": "skipped",
            "model_path": str(model_path.resolve()),
            "message": "Model already exists. Set force_retrain=True to retrain."
        }
    
    # Run training
    logger.info("🚀 Starting model training...")
    logger.info(f"   Phase 1: {num_epochs} epochs @ LR={learning_rate}")
    logger.info(f"   Phase 2: {finetune_epochs} epochs @ LR={finetune_lr}")
    start_time = time.time()
    
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
        
        # Get best metrics
        best_f1 = max(history['val_f1'])
        best_epoch = history['val_f1'].index(best_f1) + 1
        final_acc = history['val_acc'][-1]
        
        logger.info(f"✅ Training completed in {elapsed/60:.1f} minutes")
        logger.info(f"   Best F1: {best_f1:.4f} @ Epoch {best_epoch}")
        logger.info(f"   Final Accuracy: {final_acc:.4f}")
        
        return {
            "status": "success",
            "model_path": str(model_path.resolve()),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "final_accuracy": final_acc,
            "total_epochs": num_epochs + finetune_epochs,
            "elapsed_minutes": elapsed / 60
        }
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


# =============================================================================
# TASK: Model Evaluation
# =============================================================================

@task(
    name="evaluate_model",
    description="Evaluate model on test set with explainability",
    retries=1,
    log_prints=True,
    tags=["evaluation", "metrics"]
)
def evaluate_model_task(
    model_path: str = "models/best_model_resnet.pth",
    data_dir: str = "data/processed",
    output_dir: str = "reports"
) -> Dict[str, any]:
    """
    Evaluate the trained model on the test set.
    
    Args:
        model_path: Path to trained model.
        data_dir: Directory with test data.
        output_dir: Directory for evaluation outputs.
        
    Returns:
        Dict with evaluation metrics.
    """
    logger = get_run_logger()
    
    from src.models.evaluate import run_evaluation
    
    # Check model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    
    logger.info("📊 Starting model evaluation...")
    start_time = time.time()
    
    try:
        metrics = run_evaluation(
            model_path=model_path,
            data_dir=data_dir,
            output_dir=output_dir,
            num_workers=0  # Use 0 workers for Prefect compatibility
        )
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Evaluation completed in {elapsed:.1f}s")
        logger.info(f"   Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Test F1 (Weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"   Test F1 (Macro): {metrics['f1_macro']:.4f}")
        
        return {
            "status": "success",
            "test_accuracy": metrics['accuracy'],
            "test_f1_weighted": metrics['f1_weighted'],
            "test_f1_macro": metrics['f1_macro'],
            "per_class_metrics": metrics.get('per_class', {}),
            "elapsed_seconds": elapsed
        }
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise


# =============================================================================
# TASK: Send Notification
# =============================================================================

@task(
    name="send_notification",
    description="Send notification when pipeline completes",
    log_prints=True,
    tags=["notification"]
)
def send_notification_task(
    pipeline_results: Dict[str, any],
    status: str = "success"
) -> Dict[str, str]:
    """
    Send notification about pipeline completion.
    
    For now, this creates a local notification file and Prefect artifact.
    Can be extended to send Slack/Email notifications.
    
    Args:
        pipeline_results: Results from all pipeline stages.
        status: Pipeline status (success/failed).
        
    Returns:
        Dict with notification status.
    """
    logger = get_run_logger()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create notification content
    if status == "success":
        icon = "✅"
        title = "DermaOps Pipeline Completed Successfully"
        
        # Extract metrics
        train_result = pipeline_results.get("training", {})
        eval_result = pipeline_results.get("evaluation", {})
        
        # Format metrics safely
        best_f1 = train_result.get('best_f1')
        best_f1_str = f"{best_f1:.4f}" if isinstance(best_f1, (int, float)) else "N/A"
        
        elapsed_min = train_result.get('elapsed_minutes')
        elapsed_str = f"{elapsed_min:.1f}" if isinstance(elapsed_min, (int, float)) else "N/A"
        
        test_acc = eval_result.get('test_accuracy')
        test_acc_str = f"{test_acc:.4f}" if isinstance(test_acc, (int, float)) else "N/A"
        
        test_f1_w = eval_result.get('test_f1_weighted')
        test_f1_w_str = f"{test_f1_w:.4f}" if isinstance(test_f1_w, (int, float)) else "N/A"
        
        test_f1_m = eval_result.get('test_f1_macro')
        test_f1_m_str = f"{test_f1_m:.4f}" if isinstance(test_f1_m, (int, float)) else "N/A"
        
        content = f"""
# {icon} {title}

**Timestamp:** {timestamp}

## Training Results
- **Status:** {train_result.get('status', 'N/A')}
- **Best F1 Score:** {best_f1_str}
- **Training Duration:** {elapsed_str} minutes

## Test Set Performance
- **Accuracy:** {test_acc_str}
- **F1 Score (Weighted):** {test_f1_w_str}
- **F1 Score (Macro):** {test_f1_m_str}

## Artifacts Generated
- `reports/metrics_report.txt`
- `reports/figures/confusion_matrix_test.png`
- `reports/figures/explainability_samples.png`
- `models/best_model_resnet.pth`
"""
    else:
        icon = "❌"
        title = "DermaOps Pipeline Failed"
        error_msg = pipeline_results.get("error", "Unknown error")
        stage = pipeline_results.get("failed_stage", "Unknown")
        
        content = f"""
# {icon} {title}

**Timestamp:** {timestamp}
**Failed Stage:** {stage}

## Error Details
```
{error_msg}
```

## Troubleshooting
1. Check the logs in Prefect UI for detailed error messages
2. Verify network connectivity for data download issues
3. Ensure sufficient GPU memory for training
4. Check disk space for data storage
"""
    
    # Create Prefect artifact (visible in Prefect UI)
    create_markdown_artifact(
        key="pipeline-notification",
        markdown=content,
        description=f"Pipeline {status} notification"
    )
    
    # Also save to local file
    notification_dir = Path(PROJECT_ROOT) / "reports"
    notification_dir.mkdir(exist_ok=True)
    notification_path = notification_dir / "pipeline_notification.md"
    notification_path.write_text(content)
    
    logger.info(f"{icon} Pipeline {status} notification saved!")
    logger.info(f"   📄 Local: {notification_path}")
    logger.info(f"   🔗 Prefect UI: Check artifacts tab")
    
    # Print notification to console
    print("\n" + "="*60)
    print(f"📬 PIPELINE NOTIFICATION - {status.upper()}")
    print("="*60)
    print(content)
    
    return {
        "status": "sent",
        "notification_path": str(notification_path),
        "timestamp": timestamp
    }


# =============================================================================
# MAIN FLOW: DermaOps Pipeline
# =============================================================================

@flow(
    name="DermaOps-ML-Pipeline",
    description="End-to-end ML pipeline for skin cancer classification",
    version="1.0.0",
    log_prints=True,
    retries=0  # Flow-level retries disabled, task-level retries handle failures
)
def dermaops_pipeline(
    # Data ingestion params
    raw_dir: str = "data/raw",
    force_download: bool = False,
    
    # Preprocessing params
    processed_dir: str = "data/processed",
    target_size: tuple = (224, 224),
    force_reprocess: bool = False,
    
    # Training params
    model_dir: str = "models",
    num_epochs: int = 15,
    finetune_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    finetune_lr: float = 1e-5,
    force_retrain: bool = False,
    
    # Evaluation params
    output_dir: str = "reports",
    
    # Pipeline options
    skip_training: bool = False,
    skip_evaluation: bool = False
) -> Dict[str, any]:
    """
    Main DermaOps ML Pipeline.
    
    Orchestrates:
    1. Data Ingestion (with retries for network issues)
    2. Data Preprocessing (stratified split, resizing)
    3. Model Training (two-phase transfer learning)
    4. Model Evaluation (metrics, confusion matrix, Grad-CAM)
    5. Notification (pipeline completion alert)
    
    Args:
        raw_dir: Directory for raw data download.
        force_download: Force re-download of dataset.
        processed_dir: Directory for processed data.
        target_size: Image dimensions for resizing.
        force_reprocess: Force reprocessing of data.
        model_dir: Directory for model checkpoints.
        num_epochs: Epochs for frozen backbone training.
        finetune_epochs: Epochs for fine-tuning.
        batch_size: Training batch size.
        learning_rate: Learning rate for phase 1.
        finetune_lr: Learning rate for fine-tuning.
        force_retrain: Force retraining of model.
        output_dir: Directory for evaluation outputs.
        skip_training: Skip training stage.
        skip_evaluation: Skip evaluation stage.
        
    Returns:
        Dict with results from all pipeline stages.
    """
    logger = get_run_logger()
    pipeline_results = {}
    
    logger.info("="*60)
    logger.info("🚀 DERMAOPS ML PIPELINE STARTING")
    logger.info("="*60)
    
    pipeline_start = time.time()
    
    try:
        # Stage 1: Data Ingestion
        logger.info("\n📥 STAGE 1: DATA INGESTION")
        ingestion_result = ingest_data_task(
            target_dir=raw_dir,
            force_download=force_download
        )
        pipeline_results["ingestion"] = ingestion_result
        
        # Stage 2: Data Preprocessing
        logger.info("\n⚙️  STAGE 2: DATA PREPROCESSING")
        preprocess_result = preprocess_data_task(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            target_size=target_size,
            force_reprocess=force_reprocess
        )
        pipeline_results["preprocessing"] = preprocess_result
        
        # Stage 3: Model Training
        if not skip_training:
            logger.info("\n🧠 STAGE 3: MODEL TRAINING")
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
        else:
            logger.info("\n⏭️  STAGE 3: SKIPPED (skip_training=True)")
            pipeline_results["training"] = {"status": "skipped"}
        
        # Stage 4: Model Evaluation
        if not skip_evaluation:
            logger.info("\n📊 STAGE 4: MODEL EVALUATION")
            model_path = f"{model_dir}/best_model_resnet.pth"
            evaluation_result = evaluate_model_task(
                model_path=model_path,
                data_dir=processed_dir,
                output_dir=output_dir
            )
            pipeline_results["evaluation"] = evaluation_result
        else:
            logger.info("\n⏭️  STAGE 4: SKIPPED (skip_evaluation=True)")
            pipeline_results["evaluation"] = {"status": "skipped"}
        
        # Calculate total time
        total_time = time.time() - pipeline_start
        pipeline_results["total_elapsed_minutes"] = total_time / 60
        
        # Stage 5: Send Success Notification
        logger.info("\n📬 STAGE 5: SENDING NOTIFICATION")
        notification_result = send_notification_task(
            pipeline_results=pipeline_results,
            status="success"
        )
        pipeline_results["notification"] = notification_result
        
        logger.info("\n" + "="*60)
        logger.info("✅ DERMAOPS PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"   Total Duration: {total_time/60:.1f} minutes")
        logger.info("="*60)
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {str(e)}")
        
        # Determine which stage failed
        failed_stage = "Unknown"
        if "ingestion" not in pipeline_results:
            failed_stage = "Data Ingestion"
        elif "preprocessing" not in pipeline_results:
            failed_stage = "Preprocessing"
        elif "training" not in pipeline_results:
            failed_stage = "Training"
        elif "evaluation" not in pipeline_results:
            failed_stage = "Evaluation"
        
        # Send failure notification
        pipeline_results["error"] = str(e)
        pipeline_results["failed_stage"] = failed_stage
        
        try:
            send_notification_task(
                pipeline_results=pipeline_results,
                status="failed"
            )
        except Exception:
            logger.warning("Could not send failure notification")
        
        raise


# =============================================================================
# QUICK-RUN FLOWS (for specific stages)
# =============================================================================

@flow(name="DermaOps-Ingest-Only", log_prints=True)
def run_ingestion_only(
    target_dir: str = "data/raw",
    force_download: bool = False
):
    """Run only data ingestion."""
    return ingest_data_task(target_dir, force_download)


@flow(name="DermaOps-Preprocess-Only", log_prints=True)
def run_preprocessing_only(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    force_reprocess: bool = False
):
    """Run only preprocessing."""
    return preprocess_data_task(raw_dir, processed_dir, force_reprocess=force_reprocess)


@flow(name="DermaOps-Train-Only", log_prints=True)
def run_training_only(
    data_dir: str = "data/processed",
    model_dir: str = "models",
    num_epochs: int = 15,
    finetune_epochs: int = 15,
    force_retrain: bool = False
):
    """Run only model training."""
    return train_model_task(
        data_dir=data_dir,
        model_dir=model_dir,
        num_epochs=num_epochs,
        finetune_epochs=finetune_epochs,
        force_retrain=force_retrain
    )


@flow(name="DermaOps-Evaluate-Only", log_prints=True)
def run_evaluation_only(
    model_path: str = "models/best_model_resnet.pth",
    data_dir: str = "data/processed",
    output_dir: str = "reports"
):
    """Run only model evaluation."""
    return evaluate_model_task(model_path, data_dir, output_dir)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DermaOps ML Pipeline Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (skips existing data/models)
  python -m src.pipelines.orchestration
  
  # Force retraining
  python -m src.pipelines.orchestration --force-retrain
  
  # Skip training (evaluate existing model)
  python -m src.pipelines.orchestration --skip-training
  
  # Run with custom epochs
  python -m src.pipelines.orchestration --epochs 10 --finetune-epochs 10
  
  # Run specific stage only
  python -m src.pipelines.orchestration --stage ingest
  python -m src.pipelines.orchestration --stage preprocess
  python -m src.pipelines.orchestration --stage train
  python -m src.pipelines.orchestration --stage evaluate
        """
    )
    
    parser.add_argument(
        "--stage", 
        type=str, 
        choices=["full", "ingest", "preprocess", "train", "evaluate"],
        default="full",
        help="Pipeline stage to run (default: full)"
    )
    parser.add_argument("--force-download", action="store_true", help="Force data re-download")
    parser.add_argument("--force-reprocess", action="store_true", help="Force data reprocessing")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    parser.add_argument("--skip-training", action="store_true", help="Skip training stage")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation stage")
    parser.add_argument("--epochs", type=int, default=15, help="Phase 1 epochs (default: 15)")
    parser.add_argument("--finetune-epochs", type=int, default=15, help="Phase 2 epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    print("\n" + "="*60)
    print("🔬 DERMAOPS ORCHESTRATION PIPELINE")
    print("="*60)
    print(f"   Stage: {args.stage}")
    print(f"   Project Root: {PROJECT_ROOT}")
    print("="*60 + "\n")
    
    # Run appropriate flow
    if args.stage == "full":
        result = dermaops_pipeline(
            force_download=args.force_download,
            force_reprocess=args.force_reprocess,
            force_retrain=args.force_retrain,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            num_epochs=args.epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size
        )
    elif args.stage == "ingest":
        result = run_ingestion_only(force_download=args.force_download)
    elif args.stage == "preprocess":
        result = run_preprocessing_only(force_reprocess=args.force_reprocess)
    elif args.stage == "train":
        result = run_training_only(
            num_epochs=args.epochs,
            finetune_epochs=args.finetune_epochs,
            force_retrain=args.force_retrain
        )
    elif args.stage == "evaluate":
        result = run_evaluation_only()
    
    print("\n📋 Pipeline Result:")
    print(result)
