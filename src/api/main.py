"""
DermaOps FastAPI Backend

REST API for skin lesion classification using the trained ResNet50 model.

Endpoints:
    GET  /           - Health check and API info
    GET  /health     - Health status
    POST /predict    - Classify a skin lesion image

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# =============================================================================
# CONFIGURATION
# =============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Model configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model_resnet.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (alphabetically sorted as per ImageFolder)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    prediction: str
    prediction_full_name: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    model_version: str = "resnet50_finetuned_v1"


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str
    model_loaded: bool
    device: str
    model_path: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="DermaOps API",
    description="AI-powered skin lesion classification using Fine-Tuned ResNet50",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allow Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODEL LOADING
# =============================================================================

# Global model variable
model = None


def load_model() -> nn.Module:
    """
    Load the trained ResNet50 model.
    
    Returns:
        Loaded model in eval mode.
    """
    global model
    
    if model is not None:
        return model
    
    print(f"🔄 Loading model from: {MODEL_PATH}")
    print(f"🖥️  Device: {DEVICE}")
    
    # Check model file exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        # Try relative to project root
        model_path = PROJECT_ROOT / MODEL_PATH
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # Create model architecture (must match training)
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 7)
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set eval mode
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
    
    return model


def get_transform() -> transforms.Compose:
    """
    Get image transforms for inference (must match training).
    
    Returns:
        Composed transforms.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        print("🚀 DermaOps API ready!")
    except Exception as e:
        print(f"⚠️  Warning: Could not load model on startup: {e}")
        print("   Model will be loaded on first prediction request.")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "DermaOps API",
        "description": "AI-powered skin lesion classification",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        },
        "model": "ResNet50 (Fine-tuned on HAM10000)",
        "classes": CLASS_NAMES
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    """
    global model
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(DEVICE),
        model_path=MODEL_PATH
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify a skin lesion image.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        PredictionResponse with classification results.
    """
    global model
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Supported: JPEG, PNG"
        )
    
    try:
        # Load model if not already loaded
        if model is None:
            model = load_model()
        
        # Read and preprocess image
        start_time = time.time()
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get prediction
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_value = confidence.item()
        
        # Build probability dictionary
        probs_dict = {
            class_name: float(probabilities[0][i].item())
            for i, class_name in enumerate(CLASS_NAMES)
        }
        
        return PredictionResponse(
            prediction=predicted_class,
            prediction_full_name=CLASS_FULL_NAMES.get(predicted_class, predicted_class),
            confidence=confidence_value,
            probabilities=probs_dict,
            inference_time_ms=round(inference_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/classes", tags=["Info"])
async def get_classes():
    """
    Get list of supported classes.
    """
    return {
        "classes": CLASS_NAMES,
        "full_names": CLASS_FULL_NAMES,
        "count": len(CLASS_NAMES)
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    print("="*60)
    print("🩺 DERMAOPS API SERVER")
    print("="*60)
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Device: {DEVICE}")
    print("="*60)
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
