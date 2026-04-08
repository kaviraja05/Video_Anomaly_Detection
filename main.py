"""
FastAPI Backend for Video Anomaly Detection - Preprocessing Proof
Minimal endpoint to demonstrate I3D feature preprocessing.
"""

import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI(
    title="Video Anomaly Detection API",
    description="Preprocessing proof endpoint for I3D features",
    version="1.0.0"
)

# Configuration
FEATURES_DIR = Path("data/i3d_features/train")


def get_random_feature_file() -> Path:
    """
    Get a random .npy feature file from the dataset.
    
    Returns:
        Path: Path to a random feature file.
        
    Raises:
        FileNotFoundError: If directory doesn't exist or no .npy files found.
    """
    if not FEATURES_DIR.exists():
        raise FileNotFoundError(f"Features directory not found: {FEATURES_DIR}")
    
    npy_files = list(FEATURES_DIR.glob("*.npy"))
    
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {FEATURES_DIR}")
    
    return random.choice(npy_files)


def count_nans(array: np.ndarray) -> int:
    """Count NaN values in numpy array."""
    return int(np.isnan(array).sum())


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: Input feature array
        
    Returns:
        Normalized feature array
    """
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    normalized = (features - mean) / std
    return normalized


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "message": "Video Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": {
            "preprocessing_proof": "/preprocessing-proof",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/preprocessing-proof")
async def preprocessing_proof() -> Dict[str, Any]:
    """
    Preprocessing proof endpoint.
    
    Automatically picks a random I3D feature file, loads it, applies normalization,
    and returns detailed preprocessing statistics.
    
    Returns:
        JSON with file info, shapes, NaN counts, and preprocessing stats
        
    Raises:
        HTTPException: If files not found or processing fails
    """
    try:
        # Step 1: Pick random feature file
        feature_file = get_random_feature_file()
        
        # Step 2: Load features
        features = np.load(feature_file)
        
        # Step 3: Original statistics
        original_shape = features.shape
        original_dtype = str(features.dtype)
        original_nan_count = count_nans(features)
        original_min = float(np.nanmin(features))
        original_max = float(np.nanmax(features))
        original_mean = float(np.nanmean(features))
        original_std = float(np.nanstd(features))
        
        # Step 4: Apply normalization
        normalized_features = normalize_features(features)
        
        # Step 5: Normalized statistics
        normalized_nan_count = count_nans(normalized_features)
        normalized_min = float(np.nanmin(normalized_features))
        normalized_max = float(np.nanmax(normalized_features))
        normalized_mean = float(np.nanmean(normalized_features))
        normalized_std = float(np.nanstd(normalized_features))
        
        # Step 6: Build response
        response = {
            "status": "success",
            "file_info": {
                "filename": feature_file.name,
                "path": str(feature_file),
                "size_bytes": feature_file.stat().st_size
            },
            "original_features": {
                "shape": list(original_shape),
                "dtype": original_dtype,
                "total_elements": int(np.prod(original_shape)),
                "nan_count": original_nan_count,
                "statistics": {
                    "min": original_min,
                    "max": original_max,
                    "mean": original_mean,
                    "std": original_std
                }
            },
            "preprocessing": {
                "method": "normalization",
                "formula": "(x - mean) / std",
                "applied": True
            },
            "normalized_features": {
                "shape": list(normalized_features.shape),
                "nan_count": normalized_nan_count,
                "statistics": {
                    "min": normalized_min,
                    "max": normalized_max,
                    "mean": normalized_mean,
                    "std": normalized_std
                }
            },
            "validation": {
                "nan_count_unchanged": original_nan_count == normalized_nan_count,
                "shape_preserved": original_shape == normalized_features.shape,
                "mean_close_to_zero": abs(normalized_mean) < 1e-6,
                "std_close_to_one": abs(normalized_std - 1.0) < 1e-6
            }
        }
        
        return JSONResponse(content=response)
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Dataset not found",
                "message": str(e),
                "suggestion": "Ensure data/i3d_features/train directory exists with .npy files"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Processing failed",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        features_exist = FEATURES_DIR.exists()
        if features_exist:
            npy_count = len(list(FEATURES_DIR.glob("*.npy")))
        else:
            npy_count = 0
            
        return {
            "status": "healthy",
            "features_directory": str(FEATURES_DIR),
            "directory_exists": features_exist,
            "feature_files_count": npy_count
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Video Anomaly Detection API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Preprocessing Proof: http://localhost:8000/preprocessing-proof")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
