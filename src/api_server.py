"""
FastAPI server for fraud detection API.
Provides RESTful endpoints for real-time fraud prediction.
"""
import logging
import time
from typing import Dict, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from predict import FraudPredictor
from config import API_CONFIG, MODEL_CONFIG, FEATURE_NAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection for credit card transactions",
    version=MODEL_CONFIG['model_version']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


# Request/Response Models
class Transaction(BaseModel):
    """Transaction data model."""
    Time: float = Field(..., description="Time elapsed since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    
    class Config:
        schema_extra = {
            "example": {
                "Time": 406,
                "Amount": 150.50,
                "V1": -1.36,
                "V2": -0.07,
                "V3": 2.54,
                "V4": 1.38,
                "V5": -0.34,
                "V6": 0.46,
                "V7": 0.24,
                "V8": 0.10,
                "V9": 0.36,
                "V10": 0.09,
                "V11": -0.55,
                "V12": -0.62,
                "V13": -0.99,
                "V14": -0.31,
                "V15": 1.47,
                "V16": -0.47,
                "V17": 0.21,
                "V18": 0.03,
                "V19": 0.40,
                "V20": 0.25,
                "V21": -0.02,
                "V22": 0.28,
                "V23": -0.11,
                "V24": 0.07,
                "V25": 0.13,
                "V26": -0.19,
                "V27": 0.13,
                "V28": -0.02
            }
        }


class BatchTransactions(BaseModel):
    """Batch transaction request model."""
    transactions: List[Transaction] = Field(..., min_items=1, max_items=1000)


class PredictionResponse(BaseModel):
    """Prediction response model."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    prediction: Dict = Field(..., description="Prediction results")
    timestamp: str = Field(..., description="Prediction timestamp")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    batch_id: str
    predictions: List[Dict]
    total_transactions: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model info response model."""
    model_type: str
    version: str
    features: int
    feature_names: List[str]


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global predictor
    try:
        logger.info("Starting Fraud Detection API...")
        predictor = FraudPredictor()
        logger.info("Model loaded successfully")
        app.state.start_time = time.time()
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Fraud Detection API...")


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    latency = (time.time() - start_time) * 1000  # Convert to ms
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Latency: {latency:.2f}ms"
    )
    
    return response


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": MODEL_CONFIG['model_version'],
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    
    return {
        "status": "healthy" if predictor else "unhealthy",
        "model_loaded": predictor is not None,
        "version": MODEL_CONFIG['model_version'],
        "uptime_seconds": uptime
    }


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "RandomForestClassifier",
        "version": MODEL_CONFIG['model_version'],
        "features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_transaction(transaction: Transaction):
    """
    Predict fraud for a single transaction.
    
    Returns fraud probability and risk score.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert to dict
        transaction_dict = transaction.dict()
        
        # Make prediction
        prediction = predictor.predict(transaction_dict)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Generate transaction ID (in production, use proper ID generation)
        transaction_id = f"txn_{int(time.time() * 1000)}"
        
        return {
            "transaction_id": transaction_id,
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "latency_ms": latency_ms
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchTransactions):
    """
    Predict fraud for multiple transactions.
    
    Supports up to 1000 transactions per request.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert transactions to list of dicts
        transactions = [t.dict() for t in batch.transactions]
        
        # Make predictions
        predictions = predictor.predict_batch(transactions)
        
        # Add transaction IDs
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "transaction_id": f"txn_{int(time.time() * 1000)}_{i}",
                "prediction": pred
            })
        
        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000
        
        return {
            "batch_id": f"batch_{int(time.time() * 1000)}",
            "predictions": results,
            "total_transactions": len(transactions),
            "total_latency_ms": total_latency_ms
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/features", tags=["Model"])
async def get_feature_importance():
    """Get feature importance from the model."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        importance = predictor.get_feature_importance()
        return {
            "feature_importance": importance,
            "top_10_features": dict(list(importance.items())[:10])
        }
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


def main():
    """Run the API server."""
    uvicorn.run(
        "api_server:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        workers=API_CONFIG['workers'],
        reload=API_CONFIG['reload'],
        log_level=API_CONFIG['log_level']
    )


if __name__ == "__main__":
    main()
