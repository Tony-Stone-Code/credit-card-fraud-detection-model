"""
Configuration management for fraud detection system.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_path": MODEL_DIR / "fraud_detector.joblib",
    "scaler_path": MODEL_DIR / "scaler.joblib",
    "model_version": "1.0.0",
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}

# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "smote_sampling_strategy": "auto",
    "smote_random_state": 42,
}

# API configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4")),
    "reload": os.getenv("API_RELOAD", "false").lower() == "true",
    "log_level": os.getenv("LOG_LEVEL", "info"),
}

# Feature names (V1-V28 + Time + Amount)
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

# Threshold configuration
THRESHOLDS = {
    "high_risk": 0.8,      # > 80% probability = high risk
    "medium_risk": 0.5,    # 50-80% probability = medium risk
    "low_risk": 0.0,       # < 50% probability = low risk
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": LOG_DIR / "fraud_detection.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}
