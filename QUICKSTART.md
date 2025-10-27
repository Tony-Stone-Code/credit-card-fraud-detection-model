# Quick Start Guide - Fraud Detection System

## Local Development Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd credit-card-fraud-detection-model
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data
Download the Credit Card Fraud Detection dataset from Kaggle and place it in `data/creditcard.csv`:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### 3. Train Model
```bash
python src/train_model.py --data data/creditcard.csv
```

This will:
- Preprocess the data
- Train a Random Forest model with SMOTE
- Evaluate performance
- Save model to `models/fraud_detector.joblib`

### 4. Start API Server (Local)
```bash
pip install -r requirements-prod.txt
python src/api_server.py
```

API will be available at `http://localhost:8000`

### 5. Test the API
```bash
python src/demo.py --test all
```

Or test individual endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @data/sample/transaction.json
```

---

## Docker Deployment

### Build and Run with Docker
```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  --name fraud-api \
  fraud-detection:latest
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## API Documentation

Once the server is running, access interactive API docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example API Usage

**Python:**
```python
import requests

response = requests.post(
    'http://localhost:8000/api/v1/predict',
    json={
        'Time': 406,
        'Amount': 150.50,
        'V1': -1.36, 'V2': -0.07, ..., 'V28': -0.02
    }
)
result = response.json()
print(f"Fraud: {result['prediction']['is_fraud']}")
print(f"Probability: {result['prediction']['fraud_probability']}")
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406,
    "Amount": 150.50,
    "V1": -1.36,
    ...
  }'
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_predict.py -v
```

---

## Production Deployment

See `DEPLOYMENT.md` for comprehensive production deployment guide including:
- Cloud deployment (AWS, GCP, Azure)
- Kubernetes configuration
- Monitoring and logging
- Security best practices
- Scaling strategies

---

## Troubleshooting

**Model not found error:**
```bash
# Ensure model is trained
python src/train_model.py --data data/creditcard.csv
```

**API connection refused:**
```bash
# Check if server is running
curl http://localhost:8000/health
```

**Import errors:**
```bash
# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-prod.txt
```

---

## Key Files

- `Model.ipynb` - Original Jupyter notebook
- `DEPLOYMENT.md` - Comprehensive deployment guide
- `src/train_model.py` - Model training script
- `src/predict.py` - Prediction module
- `src/api_server.py` - FastAPI server
- `src/demo.py` - Demo and testing script
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Multi-service setup

---

## Next Steps

1. Review `DEPLOYMENT.md` for production deployment strategies
2. Customize model parameters in `src/config.py`
3. Implement monitoring and alerting
4. Set up CI/CD pipeline
5. Add authentication and rate limiting for production
