# Credit Card Fraud Detection - Production-Ready ML System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)

A production-ready credit card fraud detection system using machine learning. This project transforms a Jupyter notebook into a scalable, deployable real-world application with RESTful API, Docker support, and comprehensive documentation.

## 🎯 Overview

This system builds a credit card fraud detection model using **Random Forest** classification with **SMOTE** (Synthetic Minority Over-sampling Technique) to handle highly imbalanced datasets. The project includes:

- ✅ **Production-ready Python scripts** converted from Jupyter notebook
- ✅ **RESTful API** with FastAPI for real-time predictions
- ✅ **Docker & Docker Compose** for easy deployment
- ✅ **Comprehensive testing** (unit and integration tests)
- ✅ **Demo scripts** for testing and validation
- ✅ **Complete deployment documentation** for real-world use cases

## 📊 Dataset

The dataset contains credit card transactions with:
- **Features**: 28 anonymized features (`V1` to `V28`) obtained via PCA transformation
- **Time**: Seconds elapsed between transactions
- **Amount**: Transaction amount
- **Class**: Binary label (0 = legitimate, 1 = fraud)

**Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/Tony-Stone-Code/credit-card-fraud-detection-model.git
cd credit-card-fraud-detection-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
# Download dataset from Kaggle and place in data/creditcard.csv
python src/train_model.py --data data/creditcard.csv
```

### Start the API Server

```bash
# Install production dependencies
pip install -r requirements-prod.txt

# Start server
python src/api_server.py
```

API will be available at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Run Demo Tests

```bash
python src/demo.py --test all
```

## 📁 Project Structure

```
credit-card-fraud-detection-model/
├── Model.ipynb                 # Original Jupyter notebook
├── README.md                   # This file
├── DEPLOYMENT.md              # Comprehensive deployment guide
├── QUICKSTART.md              # Quick start guide
├── requirements.txt           # Python dependencies
├── requirements-prod.txt      # Production dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Multi-service Docker setup
├── .gitignore                # Git ignore rules
│
├── src/                       # Source code
│   ├── config.py             # Configuration management
│   ├── train_model.py        # Model training script
│   ├── predict.py            # Prediction module
│   ├── api_server.py         # FastAPI server
│   └── demo.py               # Demo and testing script
│
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   │   └── test_predict.py
│   └── integration/          # Integration tests
│       └── test_api.py
│
├── models/                    # Trained models (created after training)
│   ├── fraud_detector.joblib
│   └── scaler.joblib
│
├── data/                      # Data directory
│   └── sample/               # Sample data
│
└── logs/                      # Application logs (created at runtime)
```

## 🔑 Key Features

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Imbalance Handling**: SMOTE over-sampling
- **Feature Engineering**: StandardScaler for Time and Amount
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Production API
- **Framework**: FastAPI with async support
- **Endpoints**: Health check, single prediction, batch prediction
- **Validation**: Pydantic models for request/response
- **Documentation**: Auto-generated OpenAPI/Swagger docs
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging for production

### Deployment Options
- **Local**: Python virtual environment
- **Docker**: Single container deployment
- **Docker Compose**: Multi-service deployment with Redis
- **Cloud**: AWS, GCP, Azure (see DEPLOYMENT.md)
- **Kubernetes**: Scalable orchestration (see DEPLOYMENT.md)

## 💡 Real-World Use Cases

1. **Real-Time Transaction Monitoring**
   - Process transactions as they occur
   - Immediate fraud detection and alerts
   - Integration with payment gateways

2. **Batch Processing**
   - End-of-day fraud analysis
   - Historical data review
   - Compliance reporting

3. **Risk Scoring System**
   - Assign risk levels to transactions
   - Threshold-based decision making
   - Manual review queue management

4. **Customer Behavior Analytics**
   - Profile-based fraud detection
   - Anomaly detection
   - Personalized security

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed use case analysis and implementation strategies.

## 📈 Model Performance

Typical performance metrics (will vary based on dataset):
- **ROC-AUC**: > 0.95
- **Precision**: 0.85-0.92
- **Recall**: 0.88-0.95
- **F1-Score**: 0.86-0.93

## 🐳 Docker Deployment

### Using Docker

```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -d -p 8000:8000 --name fraud-api fraud-detection:latest
```

### Using Docker Compose

```bash
# Start all services (API + Redis)
docker-compose up -d

# View logs
docker-compose logs -f fraud-detection-api

# Stop services
docker-compose down
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/

# Run specific test suite
pytest tests/unit/ -v
pytest tests/integration/ -v
```

## 📚 API Usage Examples

### Python Client

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:8000/api/v1/predict',
    json={
        'Time': 406,
        'Amount': 150.50,
        'V1': -1.36,
        'V2': -0.07,
        # ... V3 to V28
    }
)

result = response.json()
print(f"Fraud: {result['prediction']['is_fraud']}")
print(f"Probability: {result['prediction']['fraud_probability']:.4f}")
print(f"Risk: {result['prediction']['risk_score']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406,
    "Amount": 150.50,
    "V1": -1.36,
    "V2": -0.07,
    "V3": 2.54,
    ...
  }'
```

### JavaScript/Node.js

```javascript
const response = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(transactionData)
});

const result = await response.json();
console.log(`Fraud: ${result.prediction.is_fraud}`);
```

## 📖 Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Comprehensive deployment guide covering:
  - Real-world use cases and business value
  - System requirements and architecture
  - Deployment strategies (Local, Docker, Kubernetes, Cloud)
  - API integration examples
  - Monitoring and maintenance
  - Security considerations
  - Cost analysis and ROI

- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide for:
  - Fast setup and installation
  - Common commands
  - API examples
  - Troubleshooting

## 🔒 Security Considerations

For production deployment, implement:
- API authentication (JWT, API keys)
- Rate limiting
- Input validation and sanitization
- HTTPS/TLS encryption
- Secure model storage
- Audit logging
- PCI-DSS compliance for payment data

See [DEPLOYMENT.md](DEPLOYMENT.md) Security section for details.

## 🛠️ Technologies Used

- **ML Framework**: scikit-learn, imbalanced-learn
- **API Framework**: FastAPI, Uvicorn
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Testing**: pytest
- **Containerization**: Docker, Docker Compose
- **Model Serving**: joblib

## 📊 Monitoring & Metrics

The system tracks:
- **Model Performance**: Accuracy, precision, recall, ROC-AUC
- **API Performance**: Latency, throughput, error rates
- **Business Metrics**: Fraud detection rate, false positives
- **System Metrics**: CPU, memory, disk usage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: [Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Inspired by real-world fraud detection systems in financial institutions

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed documentation
- Review [QUICKSTART.md](QUICKSTART.md) for common tasks

## 🎓 Learning Resources

- [Understanding Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [SMOTE for Imbalanced Classification](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Built with ❤️ for production ML deployment**
