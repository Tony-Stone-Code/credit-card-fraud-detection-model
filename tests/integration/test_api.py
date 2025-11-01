"""
Integration tests for the API server.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


@pytest.fixture
def mock_predictor():
    """Create a mock predictor."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        'is_fraud': False,
        'fraud_probability': 0.15,
        'non_fraud_probability': 0.85,
        'risk_score': 'LOW',
        'model_version': '1.0.0'
    }
    predictor.predict_batch.return_value = [
        {
            'is_fraud': False,
            'fraud_probability': 0.15,
            'non_fraud_probability': 0.85,
            'risk_score': 'LOW',
            'model_version': '1.0.0'
        }
    ] * 3
    predictor.get_feature_importance.return_value = {
        'V1': 0.05,
        'V2': 0.04,
        'Amount': 0.10
    }
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Create test client with mocked predictor."""
    with patch('api_server.FraudPredictor', return_value=mock_predictor):
        from api_server import app
        with TestClient(app) as test_client:
            yield test_client


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert data['model_loaded'] == True
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200
        data = response.json()
        assert 'model_type' in data
        assert 'version' in data
        assert 'features' in data
    
    def test_single_prediction(self, client):
        """Test single prediction endpoint."""
        transaction = {
            "Time": 406,
            "Amount": 150.50,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        
        response = client.post("/api/v1/predict", json=transaction)
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'transaction_id' in data
        assert 'timestamp' in data
        assert 'latency_ms' in data
    
    def test_batch_prediction(self, client):
        """Test batch prediction endpoint."""
        transaction = {
            "Time": 406,
            "Amount": 150.50,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        
        batch_request = {
            "transactions": [transaction] * 3
        }
        
        response = client.post("/api/v1/predict/batch", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        assert 'batch_id' in data
        assert 'predictions' in data
        assert len(data['predictions']) == 3
    
    def test_feature_importance(self, client):
        """Test feature importance endpoint."""
        response = client.get("/api/v1/model/features")
        assert response.status_code == 200
        data = response.json()
        assert 'feature_importance' in data
    
    def test_invalid_transaction(self, client):
        """Test prediction with invalid transaction."""
        invalid_transaction = {
            "Time": 406,
            "Amount": -100,  # Negative amount should fail validation
        }
        
        response = client.post("/api/v1/predict", json=invalid_transaction)
        assert response.status_code == 422  # Validation error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
