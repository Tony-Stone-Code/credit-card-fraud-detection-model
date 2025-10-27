"""
Unit tests for fraud detection prediction module.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from predict import FraudPredictor
from config import FEATURE_NAMES


class TestFraudPredictor:
    """Test cases for FraudPredictor class."""
    
    @pytest.fixture
    def sample_transaction(self):
        """Sample transaction data."""
        return {
            'Time': 406,
            'Amount': 150.50,
            **{f'V{i}': np.random.randn() for i in range(1, 29)}
        }
    
    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor with mocked model."""
        with patch('predict.joblib.load') as mock_load:
            # Mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0])
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
            
            # Mock scaler
            mock_scaler = MagicMock()
            mock_scaler.transform.return_value = np.array([[150.50, 406]])
            
            mock_load.side_effect = [mock_model, mock_scaler]
            
            predictor = FraudPredictor()
            predictor.model = mock_model
            predictor.scaler = mock_scaler
            
            return predictor
    
    def test_preprocess_input_valid(self, mock_predictor, sample_transaction):
        """Test preprocessing with valid input."""
        result = mock_predictor.preprocess_input(sample_transaction)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == len(FEATURE_NAMES)
    
    def test_preprocess_input_missing_features(self, mock_predictor):
        """Test preprocessing with missing features."""
        incomplete_transaction = {'Time': 100, 'Amount': 50}
        
        with pytest.raises(ValueError, match="Missing required features"):
            mock_predictor.preprocess_input(incomplete_transaction)
    
    def test_predict_returns_dict(self, mock_predictor, sample_transaction):
        """Test that predict returns proper dictionary."""
        result = mock_predictor.predict(sample_transaction)
        
        assert isinstance(result, dict)
        assert 'is_fraud' in result
        assert 'fraud_probability' in result
        assert 'risk_score' in result
        assert 'model_version' in result
    
    def test_predict_probability_range(self, mock_predictor, sample_transaction):
        """Test that fraud probability is in valid range."""
        result = mock_predictor.predict(sample_transaction)
        
        assert 0 <= result['fraud_probability'] <= 1
        assert 0 <= result['non_fraud_probability'] <= 1
    
    def test_risk_score_categorization(self, mock_predictor):
        """Test risk score categorization."""
        # Test high risk
        assert mock_predictor._get_risk_score(0.9) == 'HIGH'
        
        # Test medium risk
        assert mock_predictor._get_risk_score(0.6) == 'MEDIUM'
        
        # Test low risk
        assert mock_predictor._get_risk_score(0.2) == 'LOW'
    
    def test_predict_batch(self, mock_predictor, sample_transaction):
        """Test batch prediction."""
        transactions = [sample_transaction] * 3
        results = mock_predictor.predict_batch(transactions)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
