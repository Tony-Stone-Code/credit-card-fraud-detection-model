"""
Prediction module for fraud detection.
Handles loading model and making predictions.
"""
import logging
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from config import MODEL_CONFIG, FEATURE_NAMES, THRESHOLDS

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Handles fraud prediction for credit card transactions."""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to scaler file
        """
        self.model_path = Path(model_path) if model_path else MODEL_CONFIG['model_path']
        self.scaler_path = Path(scaler_path) if scaler_path else MODEL_CONFIG['scaler_path']
        self.model = None
        self.scaler = None
        self._load_model()
        
    def _load_model(self):
        """Load trained model and scaler from disk."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Loading scaler from {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            
            logger.info("Model and scaler loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, transaction: Dict) -> np.ndarray:
        """
        Preprocess transaction data for prediction.
        
        Args:
            transaction: Dictionary with transaction features
            
        Returns:
            Preprocessed feature array
        """
        # Validate features
        missing_features = set(FEATURE_NAMES) - set(transaction.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Create DataFrame with correct column order
        df = pd.DataFrame([transaction], columns=FEATURE_NAMES)
        
        # Scale Time and Amount using fitted scaler
        df[['Amount', 'Time']] = self.scaler.transform(df[['Amount', 'Time']])
        
        return df.values
    
    def predict(self, transaction: Dict) -> Dict:
        """
        Predict fraud probability for a single transaction.
        
        Args:
            transaction: Dictionary with transaction features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess
            X = self.preprocess_input(transaction)
            
            # Predict
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]
            
            # Get fraud probability (class 1)
            fraud_prob = probability[1]
            
            # Determine risk level
            risk_score = self._get_risk_score(fraud_prob)
            
            return {
                'is_fraud': bool(prediction),
                'fraud_probability': float(fraud_prob),
                'non_fraud_probability': float(probability[0]),
                'risk_score': risk_score,
                'model_version': MODEL_CONFIG['model_version']
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Predict fraud probability for multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        try:
            predictions = []
            for transaction in transactions:
                pred = self.predict(transaction)
                predictions.append(pred)
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise
    
    def _get_risk_score(self, probability: float) -> str:
        """
        Convert probability to risk score category.
        
        Args:
            probability: Fraud probability
            
        Returns:
            Risk score category (HIGH, MEDIUM, LOW)
        """
        if probability >= THRESHOLDS['high_risk']:
            return 'HIGH'
        elif probability >= THRESHOLDS['medium_risk']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
            
        importance = self.model.feature_importances_
        feature_importance = dict(zip(FEATURE_NAMES, importance))
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return feature_importance


def predict_from_csv(csv_path: str, output_path: str = None):
    """
    Make predictions for transactions in a CSV file.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to save predictions (optional)
    """
    logger.info(f"Loading transactions from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Initialize predictor
    predictor = FraudPredictor()
    
    # Make predictions
    predictions = []
    for idx, row in df.iterrows():
        transaction = row.to_dict()
        if 'Class' in transaction:
            del transaction['Class']  # Remove label if present
            
        pred = predictor.predict(transaction)
        predictions.append(pred)
    
    # Create results DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Add original data
    for col in df.columns:
        if col != 'Class':
            results_df[col] = df[col].values
    
    # Save if output path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    
    return results_df


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Make fraud predictions')
    parser.add_argument('--csv', type=str, help='Path to CSV file with transactions')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    
    args = parser.parse_args()
    
    if args.csv:
        predict_from_csv(args.csv, args.output)
    else:
        # Example single prediction
        predictor = FraudPredictor()
        
        example_transaction = {
            'Time': 406,
            'Amount': 150.50,
            'V1': -1.3598071336738,
            'V2': -0.0727811733098497,
            'V3': 2.53634673796914,
            'V4': 1.37815522427443,
            'V5': -0.338320769942518,
            'V6': 0.462387777762292,
            'V7': 0.239598554061257,
            'V8': 0.0986979012610507,
            'V9': 0.363786969611213,
            'V10': 0.0907941719789316,
            'V11': -0.551599533260813,
            'V12': -0.617800855762348,
            'V13': -0.991389847235408,
            'V14': -0.311169353699879,
            'V15': 1.46817697209427,
            'V16': -0.470400525259478,
            'V17': 0.207971241929242,
            'V18': 0.0257905801985591,
            'V19': 0.403992960255733,
            'V20': 0.251412098239705,
            'V21': -0.018306777944153,
            'V22': 0.277837575558899,
            'V23': -0.110473910188767,
            'V24': 0.0669280749146731,
            'V25': 0.128539358273528,
            'V26': -0.189114843888824,
            'V27': 0.133558376740387,
            'V28': -0.0210530531905623
        }
        
        result = predictor.predict(example_transaction)
        print("\nPrediction Result:")
        print(f"Is Fraud: {result['is_fraud']}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Risk Score: {result['risk_score']}")
