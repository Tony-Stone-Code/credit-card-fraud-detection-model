"""
Training script for fraud detection model.
Converts notebook training logic into production-ready script.
"""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODEL_CONFIG, TRAINING_CONFIG, MODEL_DIR, DATA_DIR, FEATURE_NAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """Handles training of fraud detection model."""
    
    def __init__(self, data_path: str):
        """
        Initialize trainer.
        
        Args:
            data_path: Path to training data CSV file
        """
        self.data_path = Path(data_path)
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate dataset."""
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} transactions")
        
        # Validate required columns
        required_columns = FEATURE_NAMES + ['Class']
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess data: handle missing values and scale features."""
        logger.info("Preprocessing data...")
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Scale Time and Amount features
        self.scaler = StandardScaler()
        X[['Amount', 'Time']] = self.scaler.fit_transform(X[['Amount', 'Time']])
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=TRAINING_CONFIG['test_size'],
            random_state=TRAINING_CONFIG['random_state'],
            stratify=y
        )
        
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
        logger.info(f"Fraud rate in train: {self.y_train.mean():.4f}")
        logger.info(f"Fraud rate in test: {self.y_test.mean():.4f}")
        
        return X, y
    
    def handle_imbalance(self):
        """Handle class imbalance using SMOTE."""
        logger.info("Applying SMOTE to handle class imbalance...")
        
        smote = SMOTE(
            sampling_strategy=TRAINING_CONFIG['smote_sampling_strategy'],
            random_state=TRAINING_CONFIG['smote_random_state']
        )
        
        X_train_res, y_train_res = smote.fit_resample(self.X_train, self.y_train)
        
        logger.info(f"Original training set: {len(self.X_train)} samples")
        logger.info(f"Resampled training set: {len(X_train_res)} samples")
        logger.info(f"Fraud rate after SMOTE: {y_train_res.mean():.4f}")
        
        return X_train_res, y_train_res
    
    def train(self, X_train_res, y_train_res):
        """Train Random Forest classifier."""
        logger.info("Training Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=MODEL_CONFIG['n_estimators'],
            max_depth=MODEL_CONFIG['max_depth'],
            min_samples_split=MODEL_CONFIG['min_samples_split'],
            min_samples_leaf=MODEL_CONFIG['min_samples_leaf'],
            random_state=MODEL_CONFIG['random_state'],
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train_res, y_train_res)
        logger.info("Model training completed")
        
    def evaluate(self):
        """Evaluate model on test set."""
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_prob)
        
        logger.info("\n" + "="*50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        logger.info(f"\nClassification Report:\n{class_report}")
        logger.info(f"\nROC AUC Score: {roc_auc:.4f}")
        logger.info("="*50)
        
        # Save evaluation plots
        self._save_evaluation_plots(conf_matrix, y_pred_prob)
        
        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'roc_auc': roc_auc
        }
    
    def _save_evaluation_plots(self, conf_matrix, y_pred_prob):
        """Save evaluation visualizations."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
        axes[1].plot(fpr, tpr, label=f'ROC Curve')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plot_path = MODEL_DIR / 'evaluation_plots.png'
        plt.savefig(plot_path)
        logger.info(f"Evaluation plots saved to {plot_path}")
        plt.close()
        
    def save_model(self):
        """Save trained model and scaler."""
        logger.info("Saving model and scaler...")
        
        model_path = MODEL_CONFIG['model_path']
        scaler_path = MODEL_CONFIG['scaler_path']
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        
    def run(self):
        """Execute full training pipeline."""
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess
            self.preprocess_data(df)
            
            # Handle imbalance
            X_train_res, y_train_res = self.handle_imbalance()
            
            # Train
            self.train(X_train_res, y_train_res)
            
            # Evaluate
            metrics = self.evaluate()
            
            # Save
            self.save_model()
            
            logger.info("Training pipeline completed successfully!")
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument(
        '--data',
        type=str,
        default=str(DATA_DIR / 'creditcard.csv'),
        help='Path to training data CSV file'
    )
    
    args = parser.parse_args()
    
    # Train model
    trainer = FraudDetectionTrainer(args.data)
    trainer.run()


if __name__ == '__main__':
    main()
