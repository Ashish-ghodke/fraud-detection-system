"""
Prediction Pipeline Module
=========================
This module handles the complete prediction pipeline for fraud detection.
It integrates preprocessing, feature engineering, and model inference.

Author: Fraud Detection Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Tuple, Optional
import logging

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_evaluation import ModelEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Complete prediction pipeline for fraud detection.
    
    This class orchestrates the entire prediction workflow:
    1. Data validation
    2. Data preprocessing
    3. Feature engineering
    4. Model inference
    5. Result generation
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the PredictionPipeline.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = None
        self.model_path = model_path
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        self.is_loaded = False
        self.prediction_threshold = 0.5
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the trained model.
        
        Args:
            model_path: Path to model file (overrides init path)
            
        Returns:
            True if successful, False otherwise
        """
        path = model_path or self.model_path
        
        if path is None:
            logger.error("No model path provided")
            return False
        
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False
        
        try:
            self.model = joblib.load(path)
            
            # Try to load metadata if available
            metadata_path = path.replace('.pkl', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_engineer.feature_columns = metadata.get('feature_columns', [])
            
            self.is_loaded = True
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate input data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Dataset is empty"
        
        if len(df) > 500000:
            return False, "Dataset too large. Maximum 500,000 rows allowed."
        
        # Validate schema
        is_valid, errors = self.preprocessor.validate_schema(df)
        if not is_valid:
            return False, "; ".join(errors)
        
        return True, ""
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess input data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (preprocessed DataFrame, processing report)
        """
        # Preprocess data
        df_processed, report = self.preprocessor.preprocess(
            df,
            handle_missing='mean',
            remove_duplicates=True,
            ensure_numeric=True
        )
        
        return df_processed, report
    
    def transform_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Apply feature engineering.
        
        Args:
            df: Preprocessed DataFrame
            fit: Whether to fit transformers (use False for inference)
            
        Returns:
            Transformed DataFrame
        """
        # Apply feature engineering
        df_transformed = self.feature_engineer.transform(
            df,
            apply_time_features=True,
            apply_amount_features=True,
            apply_interaction_features=True,
            scale=True,
            scaler_method='standard',
            fit=fit
        )
        
        return df_transformed
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make fraud predictions.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)[:, 1]
        
        return predictions, probabilities
    
    def run(self, df: pd.DataFrame, 
           return_probabilities: bool = True) -> Dict:
        """
        Run complete prediction pipeline.
        
        Args:
            df: Input DataFrame
            return_probabilities: Whether to return probability scores
            
        Returns:
            Prediction results dictionary
        """
        logger.info("Starting prediction pipeline")
        
        results = {
            'success': False,
            'error': None,
            'predictions': None,
            'probabilities': None,
            'processing_report': {},
            'summary': {}
        }
        
        # Step 1: Validate input
        is_valid, error = self.validate_input(df)
        if not is_valid:
            results['error'] = f"Validation failed: {error}"
            logger.error(results['error'])
            return results
        
        # Step 2: Preprocess data
        try:
            df_processed, report = self.preprocess(df)
            results['processing_report'] = report
        except Exception as e:
            results['error'] = f"Preprocessing failed: {str(e)}"
            logger.error(results['error'])
            return results
        
        # Step 3: Feature engineering
        try:
            # Check if we need to fit the scaler (use first row of data)
            # For inference, we use the existing scaler
            df_transformed = self.transform_features(df_processed, fit=False)
        except Exception as e:
            # If scaling fails, try with original features
            logger.warning(f"Feature engineering failed: {e}. Using raw features.")
            df_transformed = df_processed.drop('Class', axis=1, errors='ignore')
        
        # Get features (exclude Class if present)
        feature_cols = [col for col in df_transformed.columns if col != 'Class']
        X = df_transformed[feature_cols]
        
        # Step 4: Make predictions
        try:
            predictions, probabilities = self.predict(X)
            results['predictions'] = predictions
            results['probabilities'] = probabilities
        except Exception as e:
            results['error'] = f"Prediction failed: {str(e)}"
            logger.error(results['error'])
            return results
        
        # Step 5: Generate summary
        n_total = len(predictions)
        n_fraud = int((predictions == 1).sum())
        n_normal = n_total - n_fraud
        fraud_percentage = (n_fraud / n_total * 100) if n_total > 0 else 0
        
        results['summary'] = {
            'total_transactions': n_total,
            'fraud_transactions': n_fraud,
            'normal_transactions': n_normal,
            'fraud_percentage': fraud_percentage,
            'avg_fraud_probability': float(probabilities.mean()) if len(probabilities) > 0 else 0
        }
        
        results['success'] = True
        logger.info(f"Prediction complete. Fraud detected: {n_fraud}/{n_total} ({fraud_percentage:.2f}%)")
        
        return results
    
    def add_results_to_dataframe(self, df: pd.DataFrame,
                               predictions: np.ndarray,
                               probabilities: np.ndarray) -> pd.DataFrame:
        """
        Add prediction results to original DataFrame.
        
        Args:
            df: Original DataFrame
            predictions: Model predictions
            probabilities: Fraud probabilities
            
        Returns:
            DataFrame with added columns
        """
        df = df.copy()
        
        # Add prediction column
        df['Prediction'] = predictions
        df['Fraud_Probability'] = probabilities
        
        # Add risk category
        df['Risk_Level'] = pd.cut(
            probabilities,
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Sort by fraud probability (descending)
        df = df.sort_values('Fraud_Probability', ascending=False)
        
        return df
    
    def get_top_suspicious(self, df: pd.DataFrame, 
                         n: int = 15) -> pd.DataFrame:
        """
        Get top N most suspicious transactions.
        
        Args:
            df: DataFrame with predictions
            n: Number of transactions to return
            
        Returns:
            DataFrame with top suspicious transactions
        """
        if 'Fraud_Probability' not in df.columns:
            logger.warning("No predictions found in DataFrame")
            return pd.DataFrame()
        
        top_suspicious = df.nlargest(n, 'Fraud_Probability')
        
        return top_suspicious
    
    def export_results(self, df: pd.DataFrame, 
                     output_path: str) -> bool:
        """
        Export results to CSV.
        
        Args:
            df: DataFrame with results
            output_path: Path to save CSV
            
        Returns:
            True if successful
        """
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Results exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False


def create_prediction_pipeline(model_path: str) -> PredictionPipeline:
    """
    Create and initialize a prediction pipeline.
    
    Args:
        model_path: Path to trained model
        
    Returns:
        Initialized PredictionPipeline
    """
    pipeline = PredictionPipeline(model_path)
    pipeline.load_model()
    return pipeline


# Testing the prediction pipeline
if __name__ == "__main__":
    from data_preprocessing import create_sample_dataset
    
    print("Testing PredictionPipeline...")
    
    # Create sample data
    df = create_sample_dataset(n_samples=1000, fraud_ratio=0.02)
    print(f"Created sample dataset with shape: {df.shape}")
    
    # Note: In real usage, you would load a trained model
    # For testing, we'll simulate the pipeline
    pipeline = PredictionPipeline()
    
    # Test validation
    is_valid, error = pipeline.validate_input(df)
    print(f"Validation: {is_valid}, Error: {error}")
    
    # Test preprocessing
    df_processed, report = pipeline.preprocess(df)
    print(f"Preprocessed shape: {df_processed.shape}")
    print(f"Preprocessing report: {report.get('validation_passed', False)}")
    
    print("PredictionPipeline test completed!")
    print("Note: Full prediction requires a trained model.")
