"""
Model Training Module
====================
This module handles XGBoost model training with SMOTE for handling
imbalanced fraud detection datasets.

Author: Fraud Detection Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import logging
from typing import Dict, Tuple, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    XGBoost model trainer with SMOTE for fraud detection.
    
    This class handles all aspects of model training including:
    - Handling class imbalance with SMOTE
    - Hyperparameter tuning
    - Cross-validation
    - Model persistence
    """
    
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.model = None
        self.smote = None
        self.is_trained = False
        self.feature_columns = []
        self.class_weights = {}
        
    def calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Calculate scale_pos_weight for XGBoost.
        
        Args:
            y: Target variable
            
        Returns:
            Scale_pos_weight value
        """
        counter = Counter(y)
        neg_count = counter[0]
        pos_count = counter[1]
        
        scale_pos_weight = neg_count / pos_count
        logger.info(f"Class distribution: Normal={neg_count}, Fraud={pos_count}")
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        return scale_pos_weight
    
    def _handle_nan_values(self, X: np.ndarray) -> np.ndarray:
        """
        Handle NaN values in the feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Feature matrix with NaN filled
        """
        # Replace NaN with median for each column
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        return X_imputed
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray, 
                   sampling_strategy: float = 0.5,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            X: Feature matrix
            y: Target variable
            sampling_strategy: Ratio of minority to majority class after resampling
            random_state: Random seed
            
        Returns:
            Resampled X and y
        """
        logger.info(f"Original class distribution: {Counter(y)}")
        
        # Handle NaN values before SMOTE
        if np.isnan(X).sum() > 0:
            logger.warning("NaN values detected in features. Filling with median before SMOTE.")
            X = self._handle_nan_values(X)
        
        # Initialize SMOTE
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=5
        )
        
        # Apply SMOTE
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        logger.info(f"Resampled class distribution: {Counter(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def get_default_params(self, scale_pos_weight: float = 1.0) -> Dict:
        """
        Get default XGBoost parameters optimized for fraud detection.
        
        Args:
            scale_pos_weight: Weight for positive class
            
        Returns:
            Dictionary of XGBoost parameters
        """
        params = {
            # Model type
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            
            # Tree structure
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            
            # Learning rate
            'learning_rate': 0.1,
            'n_estimators': 200,
            
            # Regularization (prevent overfitting)
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'gamma': 0.1,  # Minimum loss reduction
            
            # Class imbalance handling
            'scale_pos_weight': scale_pos_weight,
            
            # Performance
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0,
            
            # Early stopping
            'early_stopping_rounds': 20
        }
        
        return params
    
    def train(self, X: pd.DataFrame, y: pd.Series,
             use_smote: bool = True,
             smote_ratio: float = 0.5,
             test_size: float = 0.2,
             params: Optional[Dict] = None,
             verbose: bool = True) -> Dict:
        """
        Train XGBoost model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            use_smote: Whether to apply SMOTE
            smote_ratio: SMOTE sampling strategy
            test_size: Test set proportion
            params: Model parameters (if None, use defaults)
            verbose: Whether to print training progress
            
        Returns:
            Training report dictionary
        """
        logger.info("Starting model training")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Handle any remaining NaN values in the data
        if X.isnull().sum().sum() > 0:
            logger.warning(f"Found {X.isnull().sum().sum()} NaN values in features. Filling with median.")
            X = X.fillna(X.median())
        
        # Calculate scale_pos_weight
        scale_pos_weight = self.calculate_scale_pos_weight(y.values)
        
        # Get parameters
        if params is None:
            params = self.get_default_params(scale_pos_weight)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42,
            stratify=y  # Maintain class distribution
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Apply SMOTE if requested
        if use_smote:
            X_train, y_train = self.apply_smote(
                X_train, y_train, 
                sampling_strategy=smote_ratio
            )
        
        # Initialize model
        self.model = XGBClassifier(**params)
        
        # Train model
        if verbose:
            logger.info("Training XGBoost model...")
        
        # Use early stopping
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=verbose
            )
        except Exception as e:
            # If early stopping fails, train without it
            logger.warning(f"Early stopping failed: {e}. Training without early stopping.")
            params_no_early_stop = params.copy()
            params_no_early_stop.pop('early_stopping_rounds', None)
            self.model = XGBClassifier(**params_no_early_stop)
            self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Generate report
        report = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'use_smote': use_smote,
            'smote_ratio': smote_ratio if use_smote else None,
            'n_features': len(self.feature_columns),
            'params': params
        }
        
        self.is_trained = True
        
        if verbose:
            logger.info(f"Training complete! ROC-AUC: {roc_auc:.4f}")
        
        return report
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv: int = 5, use_smote: bool = True) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of folds
            use_smote: Whether to apply SMOTE
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        # Handle NaN values
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        # Initialize model
        scale_pos_weight = self.calculate_scale_pos_weight(y.values)
        params = self.get_default_params(scale_pos_weight)
        params.pop('early_stopping_rounds', None)
        
        model = XGBClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        results = {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std()
        }
        
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, filepath)
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'model_type': 'XGBoost'
        }
        metadata_path = filepath.replace('.pkl', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file.
        
        Args:
            filepath: Path to load the model from
        """
        self.model = joblib.load(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.feature_columns = metadata.get('feature_columns', [])
            self.is_trained = metadata.get('is_trained', True)
        
        logger.info(f"Model loaded from {filepath}")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities


def train_fraud_model(X: pd.DataFrame, y: pd.Series,
                     use_smote: bool = True,
                     model_path: Optional[str] = None) -> Tuple[XGBClassifier, Dict]:
    """
    Convenience function to train a fraud detection model.
    
    Args:
        X: Features
        y: Target
        use_smote: Whether to use SMOTE
        model_path: Path to save model (optional)
        
    Returns:
        Tuple of (trained model, training report)
    """
    trainer = ModelTrainer()
    report = trainer.train(X, y, use_smote=use_smote)
    
    if model_path:
        trainer.save_model(model_path)
    
    return trainer.model, report


# Testing the model trainer
if __name__ == "__main__":
    from data_preprocessing import create_sample_dataset, DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Test with sample data
    print("Testing ModelTrainer...")
    
    # Create and preprocess sample data
    df = create_sample_dataset(n_samples=5000, fraud_ratio=0.02)
    preprocessor = DataPreprocessor()
    df_processed, _ = preprocessor.preprocess(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    X = engineer.transform(
        df_processed.drop('Class', axis=1),
        apply_time_features=True,
        apply_amount_features=True,
        scale=True,
        fit=True
    )
    y = df_processed['Class']
    
    # Train model
    trainer = ModelTrainer()
    report = trainer.train(X, y, use_smote=True, smote_ratio=0.5)
    
    print(f"\nTraining Report:")
    print(f"  ROC-AUC: {report['roc_auc']:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Train size: {report['train_size']}")
    print(f"  Test size: {report['test_size']}")
    
    # Get feature importance
    importance = trainer.get_feature_importance()
    print(f"\nTop 10 Important Features:")
    print(importance.head(10))
    
    print("\nModelTrainer test completed successfully!")
