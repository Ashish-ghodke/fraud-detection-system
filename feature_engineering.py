"""
Feature Engineering Module
=========================
This module handles all feature engineering tasks including:
- Feature scaling
- Feature transformation
- Feature selection
- Data transformation for model input

Author: Fraud Detection Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A comprehensive feature engineer for fraud detection datasets.
    
    This class handles all feature transformations required for
    optimal XGBoost model performance.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.scaler = None
        self.scaler_type = None
        self.feature_columns = []
        self.is_fitted = False
        
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature columns (exclude Class column).
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature column names
        """
        feature_cols = [col for col in df.columns if col != 'Class']
        return feature_cols
    
    def _fill_na_with_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN values with median for all numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with NaN values filled
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def apply_log_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply log transformation to specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to transform
            
        Returns:
            DataFrame with log-transformed columns
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                # Add small constant to avoid log(0)
                df[col] = np.log1p(np.abs(df[col]))
                logger.info(f"Applied log transform to {col}")
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from Time column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        if 'Time' in df.columns:
            # Fill any NaN in Time first
            df['Time'] = df['Time'].fillna(df['Time'].median())
            
            # Convert seconds to hours
            df['Time_Hours'] = df['Time'] / 3600
            
            # Extract hour of day (assuming Time is seconds from start)
            df['Hour_of_Day'] = (df['Time'] % 86400) / 3600
            
            # Is it during business hours? (9 AM - 5 PM)
            df['Is_Business_Hours'] = ((df['Hour_of_Day'] >= 9) & 
                                       (df['Hour_of_Day'] <= 17)).astype(int)
            
            # Is it weekend?
            df['Is_Weekend'] = (df['Hour_of_Day'] < 1).astype(int)
            
            logger.info("Created time-based features")
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create amount-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional amount features
        """
        df = df.copy()
        
        if 'Amount' in df.columns:
            # Fill any NaN in Amount first
            df['Amount'] = df['Amount'].fillna(df['Amount'].median())
            
            # Log transform of amount
            df['Amount_Log'] = np.log1p(df['Amount'].clip(lower=0))
            
            # Amount bins - handle NaN from pd.cut
            try:
                df['Amount_Category'] = pd.cut(
                    df['Amount'],
                    bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                    labels=[0, 1, 2, 3, 4, 5]
                ).astype(float)
            except:
                df['Amount_Category'] = 0
            
            # Is high amount? (above $200)
            df['Is_High_Amount'] = (df['Amount'] > 200).astype(int)
            
            # Is small amount? (below $10)
            df['Is_Small_Amount'] = (df['Amount'] < 10).astype(int)
            
            logger.info("Created amount-based features")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between V columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Get V columns
        v_cols = [col for col in df.columns if str(col).startswith('V')]
        
        if len(v_cols) >= 2:
            # Fill NaN in V columns first
            for col in v_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
            
            # Create sum and mean of V features
            df['V_Sum'] = df[v_cols].sum(axis=1)
            df['V_Mean'] = df[v_cols].mean(axis=1)
            df['V_Std'] = df[v_cols].std(axis=1)
            df['V_Max'] = df[v_cols].max(axis=1)
            df['V_Min'] = df[v_cols].min(axis=1)
            
            logger.info("Created V feature interactions")
        
        return df
    
    def scale_features(self, df: pd.DataFrame, 
                      method: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale features using specified method.
        
        Args:
            df: Input DataFrame
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        # Get feature columns (exclude Class)
        feature_cols = [col for col in df.columns if col != 'Class']
        
        # Handle any remaining NaN values before scaling
        nan_count = df[feature_cols].isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values before scaling. Filling with median.")
            for col in feature_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df[col] = df[col].fillna(median_val)
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}. Using StandardScaler.")
            scaler = StandardScaler()
        
        if fit:
            self.scaler = scaler
            self.scaler_type = method
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.is_fitted = True
            logger.info(f"Fitted {method} scaler on {len(feature_cols)} features")
        else:
            if self.scaler is None:
                logger.warning("Scaler not fitted. Returning original data.")
                return df
            df[feature_cols] = scaler.transform(df[feature_cols])
            logger.info(f"Transformed {len(feature_cols)} features using fitted scaler")
        
        self.feature_columns = feature_cols
        return df
    
    def apply_pca(self, df: pd.DataFrame, 
                 n_components: int = 10,
                 fit: bool = True) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            df: Input DataFrame
            n_components: Number of components to keep
            fit: Whether to fit PCA (True for training)
            
        Returns:
            DataFrame with PCA components
        """
        df = df.copy()
        
        feature_cols = self.get_feature_columns(df)
        
        if fit:
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(df[feature_cols])
            logger.info(f"Fitted PCA with {n_components} components. "
                       f"Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        else:
            if not hasattr(self, 'pca'):
                logger.warning("PCA not fitted. Returning original data.")
                return df
            pca_features = self.pca.transform(df[feature_cols])
        
        # Create new column names
        pca_cols = [f'PCA_{i+1}' for i in range(n_components)]
        
        # Create DataFrame with PCA features
        pca_df = pd.DataFrame(pca_features, columns=pca_cols, index=df.index)
        
        # Combine with original (optional: drop original features)
        df = pd.concat([df, pca_df], axis=1)
        
        return df
    
    def select_features(self, df: pd.DataFrame, 
                       feature_importance: Optional[Dict[str, float]] = None,
                       top_n: int = 20) -> pd.DataFrame:
        """
        Select top features based on importance.
        
        Args:
            df: Input DataFrame
            feature_importance: Dictionary of feature importance scores
            top_n: Number of top features to keep
             
        Returns:
            DataFrame with selected features
        """
        df = df.copy()
        
        if feature_importance is None:
            logger.info("No feature importance provided. Keeping all features.")
            return df
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        # Select top N features
        selected_features = [f[0] for f in sorted_features[:top_n]]
        
        # Always include Class column
        if 'Class' in df.columns and 'Class' not in selected_features:
            selected_features.append('Class')
        
        # Also keep non-feature columns (Time, Amount, etc.)
        non_feature_cols = ['Time', 'Amount', 'Time_Hours', 'Hour_of_Day']
        for col in non_feature_cols:
            if col in df.columns and col not in selected_features:
                selected_features.append(col)
        
        df = df[selected_features]
        logger.info(f"Selected top {top_n} features")
        
        return df
    
    def transform(self, df: pd.DataFrame,
                 apply_time_features: bool = True,
                 apply_amount_features: bool = True,
                 apply_interaction_features: bool = True,
                 scale: bool = True,
                 scaler_method: str = 'standard',
                 fit: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            apply_time_features: Whether to create time features
            apply_amount_features: Whether to create amount features
            apply_interaction_features: Whether to create interaction features
            scale: Whether to scale features
            scaler_method: Scaling method to use
            fit: Whether to fit transformers (True for training)
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting feature engineering pipeline")
        
        df = df.copy()
        
        # Fill NaN values before any transformation
        df = self._fill_na_with_median(df)
        
        # Apply feature engineering
        if apply_time_features:
            df = self.create_time_features(df)
        
        if apply_amount_features:
            df = self.create_amount_features(df)
        
        if apply_interaction_features:
            df = self.create_interaction_features(df)
        
        # Fill any NaN created by feature engineering
        df = self._fill_na_with_median(df)
        
        # Scale features
        if scale:
            df = self.scale_features(df, method=scaler_method, fit=fit)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        
        return df
    
    def save_scaler(self, filepath: str) -> None:
        """
        Save the fitted scaler to a file.
        
        Args:
            filepath: Path to save the scaler
        """
        if self.scaler is None:
            logger.warning("No scaler to save")
            return
        
        joblib.dump(self.scaler, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str) -> None:
        """
        Load a fitted scaler from a file.
        
        Args:
            filepath: Path to load the scaler from
        """
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {filepath}")


def create_feature_transformer():
    """
    Create a feature transformer with standard settings.
    
    Returns:
        FeatureEngineer instance
    """
    return FeatureEngineer()


# Testing the feature engineer
if __name__ == "__main__":
    from data_preprocessing import create_sample_dataset, DataPreprocessor
    
    # Test with sample data
    print("Testing FeatureEngineer...")
    
    # Create and preprocess sample data
    df = create_sample_dataset(n_samples=1000)
    preprocessor = DataPreprocessor()
    df_processed, _ = preprocessor.preprocess(df)
    
    # Test feature engineering
    engineer = FeatureEngineer()
    df_transformed = engineer.transform(
        df_processed,
        apply_time_features=True,
        apply_amount_features=True,
        apply_interaction_features=True,
        scale=True,
        scaler_method='standard',
        fit=True
    )
    
    print(f"Original shape: {df_processed.shape}")
    print(f"Transformed shape: {df_transformed.shape}")
    print(f"Columns: {df_transformed.columns.tolist()}")
    print("FeatureEngineer test completed successfully!")
