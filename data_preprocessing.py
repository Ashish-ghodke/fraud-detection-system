"""
Data Preprocessing Module
=========================
This module handles all data preprocessing tasks including:
- Data validation
- Missing value handling
- Duplicate removal
- Data type conversion
- Schema validation

Author: Fraud Detection Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A comprehensive data preprocessor for fraud detection datasets.
    
    This class handles all data cleaning and validation tasks required
    before feeding data into the ML pipeline.
    """
    
    # Required columns for fraud detection dataset
    REQUIRED_COLUMNS = [
        'Time', 
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
        'Amount',
        'Class'
    ]
    
    # Alternative column names that might be used
    ALTERNATIVE_COLUMNS = {
        'class': 'Class',
        'fraud': 'Class',
        'is_fraud': 'Class',
        'transaction_amount': 'Amount',
        'amount': 'Amount',
        'time': 'Time',
        'transaction_time': 'Time'
    }
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.processed_columns = []
        self.missing_values_handled = False
        self.duplicates_removed = False
        self.validation_passed = False
        self.validation_errors = []
        
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the dataset schema.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if DataFrame is empty
        if df is None or df.empty:
            errors.append("Dataset is empty")
            return False, errors
        
        # Check column count (should have at least 30 columns for V1-V28 + Time + Amount + Class)
        if len(df.columns) < 30:
            errors.append(f"Dataset has only {len(df.columns)} columns. Expected at least 30 columns.")
        
        # Check for required columns (allow flexible naming)
        available_cols = {col.lower(): col for col in df.columns}
        
        # Check for key columns
        has_amount = any(amt in available_cols for amt in ['amount', 'transaction_amount'])
        has_class = any(cls in available_cols for cls in ['class', 'fraud', 'is_fraud'])
        
        if not has_amount:
            errors.append("Missing required column: Amount (or transaction_amount)")
        if not has_class:
            errors.append("Missing required column: Class (or fraud, is_fraud)")
        
        # Check for V features (V1-V28)
        v_columns = [col for col in df.columns if str(col).upper().startswith('V')]
        if len(v_columns) < 10:
            logger.warning(f"Found only {len(v_columns)} V columns. Ideally should have 28.")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to standard format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized column names
        """
        df = df.copy()
        
        # Create column mapping
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in self.ALTERNATIVE_COLUMNS:
                column_mapping[col] = self.ALTERNATIVE_COLUMNS[col_lower]
        
        # Rename columns
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.info(f"Renamed columns: {column_mapping}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("No missing values found")
            self.missing_values_handled = True
            return df
        
        logger.info(f"Found {missing_count} missing values")
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if strategy == 'drop':
            # Drop rows with missing values
            df = df.dropna()
            logger.info("Dropped rows with missing values")
            
        elif strategy == 'mean':
            # Fill numeric columns with mean
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
            logger.info("Filled missing values with mean for numeric columns")
            
        elif strategy == 'median':
            # Fill numeric columns with median
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
            logger.info("Filled missing values with median for numeric columns")
        
        self.missing_values_handled = True
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (DataFrame without duplicates, number of duplicates removed)
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        self.duplicates_removed = True
        return df, duplicates_removed
    
    def ensure_numeric_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all feature columns are in numeric format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with numeric columns
        """
        df = df.copy()
        
        # Get all columns except Class
        feature_cols = [col for col in df.columns if col != 'Class']
        
        for col in feature_cols:
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert column {col} to numeric: {e}")
        
        # Check for non-numeric values after conversion
        non_numeric = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric:
            logger.warning(f"Non-numeric columns found: {non_numeric}")
            # Drop non-numeric columns except Class
            cols_to_drop = [col for col in non_numeric if col in feature_cols]
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped non-numeric columns: {cols_to_drop}")
        
        return df
    
    def validate_class_column(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate the Class column values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if 'Class' not in df.columns:
            return False, "Class column not found"
        
        # Check unique values
        unique_values = df['Class'].unique()
        
        # Check if binary (0 and 1)
        if set(unique_values).issubset({0, 1, 0.0, 1.0}):
            return True, ""
        
        # Check if it contains string values
        if df['Class'].dtype == 'object':
            # Try to map string values
            try:
                df['Class'] = df['Class'].map({'fraud': 1, 'non-fraud': 0, 'non fraud': 0, 
                                                'normal': 0, 'legitimate': 0})
                if df['Class'].notna().all():
                    return True, ""
            except:
                pass
            return False, f"Invalid Class values: {unique_values}. Expected 0 (non-fraud) or 1 (fraud)"
        
        return False, f"Invalid Class values: {unique_values}. Expected 0 (non-fraud) or 1 (fraud)"
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature columns for model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of feature column names
        """
        # Exclude Class column
        feature_cols = [col for col in df.columns if col != 'Class']
        return feature_cols
    
    def preprocess(self, df: pd.DataFrame, 
                   handle_missing: str = 'mean',
                   remove_duplicates: bool = True,
                   ensure_numeric: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            handle_missing: Strategy for missing values
            remove_duplicates: Whether to remove duplicates
            ensure_numeric: Whether to ensure numeric format
            
        Returns:
            Tuple of (preprocessed DataFrame, processing report)
        """
        logger.info("Starting data preprocessing pipeline")
        
        report = {
            'initial_rows': len(df),
            'initial_columns': len(df.columns),
            'duplicates_removed': 0,
            'missing_values_handled': False,
            'numeric_format_ensured': False,
            'validation_passed': False,
            'validation_errors': []
        }
        
        # Step 1: Validate schema
        is_valid, errors = self.validate_schema(df)
        if not is_valid:
            report['validation_errors'] = errors
            logger.error(f"Schema validation failed: {errors}")
            return df, report
        
        # Step 2: Normalize column names
        df = self.normalize_column_names(df)
        
        # Step 3: Validate class column
        is_valid, error = self.validate_class_column(df)
        if not is_valid:
            report['validation_errors'] = [error]
            logger.error(f"Class validation failed: {error}")
            return df, report
        
        # Step 4: Handle missing values
        df = self.handle_missing_values(df, strategy=handle_missing)
        report['missing_values_handled'] = True
        
        # Step 5: Remove duplicates
        if remove_duplicates:
            df, duplicates_removed = self.remove_duplicates(df)
            report['duplicates_removed'] = duplicates_removed
        
        # Step 6: Ensure numeric format
        if ensure_numeric:
            df = self.ensure_numeric_format(df)
            report['numeric_format_ensured'] = True
        
        # Update report
        report['final_rows'] = len(df)
        report['final_columns'] = len(df.columns)
        report['validation_passed'] = True
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        
        return df, report


def create_sample_dataset(n_samples: int = 1000, fraud_ratio: float = 0.02) -> pd.DataFrame:
    """
    Create a sample dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        fraud_ratio: Ratio of fraud transactions (0.0 to 1.0)
        
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    # Calculate fraud and non-fraud samples
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Generate V1-V28 features (simulated)
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),  # Seconds in 48 hours
        'Amount': np.random.exponential(100, n_samples),
    }
    
    # Generate V features for normal transactions
    for i in range(1, 29):
        v_col = f'V{i}'
        normal_values = np.random.normal(0, 1, n_normal)
        fraud_values = np.random.normal(1.5, 2, n_fraud)  # Slightly different for fraud
        data[v_col] = np.concatenate([normal_values, fraud_values])
    
    # Create Class column
    class_values = [0] * n_normal + [1] * n_fraud
    np.random.shuffle(class_values)
    data['Class'] = class_values
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


# Testing the preprocessor
if __name__ == "__main__":
    # Test with sample data
    print("Testing DataPreprocessor...")
    
    # Create sample dataset
    df = create_sample_dataset(n_samples=1000)
    print(f"Created sample dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Test preprocessing
    preprocessor = DataPreprocessor()
    df_processed, report = preprocessor.preprocess(df)
    
    print(f"\nPreprocessing Report:")
    print(f"  Initial rows: {report['initial_rows']}")
    print(f"  Final rows: {report['final_rows']}")
    print(f"  Duplicates removed: {report['duplicates_removed']}")
    print(f"  Validation passed: {report['validation_passed']}")
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print("DataPreprocessor test completed successfully!")
