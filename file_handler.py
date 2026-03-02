"""
File Handler Module
==================
This module handles file upload, validation, and processing.

Author: Fraud Detection Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, Dict
import logging

from .constants import (
    MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    MSG_FILE_TOO_LARGE,
    MSG_INVALID_FILE_TYPE,
    MSG_EMPTY_FILE
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileHandler:
    """
    File handler for uploading and processing CSV/Excel files.
    
    This class handles:
    - File type validation
    - File size validation
    - Data reading (CSV and Excel)
    - Error handling
    """
    
    def __init__(self):
        """Initialize the FileHandler."""
        self.filename = None
        self.file_size = 0
        self.file_extension = None
        self.last_loaded_data = None
        
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Get file name and size
        self.filename = uploaded_file.name
        self.file_size = uploaded_file.size
        
        # Check file size
        if self.file_size > MAX_FILE_SIZE_BYTES:
            return False, MSG_FILE_TOO_LARGE
        
        # Check file extension
        self.file_extension = os.path.splitext(self.filename)[1].lower().replace('.', '')
        
        if self.file_extension not in ALLOWED_EXTENSIONS:
            return False, MSG_INVALID_FILE_TYPE
        
        return True, ""
    
    def read_file(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Read file into DataFrame.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (DataFrame, error_message)
        """
        # Validate file first
        is_valid, error = self.validate_file(uploaded_file)
        if not is_valid:
            return None, error
        
        try:
            df = None
            
            if self.file_extension == 'csv':
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
            elif self.file_extension in ['xlsx', 'xls']:
                # Read Excel file
                df = pd.read_excel(uploaded_file)
            
            # Check if DataFrame is empty
            if df is None or df.empty:
                return None, MSG_EMPTY_FILE
            
            # Store reference
            self.last_loaded_data = df
            
            logger.info(f"Successfully loaded file: {self.filename}")
            logger.info(f"Shape: {df.shape}")
            
            return df, ""
            
        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def get_dataframe_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with DataFrame information
        """
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        return info
    
    def save_dataframe(self, df: pd.DataFrame, filepath: str, format: str = 'csv') -> bool:
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            format: Output format ('csv' or 'excel')
            
        Returns:
            True if successful
        """
        try:
            if format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'excel':
                df.to_excel(filepath, index=False)
            
            logger.info(f"DataFrame saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return False
    
    def convert_to_csv(self, df: pd.DataFrame) -> bytes:
        """
        Convert DataFrame to CSV bytes for download.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            CSV as bytes
        """
        return df.to_csv(index=False).encode('utf-8')


def load_sample_data() -> pd.DataFrame:
    """
    Load sample transaction data for demonstration.
    
    Returns:
        Sample DataFrame
    """
    # Create sample data with realistic patterns
    np.random.seed(42)
    
    n_samples = 1000
    n_fraud = int(n_samples * 0.02)  # 2% fraud
    
    # Generate features
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),
        'Amount': np.random.exponential(100, n_samples),
    }
    
    # Generate V1-V28 features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Adjust some features for fraud cases
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    for i in [1, 2, 3, 4, 14]:
        data[f'V{i}'][fraud_indices] += np.random.normal(2, 1, n_fraud)
    
    data['Amount'][fraud_indices] *= np.random.uniform(2, 5, n_fraud)
    
    # Create class labels
    class_labels = np.zeros(n_samples)
    class_labels[fraud_indices] = 1
    data['Class'] = class_labels
    
    df = pd.DataFrame(data)
    
    return df


# Testing the file handler
if __name__ == "__main__":
    print("Testing FileHandler...")
    
    handler = FileHandler()
    
    # Test loading sample data
    df = load_sample_data()
    print(f"Sample data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get info
    info = handler.get_dataframe_info(df)
    print(f"\nDataFrame Info:")
    print(f"  Rows: {info['rows']}")
    print(f"  Columns: {info['columns']}")
    
    print("\nFileHandler test completed successfully!")
