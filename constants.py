"""
Application Constants
====================
Contains all application-wide constants and configuration values.

Author: Fraud Detection Team
Version: 1.0.0
"""

# Application Info
APP_NAME = "Credit Card Fraud Detection System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-powered fraud detection using XGBoost and SMOTE"

# File Upload Constraints
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']
ALLOWED_FILE_TYPES = ['text/csv', 'application/vnd.ms-excel', 
                      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']

# Data Constraints
MAX_ROWS = 500000
MIN_ROWS = 10
REQUIRED_MIN_COLUMNS = 5

# Model Configuration
DEFAULT_PREDICTION_THRESHOLD = 0.5
DEFAULT_SMOTE_RATIO = 0.5
DEFAULT_TEST_SIZE = 0.2

# Dashboard Configuration
TOP_SUSPICIOUS_TRANSACTIONS = 15
CHART_HEIGHT = 400
CHART_WIDTH = None  # None means auto

# Colors
COLOR_FRAUD = "#FF4B4B"
COLOR_NORMAL = "#00CC96"
COLOR_WARNING = "#FFA15A"
COLOR_CRITICAL = "#FF6692"

# Feature Column Prefixes
V_FEATURE_PREFIX = "V"
AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"
CLASS_COLUMN = "Class"

# Prediction Labels
LABEL_FRAUD = "Fraud"
LABEL_NORMAL = "Normal"

# Risk Levels
RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"
RISK_CRITICAL = "Critical"

# Validation Messages
MSG_FILE_TOO_LARGE = f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
MSG_INVALID_FILE_TYPE = f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
MSG_EMPTY_FILE = "The uploaded file is empty"
MSG_INVALID_SCHEMA = "Invalid dataset schema. Please check the format requirements."
MSG_NO_FRAUD_DETECTED = "No fraud detected in the dataset"
MSG_ANALYSIS_COMPLETE = "Dataset analyzed successfully"

# UI Labels
BTN_UPLOAD_AND_ANALYZE = "Upload and Analyze"
BTN_VIEW_DASHBOARD = "View Dashboard"
BTN_DOWNLOAD_RESULTS = "Download Results"
BTN_RE_UPLOAD = "Upload New File"

# Page Titles
PAGE_HOME = "Home"
PAGE_UPLOAD = "Upload & Analyze"
PAGE_DASHBOARD = "Dashboard"

# Streamlit Config
STREAMLIT_PAGE_ICON = "🔒"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_INITIAL_SIDEBAR = True

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Model Paths (relative to app directory)
MODEL_DIR = "models"
MODEL_FILENAME = "xgboost_fraud_model.pkl"
SCALER_FILENAME = "scaler.pkl"
METADATA_FILENAME = "model_metadata.pkl"

# Sample Data
SAMPLE_DATA_ROWS = 1000
SAMPLE_FRAUD_RATIO = 0.02

# Feature Engineering
TIME_FEATURES = ['Time_Hours', 'Hour_of_Day', 'Is_Business_Hours', 'Is_Weekend']
AMOUNT_FEATURES = ['Amount_Log', 'Amount_Category', 'Is_High_Amount', 'Is_Small_Amount']
INTERACTION_FEATURES = ['V_Sum', 'V_Mean', 'V_Std', 'V_Max', 'V_Min']

# Metrics Display
METRIC_FORMAT = "{:.2f}"
PERCENTAGE_FORMAT = "{:.2f}%"
