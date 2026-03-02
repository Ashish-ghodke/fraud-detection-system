# 🔒 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-level web-based Credit Card Fraud Detection System using XGBoost and SMOTE for handling imbalanced datasets.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Format](#dataset-format)
- [Dashboard](#dashboard)
- [Deployment](#deployment)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This system provides a complete end-to-end solution for detecting credit card fraud using machine learning. It features:

- **User-friendly Web Interface**: Built with Streamlit
- **Advanced ML Pipeline**: XGBoost with SMOTE for imbalanced data handling
- **Interactive Dashboard**: Real-time visualizations with Plotly
- **Production-Ready**: Error handling, validation, and deployment-safe code

---

## Features

### 🔍 Fraud Detection
- Automatic data validation and preprocessing
- Feature engineering (time, amount, interaction features)
- XGBoost model with SMOTE for class imbalance
- Fraud probability scoring

### 📊 Interactive Dashboard
- Total transaction metrics
- Fraud vs Non-Fraud distribution
- Transaction amount histograms
- Top suspicious transactions table
- Risk level categorization

### 🛡️ Data Security
- All processing done locally
- No data sent to external servers
- Secure file handling

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Backend | Python 3.9+ |
| ML Model | XGBoost |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn |
| Imbalance Handling | SMOTE |
| Visualization | Plotly |
| Model Serialization | Joblib |

---

## Project Structure

```
credit_card_fraud_system/
├── fraud_detection_system/
│   ├── __init__.py
│   ├── app.py                    # Main Streamlit application
│   ├── ml_pipeline/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py # Data validation & cleaning
│   │   ├── feature_engineering.py # Feature transformations
│   │   ├── model_training.py     # XGBoost training with SMOTE
│   │   ├── model_evaluation.py    # Model metrics
│   │   └── prediction_pipeline.py # Inference pipeline
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py          # App constants
│   │   ├── file_handler.py       # File upload & validation
│   │   └── chart_generator.py    # Dashboard charts
│   └── data/
│       └── sample_transaction.csv
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── ARCHITECTURE.md                # System architecture
├── DEPLOYMENT.md                 # Deployment guide
└── TODO.md                        # Task tracker
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository

```
bash
git clone https://github.com/YOUR_USERNAME/credit_card_fraud_system.git
cd credit_card_fraud_system
```

### Step 2: Create Virtual Environment (Recommended)

```
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```
bash
pip install -r requirements.txt
```

---

## Usage

### Running Locally

```
bash
streamlit run fraud_detection_system/app.py
```

The application will open in your browser at `http://localhost:8501`.

### Using the Application

1. **Home Page**: Learn about the project, XGBoost, and SMOTE
2. **Upload**: Upload your CSV or Excel file with transaction data
3. **Analyze**: Click "Upload and Analyze" to process the data
4. **Dashboard**: View interactive charts and fraud predictions
5. **Download**: Export results as CSV

---

## Dataset Format

Your dataset should contain:

| Column | Description | Required |
|--------|-------------|----------|
| Time | Seconds from first transaction | Yes |
| V1-V28 | Anonymized PCA features | Yes |
| Amount | Transaction amount | Yes |
| Class | 0=Normal, 1=Fraud | For training |

### Example CSV Format:

```
csv
Time,V1,V2,V3,...,V28,Amount,Class
0.0,-1.36,-0.07,2.54,...,0.17,149.62,0
1.0,-1.36,1.19,0.24,...,0.08,2.69,0
```

---

## Dashboard

The dashboard provides:

- 📈 **Metrics Cards**: Total transactions, fraud count, fraud percentage
- 📊 **Bar Chart**: Fraud vs Non-Fraud transactions
- 🥧 **Pie Chart**: Fraud distribution percentage
- 📉 **Histogram**: Transaction amount distribution
- ⚠️ **Top Suspicious**: Table of highest risk transactions

---

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## Model Details

### Why XGBoost?

- **Handles Imbalance**: Built-in `scale_pos_weight` parameter
- **High Performance**: Fast training and prediction
- **Regularization**: L1/L2 to prevent overfitting
- **Missing Values**: Native handling
- **Interpretability**: Feature importance scores

### Why SMOTE?

- **Synthetic Oversampling**: Creates new minority class samples
- **Prevents Overfitting**: More diverse than simple duplication
- **Improves Recall**: Better fraud detection

### Model Performance

The model is evaluated using:
- ROC-AUC Score
- Precision, Recall, F1-Score
- Confusion Matrix

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Built with ❤️ using XGBoost and Streamlit**
