# Credit Card Fraud Detection System - Architecture Design

## 1. Executive Summary

This document outlines the system architecture for a production-level Credit Card Fraud Detection System using XGBoost as the primary machine learning model. The system is designed to be deployed on Streamlit Cloud with a focus on stability, scalability, and deployment safety.

---

## 2. System Architecture Overview

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CREDIT CARD FRAUD DETECTION SYSTEM                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────────────┐  │
│  │   FRONTEND   │      │   BACKEND    │      │      ML PIPELINE        │  │
│  │  (Streamlit) │◄────►│  (Python)    │◄────►│     (XGBoost + SMOTE)    │  │
│  └──────────────┘      └──────────────┘      └──────────────────────────┘  │
│         │                     │                        │                    │
│         │                     │                        │                    │
│         ▼                     ▼                        ▼                    │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────────────┐  │
│  │  - Home Page │      │ - File Upload│      │  - Data Validation       │  │
│  │  - Upload    │      │ - Validation │      │  - Preprocessing         │  │
│  │  - Dashboard │      │ - Prediction │      │  - Model Inference       │  │
│  │  - Charts    │      │ - Dashboard  │      │  - Result Generation     │  │
│  └──────────────┘      └──────────────┘      └──────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Streamlit | Web UI framework |
| Backend | Python 3.9+ | Server-side logic |
| ML Model | XGBoost | Fraud classification |
| Data Processing | Pandas, NumPy | Data manipulation |
| Preprocessing | Scikit-learn | Scaling, encoding |
| Imbalance Handling | SMOTE | Synthetic oversampling |
| Model Serialization | Joblib | Model persistence |
| Visualization | Plotly, Matplotlib | Dashboard charts |

---

## 3. Component Architecture

### 3.1 Frontend Layer (Streamlit)

```
┌─────────────────────────────────────────────────────────────┐
│                      STREAMLIT APP                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  HOME PAGE  │  │ UPLOAD PAGE │  │   DASHBOARD PAGE   │  │
│  │             │  │             │  │                     │  │
│  │ - Title     │  │ - File      │  │ - Metrics Cards     │  │
│  │ - Intro     │  │   Uploader  │  │ - Bar Chart         │  │
│  │ - XGBoost   │  │ - Validate  │  │ - Pie Chart         │  │
│  │   Explain   │  │ - Process   │  │ - Histogram         │  │
│  │ - Guide     │  │ - Results   │  │ - Line Chart        │  │
│  │             │  │             │  │ - Data Table        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Backend Layer

```
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND LAYER                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │   FILE HANDLER       │  │    VALIDATION ENGINE        │ │
│  │                      │  │                              │ │
│  │ - CSV/Excel Reader   │  │ - Schema Validation         │ │
│  │ - File Size Check    │  │ - Column Verification        │ │
│  │ - Format Detection   │  │ - Data Type Check           │ │
│  │ - Error Recovery     │  │ - Missing Value Check       │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │   ML PIPELINE        │  │    RESPONSE HANDLER         │ │
│  │                      │  │                              │ │
│  │ - Data Preprocessing │  │ - JSON Response             │ │
│  │ - Feature Engineering│  │ - Success/Error Messages    │ │
│  │ - Model Loading      │  │ - Download Results          │ │
│  │ - Prediction         │  │ - Dashboard Data Prep      │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ML PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT DATA                                                              │
│      │                                                                   │
│      ▼                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐    │
│  │  LOAD DATA  │────►│   VALIDATE  │────►│    CLEAN DATA           │    │
│  │             │     │   SCHEMA    │     │                         │    │
│  │ - CSV read  │     │             │     │ - Handle missing values│    │
│  │ - Excel     │     │ - Required  │     │ - Remove duplicates    │    │
│  │   read      │     │   columns   │     │ - Data type conversion │    │
│  └─────────────┘     └─────────────┘     └─────────────────────────┘    │
│                                                    │                     │
│                                                    ▼                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐    │
│  │   SAVE &    │◄────│  PREDICT    │◄────│  FEATURE ENGINEERING    │    │
│  │   EXPORT    │     │             │     │                         │    │
│  │             │     │ - Load      │     │ - Scale numeric features│    │
│  │ - Add       │     │   model     │     │ - Encode categorical    │    │
│  │   fraud     │     │ - Predict   │     │ - Feature selection     │    │
│  │   prob      │     │ - Probality │     │ - SMOTE (if training)   │    │
│  │ - Export    │     │   scores    │     │                         │    │
│  │   results   │     └─────────────┘     └─────────────────────────┘    │
│  └─────────────┘                                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Architecture

### 4.1 User Upload Flow

```
USER UPLOADS FILE
       │
       ▼
┌──────────────────┐
│  FILE VALIDATION │
│                  │
│ - Check format   │
│ - Check size     │
│ - Check encoding │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
VALID    INVALID
    │         │
    ▼         ▼
┌────────┐  ┌─────────────┐
│PROCEED │  │SHOW ERROR   │
│TO ML   │  │+RE-UPLOAD   │
│PIPELINE│  │OPTION       │
└────────┘  └─────────────┘
```

### 4.2 Prediction Flow

```
VALIDATED DATA
      │
      ▼
┌──────────────────┐
│  PREPROCESSING   │
│                  │
│ - Clean data     │
│ - Handle missing │
│ - Scale features │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LOAD XGBOOST     │
│ MODEL            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  PREDICTION      │
│                  │
│ - Get fraud prob │
│ - Classify       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  PREPARE RESULTS │
│                  │
│ - Add columns    │
│ - Format output  │
└────────┬─────────┘
         │
         ▼
   DASHBOARD + EXPORT
```

---

## 5. Folder Structure

```
credit_card_fraud_system/
│
├── 📁 fraud_detection_system/          # Main application package
│   ├── __init__.py                     # Package initialization
│   ├── app.py                          # Main Streamlit application
│   │
│   ├── 📁 ml_pipeline/                 # ML components
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py       # Data cleaning & validation
│   │   ├── feature_engineering.py      # Feature transformations
│   │   ├── model_training.py           # XGBoost training with SMOTE
│   │   ├── model_evaluation.py         # Model metrics
│   │   └── prediction_pipeline.py      # Inference pipeline
│   │
│   ├── 📁 models/                       # Saved models
│   │   ├── xgboost_fraud_model.pkl     # Trained XGBoost model
│   │   └── scaler.pkl                  # Fitted StandardScaler
│   │
│   ├── 📁 utils/                        # Helper utilities
│   │   ├── __init__.py
│   │   ├── file_handler.py             # File upload & validation
│   │   ├── chart_generator.py          # Plotly chart generation
│   │   └── constants.py                # App constants
│   │
│   └── 📁 data/                         # Sample data
│       └── sample_transaction.csv      # Sample dataset
│
├── 📁 tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_pipeline.py
│   └── test_integration.py
│
├── requirements.txt                     # Python dependencies
├── setup.py                             # Package setup
├── README.md                            # Project documentation
├── ARCHITECTURE.md                      # This architecture doc
├── DEPLOYMENT.md                        # Deployment guide
└── TODO.md                              # Task tracker
```

---

## 6. Database & Model Storage

### 6.1 Model Storage Strategy

| Model File | Storage | Loading Method |
|------------|---------|----------------|
| XGBoost Model | `models/xgboost_fraud_model.pkl` | Joblib load |
| StandardScaler | `models/scaler.pkl` | Joblib load |
| Label Encoder | `models/label_encoder.pkl` | Joblib load |

### 6.2 Data Handling

- **Input**: CSV or Excel files (max 50MB)
- **Processing**: In-memory pandas DataFrames
- **Output**: CSV with prediction results
- **No persistent database required** (stateless design)

---

## 7. API & Integration Design

### 7.1 Streamlit Page Structure

```
python
# Page Navigation
pages = {
    "Home": "/",
    "Upload & Analyze": "/upload",
    "Dashboard": "/dashboard"
}
```

### 7.2 Session State Management

```
python
st.session_state = {
    "uploaded_file": None,
    "processed_data": None,
    "predictions": None,
    "model_metrics": None,
    "analysis_complete": False
}
```

---

## 8. Security Architecture

### 8.1 Input Validation

- File type validation (CSV, XLSX only)
- File size limit (50MB max)
- Column schema validation
- Data type enforcement
- Special character sanitization

### 8.2 Error Handling

- Graceful degradation on model failure
- User-friendly error messages
- Retry mechanism for transient failures
- Comprehensive logging

---

## 9. Performance Architecture

### 9.1 Optimization Strategies

| Component | Optimization |
|-----------|--------------|
| File Upload | Chunk reading for large files |
| Data Processing | Vectorized operations with NumPy |
| Model Inference | Batch prediction |
| Visualization | Cached chart generation |
| Memory | Delete unused DataFrames |

### 9.2 Resource Limits

- Max file size: 50MB
- Max rows: 500,000
- Max prediction time: 60 seconds
- Memory limit: 500MB

---

## 10. Deployment Architecture

### 10.1 Streamlit Cloud Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT CLOUD                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 REPOSITORY                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│  │  │ app.py  │  │  ml/    │  │ models/ │  │utils/   │  │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            BUILD & DEPLOY                           │   │
│  │  - pip install -r requirements.txt                  │   │
│  │  - python -m streamlit run app.py                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Environment Variables

```
bash
# For production deployment
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

---

## 11. Key Design Decisions

### 11.1 Why XGBoost?

| Factor | XGBoost Advantage |
|--------|-------------------|
| Imbalanced Data | Built-in scale_pos_weight parameter |
| Performance | Fast training and prediction |
| Interpretability | Feature importance scores |
| Regularization | L1/L2 regularization prevents overfitting |
| Missing Values | Native handling |

### 11.2 Why SMOTE?

- Synthetic Minority Oversampling Technique
- Creates synthetic samples for minority class
- Avoids overfitting from simple duplication
- Works well with XGBoost

### 11.3 Why Streamlit?

- Rapid prototyping
- Built-in data visualization
- Easy deployment to cloud
- No frontend expertise required
- Production-ready with proper caching

---

## 12. Scalability Considerations

### 12.1 Horizontal Scaling

- Streamlit handles concurrent users
- Stateless design allows load balancing
- Model files are read-only (thread-safe)

### 12.2 Vertical Scaling

- Model inference is fast (<1s for 10k rows)
- Data processing optimized with pandas
- Memory-efficient design

---

## 13. Monitoring & Logging

### 13.1 Application Logging

```
python
import logging

# Log levels
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log events
logger.info("File uploaded successfully")
logger.info("Prediction completed")
logger.error("Validation failed")
```

### 13.2 Metrics to Track

- File upload success rate
- Validation pass rate
- Prediction latency
- Memory usage
- Error frequency

---

## 14. Summary

This architecture provides:

1. **Separation of Concerns**: Clear boundaries between UI, business logic, and ML components
2. **Production-Ready**: Error handling, validation, and monitoring built-in
3. **Deployment-Safe**: Streamlit Cloud compatible, no local dependencies
4. **Scalable**: Stateless design, efficient processing
5. **Maintainable**: Modular structure, clear documentation
6. **Robust**: Input validation, error recovery, logging

---

**Next Steps:**
1. Create folder structure
2. Set up requirements.txt
3. Implement ML pipeline
4. Build Streamlit application
5. Test locally
6. Deploy to Streamlit Cloud

---

*Architecture Version: 1.0*
*Created: 2024*
*Last Updated: 2024*
