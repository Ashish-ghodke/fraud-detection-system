# 💳 Credit Card Fraud Detection System

A **Machine Learning based web application** that detects fraudulent credit card transactions in real-time using an **XGBoost classification model**. The system processes transaction data, performs preprocessing and feature scaling, and predicts whether a transaction is **fraudulent or legitimate** through an interactive web interface.

This project demonstrates a **complete end-to-end ML pipeline**, including data preprocessing, model training, model serialization, and real-time prediction through a web application.

---

# 📌 Project Overview

Credit card fraud is a critical problem in the financial industry. Detecting fraudulent transactions quickly helps financial institutions reduce losses and improve customer trust.

This project aims to build an intelligent fraud detection system that:

* Uses machine learning to identify suspicious transactions
* Handles **highly imbalanced datasets**
* Provides **real-time predictions through a web interface**
* Demonstrates a **production-style ML workflow**

---

# 🚀 Features

* Fraud detection using **XGBoost**
* Real-time prediction interface
* Data preprocessing and feature scaling
* Modular Python project structure
* Trained model loading for fast inference
* Interactive UI built with Streamlit

---

# 🧠 Machine Learning Model

The project uses **XGBoost (Extreme Gradient Boosting)**, a powerful ensemble learning algorithm widely used for structured data problems.

### Why XGBoost?

* Excellent performance on tabular datasets
* Handles **imbalanced classification problems**
* Fast training and prediction
* High accuracy and scalability

---

# 🛠 Technology Stack

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Core programming language      |
| Pandas       | Data processing                |
| NumPy        | Numerical computations         |
| Scikit-learn | Data preprocessing & utilities |
| XGBoost      | Machine learning model         |
| Streamlit    | Web application interface      |

---

# 📂 Project Structure

```
credit_card_fraud_system/
│
├── app.py                          # Streamlit entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── ARCHITECTURE.md                 # System architecture explanation
├── DEPLOYMENT.md                   # Deployment instructions
├── TODO.md                         # Task tracker / improvements
│
├── models/                         # Serialized ML artifacts
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── fraud_detection_system/         # Core application package
│   ├── __init__.py
│   │
│   ├── ml_pipeline/                # Machine learning pipeline
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py   # Data cleaning & validation
│   │   ├── feature_engineering.py  # Feature transformations
│   │   ├── model_training.py       # XGBoost training + SMOTE
│   │   ├── model_evaluation.py     # Metrics & evaluation
│   │   └── prediction_pipeline.py  # Inference logic
│   │
│   ├── utils/                      # Utility helpers
│   │   ├── __init__.py
│   │   ├── constants.py            # App configuration
│   │   ├── file_handler.py         # File upload validation
│   │   └── chart_generator.py      # Charts for dashboard
│   │
│   └── data/
│       └── sample_transaction.csv  # Example dataset for testing
│
└── notebooks/                      # Optional: experimentation
    └── model_training.ipynb
```

---

# 🔄 System Architecture

```
User Input
    ↓
Streamlit Web Interface
    ↓
Data Preprocessing
    ↓
Feature Scaling
    ↓
XGBoost Model
    ↓
Fraud / Legitimate Prediction
```

---

# 📊 Model Evaluation

Fraud detection datasets are typically **highly imbalanced**, so multiple evaluation metrics are used to assess performance.

Common evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

These metrics help evaluate how well the model detects fraudulent transactions while minimizing false positives.

---

# ⚙️ Installation and Setup

### 1. Clone the Repository

```
git clone https://github.com/Ashish-ghodke/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run the Application

```
streamlit run app.py
```

The application will launch in your default browser.

---

# 🖥 Application Demo

You can add screenshots of the application here.

Example:

* Transaction input interface
* Fraud prediction output
* System UI

```
(Add screenshots after uploading them to the repository)
```

---

# 🔮 Future Improvements

Possible enhancements for this project:

* Deploy the application on a cloud platform
* Create a REST API for model predictions
* Implement real-time transaction streaming
* Add advanced anomaly detection techniques
* Improve feature engineering

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Ashish Ghodke**
**Sarika Tayde**
**Gayatri Ramne**
**Sarthak Choube**

B.Sc. Data Science
Dr. Babasaheb Ambedkar Marathwada University

GitHub:
https://github.com/Ashish-ghodke

---

⭐ If you find this project helpful, please consider giving it a **star on GitHub**.

