# Deployment Guide
## Credit Card Fraud Detection System

This guide provides step-by-step instructions for deploying the Credit Card Fraud Detection System to Streamlit Cloud.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Testing](#local-testing)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Common Deployment Mistakes](#common-deployment-mistakes-to-avoid)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before deploying, ensure you have:

- Python 3.9 or higher
- Git installed
- GitHub account
- Streamlit Cloud account (free)

---

## Local Testing

### Step 1: Install Dependencies

```
bash
# Navigate to project directory
cd credit_card_fraud_system

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Locally

```
bash
# Run the Streamlit app
streamlit run fraud_detection_system/app.py
```

The application will open in your browser at `http://localhost:8501`.

### Step 3: Test the Application

1. **Home Page**: Verify all information displays correctly
2. **Upload**: Test with the sample CSV file
3. **Dashboard**: Verify all charts render properly
4. **Download**: Test the download functionality

---

## Streamlit Cloud Deployment

### Step 1: Prepare Your Repository

Your project structure should look like this:

```
credit_card_fraud_system/
├── fraud_detection_system/
│   ├── __init__.py
│   ├── app.py                    # Main Streamlit app
│   ├── ml_pipeline/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── prediction_pipeline.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── file_handler.py
│   │   └── chart_generator.py
│   └── data/
│       └── sample_transaction.csv
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── DEPLOYMENT.md
└── TODO.md
```

### Step 2: Push to GitHub

```
bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - Fraud Detection System"

# Create repository on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-system.git
git push -u origin main
```

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/fraud-detection-system`
   - Select branch: `main`
   - Main file path: `fraud_detection_system/app.py`

3. **Configure Advanced Settings** (if needed)
   - Python version: 3.9 or 3.10
   - Requirements file: `requirements.txt`

4. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete (2-5 minutes)

5. **Access Your App**
   - Once deployed, you'll get a URL like: `https://fraud-detection-system.streamlit.app`

---

## Common Deployment Mistakes to Avoid

### 1. File Path Issues ❌

**Mistake**: Using hardcoded local paths
```
python
# WRONG
model = joblib.load("C:/Users/Admin/models/model.pkl")
```

**Correct**: Use relative paths
```
python
# CORRECT
import os
model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
model = joblib.load(model_path)
```

### 2. Missing Dependencies ❌

**Mistake**: Incomplete requirements.txt

**Correct**: Ensure all dependencies are listed:
```
streamlit>=1.24.0
pandas>=1.5.0
numpy>=1.23.0
xgboost>=1.7.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
plotly>=5.14.0
joblib>=1.2.0
```

### 3. Import Errors ❌

**Mistake**: Incorrect import statements

**Correct**: Use proper relative imports:
```
python
# In app.py
from ml_pipeline.data_preprocessing import DataPreprocessor

# Or add to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

### 4. Memory Issues ❌

**Mistake**: Loading large datasets into memory

**Correct**: Add file size limits and streaming:
```
python
# Limit file size
MAX_FILE_SIZE_MB = 50

if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
    st.error("File too large!")
```

### 5. Missing __init__.py ❌

**Mistake**: Not including __init__.py files in packages

**Correct**: Always include empty __init__.py in each package directory

---

## Troubleshooting

### Issue: App Won't Start

**Solution**: Check for errors in `app.py`
```
bash
# Test locally first
streamlit run fraud_detection_system/app.py
```

### Issue: Import Errors

**Solution**: Verify package structure
```
fraud_detection_system/
├── __init__.py        # Must exist
├── app.py
└── ml_pipeline/
    └── __init__.py    # Must exist
```

### Issue: Model Not Loading

**Solution**: Check model path exists
```
python
import os
model_path = "models/xgboost_fraud_model.pkl"
if not os.path.exists(model_path):
    st.warning("Model not found. Using simulated predictions.")
```

### Issue: Charts Not Rendering

**Solution**: Verify Plotly is installed
```
bash
pip install plotly>=5.14.0
```

---

## Environment Variables

For production, you can use environment variables:

```
python
import os

# Get environment variables
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model.pkl')
API_KEY = os.environ.get('API_KEY', '')
```

---

## Security Best Practices

1. **Don't commit sensitive data**
   - Add `.env` to `.gitignore`
   - Never commit API keys or passwords

2. **Validate file uploads**
   - Check file type
   - Check file size
   - Sanitize filenames

3. **Handle errors gracefully**
   - Use try-except blocks
   - Show user-friendly error messages

---

## Performance Optimization

1. **Cache expensive operations**
```
python
@st.cache_data
def load_model():
    return joblib.load("model.pkl")
```

2. **Use session state**
```
python
if 'model' not in st.session_state:
    st.session_state.model = load_model()
```

3. **Limit data processing**
```
python
MAX_ROWS = 500000
if len(df) > MAX_ROWS:
    st.error("Dataset too large!")
```

---

## Monitoring

After deployment, monitor:

1. **App usage** - Streamlit Cloud dashboard
2. **Errors** - Check Streamlit Cloud logs
3. **Performance** - Response times

---

## Support

If you encounter issues:

1. Check the logs in Streamlit Cloud
2. Test locally first
3. Verify all dependencies in requirements.txt
4. Check GitHub Issues

---

**Happy Deployment! 🚀**
