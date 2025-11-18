"""
Compare All 3 Models Side-by-Side
ChurnGuard AI - Week 1, Day 4

Compares:
1. Logistic Regression (baseline)
2. Random Forest (ensemble - parallel)
3. XGBoost (ensemble - sequential)

Creates comprehensive comparison dashboard
"""

import logging
import sys
import pandas as pd 
import numpy as np
import joblib   
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    roc_curve
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_data():
    "load test data"
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    logger.info("Loading test data ...")
    df = pd.read_csv('data/synthetic_training_data.csv')

    feature_columns = [
        'avg_sentiment', 'sentiment_std', 'negative_ratio', 'urgent_count',
        'monthly_value', 'contract_length_months', 'days_to_renewal',
        'response_time_hours', 'meeting_attendance_rate', 'feature_adoption_score',
        'technical_issue_flag', 'pricing_concern_flag', 'competitor_mention_flag'
    ]

    X = df[feature_columns]
    y = df['churned']
    
    # Split 
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Scale for logistic regression
    scaler = StandardScaler()
    scaler.fit(X_temp)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    logger.info(f"✓ Loaded {len(X_test)} test samples")

    return X_test, X_test_scaled, y_test, feature_columns

def load_models():
    """Load all 3 trained models"""
    logger.info("\nLoading trained models ...")

    models = {}

    try:
        models['Logistic Regression'] = joblib.load('data/models/logistic_regression_model.joblib')
        logger.info("✓ Loaded Logistic Regression")
    except:
        logger.warning("⚠ Logistic Regression model not found")

    try:
        models['Random Forest'] = joblib.load('data/models/random_forest_model.joblib')
        logger.info("✓ Loaded Random Forest")
    except:
        logger.warning("⚠ Random Forest model not found")
    
    try:
        models['XGBoost'] = joblib.load('data/models/xgboost_model.joblib')
        logger.info("✓ Loaded XGBoost")
    except:
        logger.warning("⚠ XGBoost model not found")
    
    if len(models) == 0:
        raise ValueError("No models found! Please train models first.")
    
    return models

if __name__ == "__main__":
    data = load_models()
    logger.info(f"Model successfull loaded {len(data)}")


