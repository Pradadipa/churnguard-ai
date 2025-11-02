"""
Handles evaluation of machine learning models including metric computation,
visualization generation, and report compilation. 
"""



import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from model_config import (
    EVALUATION_CONFIG,
    DATA_CONFIG,
    FEATURE_DISPLAY_NAMES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluator:
    """Evaluates machine learning models and generates reports."""

    def __init__(self, config=EVALUATION_CONFIG):
        """Initializes the evaluator with configuration settings."""
        self.config = config
        self.metrics = {}

        # Create result directory if it doesn't exist
        Path(DATA_CONFIG.results_dir).mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Evaluates the model and computes metrics.

        Args:
            model: Trained machine learning model.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for test data.
            model_name (str): Name of the model for reporting.
        Returns:
            Dict[str, Any]: Computed evaluation metrics.
        """
        logger.info("="*80)
        logger.info(f"EVALUATING MODEL: {model_name.upper()}")
        logger.info("="*80)

        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {}

        # Accuracy
        if self.config.calculate_accuracy:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)

        # Precision
        if self.config.calculate_precision:
            metrics['precision'] = precision_score(y_test, y_pred)

        # Recall
        if self.config.calculate_recall:
            metrics['recall'] = recall_score(y_test, y_pred)

        # F1 Score
        if self.config.calculate_f1:
            metrics['f1_score'] = f1_score(y_test, y_pred)  
