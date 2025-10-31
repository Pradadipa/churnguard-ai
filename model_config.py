"""
    Model Configuration for Churnguard AI

    Configuration for training, evaluation, and model parameters.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""

    # Data paths
    data_path: str = 'data/synthetic_training_data.csv'
    models_dir: str = 'data/models'
    results_dir: str = 'data/results'

    # Feature Columns 
    feature_columns: List[str] = None  # To be set after data generation
    target_column: str = 'churned'

    # Train/Val/Test split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state = 42

    def __post_init__(self):
        # Define feature columns after data generation
        if self.feature_columns is None: # To be set after data generation
            self.feature_columns = [
                # sentiment features
                'avg_sentiment',
                'sentiment_std',
                'negative_ratio',
                'urgent_count',

                # business metrics
                'monthly_value',
                'contract_length_months',
                'days_to_renewal',

                # behavioral metrics
                'response_time_hours',
                'meeting_attendance_rate',
                'feature_adoption_score',

                # risk flags
                'technical_issues_flag',
                'pricing_concerns_flag',
                'competitor_activity_flag'
            ]

@dataclass
class ModelConfig:
    """Model training configuration"""

    # Class imbalance handling
    use_class_weights: bool = True

    # Cross-validation
    use_cross_validation: bool = True

    # Feature scaling
    scale_features: bool = True
    scaler_type: str = 'standard' # 'standard' or 'minmax'

    # Random state for reproducibility
    random_state: int =  42

    # Performance thresholds
    target_accuracy: float = 0.85
    target_auc: float = 0.85
    target_f1: float = 0.80

@dataclass
class LogisticRegressionConfig:
    """Logistic Regression hyperparameters"""

    # Model parameters
    penalty: str = 'l2'
    C: float = 1.0 # Invers of regularization strength
    solver: str = 'lbfgs'
    max_iter: int = 1000
    class_weight: str = 'balanced'
    random_state: int = 42

    # Grid search paramete 
    param_grid: Dict[str, List[Any]] = None

    def __post_init__(self):
        """Initialize parameter grid"""
        if self.param_grip is None:
            self.param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
