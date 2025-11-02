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
                'technical_issue_flag',
                'pricing_concern_flag',
                'competitor_mention_flag'
            ]

@dataclass
class ModelConfig:
    """Model training configuration"""

    # Class imbalance handling
    use_class_weights: bool = True

    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 5

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
    penalty: str = 'l2' # Regularization type
    C: float = 1.0 # Invers of regularization strength
    solver: str = 'lbfgs' # Optimization algorithm
    max_iter: int = 1000 # Maximum iterations for convergence
    class_weight: str = 'balanced' # Handle class imbalance
    random_state: int = 42 # For reproducibility

    # Grid search paramete 
    param_grid: Dict[str, List[Any]] = None

    def __post_init__(self):
        """Initialize parameter grid"""
        if self.param_grid is None:
            self.param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }

@dataclass
class RandomForestConfig:
    """Random Forest hyperparameters"""

    # Model parameters
    n_estimators: int = 100 # Number of trees
    max_depth: int = 10 # Maximum tree depth
    min_samples_split: int = 5 # Minimum samples to split a node
    min_samples_leaf: int = 2 # Minimum samples at a leaf node
    max_features: str = 'sqrt' # Number of features to consider at each split
    class_weight: str = 'balanced' # Handle class imbalance
    random_state: int = 42 # For reproducibility
    n_jobs: int = -1 # Use all available cores

    # Grid search parameters
    param_grid: Dict[str, List[Any]] = None

    def __post_init__(self):
        """Initialize parameter grid"""
        if self.param_grid is None:
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

@dataclass
class XGBoostConfig:
    """XGBoost hyperparameters"""

    # Model Parameters
    n_estimators: int = 100 # Number of trees
    max_depth: int = 6 # Maximum tree depth
    learning_rate: float = 0.1 # Step size
    subsample: float = 0.8 # Subsample ratio
    colsample_bytree: float = 0.8 # Feature subsample ratio
    gamma: float = 0 # Minimum loss reduction
    min_child_weight: int = 1 # Minimum sum of instance weight
    scale_pos_weight: float = 1.0 # Control balance of positive and negative weights
    random_state: int = 42
    n_jobs: int = -1 # Use all available cores

    # Grid search parameters
    param_grid: Dict[str, List[Any]] = None

    def __post_init__(self):
        """Initialize parameter grid"""
        if self.param_grid is None:
            self.param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2]
            }

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""

    # Metrics to calculate
    calculate_accuracy: bool = True # Whether to calculate accuracy
    calculate_precision: bool = True # Whether to calculate precision
    calculate_recall: bool = True # Whether to calculate recall
    calculate_f1: bool = True # Whether to calculate F1-score
    calculate_auc: bool = True # Whether to calculate AUC-ROC
    calculate_confussion_matrix: bool = True # Whether to calculate confusion matrix

    # Visualizations
    plot_confussion_matrix: bool = True # Whether to plot confusion matrix
    plot_roc_curve: bool = True # Whether to plot ROC curve
    plot_feature_importance: bool = True # Whether to plot feature importance
    plot_precision_recall_curve: bool = True # Whether to plot precision-recall curve

    # Output settings
    save_plots: bool = True # Whether to save plots to disk
    save_metrics: bool = True # Whether to save metrics to disk
    verbose: bool = True # Whether to print detailed evaluation info

# Global configuration instances
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
LOGISTIC_CONFIG = LogisticRegressionConfig()
RANDOM_FOREST_CONFIG = RandomForestConfig()
XGBOOST_CONFIG = XGBoostConfig()
EVALUATION_CONFIG = EvaluationConfig()


# Utility function to calculate scale_pos_weight for XGBoost
def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Calculate scale_pos_weight for XGBoost based on class imbalance
    
    Args:
        y_train: Training labels
        
    Returns:
        scale_pos_weight: Ratio of negative to positive samples
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    
    if n_positive == 0:
        return 1.0
    
    return n_negative / n_positive

# Model name mappings
MODEL_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost'
}


# Feature importance display names (for better visualization)
FEATURE_DISPLAY_NAMES = {
    'avg_sentiment': 'Avg Sentiment',
    'sentiment_std': 'Sentiment Std Dev',
    'negative_ratio': 'Negative Ratio',
    'urgent_count': 'Urgent Count',
    'monthly_value': 'Monthly Value',
    'contract_length_months': 'Contract Length',
    'days_to_renewal': 'Days to Renewal',
    'response_time_hours': 'Response Time',
    'meeting_attendance_rate': 'Meeting Attendance',
    'feature_adoption_score': 'Feature Adoption',
    'technical_issue_flag': 'Technical Issue',
    'pricing_concern_flag': 'Pricing Concern',
    'competitor_mention_flag': 'Competitor Mention'
}