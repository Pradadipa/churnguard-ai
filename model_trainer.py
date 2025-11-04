"""
Model Traning Module for ChurnGuard AI

Handles data loading, preprocessing, model training, and hyperparameter tuning
"""

import numpy as np
import pandas as pd 
from typing import Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from model_config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    LOGISTIC_CONFIG,
    RANDOM_FOREST_CONFIG,
    XGBOOST_CONFIG,
    calculate_scale_pos_weight
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChurnDataLoader:
    """ Load and preprocess churn data """

    # Initialization and methods for loading and preprocessing data
    def __init__(self, config=DATA_CONFIG):
        """Initialize with data configuration"""
        self.config = config
        self.scaler = None

        # Create directories if they don't exist
        Path(config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV"""
        
        logger.info(f"Loading data from {self.config.data_path}")
        df = pd.read_csv(self.config.data_path)
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        logger.info(f"Churn rate: {df[self.config.target_column].mean():.2f}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target

        Args:
            df (pd.DataFrame): Complete dataframe

        Returns:
            X: Feature dataframe
            y: Target series     
        """

        X = df[self.config.feature_columns].copy()
        y = df[self.config.target_column].copy()

        logger.info(f"Feature set shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """ 
        Split data into train/validation/test sets (70/15/15)

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test   
        """

        
        logger.info("Splitting data into train/val/test sets...")
        
        # First split into train and temp (val+test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_ratio,
            random_state=self.config.random_state,
            stratify=y
        )

        # Second split temp into val and test
        val_size_adjusted = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )

        logger.info(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"Val set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Log class distributions
        logger.info(f"Train churn rate: {y_train.mean():.2%}")
        logger.info(f"Val churn rate: {y_val.mean():.2%}")
        logger.info(f"Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        scaler_type: str = 'standard'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale feature using StandardScaler or MinMaxScaler
        Fit on train, transform train/val/test

        Returns:
            Scaled X_train, X_val, X_test
        """
        logger.info(f"Scaling features using {scaler_type} scaler...")

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        # Fit scaler on training data
        self.scaler.fit(X_train)

        # Transform all datasets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )            
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )            
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        logger.info("Feature scaling complete.")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_scaler(self, filepath: str = None):
        """Save fitted scaler for production use"""
        if filepath is None:
            filepath = f"{self.config.models_dir}/feature_scaler.joblib"

        if self.scaler is not None:
            joblib.dump(self.scaler, filepath)
            logger.info(f"Saved feature scaler to {filepath}")
        else:
            logger.warning("No scaler to save.")

class ModelTrainer:
    """Train and tune machine learning models for churn prediction"""

    def __init__(self, model_config=MODEL_CONFIG):
        """Initialize with model configuration"""
        self.config = model_config
        self.model = None
        self.best_params = None
        self.cv_scores = None

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = True
    ) -> LogisticRegression:
        """
        Train Logistic Regression model

        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning

        Returns:
            Trained Logistic Regression model
        """
        logger.info("="*80)
        logger.info("TRAINING LOGISTIC REGRESSION MODEL")
        logger.info("="*80)

        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning with GridSearchCV...")

            # Create base model
            base_model = LogisticRegression(
                random_state=LOGISTIC_CONFIG.random_state,
                max_iter=LOGISTIC_CONFIG.max_iter,
                class_weight = 'balanced' if self.config.use_class_weights else None
            )

            # Grid search
            grid_search = GridSearchCV(
                base_model,
                LOGISTIC_CONFIG.param_grid,
                cv=self.config.cv_folds if self.config.use_cross_validation else 3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        else:
            logger.info("Training with deafult hyperparameters...")

            self.model = LogisticRegression(
                C=LOGISTIC_CONFIOG.C,
                penalty=LOGISTIC_CONFIG.penalty,
                solver=LOGISTIC_CONFIG.solver,
                max_iter=LOGISTIC_CONFIG.max_iter,
                class_weight= 'balanced' if self.config.use_class_weights else None,
                random_state=LOGISTIC_CONFIG.random_state
            )

            self.model.fit(X_train, y_train)

        # Cross-validation scores
        if self.config.use_cross_validation:
            self._calculate_cv_scores(X_train, y_train)

        return self.model

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = True
    ) -> RandomForestClassifier:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning

        Returns:
            Trained Random Forest model
        """

        logger.info("="*80)
        logger.info("TRAINING RANDOM FOREST MODEL")
        logger.info("="*80)

        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning with GridSearchCV...")

            # Create base model
            base_model = RandomForestClassifier(
                random_state=RANDOM_FOREST_CONFIG.random_state,
                n_jobs=-1,
                class_weight = 'balanced' if self.config.use_class_weights else None
            )

            # Use smaller param grid for faster tuning
            param_grid_small = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

            grid_search = GridSearchCV(
                base_model,
                param_grid_small,
                cv=self.config.cv_folds if self.config.use_cross_validation else 3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        else:
            logger.info("Training with deafult hyperparameters...")

            self.model = RandomForestClassifier(
                n_estimators=RANDOM_FOREST_CONFIG.n_estimators,
                max_depth=RANDOM_FOREST_CONFIG.max_depth,
                min_samples_split=RANDOM_FOREST_CONFIG.min_samples_split,
                min_samples_leaf=RANDOM_FOREST_CONFIG.min_samples_leaf,
                max_features=RANDOM_FOREST_CONFIG.max_features,
                class_weight= 'balanced' if self.config.use_class_weights else None,
                random_state=RANDOM_FOREST_CONFIG.random_state,
                n_jobs=-1
            )

            self.model.fit(X_train, y_train)

        # Cross-validation scores`
        if self.config.use_cross_validation:
            self._calculate_cv_scores(X_train, y_train)

        return self.model            


    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        tune_hyperparameters: bool = True
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training target
            tune_hyperparameters: Whether to perform hyperparameter tuning

        Returns:
            Trained XGBoost model
        """
        logger.info("="*80)
        logger.info("TRAINING XGBOOST MODEL")
        logger.info("="*80)

        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = calculate_scale_pos_weight(y_train.values)
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning with GridSearchCV...")

            # Create base model
            base_model = xgb.XGBClassifier(
                random_state=XGBOOST_CONFIG.random_state,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                eval_metric='logloss'
            )

            # Use smaller param grid for faster tuning
            param_grid_small = {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9]
            }

            grid_search = GridSearchCV(
                base_model,
                param_grid_small,
                cv=self.config.cv_folds if self.config.use_cross_validation else 3,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")

        else:
            logger.info("Training with deafult hyperparameters...")

            self.model = xgb.XGBClassifier(
                n_estimators=XGBOOST_CONFIG.n_estimators,
                max_depth=XGBOOST_CONFIG.max_depth,
                learning_rate=XGBOOST_CONFIG.learning_rate,
                subsample=XGBOOST_CONFIG.subsample,
                colsample_bytree=XGBOOST_CONFIG.colsample_bytree,
                gamma=XGBOOST_CONFIG.gamma,
                scale_pos_weight=scale_pos_weight,
                random_state=XGBOOST_CONFIG.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )

            self.model.fit(X_train, y_train)

        # Cross-validation scores
        if self.config.use_cross_validation:
            self._calculate_cv_scores(X_train, y_train)

        return self.model            
                


    def _calculate_cv_scores(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):
        """Calculate cross-validation scores for the trained model"""
        logger.info(f"Calculating {self.config.cv_folds}-fold cross-validation scores...")

        cv_score = cross_val_score(
            self.model, X_train, y_train,
            cv=self.config.cv_folds,
            scoring='f1',
            n_jobs=-1
        )

        self.cv_scores = cv_score
        
        logger.info(f"Cross-validation F1 scores: {cv_score}")
        logger.info(f"Mean CV F1 score: {cv_score.mean():.4f} ± {cv_score.std()*2:.4f}")    


    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            logger.info(f"Saved model to {filepath}")
        else:
            logger.warning("No model to save.")

    def load_model(self, filepath: str):
        """Load trained model from disk"""
        self.model = joblib.load(filepath)
        logger.info(f"Loaded model from {filepath}")
        return self.model  

# Test code
if __name__ == "__main__":
    # Example usage
    data_loader = ChurnDataLoader()
    df = data_loader.load_data()
    X, y = data_loader.prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = data_loader.scale_features(X_train, X_val, X_test)

    trainer = ModelTrainer()

    # Train Logistic Regression
    lr_model = trainer.train_logistic_regression(X_train_scaled, y_train, tune_hyperparameters=True)
    trainer.save_model(f"{DATA_CONFIG.models_dir}/logistic_regression_model.joblib")

#     # # Train Random Forest
#     # rf_model = trainer.train_random_forest(X_train_scaled, y_train, tune_hyperparameters=True)
#     # trainer.save_model(f"{DATA_CONFIG.models_dir}/random_forest_model.joblib")

#     # Train XGBoost
#     xgb_model = trainer.train_xgboost(X_train_scaled, y_train, tune_hyperparameters=True)
#     trainer.save_model(f"{DATA_CONFIG.models_dir}/xgboost_model.joblib")