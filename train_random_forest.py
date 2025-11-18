
"""
Train Random Forest Model
ChurnGuard AI - Week 1, Day 4

Random Forest is an ENSEMBLE of decision trees that:
1. Creates multiple trees (n_estimators=200)
2. Each tree sees random subset of data (bootstrap sampling)
3. Each split considers random subset of features
4. Final prediction = majority vote of all trees

Why better than Logistic Regression?
- Captures non-linear relationships (e.g., sentiment * days_to_renewal)
- Handles feature interactions automatically
- More robust to outliers
- NO SCALING NEEDED (trees are scale-invariant)
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('random_forest_training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """"Load and prepare data"""
    logger.info("="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)

    # Load CSV
    df = pd.read_csv('data/synthetic_training_data.csv')
    logger.info(f"Loaded {len(df)} records")

    # Define feature column
    feature_columns = [
        'avg_sentiment', 'sentiment_std', 'negative_ratio', 'urgent_count',
        'monthly_value', 'contract_length_months', 'days_to_renewal',
        'response_time_hours', 'meeting_attendance_rate', 'feature_adoption_score',
        'technical_issue_flag', 'pricing_concern_flag', 'competitor_mention_flag'
    ]

    X = df[feature_columns]
    y = df['churned']

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Churn rate: {y.mean():.2%}")

    return X, y, feature_columns

def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """Split into train/val/test (70/15/15)"""

    from sklearn.model_selection import train_test_split

    logger.info("\n" + "="*80)
    logger.info("STEP 2: SPLITTING DATA")
    logger.info("="*80)

    # First split: Train+Val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y
    )

    # Second split: train vs val
    val_size_adjusted = val_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )

    logger.info(f"Train set: {len(X_train)} samples {len(X_train)/len(X)*100:.1f}%")
    logger.info(f"Val set: {len(X_val)} samples {len(X_val)/len(X)*100:.1f}%")
    logger.info(f"Test set: {len(X_test)} samples {len(X_test)/len(X)*100:.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_random_forest(X_train, y_train, tune_hyperparameters=True):
    """
    Train Random Forest with optional hyperparameter tuning
    
    Key Hyperparameters:
    - n_estimators: Number of trees (more = better, but slower)
    - max_depth: Maximum tree depth (controls overfitting)
    - min_samples_split: Min samples to split a node
    - min_samples_leaf: Min samples in leaf node
    - max_features: Features to consider for best split
    - class_weight: Handle class imbalance
    """
    logger.info("="*80)
    logger.info("STEP 3: TRAINING RANDOM FOREST")
    logger.info("="*80)
    
    if tune_hyperparameters:
        logger.info("Performing hyperparameter tuning with GridSearchCV...")

        # Create base model
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        )

        # Parameter grid to search
        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [10,15,20],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2],
            'max_features': ['sqrt', 'log2']
        }

        logger.info(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinations...")

        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        logger.info(f"\n Best parameters found: {best_params}")
        logger.info(f" Best CV F1 score: {grid_search.best_score_:.4f}")
    
    else:
        # Train with sensible deafults
        logger.info("Training with default parameters (no tuning)...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        logger.info(" Model trained")

    return model

def evaluate_model(model, X_test, y_test, model_name="Random Forest"):
    """Comprehensive evaluation"""
    logger.info("\n"+"="*80)
    logger.info(f"STEP 4: EVALUATING {model_name.upper()}")
    logger.info("="*80)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precission': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    # Print results
    logger.info(f"\n{model_name} Performance Metrics:")
    logger.info("-" * 50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"Precision: {metrics['precission']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    logger.info(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
    
    # Detailed report
    logger.info("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))
    
    return metrics

def plot_feature_importance(model, feature_name, save_path='data/results/rf_feature_importance.png'):
    """
    Plot feature importance
    
    Random Forest importance = Average decrease in impurity across all trees
    Higher value = more important for splitting decisions
    """

    logger.info("="*80)
    logger.info("STEP 5: FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)

    # Get importance
    importances = model.feature_importances_

    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_name,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Print top 5
    logger.info("\nTop 5 Most importance feature:")
    for idx, row in importance_df.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Plot
    plt.figure(figsize=(10,8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))

    plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Save
    Path('data/results').mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f" Saved feature importance plot to {save_path}")
    plt.close()

def plot_confusion_matrix(cm, save_path='data/results/rf_confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Not Churned', 'Churned'],
        yticklabels=['Not Churned', 'Churned'],
        cbar_kws={'label':'Count'}
    )
    plt.title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    # Add Percentage
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i,j]/total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.2f}%)',
                ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f" Saved confusion matrix to {save_path}")
    plt.close()

def main():
    """Main Execution"""
    logger.info("="*80)
    logger.info("CHURNGUARD AI - RANDOM FOREST MODEL TRAINING")
    logger.info("="*80)

    try:
        # Load data
        X, y, feature_columns = load_data()

        # Split Data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Train Model
        model = train_random_forest(
            X_train, 
            y_train,
            tune_hyperparameters=True
        )

        # Evaluate on validation set first
        logger.info("="*80)
        logger.info("VALIDATION SET EVALUATION")
        logger.info("="*80)
        val_metrics = evaluate_model(model, X_val, y_val, "Random Forest (Validation)")

        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test, "Random Forest (Test)")

        # Plot Feature importance
        plot_feature_importance(model, feature_columns)

        # Plot confusion matrix
        plot_confusion_matrix(test_metrics['confusion_matrix'])

        # Save model
        Path('data/models').mkdir(parents=True, exist_ok=True)
        model_path = 'data/models/random_forest_model.joblib'
        joblib.dump(model, model_path)
        logger.info(f"\n Saved model to {model_path}")

        # Compare to target
        logger.info("="*80)
        logger.info("TARGET METRICS CHECK")
        logger.info("="*80)

        logger.info(f"\nTarget vs Actual (Test Set):")
        logger.info(f"Accuracy: 85.00% target : {test_metrics['accuracy']*100:.2f}% actual " + 
                    ("Pass" if test_metrics['accuracy'] >= 0.85 else "Not Pass"))
        logger.info(f"AUC-ROC:  85.00% target : {test_metrics['auc_roc']*100:.2f}% actual " + 
                    ("Pass" if test_metrics['auc_roc'] >= 0.85 else "Not Pass"))
        logger.info(f"F1 Score: 80.00% target : {test_metrics['f1']*100:.2f}% actual " + 
                    ("Pass" if test_metrics['f1'] >= 0.80 else "Not Pass"))


        # Success summary
        logger.info("\n" + "="*80)
        logger.info("RANDOM FOREST TRAINING COMPLETE!")
        logger.info("="*80)
        
        logger.info(f"\nGenerated Files:")
        logger.info(f"  - {model_path}")
        logger.info(f"  - data/results/rf_feature_importance.png")
        logger.info(f"  - data/results/rf_confusion_matrix.png")
        
        logger.info(f"\nNext Steps:")
        logger.info(f"  1. Review feature importance - did it change from Logistic Regression?")
        logger.info(f"  2. Run: python train_xgboost.py")
        logger.info(f"  3. Then: python compare_models.py")

        return 0
    
    except Exception as e:
        logger.info(f"\nERROR: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())