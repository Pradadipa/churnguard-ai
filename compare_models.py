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

def evaluate_all_model(models, X_test, X_test_scaled, y_test):
    """Evaluate all models on same test set"""
    logger.info("\n"+"="*80)
    logger.info("EVALUATING ALL MODELS")
    logger.info("="*80)

    results = []

    for name, model in models.items():
        logger.info(f"\nEvaluating {name}...")

        # Use scaled data for logistic regression only
        X_input = X_test_scaled if name == 'Logistic Regression' else X_test

        # Predictions
        y_pred = model.predict(X_input)
        y_pred_boba = model.predict_proba(X_input)[:, 1]

        # Metrics
        metrics = {
            'Model':name,
            'Accuracy':accuracy_score(y_test, y_pred),
            'Precision':precision_score(y_test, y_pred),
            'Recall':recall_score(y_test, y_pred),
            'F1':f1_score(y_test, y_pred),
            'AUC':roc_auc_score(y_test, y_pred_boba)
        }

        results.append(metrics)

        logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")
        logger.info(f"  F1:       {metrics['F1']:.4f}")
        logger.info(f"  AUC:      {metrics['AUC']:.4f}")
    
    return pd.DataFrame(results)

def print_comparison_table(comparison_df):
    """Print formatted comparison table"""
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("="*80)

    print("\n" + comparison_df.to_string(index=False, float_format='%.4f'))

    # Find best model for each metrics
    logger.info("\n" + "="*80)
    logger.info("BEST MODEL BY METRICS")
    logger.info("="*80)

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
        best_idx = comparison_df[metric].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_score = comparison_df.loc[best_idx, metric]
        logger.info(f"  {metric:12s}: {best_model:20s}: ({best_score:.4f})")

    # Overall best (by F1)
    best_overall_idx = comparison_df['F1'].idxmax()
    best_overall = comparison_df.loc[best_overall_idx, 'Model']
    best_f1 = comparison_df.loc[best_overall_idx, 'F1']

    logger.info("\n" + "="*80)
    logger.info(f"BEST OVERALL MODEL: {best_overall} (F1 = {best_f1:.4f})")
    logger.info("="*80)

def plot_comparison_dashboard(comparison_df, save_path='data/results/model_comparison_dashboard.png'):
    """Create comprehensive comparison dashboard"""
    logger.info("\nCreating comparison dashboard...")

    fig, axes = plt.subplots(2,2, figsize=(14,10))

    # Plot 1: All metrics bar chart
    ax1 = axes[0,0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    x = np.arange(len(metrics))
    width = 0.25

    for i, (idx, row) in enumerate(comparison_df.iterrows()):
        offset = (i-1) * width
        ax1.bar(x+offset, [row[m] for m in metrics], width, label=row['Model'])

    ax1.set_xlabel('Metrics', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0.6, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: F1 Score Focus
    ax2 = axes[0, 1]
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(comparison_df)]
    bars = ax2.barh(comparison_df['Model'], comparison_df['F1'], color=colors)

    # Add Target line
    ax2.axvline(x=0.80, color='red', linestyle='--', linewidth=2, label='Target (80%)')

    ax2.set_xlabel('F1 Score', fontsize=11)
    ax2.set_title('F1 Score Comparison (Target: 80%)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0.7, 1.0)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar in bars:
        weidth = bar.get_width()
        ax2.text(weidth + 0.01, bar.get_y() + bar.get_height()/2,
        f'{weidth:.3f}', ha='left', va='center',fontsize=10)
    
    # Plot 3: AUC Score
    ax3 = axes[1,0]
    bars = ax3.barh(comparison_df['Model'], comparison_df['AUC'], color=colors)
    ax3.axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
    ax3.set_xlabel('AUC-ROC', fontsize=11)
    ax3.set_title('AUC-ROC Comparison (Target: 85%)', fontsize=12, fontweight='bold')
    ax3.set_xlim(0.85, 1.0)
    ax3.legend()
    ax3.grid(axis='x', alpha=0.3)

    for bar in bars:
        weidth = bar.get_width()
        ax3.text(weidth + 0.002, bar.get_y() + bar.get_height()/2,
        f'{weidth:.3f}', ha='left', va='center',fontsize=10)
    
    # Plot 4: Precision vs Recall
    ax4 = axes[1, 1]
    
    for idx, row in comparison_df.iterrows():
        ax4.scatter(row['Recall'], row['Precision'], 
                    s=200, label=row['Model'], alpha=0.7)
        ax4.annotate(row['Model'], 
                    (row['Recall'], row['Precision']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)
    
    ax4.set_xlabel('Recall', fontsize=11)
    ax4.set_ylabel('Precision', fontsize=11)
    ax4.set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    ax4.set_xlim(0.7, 1.0)
    ax4.set_ylim(0.6, 1.0)
    ax4.grid(alpha=0.3)
    ax4.legend()
    
    plt.suptitle('ChurnGuard AI - Model Comparison Dashboard', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved dashboard to {save_path}")
    plt.close()

def plot_roc_curves(models, X_test, X_test_scaled, y_test, save_path='data/results/roc_curves_comparison.png'):
    """Plot ROC curves for all models"""
    logger.info("\nCreating ROC curves comparison...")

    plt.figure(figsize=(10, 8))

    colors = ['blue', 'green', 'red']

    for (name, model), color in zip(models.items(), colors):
        # Use appropiate input
        X_input = X_test_scaled if name == 'Logistic Regression' else X_test

        # Get probabilities
        y_pred_proba = model.predict_proba(X_input)[:,1]

        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Plot
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0,1], [0,1], 'k--', lw=1, label='Random Classifier')

    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('ROC Curve - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved ROC curves to {save_path}")
    plt.close()



def main():
    """Main execution"""
    logger.info("="*80)
    logger.info("CHURNGUARD AI - MODEL COMPARISON")
    logger.info("="*80)

    try:
        # Load data
        X_test, X_test_scaled, y_test, feature_columns = load_data()

        # Load Models
        models = load_models()

        # Evaluate All
        comparison_df = evaluate_all_model(models, X_test, X_test_scaled, y_test)

        # Print comparison
        print_comparison_table(comparison_df)

        # Create visualizations
        plot_comparison_dashboard(comparison_df)
        plot_roc_curves(models, X_test, X_test_scaled, y_test)

        # Save comparison to CSV
        comparison_df.to_csv('data/results/model_comparison.csv', index=False)
        logger.info("\n✓ Saved comparison table to data/results/model_comparison.csv")
        
        # Final recommendation
        best_model = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATION")
        logger.info("="*80)
        logger.info(f"\n🎯 For production deployment, use: {best_model}")
        logger.info(f"   Reason: Best F1 score (balanced precision/recall)")
        logger.info(f"\n✓ Week 1, Day 4 COMPLETE!")
        logger.info(f"✓ All 3 models trained and compared")
        
        logger.info(f"\nNext Steps:")
        logger.info(f"  1. Review comparison dashboard: data/results/model_comparison_dashboard.png")
        logger.info(f"  2. Choose best model for integration")
        logger.info(f"  3. Week 1, Day 5: Integrate with ChurnGuard AI agents!")
        
        return 0
    
    except Exception as e:
        logger.error(f"\nERROR: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())