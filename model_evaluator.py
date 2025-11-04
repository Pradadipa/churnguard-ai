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
    precision_recall_curve,
    roc_curve
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

        # ROC AUC
        if self.config.calculate_auc:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion Matrix
        if self.config.calculate_confusion_matrix:
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

        # Store for late use
        self.metrics[model_name] = metrics

        # Print metrics
        self._print_metrics(metrics, model_name)

        # Generate classification report
        if self.config.verbose:
            logger.info("\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))

        # Create visualizations
        if self.config.plot_confusion_matrix:
            self.plot_confusion_matrix(
                metrics['confusion_matrix'], 
                model_name,
                save=self.config.save_plots
            )

        if self.config.plot_roc_curve:
            self.plot_roc_curve(
                y_test, 
                y_pred_proba, 
                model_name,
                save=self.config.save_plots
            )

        if self.config.plot_precision_recall_curve:
            self.plot_precision_recall_curve(
                y_test, 
                y_pred_proba, 
                model_name,
                save=self.config.save_plots
            )

        # Save metrics to JSON
        if self.config.save_metrics:
            self._save_metrics_json(metrics, model_name)

        return metrics

    def _print_metrics(self, metrics: Dict[str, Any], model_name: str):
        """Prints metrics in a nice format."""
        logger.info(f"\nEvaluation Metrics for {model_name}:")  
        logger.info("-"*40)

        if 'accuracy' in metrics:
            logger.info(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)") 

        if 'precision' in metrics:
            logger.info(f"Precision: {metrics['precision']:.4f}")

        if 'recall' in metrics:
            logger.info(f"Recall: {metrics['recall']:.4f}")   

        if 'f1' in metrics:
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")

        if 'auc_roc' in metrics:
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            logger.info("Confusion Matrix:")
            logger.info(f"  TN: {cm[0] [0]:4d}  |  FP: {cm[0] [1]:4d}")
            logger.info(f"  FN: {cm[1] [0]:4d}  |  TP: {cm[1] [1]:4d}")

    def plot_confusion_matrix(
            self,
            cm: np.ndarray,
            model_name: str,
                save: bool = True
    ):
        """Plots and saves the confusion matrix."""
        plt.figure(figsize=(8, 6))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Not Churn', 'Churn'],
            yticklabels=['Not Churn', 'Churn'],
            cbar_kws={'label': 'Count'}
        )

        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        # Add percentages
        cm = np.array(cm)  
        total = cm.sum() 
        for i in range(2):
            for j in range(2):
                percentage = cm[i,j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center',va='center' , color='black', fontsize=10) 
        
        plt.tight_layout()

        if save:
            filepath = f"{DATA_CONFIG.results_dir}/{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {filepath}")

        plt.close()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save: bool = True
    ):
        """Plots and saves the ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))

        # Plot ROC curve
        plt.plot(fpr, tpr, color='blue',lw=2, label=f'{model_name} (AUC = {auc:.3f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = f"{DATA_CONFIG.results_dir}/{model_name.lower().replace(' ', '_')}_roc_curve.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve saved to {filepath}")
        
        plt.close()

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save: bool = True
    ):
        """Plots and saves the Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))

        # Plot Precision-Recall curve
        plt.plot(recall, precision, color='green', lw=2, label=f'{model_name} ')

        # Plot baseline
        baseline = y_true.mean()
        plt.plot([0,1], [baseline, baseline], color='gray', lw=1, linestyle='--', label=f'Baseline ({baseline:.3f})')
        plt.xlim([0.0, 1.0])    
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'{model_name} - Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3) 

        plt.tight_layout()

        if save:
            filepath = f"{DATA_CONFIG.results_dir}/{model_name.lower().replace(' ', '_')}_precision_recall_curve.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {filepath}")
        plt.close()

    def plot_feature_importance(
        self,
        model,
        feature_names: List,
        model_name: str,
        top_n: int = 13,
        save: bool = True
    ):
        """Plots and saves the feature importance chart."""
        logger.info(f"Plotting feature importance for {model_name}...")  

        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_type = 'Importance'
        elif hasattr(model, 'coef_'):
            # Logistic Regression coefficients
            importances = np.abs(model.coef_[0])
            importance_type = 'Absolute Coefficient'
        else:
            logger.warning(f"Model {model_name} does not have feature importances or coefficients.")
            return

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        # Sort by importance
        importance_df = importance_df.sort_values(by='importance', ascending=False).head(top_n) 

        # Map to display names
        importance_df['display_name'] = importance_df['feature'].map(
            lambda x: FEATURE_DISPLAY_NAMES.get(x, x)
        )

        # Plot
        plt.figure(figsize=(10, 8))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))

        plt.barh(
            range(len(importance_df)),
            importance_df['importance'],
            color=colors
        ) 

        plt.yticks(range(len(importance_df)), importance_df['display_name'])
        plt.xlabel(importance_type, fontsize=12)
        plt.title(f'{model_name} - Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest importance on top
        plt.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = f"{DATA_CONFIG.results_dir}/{model_name.lower().replace(' ', '_')}_feature_importance.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {filepath}")
        plt.close()

        # log top features
        logger.info(f"\nTop 5 Features for {model_name}:")
        for idx, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['display_name']}: {row['importance']:.4f}")

    def _save_metrics_json(self, metrics: Dict[str, Any], model_name: str):
        """Save metrics to Json file."""
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}

        for key, value in metrics.items():
            if key == 'confusion_matrix':
                metrics_serializable[key] = value
            elif isinstance(value, np.ndarray):
                metrics_serializable[key] = value 
            elif isinstance(value, (np.float32, np.float64)):
                metrics_serializable[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                metrics_serializable[key] = int(value)
            else:
                metrics_serializable[key] = value

        filepath = f"{DATA_CONFIG.results_dir}/{model_name.lower().replace(' ', '_')}_metrics.json"
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        logger.info(f"Metrics saved to {filepath}")

    def compare_models(self, models_metrics: Dict[str, Dict], save: bool = True):
        """Compares multiple models based on their evaluation metrics.

        Args:
            models_metrics (Dict[str, Dict]): Dictionary with model names as keys and their metrics as values.
            save (bool): Whether to save the comparison report.
        """
        logger.info("="*80)
        logger.info("MODEL COMPARISON REPORT")  
        logger.info("="*80) 

        # Create comparison DataFrame
        comparison_data = []

        for model_name, metrics in models_metrics.items():
            row = {'Model:': model_name}
            row.update({
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),   
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1', 0),
                'AUC-ROC': metrics.get('auc_roc', 0)
            })
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)

        # Print comparison table
        logger.info("\nModel Comparison Table:")
        logger.info(comparison_df.to_string(index=False, float_format="{:.4f}".format))

        # plot comparison bar chart
        self._plot_model_comparison(comparison_df, save=save)

        # Identify best model
        best_model = comparison_df.loc[comparison_df['F1'].idxmax(), 'Model']
        best_f1 = comparison_df['F1'].max()

        logger.info(f"\nBest Model based on F1 Score: {best_model} (F1: {best_f1:.4f})")

        return comparison_df

    def _plot_model_comparison(self, comparison_df: pd.DataFrame, save: bool = True):
        """Plot model comparison bar chart"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
        
        # Plot 1: All metrics
        comparison_df_melted = comparison_df.melt(
            id_vars=['Model'],
            value_vars=metrics,
            var_name='Metric',
            value_name='Score'
        )
        
        sns.barplot(
            data=comparison_df_melted,
            x='Metric',
            y='Score',
            hue='Model',
            ax=axes[0]
        )
        
        axes[0].set_title('Model Comparison - All Metrics', fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, 1.0)
        axes[0].legend(loc='lower right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: F1 and AUC focus
        f1_auc_df = comparison_df[['Model', 'F1', 'AUC-ROC']].melt(
            id_vars=['Model'],
            var_name='Metric',
            value_name='Score'
        )
        
        sns.barplot(
            data=f1_auc_df,
            x='Model',
            y='Score',
            hue='Metric',
            ax=axes[1]
        )
        
        axes[1].set_title('F1 Score vs AUC-ROC', fontsize=12, fontweight='bold')
        axes[1].set_ylim(0, 1.0)
        axes[1].legend(loc='lower right')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = f"{DATA_CONFIG.results_dir}/model_comparison.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {filepath}")
        
        plt.close()