"""
Train Baseline Model - Logistic Regression ChutnGiard AI - Week 1, Day 3

Main Script to train and evaluate the baseline Logistic Regression model
"""
import logging
import sys
from pathlib import Path

from model_trainer import ChurnDataLoader, ModelTrainer # Importing ModelTrainer class
from model_evaluator import ModelEvaluator # Importing ModelEvaluator class
from model_config import DATA_CONFIG, MODEL_CONFIG # Importing configuration dictionaries

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('baseline_training.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """ Main execution function for baseline model training"""

    logger.info("="*80)
    logger.info("CHURNGUARD AI - BASELINE MODEL TRAINING (LOGISTIC REGRESSION)")
    logger.info("="*80)

    try:
        # ===================================================
        # Step 1: LOAD AND PREPARE DATA
        # ===================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 1: LOAD AND PREPARE DATA")
        logger.info("="*80)

        data_loader = ChurnDataLoader()

        # Load dara
        df = data_loader.load_data()

        # Prepare features and target
        X, y = data_loader.prepare_features(df)

        # Split data (70/15/15)
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(X, y)

        # Scale features for Logistic Regression
        if MODEL_CONFIG.scale_features:
            X_train_scaled, X_val_scaled, X_test_scaled = data_loader.scale_features(
                X_train, X_val, X_test,
                scaler_type=MODEL_CONFIG.scaler_type
            )

            # Safe scaler for production 
            data_loader.save_scaler()
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test
        
        # ===================================================
        # Step 2: TRAIN BASELINE MODEL  
        # ===================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 2: TRAIN LOGISTIC REGRESSION MODEL")
        logger.info("="*80)

        trainer = ModelTrainer()

        # Train with hyperparameter tuning
        model = trainer.train_logistic_regression(
            X_train_scaled,
            y_train,
            tune_hyperparameters=True  
        )

        # Save model
        model_path = f"{DATA_CONFIG.models_dir}/logistic_regression_model.joblib"
        trainer.save_model(model_path)

        # ====================================================
        # STEP 3: EVALUATE ON VALIDATION SET
        # ====================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 3: VALIDATION SET EVALUATION")
        logger.info("="*80)

        evaluator = ModelEvaluator()

        val_metrics = evaluator.evaluate_model(
            model,
            X_val_scaled,
            y_val,
            model_name="Logistic Regression (validation)"
        )

        # ================================================
        # STEP 4: FINAL EVALUATION ON TEST SET
        # ================================================
        logger.info("\n" + "="*80)
        logger.info("STEP 4: TEST SET EVALUATION (FINAL)")
        logger.info("="*80)

        test_metrics = evaluator.evaluate_model(
            model,
            X_test_scaled,
            y_test,
            model_name="Logistic Regression (Test)"
        )
        
        # ===============================================
        # STEP 5: FEATURE IMPORTANCE ANALYSIS
        # ===============================================
        logger.info("\n" + "="*80)
        logger.info("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*80)

        evaluator.plot_feature_importance(
            model,
            DATA_CONFIG.feature_columns,
            "Logistic Regression",
            save=True
        )

        # ===============================================
        # STEP 6: CHECK TARGET THRESHOLDS
        # ===============================================
        logger.info("\n" + "="*80)
        logger.info("STEP 6: CHECK TARGET THRESHOLDS")
        logger.info("="*80)

        accuracy_target = MODEL_CONFIG.target_accuracy
        auc_target= MODEL_CONFIG.target_auc
        f1_target= MODEL_CONFIG.target_f1

        logger.info(f"\nTarget Metrics:")
        logger.info(f" Accuracy: {accuracy_target: .2%}")
        logger.info(f" AUC-ROC: {auc_target: .2%}")
        logger.info(f" F1 Score: {f1_target: .2%}")

        logger.info(f"\nActual Test Set Performance")
        logger.info(f" Accuracy: {test_metrics['accuracy']:.2%} " +
                    ("✓" if test_metrics['accuracy'] >= accuracy_target else "X"))
        logger.info(f" AUC-ROC: {test_metrics['roc_auc']:.2%} " +
                    ("✓" if test_metrics['roc_auc'] >= auc_target else "X"))
        logger.info(f" F1 Score: {test_metrics['f1_score']:.2%} " +
                    ("✓" if test_metrics['f1_score'] >= f1_target else "X"))
        
        # ===============================================
        # SUCCSS SUMMARY
        # ===============================================
        logger.info("\n" + "="*80)
        logger.info("✓ BASELINE MODEL TRAINING COMPLETE!")
        logger.info("="*80)

        logger.info(f"\nModel saved: {model_path}")
        logger.info(f"Scaler saved: {DATA_CONFIG.models_dir}/feature_scaler.joblib")
        logger.info(f"Results saved: {DATA_CONFIG.results_dir}/")

        logger.info("\nGenerated Files:")
        logger.info(" - logistic_regression_(test)_confusion_matrix.png")
        logger.info(" - logistic_regression_(test)_roc_curve.png")
        logger.info(" - logistic_regression_(test)_pr_curve.png")
        logger.info(" - logistic_regression_feature_importance.png")
        logger.info(" - logistic_regression_(test)_metrics.json")

        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS:")
        logger.info("="*80)
        logger.info("✓ Day 3 Complete: Baseline model trained and evaluated")
        logger.info("→ Day 4: Train advanced models (Random Forest + XGBoost)")
        logger.info("  Command: python train_advanced.py")

        # Determine if targets met
        targets_met = (
            test_metrics['accuracy'] >= accuracy_target and
            test_metrics['roc_auc']  >= auc_target and
            test_metrics['f1_score'] >= f1_target
        )

        if targets_met:
            logger.info("\n🎉 Baseline model EXCEEDS all target metrics!")
            logger.info("   Advanced models (RF/XGBoost) should perform even better.")
        else:
            logger.info("\n⚠ Baseline model below some targets - this is normal.")
            logger.info("   Advanced models (RF/XGBoost) will likely improve performance.")
        
        return 0
    
    except Exception as e:
        logger.error(f"\n✗ ERROR during training: {str(e)}", exc_info=True)
        return 1
    
if __name__ == "__main__":
    sys.exit(main())