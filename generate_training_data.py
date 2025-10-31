"""
Main script to generate synthetic training data
ChurnGuard AI - Week 1, Days 1-2

Usage:
    python generate_training_data.py
"""

import logging
import sys
from synthetic_data_generator import SyntheticDataGenerator
from data_generator_config import OUTPUT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_generation.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    
    logger.info("="*80)
    logger.info("CHURNGUARD AI - SYNTHETIC TRAINING DATA GENERATION")
    logger.info("="*80)
    
    try:
        # Initialize generator
        generator = SyntheticDataGenerator(
            n_samples=OUTPUT_CONFIG['n_samples'],
            random_seed=OUTPUT_CONFIG['random_seed']
        )
        
        # Generate data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Generating synthetic data...")
        logger.info("="*80)
        df = generator.generate_data()
        
        # Validate data
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Validating data quality...")
        logger.info("="*80)
        validation_report = generator.validate_data(df)
        
        # Display key metrics
        logger.info("\n" + "="*80)
        logger.info("KEY METRICS")
        logger.info("="*80)
        logger.info(f"Total samples: {validation_report['n_samples']}")
        logger.info(f"Overall churn rate: {validation_report['churn_rate']:.2%}")
        logger.info("\nCustomer distribution:")
        for ctype, count in validation_report['customer_type_distribution'].items():
            pct = count / validation_report['n_samples']
            logger.info(f"  - {ctype}: {count} ({pct:.1%})")
        
        logger.info("\nKey correlations with churn:")
        for corr_name, corr_value in validation_report['correlations'].items():
            logger.info(f"  - {corr_name}: {corr_value:.3f}")
        
        # Save data
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Saving data and reports...")
        logger.info("="*80)
        generator.save_data(df, validation_report)
        
        # Success summary
        logger.info("\n" + "="*80)
        logger.info("DATA GENERATION COMPLETE!")
        logger.info("="*80)
        logger.info(f"CSV saved: {OUTPUT_CONFIG['csv_path']}")
        logger.info(f"Pickle saved: {OUTPUT_CONFIG['pickle_path']}")
        logger.info(f"Validation report: {OUTPUT_CONFIG['validation_report_path']}")
        logger.info("\nNext steps:")
        logger.info("1. Review the validation report: cat data/validation_report.txt")
        logger.info("2. Inspect the CSV: head data/synthetic_training_data.csv")
        logger.info("3. Proceed to Week 1, Day 3: Build baseline model")
        
        return 0
        
    except Exception as e:
        logger.error(f"\nERROR during data generation: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
