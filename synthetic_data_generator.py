"""
Synthetic Training Data Generator
ChurnGuard AI - Week 1, Days 1-2
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from dataclasses import asdict
import pickle

from data_generator_config import (
    CUSTOMER_TYPES,
    FeatureRanges,
    OUTPUT_CONFIG,
    TARGET_CORRELATIONS,
    apply_renewal_urgency
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate realistic synthetic customer data with proper correlations"""
    
    def __init__(self, n_samples: int = 1000, random_seed: int = 42):
        """
        Initialize the data generator
        
        Args:
            n_samples: Total number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.feature_ranges = FeatureRanges()
        
        np.random.seed(random_seed)
        logger.info(f"Initialized SyntheticDataGenerator with {n_samples} samples, seed={random_seed}")
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate complete synthetic dataset
        
        Returns:
            DataFrame with all features and churn labels
        """
        logger.info("Starting data generation...")
        
        # Step 1: Assign customer types
        customer_types = self._assign_customer_types()
        logger.info(f"Customer type distribution: {pd.Series(customer_types).value_counts().to_dict()}")
        
        # Step 2: Generate features for each type
        data_dict = {
            'customer_id': [f"CUST_{i:05d}" for i in range(self.n_samples)],
            'customer_type': customer_types
        }
        
        # Step 3: Generate all features
        for i, customer_type in enumerate(CUSTOMER_TYPES):
            mask = np.array(customer_types) == customer_type.name
            n_type = mask.sum()
            
            if n_type == 0:
                continue
            
            logger.info(f"Generating features for {n_type} {customer_type.name} customers...")
            type_data = self._generate_features_for_type(customer_type, n_type)
            
            # Fill in the data for this customer type
            for feature, values in type_data.items():
                if feature not in data_dict:
                    data_dict[feature] = np.zeros(self.n_samples)
                data_dict[feature][mask] = values
        
        # Step 4: Apply temporal effects
        data_dict = self._apply_temporal_effects(data_dict)
        
        # Step 5: Generate churn labels
        data_dict['churned'] = self._generate_churn_labels(data_dict, customer_types)
        
        # Convert to DataFrame
        df = pd.DataFrame(data_dict)
        
        # Step 6: Final cleanup and type conversion
        df = self._finalize_data(df)
        
        logger.info(f"Data generation complete! Shape: {df.shape}")
        logger.info(f"Churn rate: {df['churned'].mean():.2%}")
        
        return df
    
    def _assign_customer_types(self) -> List[str]:
        """Assign customer types based on proportions"""
        types = []
        proportions = [ct.proportion for ct in CUSTOMER_TYPES]
        type_names = [ct.name for ct in CUSTOMER_TYPES]
        
        types = np.random.choice(
            type_names,
            size=self.n_samples,
            p=proportions
        )
        
        return types.tolist()
    
    def _generate_features_for_type(self, config, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate features for a specific customer type"""
        data = {}
        
        # Sentiment features
        data['avg_sentiment'] = self._generate_with_noise(
            config.avg_sentiment_mean, config.noise_level, n_samples, (-1.0, 1.0)
        )
        data['sentiment_std'] = self._generate_with_noise(
            config.sentiment_std_mean, config.noise_level, n_samples, (0.0, 0.5)
        )
        data['negative_ratio'] = self._generate_with_noise(
            config.negative_ratio_mean, config.noise_level, n_samples, (0.0, 1.0)
        )
        data['urgent_count'] = np.maximum(0, np.random.poisson(
            config.urgent_count_mean, n_samples
        ))
        
        # Business metrics - monthly_value is log-normal
        data['monthly_value'] = np.random.lognormal(
            mean=np.log(config.monthly_value_mean),
            sigma=config.noise_level * 0.5,
            size=n_samples
        )
        data['monthly_value'] = np.clip(
            data['monthly_value'],
            self.feature_ranges.monthly_value[0],
            self.feature_ranges.monthly_value[1]
        )
        
        # Contract length - discrete values
        data['contract_length_months'] = np.random.choice(
            self.feature_ranges.contract_length_months,
            size=n_samples,
            p=[0.05, 0.10, 0.15, 0.35, 0.10, 0.15, 0.10]  # Weighted towards 12 months
        )
        
        # Days to renewal
        data['days_to_renewal'] = self._generate_with_noise(
            config.days_to_renewal_mean, config.noise_level, n_samples, 
            self.feature_ranges.days_to_renewal
        ).astype(int)
        
        # Behavioral metrics
        data['response_time_hours'] = self._generate_with_noise(
            config.response_time_mean, config.noise_level, n_samples,
            self.feature_ranges.response_time_hours
        )
        data['meeting_attendance_rate'] = self._generate_with_noise(
            config.meeting_attendance_mean, config.noise_level, n_samples, (0.0, 1.0)
        )
        data['feature_adoption_score'] = self._generate_with_noise(
            config.feature_adoption_mean, config.noise_level, n_samples, (0.0, 1.0)
        )
        
        # Risk flags - binary
        data['technical_issue_flag'] = (
            np.random.random(n_samples) < config.technical_issue_prob
        ).astype(int)
        data['pricing_concern_flag'] = (
            np.random.random(n_samples) < config.pricing_concern_prob
        ).astype(int)
        data['competitor_mention_flag'] = (
            np.random.random(n_samples) < config.competitor_mention_prob
        ).astype(int)
        
        return data
    
    def _generate_with_noise(
        self, 
        mean: float, 
        noise_level: float, 
        n_samples: int,
        clip_range: Tuple[float, float]
    ) -> np.ndarray:
        """Generate values with gaussian noise and clipping"""
        std = abs(mean * noise_level)
        if std == 0:
            std = noise_level
        
        values = np.random.normal(mean, std, n_samples)
        return np.clip(values, clip_range[0], clip_range[1])
    
    def _apply_temporal_effects(self, data_dict: Dict) -> Dict:
        """Apply temporal urgency effects based on days_to_renewal"""
        logger.info("Applying temporal urgency effects...")
        
        days_to_renewal = np.array(data_dict['days_to_renewal'])
        urgency_multiplier = apply_renewal_urgency(days_to_renewal)
        
        # Increase urgent_count for customers near/past renewal
        data_dict['urgent_count'] = np.array(data_dict['urgent_count']) * urgency_multiplier
        
        # Worsen sentiment for expired contracts
        expired_mask = days_to_renewal < 0
        data_dict['avg_sentiment'][expired_mask] -= 0.3
        data_dict['avg_sentiment'] = np.clip(data_dict['avg_sentiment'], -1.0, 1.0)
        
        return data_dict
    
    def _generate_churn_labels(self, data_dict: Dict, customer_types: List[str]) -> np.ndarray:
        """Generate churn labels based on customer type and features"""
        logger.info("Generating churn labels...")
        
        churned = np.zeros(self.n_samples)
        
        for customer_type in CUSTOMER_TYPES:
            mask = np.array(customer_types) == customer_type.name
            n_type = mask.sum()
            
            if n_type == 0:
                continue
            
            # Base churn probability from customer type
            base_churn_prob = customer_type.churn_rate
            
            # Adjust based on features (add some correlation)
            indices = np.where(mask)[0]
            churn_probs = np.ones(n_type) * base_churn_prob
            
            # Increase churn prob for very negative sentiment
            sentiment = np.array(data_dict['avg_sentiment'])[indices]
            churn_probs += np.maximum(0, -sentiment - 0.3) * 0.2
            
            # Increase churn prob for near/expired renewals
            days = np.array(data_dict['days_to_renewal'])[indices]
            churn_probs[days < 30] += 0.1
            churn_probs[days < 0] += 0.2
            
            # Clip probabilities
            churn_probs = np.clip(churn_probs, 0.0, 0.95)
            
            # Generate binary labels
            churned[indices] = (np.random.random(n_type) < churn_probs).astype(int)
        
        return churned.astype(int)
    
    def _finalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and type conversions"""
        
        # Convert integer columns
        int_columns = [
            'urgent_count', 'contract_length_months', 'days_to_renewal',
            'technical_issue_flag', 'pricing_concern_flag', 
            'competitor_mention_flag', 'churned'
        ]
        for col in int_columns:
            df[col] = df[col].astype(int)
        
        # Round float columns to reasonable precision
        float_columns = [
            'avg_sentiment', 'sentiment_std', 'negative_ratio',
            'monthly_value', 'response_time_hours', 
            'meeting_attendance_rate', 'feature_adoption_score'
        ]
        for col in float_columns:
            df[col] = df[col].round(4)
        
        # Reorder columns
        feature_order = [
            'customer_id', 'customer_type',
            'avg_sentiment', 'sentiment_std', 'negative_ratio', 'urgent_count',
            'monthly_value', 'contract_length_months', 'days_to_renewal',
            'response_time_hours', 'meeting_attendance_rate', 'feature_adoption_score',
            'technical_issue_flag', 'pricing_concern_flag', 'competitor_mention_flag',
            'churned'
        ]
        
        return df[feature_order]
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and correlations"""
        logger.info("Validating generated data...")
        
        validation_report = {
            'n_samples': len(df),
            'churn_rate': df['churned'].mean(),
            'customer_type_distribution': df['customer_type'].value_counts().to_dict(),
            'correlations': {},
            'feature_stats': {}
        }
        
        # Check key correlations
        validation_report['correlations'] = {
            'avg_sentiment_vs_churn': df['avg_sentiment'].corr(df['churned']),
            'negative_ratio_vs_churn': df['negative_ratio'].corr(df['churned']),
            'days_to_renewal_vs_churn': df['days_to_renewal'].corr(df['churned']),
            'technical_issue_vs_churn': df['technical_issue_flag'].corr(df['churned']),
            'meeting_attendance_vs_churn': df['meeting_attendance_rate'].corr(df['churned']),
            'feature_adoption_vs_churn': df['feature_adoption_score'].corr(df['churned']),
        }
        
        # Feature statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['churned']:
                validation_report['feature_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return validation_report
    
    def save_data(self, df: pd.DataFrame, validation_report: Dict):
        """Save generated data and validation report"""
        import os
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save CSV
        csv_path = OUTPUT_CONFIG['csv_path']
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to {csv_path}")
        
        # Save pickle
        pickle_path = OUTPUT_CONFIG['pickle_path']
        with open(pickle_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Saved pickle to {pickle_path}")
        
        # Save validation report
        report_path = OUTPUT_CONFIG['validation_report_path']
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CHURNGUARD AI - SYNTHETIC DATA VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Samples: {validation_report['n_samples']}\n")
            f.write(f"Overall Churn Rate: {validation_report['churn_rate']:.2%}\n\n")
            
            f.write("Customer Type Distribution:\n")
            for ctype, count in validation_report['customer_type_distribution'].items():
                pct = count / validation_report['n_samples']
                f.write(f"  {ctype}: {count} ({pct:.1%})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("CORRELATION ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            for corr_name, corr_value in validation_report['correlations'].items():
                target = TARGET_CORRELATIONS.get(corr_name, None)
                status = "✓" if target and target[0] <= corr_value <= target[1] else "⚠"
                target_str = f"[Target: {target[0]:.2f} to {target[1]:.2f}]" if target else ""
                f.write(f"{status} {corr_name}: {corr_value:.3f} {target_str}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("FEATURE STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            for feature, stats in validation_report['feature_stats'].items():
                f.write(f"{feature}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\n")
                f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
        
        logger.info(f"Saved validation report to {report_path}")
