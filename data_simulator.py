"""
Data Simulator for ML Model Monitoring System
Generates realistic model predictions, metrics, and drift scenarios
NO ACTUAL MODEL TRAINING REQUIRED - Perfect for assignment demo
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import random


@dataclass
class ModelConfig:
    """Configuration for a simulated model"""
    model_id: str
    model_name: str
    model_type: str  # 'image_classifier', 'fraud_detector', etc.
    baseline_accuracy: float
    baseline_latency_ms: float
    deployment_date: datetime
    traffic_per_day: int


class ModelDataSimulator:
    """
    Simulates realistic ML model behavior over time
    Includes normal operation, drift scenarios, and incidents
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Define models to simulate
        self.models = [
            ModelConfig(
                model_id='pneumonia_classifier_v1',
                model_name='Chest X-Ray Pneumonia Detector',
                model_type='image_classifier',
                baseline_accuracy=0.87,
                baseline_latency_ms=45.0,
                deployment_date=datetime.now() - timedelta(days=90),
                traffic_per_day=5000
            ),
            ModelConfig(
                model_id='fraud_detector_v2',
                model_name='Transaction Fraud Detector',
                model_type='binary_classifier',
                baseline_accuracy=0.92,
                baseline_latency_ms=12.0,
                deployment_date=datetime.now() - timedelta(days=60),
                traffic_per_day=50000
            ),
            ModelConfig(
                model_id='object_detector_v1',
                model_name='Manufacturing Defect Detector',
                model_type='object_detector',
                baseline_accuracy=0.85,
                baseline_latency_ms=85.0,
                deployment_date=datetime.now() - timedelta(days=45),
                traffic_per_day=10000
            )
        ]
    
    def generate_30_day_simulation(
        self, 
        model_config: ModelConfig
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate 30 days of model data with realistic drift patterns
        """
        
        # Generate baseline period (days 0-10)
        baseline_data = self._generate_baseline_period(model_config, days=10)
        
        # Generate drift period (days 11-20)
        drift_data = self._generate_drift_period(model_config, days=10)
        
        # Generate critical period (days 21-25)
        critical_data = self._generate_critical_period(model_config, days=5)
        
        # Generate recovery period (days 26-30)
        recovery_data = self._generate_recovery_period(model_config, days=5)
        
        # Combine all periods
        predictions_df = pd.concat([
            baseline_data['predictions'],
            drift_data['predictions'],
            critical_data['predictions'],
            recovery_data['predictions']
        ], ignore_index=True)
        
        metrics_df = pd.concat([
            baseline_data['metrics'],
            drift_data['metrics'],
            critical_data['metrics'],
            recovery_data['metrics']
        ], ignore_index=True)
        
        drift_scores_df = pd.concat([
            baseline_data['drift_scores'],
            drift_data['drift_scores'],
            critical_data['drift_scores'],
            recovery_data['drift_scores']
        ], ignore_index=True)
        
        return {
            'predictions': predictions_df,
            'metrics': metrics_df,
            'drift_scores': drift_scores_df,
            'model_config': model_config
        }
    
    def _generate_baseline_period(
        self, 
        config: ModelConfig, 
        days: int
    ) -> Dict[str, pd.DataFrame]:
        """Generate normal operation data"""
        
        predictions = []
        metrics = []
        drift_scores = []
        
        start_date = datetime.now() - timedelta(days=30)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Daily predictions
            for pred_id in range(config.traffic_per_day):
                timestamp = current_date + timedelta(
                    seconds=random.randint(0, 86400)
                )
                
                # Simulate prediction with slight random variation
                true_label = random.choice([0, 1])
                prediction = self._simulate_prediction(
                    true_label=true_label,
                    accuracy=config.baseline_accuracy,
                    noise_level=0.05
                )
                
                predictions.append({
                    'prediction_id': f"{config.model_id}_{day}_{pred_id}",
                    'model_id': config.model_id,
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'confidence': random.uniform(0.7, 0.99),
                    'true_label': true_label,
                    'latency_ms': config.baseline_latency_ms + np.random.normal(0, 5),
                    'features_hash': random.randint(1000000, 9999999)
                })
            
            # Daily metrics
            daily_predictions = [p for p in predictions if p['timestamp'].date() == current_date.date()]
            daily_accuracy = np.mean([
                1 if p['prediction'] == p['true_label'] else 0 
                for p in daily_predictions
            ])
            
            metrics.append({
                'model_id': config.model_id,
                'date': current_date.date(),
                'accuracy': daily_accuracy,
                'precision': daily_accuracy + np.random.normal(0, 0.02),
                'recall': daily_accuracy + np.random.normal(0, 0.02),
                'f1_score': daily_accuracy + np.random.normal(0, 0.01),
                'avg_latency_ms': config.baseline_latency_ms + np.random.normal(0, 3),
                'p99_latency_ms': config.baseline_latency_ms * 1.8 + np.random.normal(0, 10),
                'throughput': config.traffic_per_day + np.random.randint(-100, 100),
                'error_rate': 0.001 + np.random.uniform(0, 0.002)
            })
            
            # Drift scores (low during baseline)
            drift_scores.append({
                'model_id': config.model_id,
                'date': current_date.date(),
                'covariate_drift': np.random.uniform(0.01, 0.05),
                'prediction_drift': np.random.uniform(0.01, 0.04),
                'concept_drift': np.random.uniform(0.01, 0.03),
                'overall_drift_score': np.random.uniform(0.02, 0.06)
            })
        
        return {
            'predictions': pd.DataFrame(predictions),
            'metrics': pd.DataFrame(metrics),
            'drift_scores': pd.DataFrame(drift_scores)
        }
    
    def _generate_drift_period(
        self, 
        config: ModelConfig, 
        days: int
    ) -> Dict[str, pd.DataFrame]:
        """Generate data with increasing drift"""
        
        predictions = []
        metrics = []
        drift_scores = []
        
        start_date = datetime.now() - timedelta(days=20)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Drift severity increases over time
            drift_severity = day / days  # 0.0 to 1.0
            
            # Accuracy degrades
            degraded_accuracy = config.baseline_accuracy - (drift_severity * 0.10)
            
            # Daily predictions
            for pred_id in range(config.traffic_per_day):
                timestamp = current_date + timedelta(
                    seconds=random.randint(0, 86400)
                )
                
                true_label = random.choice([0, 1])
                prediction = self._simulate_prediction(
                    true_label=true_label,
                    accuracy=degraded_accuracy,
                    noise_level=0.05 + drift_severity * 0.10
                )
                
                predictions.append({
                    'prediction_id': f"{config.model_id}_drift_{day}_{pred_id}",
                    'model_id': config.model_id,
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'confidence': random.uniform(0.6, 0.90) - drift_severity * 0.1,
                    'true_label': true_label,
                    'latency_ms': config.baseline_latency_ms + np.random.normal(0, 5) + drift_severity * 15,
                    'features_hash': random.randint(1000000, 9999999)
                })
            
            # Daily metrics
            daily_predictions = [p for p in predictions if p['timestamp'].date() == current_date.date()]
            daily_accuracy = np.mean([
                1 if p['prediction'] == p['true_label'] else 0 
                for p in daily_predictions
            ])
            
            metrics.append({
                'model_id': config.model_id,
                'date': current_date.date(),
                'accuracy': daily_accuracy,
                'precision': daily_accuracy + np.random.normal(0, 0.03),
                'recall': daily_accuracy + np.random.normal(0, 0.03),
                'f1_score': daily_accuracy + np.random.normal(0, 0.02),
                'avg_latency_ms': config.baseline_latency_ms + drift_severity * 15,
                'p99_latency_ms': config.baseline_latency_ms * 1.8 + drift_severity * 30,
                'throughput': config.traffic_per_day + np.random.randint(-200, 0),
                'error_rate': 0.001 + drift_severity * 0.015
            })
            
            # Drift scores increase
            drift_scores.append({
                'model_id': config.model_id,
                'date': current_date.date(),
                'covariate_drift': 0.05 + drift_severity * 0.20,
                'prediction_drift': 0.04 + drift_severity * 0.18,
                'concept_drift': 0.03 + drift_severity * 0.25,
                'overall_drift_score': 0.06 + drift_severity * 0.30
            })
        
        return {
            'predictions': pd.DataFrame(predictions),
            'metrics': pd.DataFrame(metrics),
            'drift_scores': pd.DataFrame(drift_scores)
        }
    
    def _generate_critical_period(
        self, 
        config: ModelConfig, 
        days: int
    ) -> Dict[str, pd.DataFrame]:
        """Generate data with severe degradation"""
        
        predictions = []
        metrics = []
        drift_scores = []
        
        start_date = datetime.now() - timedelta(days=10)
        
        # Severe degradation
        critical_accuracy = config.baseline_accuracy - 0.12
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Daily predictions with poor performance
            for pred_id in range(config.traffic_per_day):
                timestamp = current_date + timedelta(
                    seconds=random.randint(0, 86400)
                )
                
                true_label = random.choice([0, 1])
                prediction = self._simulate_prediction(
                    true_label=true_label,
                    accuracy=critical_accuracy,
                    noise_level=0.20
                )
                
                predictions.append({
                    'prediction_id': f"{config.model_id}_critical_{day}_{pred_id}",
                    'model_id': config.model_id,
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'confidence': random.uniform(0.5, 0.75),
                    'true_label': true_label,
                    'latency_ms': config.baseline_latency_ms + np.random.normal(10, 10),
                    'features_hash': random.randint(1000000, 9999999)
                })
            
            # Daily metrics - poor performance
            daily_predictions = [p for p in predictions if p['timestamp'].date() == current_date.date()]
            daily_accuracy = np.mean([
                1 if p['prediction'] == p['true_label'] else 0 
                for p in daily_predictions
            ])
            
            metrics.append({
                'model_id': config.model_id,
                'date': current_date.date(),
                'accuracy': daily_accuracy,
                'precision': daily_accuracy + np.random.normal(0, 0.05),
                'recall': daily_accuracy + np.random.normal(0, 0.05),
                'f1_score': daily_accuracy + np.random.normal(0, 0.04),
                'avg_latency_ms': config.baseline_latency_ms + 20,
                'p99_latency_ms': config.baseline_latency_ms * 2.0 + 40,
                'throughput': config.traffic_per_day - 500,
                'error_rate': 0.001 + 0.020
            })
            
            # High drift scores
            drift_scores.append({
                'model_id': config.model_id,
                'date': current_date.date(),
                'covariate_drift': np.random.uniform(0.30, 0.45),
                'prediction_drift': np.random.uniform(0.28, 0.42),
                'concept_drift': np.random.uniform(0.32, 0.48),
                'overall_drift_score': np.random.uniform(0.35, 0.50)
            })
        
        return {
            'predictions': pd.DataFrame(predictions),
            'metrics': pd.DataFrame(metrics),
            'drift_scores': pd.DataFrame(drift_scores)
        }
    
    def _generate_recovery_period(
        self, 
        config: ModelConfig, 
        days: int
    ) -> Dict[str, pd.DataFrame]:
        """Generate data showing recovery after remediation"""
        
        predictions = []
        metrics = []
        drift_scores = []
        
        start_date = datetime.now() - timedelta(days=5)
        
        # Define accuracy levels
        critical_accuracy = config.baseline_accuracy - 0.12  # Same as critical period
        recovered_accuracy = config.baseline_accuracy + 0.01  # Slightly better than baseline
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Recovery improves over days
            recovery_factor = day / days
            current_accuracy = critical_accuracy + (recovered_accuracy - critical_accuracy) * recovery_factor
            
            # Daily predictions
            for pred_id in range(config.traffic_per_day):
                timestamp = current_date + timedelta(
                    seconds=random.randint(0, 86400)
                )
                
                true_label = random.choice([0, 1])
                prediction = self._simulate_prediction(
                    true_label=true_label,
                    accuracy=current_accuracy,
                    noise_level=0.05
                )
                
                predictions.append({
                    'prediction_id': f"{config.model_id}_recovery_{day}_{pred_id}",
                    'model_id': config.model_id,
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'confidence': random.uniform(0.75, 0.95),
                    'true_label': true_label,
                    'latency_ms': config.baseline_latency_ms + np.random.normal(0, 5),
                    'features_hash': random.randint(1000000, 9999999)
                })
            
            # Daily metrics - improving
            daily_predictions = [p for p in predictions if p['timestamp'].date() == current_date.date()]
            daily_accuracy = np.mean([
                1 if p['prediction'] == p['true_label'] else 0 
                for p in daily_predictions
            ])
            
            metrics.append({
                'model_id': config.model_id,
                'date': current_date.date(),
                'accuracy': daily_accuracy,
                'precision': daily_accuracy + np.random.normal(0, 0.02),
                'recall': daily_accuracy + np.random.normal(0, 0.02),
                'f1_score': daily_accuracy + np.random.normal(0, 0.01),
                'avg_latency_ms': config.baseline_latency_ms + np.random.normal(0, 3),
                'p99_latency_ms': config.baseline_latency_ms * 1.8 + np.random.normal(0, 10),
                'throughput': config.traffic_per_day + np.random.randint(-100, 100),
                'error_rate': 0.001 + np.random.uniform(0, 0.002)
            })
            
            # Drift scores decrease
            drift_scores.append({
                'model_id': config.model_id,
                'date': current_date.date(),
                'covariate_drift': 0.40 - recovery_factor * 0.38,
                'prediction_drift': 0.38 - recovery_factor * 0.36,
                'concept_drift': 0.42 - recovery_factor * 0.40,
                'overall_drift_score': 0.45 - recovery_factor * 0.42
            })
        
        return {
            'predictions': pd.DataFrame(predictions),
            'metrics': pd.DataFrame(metrics),
            'drift_scores': pd.DataFrame(drift_scores)
        }
    
    def _simulate_prediction(
        self, 
        true_label: int, 
        accuracy: float, 
        noise_level: float
    ) -> int:
        """Simulate a model prediction based on accuracy"""
        if random.random() < accuracy:
            return true_label
        else:
            return 1 - true_label
    
    def generate_all_models_data(self) -> Dict[str, Dict]:
        """Generate data for all models"""
        all_data = {}
        
        for model_config in self.models:
            print(f"Generating data for {model_config.model_name}...")
            model_data = self.generate_30_day_simulation(model_config)
            all_data[model_config.model_id] = model_data
        
        return all_data
    
    def save_to_files(self, output_dir: str = './simulated_data'):
        """Save all simulated data to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = self.generate_all_models_data()
        
        for model_id, data in all_data.items():
            model_dir = os.path.join(output_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save dataframes
            data['predictions'].to_csv(
                os.path.join(model_dir, 'predictions.csv'), 
                index=False
            )
            data['metrics'].to_csv(
                os.path.join(model_dir, 'metrics.csv'), 
                index=False
            )
            data['drift_scores'].to_csv(
                os.path.join(model_dir, 'drift_scores.csv'), 
                index=False
            )
            
            # Save config
            with open(os.path.join(model_dir, 'config.json'), 'w') as f:
                json.dump(asdict(data['model_config']), f, indent=2, default=str)
        
        print(f"All data saved to {output_dir}")
        return all_data


if __name__ == "__main__":
    # Generate all data
    simulator = ModelDataSimulator(seed=42)
    all_data = simulator.save_to_files()
    
    # Print summary
    print("\n" + "="*80)
    print("DATA GENERATION SUMMARY")
    print("="*80)
    
    for model_id, data in all_data.items():
        config = data['model_config']
        print(f"\n{config.model_name} ({model_id}):")
        print(f"  - Total Predictions: {len(data['predictions']):,}")
        print(f"  - Date Range: {data['metrics']['date'].min()} to {data['metrics']['date'].max()}")
        print(f"  - Baseline Accuracy: {config.baseline_accuracy:.2%}")
        print(f"  - Worst Accuracy: {data['metrics']['accuracy'].min():.2%}")
        print(f"  - Best Accuracy: {data['metrics']['accuracy'].max():.2%}")
        print(f"  - Max Drift Score: {data['drift_scores']['overall_drift_score'].max():.3f}")
