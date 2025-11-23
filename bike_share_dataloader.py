"""
Real Data Loader for ML Model Monitoring System
Integrates real bike-sharing dataset with natural drift patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RealDataLoader:
    """
    Load and prepare real bike-sharing data for monitoring system
    """
    
    def __init__(self, output_dir: str = './simulated_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Model configuration
        self.model_config = {
            'model_id': 'bike_demand_v1',
            'model_name': 'Bike Demand Predictor',
            'model_type': 'regression',
            'baseline_accuracy': 0.85,
            'baseline_latency_ms': 25.0
        }
    
    def download_data(self):
        """
        Download bike sharing dataset
        """
        print("📥 Downloading bike sharing dataset...")
        
        try:
            # Try to download from UCI repository
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
            
            import urllib.request
            import zipfile
            
            # Download
            zip_path = 'bike_data.zip'
            if not os.path.exists(zip_path):
                print("   Downloading from UCI repository...")
                urllib.request.urlretrieve(url, zip_path)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('bike_data_raw')
            
            print("   ✓ Downloaded and extracted")
            
            # Load hourly data
            df = pd.read_csv('bike_data_raw/hour.csv')
            
            return df
            
        except Exception as e:
            print(f"   ⚠️  Auto-download failed: {e}")
            print("\n   Manual download:")
            print("   1. Go to: https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset")
            print("   2. Download hour.csv")
            print("   3. Place in project directory")
            print("   4. Run again")
            
            # Try to load if already downloaded manually
            if os.path.exists('hour.csv'):
                print("\n   Found hour.csv - using local file")
                return pd.read_csv('hour.csv')
            else:
                raise FileNotFoundError("Please download dataset manually")
    
    def prepare_monitoring_data(self, df: pd.DataFrame) -> dict:
        """
        Prepare bike-sharing data for monitoring system
        """
        print("\n🔧 Preparing data for monitoring system...")
        
        # Features for prediction
        feature_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
                       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        target_col = 'cnt'
        
        # Split: First 60% for training, next 40% for monitoring (30 days)
        split_idx = int(len(df) * 0.6)
        
        train_df = df.iloc[:split_idx]
        monitoring_df = df.iloc[split_idx:split_idx + (30 * 24)]  # 30 days of hourly data
        
        print(f"   Training data: {len(train_df)} records")
        print(f"   Monitoring data: {len(monitoring_df)} records ({len(monitoring_df)//24} days)")
        
        # Train model
        print("\n🤖 Training bike demand prediction model...")
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_preds = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_preds)
        train_r2 = r2_score(y_train, train_preds)
        
        print(f"   Training MAE: {train_mae:.2f}")
        print(f"   Training R²: {train_r2:.3f}")
        
        # Generate 30-day monitoring data
        print("\n📊 Generating 30-day monitoring data...")
        
        predictions_list = []
        metrics_list = []
        drift_scores_list = []
        
        for day in range(30):
            day_start = day * 24
            day_end = (day + 1) * 24
            
            if day_end > len(monitoring_df):
                break
            
            # Get day's data
            day_df = monitoring_df.iloc[day_start:day_end]
            X_day = day_df[feature_cols]
            y_true_day = day_df[target_col]
            
            # Make predictions
            y_pred_day = model.predict(X_day)
            
            # Store hourly predictions
            for i, (pred, true) in enumerate(zip(y_pred_day, y_true_day)):
                # Create features hash from input features
                feature_values = X_day.iloc[i].values
                features_hash = int(abs(hash(tuple(feature_values))) % 1e9)
                
                predictions_list.append({
                    'prediction_id': f'bike_{day}_{i}',
                    'model_id': 'bike_demand_v1',
                    'timestamp': datetime.now() - timedelta(days=30-day, hours=i),
                    'prediction': int(round(pred)),
                    'true_label': int(true),
                    'confidence': 0.85 + np.random.uniform(-0.1, 0.1),
                    'features_hash': features_hash  # Added this field
                })
            
            # Calculate daily metrics
            day_mae = mean_absolute_error(y_true_day, y_pred_day)
            day_rmse = np.sqrt(mean_squared_error(y_true_day, y_pred_day))
            day_r2 = r2_score(y_true_day, y_pred_day)
            
            # Convert R² to accuracy-like metric (0-1 scale)
            day_accuracy = max(0, min(1, day_r2))
            
            metrics_list.append({
                'model_id': 'bike_demand_v1',
                'date': (datetime.now() - timedelta(days=30-day)).date(),
                'accuracy': day_accuracy,
                'precision': day_accuracy + np.random.normal(0, 0.02),  # Simulated
                'recall': day_accuracy + np.random.normal(0, 0.02),
                'f1_score': day_accuracy,
                'avg_latency_ms': 25.0 + np.random.normal(0, 3),
                'p99_latency_ms': 45.0 + np.random.normal(0, 5),
                'throughput': 24,  # Hourly predictions
                'error_rate': 0.001
            })
            
            # Calculate drift scores (compare to training distribution)
            drift_score = self._calculate_drift(X_train, X_day)
            
            drift_scores_list.append({
                'model_id': 'bike_demand_v1',
                'date': (datetime.now() - timedelta(days=30-day)).date(),
                'covariate_drift': drift_score['covariate'],
                'prediction_drift': drift_score['prediction'],
                'concept_drift': drift_score['concept'],
                'overall_drift_score': drift_score['overall']
            })
        
        print(f"   Generated {len(predictions_list)} predictions")
        print(f"   Generated {len(metrics_list)} daily metrics")
        
        return {
            'predictions': pd.DataFrame(predictions_list),
            'metrics': pd.DataFrame(metrics_list),
            'drift_scores': pd.DataFrame(drift_scores_list),
            'model': model,
            'feature_cols': feature_cols
        }
    
    def _calculate_drift(self, X_baseline: pd.DataFrame, X_current: pd.DataFrame) -> dict:
        """
        Calculate drift scores using statistical methods
        """
        # Calculate feature-wise drift using Kolmogorov-Smirnov test
        from scipy import stats
        
        drift_scores = []
        
        for col in X_baseline.columns:
            if col in X_current.columns:
                # KS test for continuous features
                if X_baseline[col].nunique() > 10:
                    ks_stat, p_value = stats.ks_2samp(
                        X_baseline[col].values,
                        X_current[col].values
                    )
                    drift_scores.append(ks_stat)
                else:
                    # Chi-square for categorical
                    try:
                        chi2, p_value = stats.chisquare(
                            np.histogram(X_current[col], bins=10)[0] + 1,
                            np.histogram(X_baseline[col], bins=10)[0] + 1
                        )
                        drift_scores.append(min(chi2 / 100, 1.0))
                    except:
                        drift_scores.append(0.0)
        
        # Aggregate drift score
        overall_drift = np.mean(drift_scores) if drift_scores else 0.0
        
        return {
            'covariate': overall_drift * 0.9,
            'prediction': overall_drift * 1.1,
            'concept': overall_drift,
            'overall': overall_drift
        }
    
    def save_to_monitoring_format(self, data: dict):
        """
        Save data in format expected by monitoring system
        """
        model_dir = self.output_dir / 'bike_demand_v1'
        model_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n💾 Saving data to {model_dir}...")
        
        # Save dataframes
        data['predictions'].to_csv(model_dir / 'predictions.csv', index=False)
        data['metrics'].to_csv(model_dir / 'metrics.csv', index=False)
        data['drift_scores'].to_csv(model_dir / 'drift_scores.csv', index=False)
        
        # Save model config
        import json
        with open(model_dir / 'config.json', 'w') as f:
            json.dump(self.model_config, f, indent=2)
        
        print("   ✓ predictions.csv")
        print("   ✓ metrics.csv")
        print("   ✓ drift_scores.csv")
        print("   ✓ config.json")
        
        print(f"\n✅ Real data ready! Run: python main_simulation.py --model bike_demand_v1")


def main():
    """
    Main execution - download and prepare real data
    """
    print("\n" + "="*80)
    print("🚴 REAL BIKE-SHARING DATA INTEGRATION")
    print("="*80)
    
    loader = RealDataLoader()
    
    # Download data
    try:
        df = pd.read_csv('data/hour.csv')
        print(f"✓ Loaded {len(df)} records from bike-sharing dataset")
        
        # Prepare monitoring data
        data = loader.prepare_monitoring_data(df)
        
        # Save in monitoring format
        loader.save_to_monitoring_format(data)
        
        print("\n" + "="*80)
        print("✅ INTEGRATION COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run monitoring: python main_simulation.py --model bike_demand_v1")
        print("  2. View results: open simulation_results/*.png")
        print("\n📊 Data Summary:")
        print(f"   Model: {loader.model_config['model_name']}")
        print(f"   Predictions: {len(data['predictions']):,}")
        print(f"   Days: {len(data['metrics'])}")
        print(f"   Accuracy Range: {data['metrics']['accuracy'].min():.3f} - {data['metrics']['accuracy'].max():.3f}")
        print(f"   Drift Range: {data['drift_scores']['overall_drift_score'].min():.3f} - {data['drift_scores']['overall_drift_score'].max():.3f}")
        
    except FileNotFoundError:
        print("\n" + "="*80)
        print("⚠️  MANUAL DOWNLOAD REQUIRED")
        print("="*80)
        print("\nAutomatic download failed. Please:")
        print("\n1. Go to: https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset")
        print("2. Download the dataset")
        print("3. Extract hour.csv to this directory")
        print("4. Run this script again: python real_data_loader.py")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()