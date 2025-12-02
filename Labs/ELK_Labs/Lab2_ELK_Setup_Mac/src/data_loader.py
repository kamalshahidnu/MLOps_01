"""
Data loader for Wine Quality dataset with ELK logging support
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class WineQualityDataLoader:
    """Data loader for Wine Quality dataset"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def load_data(self):
        """
        Load and preprocess Wine Quality dataset
        Uses white wine dataset for regression/classification
        """
        # Wine Quality dataset URL
        data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        
        try:
            df = pd.read_csv(data_url, sep=';')
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Creating synthetic wine quality data...")
            df = self._create_synthetic_data()
        
        # Store feature names (all columns except quality)
        self.feature_names = [col for col in df.columns if col != 'quality']
        
        # For binary classification: quality >= 7 is good wine (1), else bad (0)
        df['label'] = (df['quality'] >= 7).astype(int)
        
        # Split features and target
        X = df[self.feature_names]
        y = df['label']
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.is_fitted = True
        
        return (
            pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index),
            pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index),
            y_train,
            y_test
        )
    
    def _create_synthetic_data(self, n_samples=4898):
        """Create synthetic wine quality data if download fails"""
        np.random.seed(self.random_state)
        
        data = {
            'fixed acidity': np.random.uniform(3.8, 14.2, n_samples),
            'volatile acidity': np.random.uniform(0.08, 1.1, n_samples),
            'citric acid': np.random.uniform(0.0, 1.66, n_samples),
            'residual sugar': np.random.uniform(0.6, 65.8, n_samples),
            'chlorides': np.random.uniform(0.009, 0.346, n_samples),
            'free sulfur dioxide': np.random.uniform(2, 289, n_samples),
            'total sulfur dioxide': np.random.uniform(9, 440, n_samples),
            'density': np.random.uniform(0.98711, 1.03898, n_samples),
            'pH': np.random.uniform(2.72, 3.82, n_samples),
            'sulphates': np.random.uniform(0.22, 1.08, n_samples),
            'alcohol': np.random.uniform(8.0, 14.9, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create quality scores (0-10) based on features
        quality_score = (
            (df['alcohol'] - 8.0) / 6.9 * 2.0 +
            (1.1 - df['volatile acidity']) / 1.02 * 1.5 +
            (df['citric acid'] / 1.66) * 1.0 +
            np.random.uniform(-1.0, 1.0, n_samples)
        )
        df['quality'] = np.clip(np.round(quality_score + 4), 3, 9).astype(int)
        
        return df
    
    def get_feature_names(self):
        """Get feature names"""
        if self.feature_names is None:
            raise ValueError("Data must be loaded first")
        return self.feature_names
    
    def save_scaler(self, path='models/scaler.pkl'):
        """Save the fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before saving")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")
    
    def load_scaler(self, path='models/scaler.pkl'):
        """Load a saved scaler"""
        self.scaler = joblib.load(path)
        self.is_fitted = True
        return self.scaler

