"""
Data loader for Heart Disease dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


class HeartDiseaseDataLoader:
    """Data loader for Heart Disease classification dataset"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def load_data(self):
        """
        Load and preprocess Heart Disease dataset
        Dataset columns:
        - age: age in years
        - sex: sex (1 = male; 0 = female)
        - cp: chest pain type (0-3)
        - trestbps: resting blood pressure
        - chol: serum cholesterol in mg/dl
        - fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        - restecg: resting electrocardiographic results (0-2)
        - thalach: maximum heart rate achieved
        - exang: exercise induced angina (1 = yes; 0 = no)
        - oldpeak: ST depression induced by exercise relative to rest
        - slope: the slope of the peak exercise ST segment (0-2)
        - ca: number of major vessels colored by flourosopy (0-3)
        - thal: thalassemia (0-3)
        - target: presence of heart disease (0 = no, 1 = yes)
        """
        # Create synthetic Heart Disease dataset (UCI Heart Disease inspired)
        # In production, you would load from a file or database
        np.random.seed(self.random_state)
        n_samples = 1025
        
        data = {
            'age': np.random.randint(29, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(94, 200, n_samples),
            'chol': np.random.randint(126, 564, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(71, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.round(np.random.uniform(0, 6.2, n_samples), 1),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 4, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target with some correlation to features
        # Higher risk: older age, higher cholesterol, lower thalach, higher oldpeak
        risk_score = (
            (df['age'] - 29) / 51 * 0.2 +
            (df['chol'] - 126) / 438 * 0.2 +
            (202 - df['thalach']) / 131 * 0.3 +
            df['oldpeak'] / 6.2 * 0.2 +
            df['exang'] * 0.1
        )
        df['target'] = (risk_score > 0.5).astype(int)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != 'target']
        
        # Split features and target
        X = df[self.feature_names]
        y = df['target']
        
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

