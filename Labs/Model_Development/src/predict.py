"""
Prediction script for Heart Disease model
"""
import numpy as np
import joblib
import os
import sys
from model import HeartDiseaseModel
from data_loader import HeartDiseaseDataLoader


class HeartDiseasePredictor:
    """Predictor class for Heart Disease model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_model_and_scaler(self, model_path='models/heart_disease_model.pkl', 
                              scaler_path='models/scaler.pkl'):
        """Load model and scaler"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
        # Load model
        self.model = HeartDiseaseModel()
        self.model.load_model(model_path)
        
        # Load scaler
        data_loader = HeartDiseaseDataLoader()
        self.scaler = data_loader.load_scaler(scaler_path)
        
        # Get feature names
        self.feature_names = data_loader.get_feature_names()
        
    def predict_single(self, feature_values):
        """
        Predict for a single sample
        Args:
            feature_values: List of feature values in order
        Returns:
            prediction: 0 (no disease) or 1 (disease)
            probability: Probability of disease
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model and scaler must be loaded first")
            
        # Convert to numpy array and reshape
        features = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        return prediction, probability
    
    def predict_batch(self, X):
        """
        Predict for multiple samples
        Args:
            X: Array-like of feature values
        Returns:
            predictions: Array of predictions
            probabilities: Array of probabilities
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model and scaler must be loaded first")
            
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities

