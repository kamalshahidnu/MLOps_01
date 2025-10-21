"""
Prediction script for Boston Housing price prediction
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from data_loader import BostonHousingDataLoader
from model import BostonHousingModel


class BostonHousingPredictor:
    """Predictor class for Boston Housing price prediction"""
    
    def __init__(self, model_path='models/boston_housing_model.pkl', 
                 scaler_path='models/scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_model_and_scaler(self):
        """Load the trained model and scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Get feature names
        data_loader = BostonHousingDataLoader()
        self.feature_names = data_loader.get_feature_names()
        
    def predict_single(self, features):
        """Predict price for a single house"""
        if self.model is None or self.scaler is None:
            self.load_model_and_scaler()
            
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        return prediction
    
    def predict_batch(self, features_list):
        """Predict prices for multiple houses"""
        if self.model is None or self.scaler is None:
            self.load_model_and_scaler()
            
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        return predictions


def main():
    """Main prediction function"""
    print("Boston Housing Price Prediction")
    print("=" * 40)
    
    # Initialize predictor
    predictor = BostonHousingPredictor()
    
    try:
        # Load model and scaler
        predictor.load_model_and_scaler()
        print("Model and scaler loaded successfully!")
        
        # Example prediction with sample data
        # Sample house features: [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
        sample_house = [0.02731, 0.0, 7.07, 0.0, 0.469, 6.421, 78.9, 4.9671, 2.0, 242.0, 17.8, 396.90, 9.14]
        
        print(f"\nSample house features:")
        for i, (feature, value) in enumerate(zip(predictor.feature_names, sample_house)):
            print(f"{feature}: {value}")
        
        # Make prediction
        prediction = predictor.predict_single(sample_house)
        print(f"\nPredicted house price: ${prediction:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to train the model first by running train.py")


if __name__ == "__main__":
    main()
