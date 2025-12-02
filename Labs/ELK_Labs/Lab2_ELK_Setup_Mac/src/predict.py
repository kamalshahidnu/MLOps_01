"""
Prediction script with ELK logging
"""
import os
import sys
import pandas as pd
import numpy as np
from data_loader import WineQualityDataLoader
from model import WineQualityModel
from logger import ELKLogger


def load_model_and_scaler():
    """Load trained model and scaler"""
    model = WineQualityModel()
    model.load_model('models/wine_quality_model.pkl')
    
    data_loader = WineQualityDataLoader()
    data_loader.load_scaler('models/scaler.pkl')
    feature_names = data_loader.get_feature_names()
    
    return model, data_loader, feature_names


def predict_single(model, scaler, feature_names, features_dict):
    """Make a single prediction"""
    # Convert features dict to array in correct order
    features_array = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
    
    return prediction, probability


def main():
    """Main prediction function"""
    print("=" * 60)
    print("Wine Quality Prediction with ELK Logging")
    print("=" * 60)
    
    # Initialize logger
    elk_logger = ELKLogger(log_file='logs/ml_model.log')
    
    try:
        # Load model and scaler
        print("\nLoading model and scaler...")
        model, data_loader, feature_names = load_model_and_scaler()
        scaler = data_loader.scaler
        
        print(f"Model loaded successfully!")
        print(f"Features: {', '.join(feature_names)}")
        
        # Sample wine features for prediction (you can modify these)
        sample_wine = {
            'fixed acidity': 7.0,
            'volatile acidity': 0.27,
            'citric acid': 0.36,
            'residual sugar': 20.7,
            'chlorides': 0.045,
            'free sulfur dioxide': 45,
            'total sulfur dioxide': 170,
            'density': 0.9970,
            'pH': 3.0,
            'sulphates': 0.45,
            'alcohol': 8.8
        }
        
        print("\nMaking prediction for sample wine:")
        for key, value in sample_wine.items():
            print(f"  {key}: {value}")
        
        prediction, probability = predict_single(model, scaler, feature_names, sample_wine)
        
        result = "Good Quality (Quality >= 7)" if prediction == 1 else "Poor Quality (Quality < 7)"
        
        print(f"\nPrediction: {result}")
        print(f"Probability: {probability:.4f}")
        
        # Log prediction
        elk_logger.log_prediction(
            model_name='wine_quality_rf',
            features=sample_wine,
            prediction=prediction,
            probability=probability
        )
        
        print("\nPrediction logged to ELK stack!")
        print(f"Log file: logs/ml_model.log")
        
        # Make multiple predictions from test set
        print("\n" + "=" * 60)
        print("Making batch predictions from test set...")
        
        # Load test data
        _, X_test, _, y_test = data_loader.load_data()
        
        num_predictions = min(20, X_test.shape[0])
        print(f"Making {num_predictions} predictions...")
        
        for i in range(num_predictions):
            features_dict = {name: float(X_test.iloc[i][name]) 
                           for name in feature_names}
            prediction, probability = predict_single(model, scaler, feature_names, 
                                                   {name: X_test.iloc[i][name] 
                                                    for name in feature_names})
            
            elk_logger.log_prediction(
                model_name='wine_quality_rf',
                features=features_dict,
                prediction=prediction,
                probability=probability,
                prediction_id=f'test_{i}'
            )
        
        print(f"Logged {num_predictions} predictions to ELK stack!")
        print("=" * 60)
        
    except Exception as e:
        import traceback
        error_msg = f"Prediction failed: {str(e)}"
        print(f"\nERROR: {error_msg}")
        elk_logger.log_error(
            model_name='wine_quality_rf',
            error_message=error_msg,
            stack_trace=traceback.format_exc()
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

