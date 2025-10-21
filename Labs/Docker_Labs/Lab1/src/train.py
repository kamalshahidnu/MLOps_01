"""
Training script for Boston Housing price prediction model
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from data_loader import BostonHousingDataLoader
from model import BostonHousingModel


def main():
    """Main training function"""
    print("Starting Boston Housing price prediction model training...")
    
    # Initialize data loader
    data_loader = BostonHousingDataLoader(test_size=0.2, random_state=42)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = data_loader.load_data()
    
    # Save scaler for later use
    data_loader.save_scaler('models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # Initialize and train model
    print("Training Ridge regression model...")
    model = BostonHousingModel(alpha=1.0)
    model.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nModel Performance:")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    # Save model
    model.save_model('models/boston_housing_model.pkl')
    print("Model saved to models/boston_housing_model.pkl")
    
    # Save feature importance
    feature_names = data_loader.get_feature_names()
    feature_importance = model.get_feature_importance()
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': feature_importance
    }).sort_values('coefficient', key=abs, ascending=False)
    
    feature_importance_df.to_csv('models/feature_importance.csv', index=False)
    print("Feature importance saved to models/feature_importance.csv")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
