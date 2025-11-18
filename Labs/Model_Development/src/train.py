"""
Training script for Heart Disease prediction model
"""
import os
import sys
import pandas as pd
from data_loader import HeartDiseaseDataLoader
from model import HeartDiseaseModel


def main():
    """Main training function"""
    print("=" * 60)
    print("Heart Disease Prediction Model Training")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = HeartDiseaseDataLoader(test_size=0.2, random_state=42)
    
    # Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = data_loader.load_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(data_loader.get_feature_names())}")
    print(f"   Class distribution (train): {y_train.value_counts().to_dict()}")
    
    # Save scaler for later use
    data_loader.save_scaler('models/scaler.pkl')
    
    # Initialize and train model
    print("\n[2/4] Training XGBoost model...")
    model = HeartDiseaseModel(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.train(X_train, y_train)
    
    # Evaluate model
    print("\n[3/4] Evaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Model Performance Metrics")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"              No    Yes")
    print(f"Actual No   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"      Yes   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Save model
    print("\n[4/4] Saving model...")
    model.save_model('models/heart_disease_model.pkl')
    
    # Save feature importance
    feature_names = data_loader.get_feature_names()
    feature_importance = model.get_feature_importance()
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv('models/feature_importance.csv', index=False)
    print("Feature importance saved to models/feature_importance.csv")
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc']
    }])
    metrics_df.to_csv('models/metrics.csv', index=False)
    print("Metrics saved to models/metrics.csv")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

