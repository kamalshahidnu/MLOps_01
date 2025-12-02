"""
Training script for Wine Quality model with ELK logging
"""
import os
import sys
import time
import pandas as pd
from data_loader import WineQualityDataLoader
from model import WineQualityModel
from logger import ELKLogger


def main():
    """Main training function"""
    print("=" * 60)
    print("Wine Quality Model Training with ELK Logging")
    print("=" * 60)
    
    # Initialize logger
    elk_logger = ELKLogger(log_file='logs/ml_model.log')
    elk_logger.log_system_event('training_start', 'Starting model training')
    
    try:
        # Initialize data loader
        print("\n[1/5] Loading and preprocessing data...")
        data_loader = WineQualityDataLoader(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = data_loader.load_data()
        
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Save scaler
        data_loader.save_scaler('models/scaler.pkl')
        feature_names = data_loader.get_feature_names()
        
        # Initialize and train model
        print("\n[2/5] Training Random Forest model...")
        model = WineQualityModel(n_estimators=200, max_depth=None, random_state=42)
        
        start_time = time.time()
        model.train(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        print("\n[3/5] Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        
        print("\nModel Performance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Log training metrics
        training_metrics = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc'])
        }
        
        elk_logger.log_training_metrics(
            model_name='wine_quality_rf',
            metrics=training_metrics,
            training_time=training_time,
            dataset_size=X_train.shape[0]
        )
        
        elk_logger.log_evaluation(
            model_name='wine_quality_rf',
            metrics=training_metrics,
            dataset_type='test'
        )
        
        # Save model
        print("\n[4/5] Saving model...")
        model.save_model('models/wine_quality_model.pkl')
        
        # Save feature importance
        feature_importance = model.get_feature_importance()
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        os.makedirs('models', exist_ok=True)
        feature_importance_df.to_csv('models/feature_importance.csv', index=False)
        print("Feature importance saved to models/feature_importance.csv")
        
        print("\n[5/5] Logging sample predictions...")
        # Log some sample predictions
        sample_size = min(10, X_test.shape[0])
        sample_indices = range(sample_size)
        
        for idx in sample_indices:
            features_dict = {name: float(X_test.iloc[idx][name]) 
                           for name in feature_names}
            prediction = model.predict(X_test.iloc[idx:idx+1])[0]
            probabilities = model.predict_proba(X_test.iloc[idx:idx+1])[0]
            probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
            
            elk_logger.log_prediction(
                model_name='wine_quality_rf',
                features=features_dict,
                prediction=prediction,
                probability=probability,
                prediction_id=f'sample_{idx}'
            )
        
        elk_logger.log_system_event('training_complete', 
                                   f'Training completed successfully. Accuracy: {metrics["accuracy"]:.4f}')
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Logs written to: logs/ml_model.log")
        print("=" * 60)
        
    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}"
        print(f"\nERROR: {error_msg}")
        elk_logger.log_error(
            model_name='wine_quality_rf',
            error_message=error_msg,
            stack_trace=traceback.format_exc()
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

