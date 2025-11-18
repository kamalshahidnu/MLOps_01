"""
Comprehensive Training Script
Integrates all model development components:
- Feature selection
- Hyperparameter tuning
- Model validation
- Bias checking
- Model selection after bias checking
- Model registry
- Rollback mechanism
"""
import os
import sys
import pandas as pd
import numpy as np
from data_loader import HeartDiseaseDataLoader
from model import HeartDiseaseModel
from feature_selection import FeatureSelector
from hyperparameter_tuning import HyperparameterTuner
from model_validation import ModelValidator
from bias_checking import BiasChecker
from model_registry import ModelRegistry
from rollback import RollbackManager


def main():
    """Comprehensive training pipeline"""
    print("=" * 80)
    print("COMPREHENSIVE MODEL DEVELOPMENT PIPELINE")
    print("=" * 80)
    
    # Step 1: Load and preprocess data
    print("\n[1/8] Loading and preprocessing data...")
    data_loader = HeartDiseaseDataLoader(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = data_loader.load_data()
    
    # Split validation set from test set
    from sklearn.model_selection import train_test_split
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(data_loader.get_feature_names())}")
    
    data_loader.save_scaler('models/scaler.pkl')
    
    # Step 2: Feature selection
    print("\n[2/8] Performing feature selection...")
    feature_selector = FeatureSelector(method='mutual_info', k=None)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_val_selected = feature_selector.transform(X_val)
    X_test_selected = feature_selector.transform(X_test)
    
    selected_features = feature_selector.get_selected_features()
    print(f"   Selected {len(selected_features)} features out of {len(X_train.columns)}")
    feature_selector.save('models/feature_selector.pkl')
    
    # Step 3: Hyperparameter tuning
    print("\n[3/8] Hyperparameter tuning...")
    tuner = HyperparameterTuner(
        model_class=HeartDiseaseModel,
        X_train=X_train_selected,
        y_train=y_train,
        X_val=X_val_selected,
        y_val=y_val,
        n_trials=20,  # Reduced for faster execution
        cv=3,
        scoring='roc_auc'
    )
    
    best_params, best_model = tuner.tune()
    print(f"   Best parameters: {best_params}")
    tuner.save_study('models/optuna_study.pkl')
    
    # Step 4: Model validation
    print("\n[4/8] Validating model...")
    validator = ModelValidator(
        best_model, X_train_selected, y_train, 
        X_val_selected, y_val, X_test_selected, y_test
    )
    validation_results = validator.validate()
    
    is_valid = validator.is_model_valid()
    print(f"   Model valid: {is_valid['is_valid']}")
    print(f"   Validation accuracy: {validation_results['val_metrics']['accuracy']:.4f}")
    validator.save_validation_report('models/validation_report.json')
    
    if not is_valid['is_valid']:
        print("   ⚠️  Model validation failed. Consider retraining with different parameters.")
    
    # Step 5: Bias checking
    print("\n[5/8] Checking for bias...")
    # Identify sensitive features (categorical or low cardinality)
    sensitive_features = [
        col for col in X_train_selected.columns 
        if X_train_selected[col].nunique() <= 5
    ]
    
    bias_checker = BiasChecker(
        best_model, X_test_selected, y_test, 
        sensitive_features=sensitive_features
    )
    bias_results = bias_checker.check_all_bias()
    bias_report = bias_checker.generate_bias_report()
    
    print(f"   Checked bias for {len(sensitive_features)} sensitive features")
    unfair_features = [
        feat for feat, summary in bias_report['summary'].items()
        if not summary['is_fair']
    ]
    if unfair_features:
        print(f"   ⚠️  Potential bias detected in: {unfair_features}")
    else:
        print("   ✅ No significant bias detected")
    
    # Step 6: Model selection after bias checking
    print("\n[6/8] Final model selection...")
    # Combine validation metrics and bias results
    final_score = validation_results['val_metrics']['roc_auc']
    if unfair_features:
        # Penalize models with bias
        final_score *= 0.9
    
    print(f"   Final model score: {final_score:.4f}")
    
    # Step 7: Model registry
    print("\n[7/8] Registering model...")
    registry = ModelRegistry(registry_path='models/registry')
    
    model_metadata = {
        'validation_metrics': validation_results['val_metrics'],
        'bias_report': bias_report['summary'],
        'selected_features': selected_features,
        'hyperparameters': best_params,
        'feature_selector_method': feature_selector.method
    }
    
    version, model_path = registry.register_model(
        best_model,
        model_name='heart_disease_classifier',
        metadata=model_metadata
    )
    print(f"   Model registered: heart_disease_classifier v{version}")
    
    # Step 8: Rollback check
    print("\n[8/8] Checking for rollback...")
    rollback_manager = RollbackManager(registry_path='models/registry')
    
    # Get previous model metrics if available
    try:
        previous_models = registry.list_models().get('heart_disease_classifier', {})
        if len(previous_models) > 1:
            # Get previous version metrics
            prev_versions = sorted(previous_models.keys(), reverse=True)
            prev_metadata = previous_models[prev_versions[1]]
            prev_metrics = prev_metadata.get('metadata', {}).get('validation_metrics', {})
            
            should_rollback, rollback_info = rollback_manager.should_rollback(
                validation_results['val_metrics'],
                prev_metrics
            )
            
            if should_rollback:
                print("   ⚠️  New model performs worse. Rolling back...")
                model, metadata = rollback_manager.rollback('heart_disease_classifier')
                print(f"   Rolled back to version: {metadata['version']}")
            else:
                print("   ✅ New model performs better. Keeping new version.")
        else:
            print("   ✅ First model version. No rollback needed.")
    except Exception as e:
        print(f"   Note: Could not check previous model: {e}")
    
    # Save final model
    best_model.save_model('models/heart_disease_model.pkl')
    
    # Save feature importance
    feature_importance = best_model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    feature_importance_df.to_csv('models/feature_importance.csv', index=False)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nModel saved to: models/heart_disease_model.pkl")
    print(f"Model version: {version}")
    print(f"Validation Accuracy: {validation_results['val_metrics']['accuracy']:.4f}")
    print(f"Validation ROC-AUC: {validation_results['val_metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

