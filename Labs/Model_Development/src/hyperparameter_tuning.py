"""
Hyperparameter Tuning Module
Supports Optuna and Ray Tune
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import optuna
from optuna.visualization import (
    plot_optimization_history, plot_param_importances, plot_parallel_coordinate
)
import joblib
import os


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna"""
    
    def __init__(self, model_class, X_train, y_train, X_val=None, y_val=None, 
                 n_trials=50, cv=5, scoring='roc_auc', direction='maximize'):
        """
        Initialize hyperparameter tuner
        Args:
            model_class: Model class to tune
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            n_trials: Number of optimization trials
            cv: Cross-validation folds
            scoring: Scoring metric
            direction: 'maximize' or 'minimize'
        """
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.direction = direction
        self.study = None
        self.best_params = None
        self.best_model = None
        self.trial_results = []
        
    def _objective(self, trial):
        """Objective function for Optuna"""
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'random_state': 42
        }
        
        # Create and train model
        model = self.model_class(**params)
        
        # Evaluate using cross-validation
        scores = cross_val_score(
            model.model, self.X_train, self.y_train,
            cv=self.cv, scoring=self.scoring, n_jobs=-1
        )
        
        score = scores.mean()
        
        # Store trial results
        self.trial_results.append({
            'trial': trial.number,
            'params': params,
            'score': score
        })
        
        return score
    
    def tune(self):
        """Run hyperparameter tuning"""
        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        self.best_params['random_state'] = 42
        
        # Train best model
        self.best_model = self.model_class(**self.best_params)
        self.best_model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        return self.best_params, self.best_model
    
    def get_best_params(self):
        """Get best hyperparameters"""
        return self.best_params
    
    def get_best_model(self):
        """Get best trained model"""
        return self.best_model
    
    def get_trial_results(self):
        """Get all trial results"""
        return pd.DataFrame(self.trial_results)
    
    def save_study(self, path='models/optuna_study.pkl'):
        """Save Optuna study"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.study, path)
    
    def plot_optimization_history(self):
        """Plot optimization history"""
        return plot_optimization_history(self.study)
    
    def plot_param_importances(self):
        """Plot parameter importances"""
        return plot_param_importances(self.study)
    
    def plot_parallel_coordinate(self):
        """Plot parallel coordinate plot"""
        return plot_parallel_coordinate(self.study)


class RayTuneHyperparameterTuner:
    """Hyperparameter tuning using Ray Tune (for distributed tuning)"""
    
    def __init__(self, model_class, X_train, y_train, num_samples=50):
        """
        Initialize Ray Tune hyperparameter tuner
        Args:
            model_class: Model class to tune
            X_train, y_train: Training data
            num_samples: Number of samples to try
        """
        try:
            from ray import tune
            from ray.tune.schedulers import ASHAScheduler
        except ImportError:
            raise ImportError("Ray Tune is not installed. Install with: pip install ray[tune]")
        
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = num_samples
        self.best_result = None
        
    def _trainable(self, config):
        """Trainable function for Ray Tune"""
        from ray import tune
        
        # Create model with config
        model = self.model_class(**config)
        
        # Train model
        model.train(self.X_train, self.y_train)
        
        # Evaluate (simplified - use cross-validation in production)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(
            model.model, self.X_train, self.y_train,
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        # Report score to Ray Tune
        tune.report(mean_score=scores.mean(), std_score=scores.std())
    
    def tune(self):
        """Run distributed hyperparameter tuning with Ray"""
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        
        # Define search space
        config = {
            'n_estimators': tune.choice([50, 100, 150, 200, 300]),
            'max_depth': tune.choice([3, 4, 5, 6, 7, 8, 9, 10]),
            'learning_rate': tune.loguniform(0.01, 0.3),
            'subsample': tune.uniform(0.6, 1.0),
            'colsample_bytree': tune.uniform(0.6, 1.0),
            'min_child_weight': tune.choice([1, 2, 3, 4, 5, 6, 7]),
            'random_state': 42
        }
        
        # Create scheduler
        scheduler = ASHAScheduler(metric='mean_score', mode='max')
        
        # Run tuning
        analysis = tune.run(
            self._trainable,
            config=config,
            num_samples=self.num_samples,
            scheduler=scheduler,
            verbose=1
        )
        
        self.best_result = analysis.best_result
        return analysis.best_config, analysis.best_result

