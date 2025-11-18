"""
Distributed Training Module
Using Ray for distributed model training
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class DistributedTrainer:
    """Distributed training using Ray"""
    
    def __init__(self, n_workers=4, use_gpu=False):
        """
        Initialize distributed trainer
        Args:
            n_workers: Number of worker processes
            use_gpu: Whether to use GPU
        """
        try:
            import ray
            ray.init(num_cpus=n_workers, ignore_reinit_error=True)
            self.ray_available = True
        except ImportError:
            print("Ray not available. Install with: pip install ray")
            self.ray_available = False
            return
        
        self.n_workers = n_workers
        self.use_gpu = use_gpu
    
    def train_distributed(self, X_train, y_train, model_params=None):
        """
        Train model using distributed training
        Args:
            X_train: Training features
            y_train: Training labels
            model_params: Model parameters
        Returns:
            Trained model
        """
        if not self.ray_available:
            raise RuntimeError("Ray is not available")
        
        import ray
        
        # Split data across workers
        data_splits = self._split_data(X_train, y_train, self.n_workers)
        
        # Train on each worker
        @ray.remote
        def train_worker(X_part, y_part, params):
            model = XGBClassifier(**params)
            model.fit(X_part, y_part)
            return model
        
        # Train models in parallel
        futures = [
            train_worker.remote(X_part, y_part, model_params or {})
            for X_part, y_part in data_splits
        ]
        
        models = ray.get(futures)
        
        # Aggregate models (simplified - in production use proper ensemble)
        # For now, return the first model (in production, use model averaging)
        return models[0]
    
    def _split_data(self, X, y, n_splits):
        """Split data into n splits"""
        split_size = len(X) // n_splits
        splits = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X)
            splits.append((X.iloc[start_idx:end_idx], y.iloc[start_idx:end_idx]))
        
        return splits
    
    def shutdown(self):
        """Shutdown Ray"""
        if self.ray_available:
            import ray
            ray.shutdown()

