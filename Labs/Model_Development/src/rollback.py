"""
Rollback Mechanism Module
Implements rollback to previous model if new model performs worse
"""
import os
import json
from pathlib import Path
from model_registry import ModelRegistry


class RollbackManager:
    """Manages model rollback mechanism"""
    
    def __init__(self, registry_path='models/registry', metrics_threshold=0.05):
        """
        Initialize rollback manager
        Args:
            registry_path: Path to model registry
            metrics_threshold: Minimum improvement threshold (e.g., 0.05 = 5% improvement required)
        """
        self.registry = ModelRegistry(registry_path=registry_path)
        self.metrics_threshold = metrics_threshold
        self.rollback_history = []
        
    def should_rollback(self, new_model_metrics, previous_model_metrics, metric='accuracy'):
        """
        Determine if rollback is needed
        Args:
            new_model_metrics: Metrics from newly trained model
            previous_model_metrics: Metrics from previous model
            metric: Metric to compare (default: 'accuracy')
        Returns:
            Boolean indicating if rollback is needed
        """
        if previous_model_metrics is None:
            # No previous model, no rollback needed
            return False
        
        new_score = new_model_metrics.get(metric, 0)
        prev_score = previous_model_metrics.get(metric, 0)
        
        # Check if new model is worse
        is_worse = new_score < prev_score
        
        # Check if improvement is below threshold
        improvement = new_score - prev_score
        below_threshold = improvement < self.metrics_threshold
        
        should_rollback = is_worse or (improvement > 0 and below_threshold)
        
        return should_rollback, {
            'new_score': new_score,
            'prev_score': prev_score,
            'improvement': improvement,
            'is_worse': is_worse,
            'below_threshold': below_threshold,
            'should_rollback': should_rollback
        }
    
    def rollback(self, model_name, target_version=None):
        """
        Rollback to a previous model version
        Args:
            model_name: Name of the model
            target_version: Target version to rollback to (None = previous version)
        Returns:
            Rolled back model and metadata
        """
        if target_version is None:
            # Get previous version
            all_versions = list(self.registry.list_models().get(model_name, {}).keys())
            if len(all_versions) < 2:
                raise ValueError("No previous version to rollback to")
            
            all_versions.sort(reverse=True)
            target_version = all_versions[1]  # Second most recent
        
        model, metadata = self.registry.get_model(model_name, target_version)
        
        # Record rollback
        rollback_record = {
            'model_name': model_name,
            'rolled_back_to': target_version,
            'timestamp': metadata.get('timestamp'),
            'reason': 'New model performed worse than previous model'
        }
        
        self.rollback_history.append(rollback_record)
        self._save_rollback_history()
        
        print(f"Rolled back {model_name} to version {target_version}")
        return model, metadata
    
    def _save_rollback_history(self):
        """Save rollback history"""
        history_path = Path(self.registry.registry_path) / 'rollback_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.rollback_history, f, indent=2)
    
    def get_rollback_history(self):
        """Get rollback history"""
        return self.rollback_history
    
    def deploy_with_rollback_check(self, model_name, new_model, new_metrics, 
                                   previous_model_metrics=None):
        """
        Deploy model with automatic rollback check
        Args:
            model_name: Name of the model
            new_model: Newly trained model
            new_metrics: Metrics from new model
            previous_model_metrics: Metrics from previous model (None if first deployment)
        Returns:
            Tuple of (deployed_model, deployed_version, was_rollback)
        """
        should_rollback, rollback_info = self.should_rollback(
            new_metrics, previous_model_metrics
        )
        
        if should_rollback:
            print(f"⚠️  New model performs worse. Rolling back...")
            print(f"   Previous: {rollback_info['prev_score']:.4f}")
            print(f"   New:      {rollback_info['new_score']:.4f}")
            print(f"   Improvement: {rollback_info['improvement']:.4f}")
            
            # Rollback to previous version
            model, metadata = self.rollback(model_name)
            return model, metadata['version'], True
        else:
            # Register and deploy new model
            version, _ = self.registry.register_model(
                new_model, 
                model_name,
                metadata={'metrics': new_metrics}
            )
            print(f"✅ New model deployed: {model_name} v{version}")
            return new_model, version, False

