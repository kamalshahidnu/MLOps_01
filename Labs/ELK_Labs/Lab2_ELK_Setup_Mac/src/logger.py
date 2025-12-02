"""
ELK Logger for ML model events
Logs model predictions, training metrics, and system events to a JSON log file
that will be processed by Logstash and sent to Elasticsearch
"""
import json
import logging
from datetime import datetime
from pathlib import Path
import os
import numpy as np


class ELKLogger:
    """Logger that writes structured JSON logs for ELK stack"""
    
    def __init__(self, log_file='logs/ml_model.log', log_level=logging.INFO):
        self.log_file = log_file
        self.log_dir = Path(log_file).parent
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup standard Python logger
        self.logger = logging.getLogger('MLModelLogger')
        self.logger.setLevel(log_level)
        
        # File handler for JSON logs (to be consumed by Logstash)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # JSON formatter
        json_formatter = JSONFormatter()
        file_handler.setFormatter(json_formatter)
        
        self.logger.addHandler(file_handler)
        
    def log_prediction(self, model_name, features, prediction, probability=None, 
                      prediction_id=None, timestamp=None):
        """Log a model prediction"""
        log_entry = {
            'event_type': 'prediction',
            'timestamp': timestamp or datetime.utcnow().isoformat() + 'Z',
            'model_name': model_name,
            'prediction_id': prediction_id or self._generate_id(),
            'features': features if isinstance(features, dict) else self._features_to_dict(features),
            'prediction': int(prediction) if hasattr(prediction, '__iter__') and len(prediction) == 1 else int(prediction),
            'probability': float(probability) if probability is not None else None
        }
        
        self.logger.info('', extra={'json': log_entry})
        return log_entry
    
    def log_training_metrics(self, model_name, metrics, training_time=None, 
                            dataset_size=None, timestamp=None):
        """Log training metrics"""
        log_entry = {
            'event_type': 'training',
            'timestamp': timestamp or datetime.utcnow().isoformat() + 'Z',
            'model_name': model_name,
            'metrics': metrics,
            'training_time_seconds': training_time,
            'dataset_size': dataset_size
        }
        
        self.logger.info('', extra={'json': log_entry})
        return log_entry
    
    def log_evaluation(self, model_name, metrics, dataset_type='test', 
                      timestamp=None):
        """Log evaluation metrics"""
        log_entry = {
            'event_type': 'evaluation',
            'timestamp': timestamp or datetime.utcnow().isoformat() + 'Z',
            'model_name': model_name,
            'dataset_type': dataset_type,
            'metrics': metrics
        }
        
        self.logger.info('', extra={'json': log_entry})
        return log_entry
    
    def log_error(self, model_name, error_message, error_type=None, 
                 stack_trace=None, timestamp=None):
        """Log an error event"""
        log_entry = {
            'event_type': 'error',
            'timestamp': timestamp or datetime.utcnow().isoformat() + 'Z',
            'model_name': model_name,
            'error_message': str(error_message),
            'error_type': error_type or type(error_message).__name__,
            'stack_trace': str(stack_trace) if stack_trace else None
        }
        
        self.logger.error('', extra={'json': log_entry})
        return log_entry
    
    def log_system_event(self, event_type, message, metadata=None, timestamp=None):
        """Log a system event"""
        log_entry = {
            'event_type': 'system',
            'timestamp': timestamp or datetime.utcnow().isoformat() + 'Z',
            'system_event_type': event_type,
            'message': message,
            'metadata': metadata or {}
        }
        
        self.logger.info('', extra={'json': log_entry})
        return log_entry
    
    def _features_to_dict(self, features):
        """Convert features array to dictionary"""
        if isinstance(features, dict):
            return features
        if isinstance(features, (list, np.ndarray)):
            # Use generic feature names
            return {f'feature_{i}': float(f) for i, f in enumerate(features)}
        return {'value': float(features)}
    
    def _generate_id(self):
        """Generate a unique ID for predictions"""
        import uuid
        return str(uuid.uuid4())


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON logs"""
    
    def format(self, record):
        if hasattr(record, 'json'):
            return json.dumps(record.json, ensure_ascii=False)
        else:
            # Fallback to standard format
            log_obj = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'level': record.levelname,
                'message': record.getMessage()
            }
            return json.dumps(log_obj, ensure_ascii=False)



