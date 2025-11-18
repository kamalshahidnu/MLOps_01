"""
Data Pipeline Integration Module
Code for loading data from data pipeline with transformations and versioning
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class DataPipelineLoader:
    """Load data from data pipeline with versioning and transformations"""
    
    def __init__(self, pipeline_path='data/pipeline', data_version=None):
        """
        Initialize data pipeline loader
        Args:
            pipeline_path: Path to data pipeline output
            data_version: Specific data version to load (None = latest)
        """
        self.pipeline_path = Path(pipeline_path)
        self.data_version = data_version
        self.metadata = {}
    
    def load_from_pipeline(self, dataset_name='heart_disease'):
        """
        Load data from data pipeline
        Args:
            dataset_name: Name of the dataset
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        # In production, this would connect to actual data pipeline
        # For now, simulate loading from pipeline structure
        
        if self.data_version is None:
            # Get latest version
            versions = self._get_available_versions(dataset_name)
            if not versions:
                raise FileNotFoundError(f"No data versions found for {dataset_name}")
            self.data_version = max(versions)
        
        data_dir = self.pipeline_path / dataset_name / self.data_version
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data version {self.data_version} not found")
        
        # Load metadata
        metadata_path = data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load transformed data
        train_path = data_dir / 'train.parquet'
        test_path = data_dir / 'test.parquet'
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
        else:
            # Fallback to CSV
            train_path = data_dir / 'train.csv'
            test_path = data_dir / 'test.csv'
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        
        # Split features and target
        target_col = self.metadata.get('target_column', 'target')
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        print(f"Loaded data version {self.data_version}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(X_train.columns)}")
        print(f"  Transformations applied: {self.metadata.get('transformations', [])}")
        
        return X_train, X_test, y_train, y_test, self.metadata
    
    def _get_available_versions(self, dataset_name):
        """Get list of available data versions"""
        dataset_dir = self.pipeline_path / dataset_name
        if not dataset_dir.exists():
            return []
        
        versions = []
        for item in dataset_dir.iterdir():
            if item.is_dir():
                try:
                    # Try to parse as version (timestamp format)
                    datetime.strptime(item.name, "%Y%m%d_%H%M%S")
                    versions.append(item.name)
                except ValueError:
                    continue
        
        return versions
    
    def list_versions(self, dataset_name='heart_disease'):
        """List all available data versions"""
        versions = self._get_available_versions(dataset_name)
        return sorted(versions, reverse=True)
    
    def get_data_info(self, dataset_name='heart_disease', version=None):
        """Get information about a specific data version"""
        if version is None:
            version = self.data_version or max(self._get_available_versions(dataset_name))
        
        data_dir = self.pipeline_path / dataset_name / version
        metadata_path = data_dir / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return {}


# Example: Integration with data pipeline
def load_data_from_pipeline(pipeline_path='data/pipeline', version=None):
    """
    Convenience function to load data from pipeline
    This function should be called from training scripts
    """
    loader = DataPipelineLoader(pipeline_path=pipeline_path, data_version=version)
    return loader.load_from_pipeline()

