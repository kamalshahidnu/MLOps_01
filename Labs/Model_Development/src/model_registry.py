"""
Model Registry Module
Integration with GCP Artifact Registry and local model registry
"""
import os
import json
import joblib
from datetime import datetime
from pathlib import Path
import hashlib


class ModelRegistry:
    """Model registry for versioning and tracking models"""
    
    def __init__(self, registry_path='models/registry', use_gcp=False, gcp_project=None, gcp_location=None):
        """
        Initialize model registry
        Args:
            registry_path: Local path for model registry
            use_gcp: Whether to use GCP Artifact Registry
            gcp_project: GCP project ID
            gcp_location: GCP location (e.g., 'us-central1')
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.use_gcp = use_gcp
        self.gcp_project = gcp_project
        self.gcp_location = gcp_location
        
        if use_gcp:
            try:
                from google.cloud import artifactregistry
                self.gcp_client = artifactregistry.ArtifactRegistryClient()
            except ImportError:
                raise ImportError("Google Cloud Artifact Registry not installed. Install with: pip install google-cloud-artifact-registry")
    
    def register_model(self, model, model_name, version=None, metadata=None):
        """
        Register a model in the registry
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Version string (auto-generated if None)
            metadata: Additional metadata dictionary
        Returns:
            Model version and path
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create version directory
        version_dir = self.registry_path / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / f"{model_name}_{version}.pkl"
        joblib.dump(model, model_path)
        
        # Calculate model hash
        model_hash = self._calculate_hash(model_path)
        
        # Create metadata
        model_metadata = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'model_hash': model_hash,
            'metadata': metadata or {}
        }
        
        # Save metadata
        metadata_path = version_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Update registry index
        self._update_registry_index(model_name, version, model_metadata)
        
        # Push to GCP if enabled
        if self.use_gcp:
            self._push_to_gcp(model_path, model_name, version)
        
        print(f"Model registered: {model_name} v{version}")
        return version, model_path
    
    def _calculate_hash(self, file_path):
        """Calculate SHA256 hash of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _update_registry_index(self, model_name, version, metadata):
        """Update registry index file"""
        index_path = self.registry_path / 'index.json'
        
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {}
        
        if model_name not in index:
            index[model_name] = {}
        
        index[model_name][version] = metadata
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _push_to_gcp(self, model_path, model_name, version):
        """Push model to GCP Artifact Registry"""
        # This is a simplified version - in production, use proper GCP SDK
        print(f"Pushing {model_name} v{version} to GCP Artifact Registry...")
        print("Note: Full GCP integration requires proper authentication and configuration")
        # TODO: Implement full GCP Artifact Registry push
    
    def get_model(self, model_name, version='latest'):
        """Get a model from registry"""
        if version == 'latest':
            version = self.get_latest_version(model_name)
        
        model_path = self.registry_path / model_name / version / f"{model_name}_{version}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} v{version} not found")
        
        model = joblib.load(model_path)
        metadata_path = model_path.parent / 'metadata.json'
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def get_latest_version(self, model_name):
        """Get latest version of a model"""
        index_path = self.registry_path / 'index.json'
        
        if not index_path.exists():
            raise ValueError(f"Model {model_name} not found in registry")
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        if model_name not in index:
            raise ValueError(f"Model {model_name} not found in registry")
        
        versions = list(index[model_name].keys())
        return max(versions)  # Assuming timestamp-based versioning
    
    def list_models(self):
        """List all models in registry"""
        index_path = self.registry_path / 'index.json'
        
        if not index_path.exists():
            return {}
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        return index
    
    def compare_models(self, model_name, version1, version2):
        """Compare two model versions"""
        _, metadata1 = self.get_model(model_name, version1)
        _, metadata2 = self.get_model(model_name, version2)
        
        comparison = {
            'model_name': model_name,
            'version1': {
                'version': version1,
                'timestamp': metadata1.get('timestamp'),
                'metadata': metadata1.get('metadata', {})
            },
            'version2': {
                'version': version2,
                'timestamp': metadata2.get('timestamp'),
                'metadata': metadata2.get('metadata', {})
            }
        }
        
        return comparison

