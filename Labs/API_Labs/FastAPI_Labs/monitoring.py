"""
Basic monitoring utilities for FastAPI Labs
"""
import time
import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def add_monitoring_middleware(app: FastAPI):
    """Add monitoring middleware to FastAPI app"""
    
    @app.middleware("http")
    async def monitor_requests(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = str(hash(str(request.url)))
        
        return response

def get_system_metrics():
    """Get basic system metrics"""
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "timestamp": time.time()
    }

def add_health_endpoints(app: FastAPI):
    """Add enhanced health check endpoints"""
    
    @app.get("/health/detailed")
    async def detailed_health():
        """Detailed health check with system metrics"""
        metrics = get_system_metrics()
        return {
            "status": "healthy",
            "timestamp": metrics["timestamp"],
            "system": metrics
        }
    
    @app.get("/health/ready")
    async def readiness_check():
        """Kubernetes-style readiness check"""
        try:
            # Check if model exists
            from src.predict import load_model
            load_model()
            return {"status": "ready"}
        except Exception as e:
            return {"status": "not_ready", "error": str(e)}
