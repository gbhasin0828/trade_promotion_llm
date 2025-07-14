"""
Main FastAPI server for Trade Promotion AI System
Connects frontend HTML to API endpoints and Excel data

File: trade_llm/main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from pathlib import Path
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import API endpoints
try:
    from api.api_endpoints import app as api_app
    logger.info("Successfully imported API endpoints")
except ImportError as e:
    logger.error(f"Failed to import API endpoints: {e}")
    # Create minimal API app as fallback
    api_app = FastAPI()
    
    @api_app.get("/health")
    async def health():
        return {"status": "api_fallback", "message": "Main API not available"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("=" * 60)
    logger.info("STARTING TRADE PROMOTION AI SYSTEM")
    logger.info("=" * 60)
    
    # Check critical files
    excel_path = Path(__file__).parent / "Raw_Input_Data.xlsx"
    if excel_path.exists():
        logger.info(f"✅ Excel file found: {excel_path}")
        logger.info(f"   File size: {excel_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        logger.error(f"❌ Excel file missing: {excel_path}")
    
    # Check ML models
    models_path = Path(__file__).parent / "models" / "saved"
    if models_path.exists():
        model_files = list(models_path.glob("*.joblib"))
        logger.info(f"✅ Models directory found: {len(model_files)} model files")
        for model_file in model_files:
            logger.info(f"   - {model_file.name}")
    else:
        logger.warning(f"⚠️  Models directory missing: {models_path}")
    
    # Check frontend
    frontend_path = Path(__file__).parent / "frontend"
    if frontend_path.exists():
        html_files = list(frontend_path.glob("*.html"))
        logger.info(f"✅ Frontend found: {len(html_files)} HTML files")
        for html_file in html_files:
            logger.info(f"   - {html_file.name}")
    else:
        logger.error(f"❌ Frontend missing: {frontend_path}")
    
    logger.info("=" * 60)
    logger.info("🚀 SYSTEM READY!")
    logger.info("   Frontend: http://localhost:8000/")
    logger.info("   API: http://localhost:8000/api/")
    logger.info("   Admin: http://localhost:8000/admin")
    logger.info("   Health: http://localhost:8000/health")
    logger.info("=" * 60)
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Trade Promotion AI System...")


# Create main FastAPI application
app = FastAPI(
    title="Trade Promotion AI System",
    description="Complete system with frontend and API for trade promotion optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API endpoints under /api path
app.mount("/api", api_app)
logger.info("Mounted API endpoints at /api")

# Check if frontend directory exists and mount it
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    # Mount static files (HTML, CSS, JS)
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    logger.info(f"Mounted frontend at / from {frontend_path}")
else:
    logger.error(f"Frontend directory not found: {frontend_path}")
    
    # Create a fallback route
    @app.get("/")
    async def root():
        return {
            "message": "Frontend not found", 
            "expected_path": str(frontend_path),
            "available_endpoints": ["/api/health", "/admin", "/health"]
        }


# Health check for the main application
@app.get("/health")
async def main_health():
    """Main application health check"""
    try:
        # Check if Excel file exists
        excel_path = Path(__file__).parent / "Raw_Input_Data.xlsx"
        excel_exists = excel_path.exists()
        
        # Check if ML models exist
        models_path = Path(__file__).parent / "models" / "saved"
        models_exist = models_path.exists()
        
        # Check frontend
        frontend_exists = frontend_path.exists()
        
        return {
            "status": "healthy",
            "components": {
                "frontend": "available" if frontend_exists else "missing",
                "api": "mounted",
                "excel_data": "available" if excel_exists else "missing",
                "ml_models": "available" if models_exist else "missing"
            },
            "paths": {
                "frontend_url": "http://localhost:8000/",
                "api_base": "http://localhost:8000/api/",
                "excel_file": str(excel_path),
                "models_dir": str(models_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}


# Admin status page
@app.get("/admin")
async def admin_status():
    """Admin status page with detailed system information"""
    try:
        # Get system status
        health_info = await main_health()
        
        # Get file sizes and details
        excel_path = Path(__file__).parent / "Raw_Input_Data.xlsx"
        excel_size = excel_path.stat().st_size if excel_path.exists() else 0
        
        models_path = Path(__file__).parent / "models" / "saved"
        model_files = list(models_path.glob("*.joblib")) if models_path.exists() else []
        
        frontend_files = list(frontend_path.glob("*.*")) if frontend_path.exists() else []
        
        return {
            "system_status": health_info,
            "file_info": {
                "excel_file_size_mb": round(excel_size / (1024*1024), 2),
                "model_files": [f.name for f in model_files],
                "total_model_files": len(model_files),
                "frontend_files": [f.name for f in frontend_files],
                "total_frontend_files": len(frontend_files)
            },
            "endpoints": {
                "frontend": "http://localhost:8000/",
                "api_health": "http://localhost:8000/api/health",
                "dropdown_data": "http://localhost:8000/api/data/options",
                "ml_predict": "http://localhost:8000/api/predict",
                "sample_data": "http://localhost:8000/api/data/sample",
                "query_processing": "http://localhost:8000/api/query"
            },
            "system_info": {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "total_endpoints": len(app.routes)
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


# Error handler for 404s
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler with helpful information"""
    return {
        "error": "Not Found",
        "message": f"Path '{request.url.path}' not found",
        "available_paths": [
            "/",
            "/health",
            "/admin",
            "/api/health",
            "/api/data/options", 
            "/api/predict",
            "/api/query"
        ],
        "suggestion": "Try visiting the frontend at http://localhost:8000/ or check /admin for system status"
    }


# Utility functions
def check_file_structure():
    """Check if all required files are in place"""
    base_path = Path(__file__).parent
    
    required_files = {
        "excel_data": base_path / "Raw_Input_Data.xlsx",
        "frontend_html": base_path / "frontend" / "index.html", 
        "api_endpoints": base_path / "api" / "api_endpoints.py",
        "data_service": base_path / "api" / "data_service.py"
    }
    
    status = {}
    for name, path in required_files.items():
        status[name] = {
            "exists": path.exists(),
            "path": str(path),
            "size": path.stat().st_size if path.exists() else 0
        }
    
    return status


def get_system_info():
    """Get comprehensive system information"""
    return {
        "file_structure": check_file_structure(),
        "python_path": sys.path,
        "working_directory": os.getcwd(),
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform
        }
    }


# Run the application
if __name__ == "__main__":
    try:
        logger.info("Starting Trade Promotion AI System...")
        
        # Quick system check before starting
        logger.info("Performing system check...")
        info = get_system_info()
        
        logger.info(f"Excel file exists: {info['file_structure']['excel_data']['exists']}")
        logger.info(f"Frontend exists: {info['file_structure']['frontend_html']['exists']}")
        logger.info(f"API exists: {info['file_structure']['api_endpoints']['exists']}")
        
        # Run with uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise