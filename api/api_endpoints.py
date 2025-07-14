"""
FastAPI endpoints for Trade Promotion AI system

File: trade_llm/api/api_endpoints.py
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from data_service import get_data_service, DataService
from trade_promotion_optimizer.models.demand.base_units_predictor import BaseUnitsPredictor

# Setup logging BEFORE imports that use logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the actual query agent using importlib (works for both IDE and runtime)
TradePromotionQueryAgent = None
try:
    import importlib.util
    
    # Direct path to the query agent file
    query_agent_path = Path(__file__).parent.parent / "trade_promotion_ai" / "agents" / "query_agent.py"
    
    if query_agent_path.exists():
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("query_agent", query_agent_path)
        if spec and spec.loader:
            query_agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(query_agent_module)
            TradePromotionQueryAgent = query_agent_module.TradePromotionQueryAgent
            logger.info("Successfully imported TradePromotionQueryAgent using importlib")
        else:
            logger.error("Could not create module spec for query_agent")
    else:
        logger.error(f"Query agent file not found: {query_agent_path}")
        
except Exception as e:
    logger.error(f"Error importing TradePromotionQueryAgent: {e}")
    TradePromotionQueryAgent = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Trade Promotion AI API...")
    
    # Initialize data service
    data_service = get_data_service()
    options = data_service.get_dropdown_options()
    if options['success']:
        logger.info(f"Loaded dropdown data: {options['message']}")
    else:
        logger.warning("Failed to load dropdown data from Excel")
    
    # Initialize ML model
    model = get_ml_model()
    if model and model.is_trained:
        logger.info("ML model ready for predictions")
    else:
        logger.warning("ML model not available - predictions will fail")
    
    # Initialize query agent
    agent = get_query_agent()
    if agent:
        logger.info("Query agent ready for processing")
    else:
        logger.warning("Query agent not available - query processing will fail")
    
    logger.info("API startup complete!")
    
    yield  # Application runs here
    
    # Shutdown (if needed)
    logger.info("Shutting down Trade Promotion AI API...")

# FastAPI app with lifespan
app = FastAPI(
    title="Trade Promotion AI API",
    description="API for trade promotion optimization and prediction",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ML model and query agent instances
ml_model = None
query_agent = None

# Pydantic models for requests/responses
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "anonymous"

class MLPredictionRequest(BaseModel):
    scenario: Dict[str, Any]

class MLPredictionResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    message: str
    timestamp: datetime

class DataOptionsResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    message: str


# Dependency to get data service
def get_data_service_dep() -> DataService:
    return get_data_service()


# Dependency to get query agent
def get_query_agent():  # Removed type hint to avoid IDE issues
    global query_agent
    if query_agent is None and TradePromotionQueryAgent is not None:
        try:
            query_agent = TradePromotionQueryAgent("trade_query_agent")
            logger.info("Created TradePromotionQueryAgent instance")
        except Exception as e:
            logger.error(f"Error creating query agent: {e}")
            query_agent = None
    return query_agent


# Dependency to get ML model
def get_ml_model() -> BaseUnitsPredictor:
    global ml_model
    if ml_model is None:
        try:
            ml_model = BaseUnitsPredictor()
            # Try to load saved model
            model_path = Path(__file__).parent.parent / "models" / "saved"
            if model_path.exists():
                ml_model.load_models(str(model_path))
                logger.info("Loaded saved ML model")
            else:
                logger.warning("No saved model found - will need to train first")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            ml_model = None
    return ml_model





@app.get("/health")
async def health_check():
    """Health check endpoint"""
    data_service = get_data_service()
    ml_model = get_ml_model()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "data_service": "available",
            "ml_model": "available" if ml_model and ml_model.is_trained else "not_trained",
            "query_agent": "available" if get_query_agent() else "not_available",
            "excel_data": "loaded" if data_service.df is not None else "not_loaded"
        }
    }


@app.get("/api/data/options", response_model=DataOptionsResponse)
async def get_dropdown_options(data_service: DataService = Depends(get_data_service_dep)):
    """Get dropdown options from Excel data"""
    try:
        logger.info("Fetching dropdown options from Excel data")
        
        options = data_service.get_dropdown_options()
        
        if options['success']:
            logger.info(f"Successfully returned dropdown options: {len(options['result']['customers'])} customers, {len(options['result']['products'])} products")
        else:
            logger.warning("Using fallback dropdown options")
        
        return DataOptionsResponse(
            success=options['success'],
            result=options['result'],
            message=options['message']
        )
        
    except Exception as e:
        logger.error(f"Error getting dropdown options: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/sample")
async def get_sample_data(
    limit: int = 5,
    data_service: DataService = Depends(get_data_service_dep)
):
    """Get sample data from Excel for testing"""
    try:
        sample = data_service.get_sample_data(limit)
        return sample
        
    except Exception as e:
        logger.error(f"Error getting sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=MLPredictionResponse)
async def predict_ml(
    request: MLPredictionRequest,
    ml_model: BaseUnitsPredictor = Depends(get_ml_model),
    data_service: DataService = Depends(get_data_service_dep)
):
    """Make ML predictions for promotion scenario"""
    try:
        logger.info(f"Processing ML prediction request: {request.scenario}")
        
        # Check if ML model is available
        if not ml_model or not ml_model.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="ML model is not trained. Please train the model first."
            )
        
        # Validate input data against Excel schema
        validation = data_service.validate_input_data(request.scenario)
        if not validation['success']:
            logger.warning(f"Input validation failed: {validation['message']}")
        
        # Prepare data for ML model
        import pandas as pd
        scenario_df = pd.DataFrame([request.scenario])
        
        # Make prediction
        logger.info("Running ML model prediction...")
        predictions = ml_model.predict(scenario_df)
        
        if len(predictions) == 0:
            raise HTTPException(status_code=500, detail="ML model returned no predictions")
        
        # Extract prediction results
        pred_row = predictions.iloc[0]
        
        # Calculate business metrics
        units_predicted = pred_row.get('Units_Predicted', 0)
        base_units = pred_row.get('Base_Units', 0)
        
        # Calculate lift percentage
        lift_pct = 0
        if base_units > 0:
            lift_pct = ((units_predicted - base_units) / base_units) * 100
        
        # Calculate additional metrics
        base_price = request.scenario.get('Base_Price', 0)
        actual_price = request.scenario.get('Actual_Price', 0)
        discount_pct = 0
        if base_price > 0:
            discount_pct = ((base_price - actual_price) / base_price) * 100
        
        result = {
            "units_predicted": float(units_predicted),
            "base_units": float(base_units),
            "lift_pct": float(lift_pct),
            "discount_pct": float(discount_pct),
            "business_metrics": {
                "incremental_units": float(units_predicted - base_units),
                "discount_percentage": float(discount_pct),
                "scenario_type": request.scenario.get('Week_Type', 'Unknown')
            },
            "input_scenario": request.scenario,
            "validation_warnings": validation.get('result', {}).get('warnings', []) if validation['success'] else []
        }
        
        logger.info(f"ML prediction completed: {units_predicted} units, {lift_pct:.1f}% lift")
        
        return MLPredictionResponse(
            success=True,
            result=result,
            message="Prediction completed successfully",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ML prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/query")
async def process_query(
    request: QueryRequest, 
    query_agent = Depends(get_query_agent)  # Removed type hint
):
    """Process natural language queries using TradePromotionQueryAgent"""
    try:
        logger.info(f"Processing query with TradePromotionQueryAgent: {request.query}")
        
        # Check if query agent is available
        if not query_agent:
            raise HTTPException(
                status_code=503, 
                detail="Query agent is not available. Please check system configuration."
            )
        
        # Use the actual query agent we built
        result = await query_agent.process_request({
            "query": request.query,
            "user_id": request.user_id
        })
        
        logger.info(f"Query processed successfully: {result['interpretation']['primary_intent']}")
        
        return {
            "success": True,
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/api/model/status")
async def get_model_status(ml_model: BaseUnitsPredictor = Depends(get_ml_model)):
    """Get ML model status and information"""
    try:
        if not ml_model:
            return {
                "status": "not_loaded",
                "message": "ML model is not available"
            }
        
        status = {
            "status": "trained" if ml_model.is_trained else "not_trained",
            "model_available": ml_model is not None,
            "feature_columns": ml_model.feature_columns if ml_model.is_trained else [],
            "feature_count": len(ml_model.feature_columns) if ml_model.is_trained else 0
        }
        
        if ml_model.is_trained and ml_model.feature_importance:
            # Get top 5 important features
            all_importance = {}
            for model_type, importance_dict in ml_model.feature_importance.items():
                for feature, importance in importance_dict.items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)
            
            avg_importance = {k: sum(v)/len(v) for k, v in all_importance.items()}
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            status["top_features"] = dict(top_features)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/train")
async def train_model():
    """Train the ML model (placeholder - would trigger training pipeline)"""
    return {
        "message": "Model training would be triggered here",
        "status": "not_implemented"
    }


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )