"""
Azure Functions App for Trade Promotion ML Predictions
File: function_app.py
"""

import azure.functions as func
import logging
import json
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Initialize the Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Global ML model instance (will be loaded once)
ml_model = None

def initialize_ml_model():
    """Initialize ML model - simplified for demo"""
    global ml_model
    
    if ml_model is None:
        try:
            # For now, create a mock model that simulates predictions
            logging.info("Initializing ML model (demo version)...")
            ml_model = {
                "is_trained": True,
                "model_type": "Demo XGBoost + Random Forest",
                "version": "1.0.0"
            }
            logging.info("✅ Demo ML model initialized")
        except Exception as e:
            logging.error(f"❌ Failed to initialize ML model: {e}")

@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    
    try:
        initialize_ml_model()
        
        status = {
            "status": "healthy",
            "service": "Trade Promotion ML Function",
            "ml_model_loaded": ml_model is not None,
            "function_runtime": "Azure Functions",
            "python_version": sys.version,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(status, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="predict", methods=["POST"])
def predict_units(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function for ML predictions
    Simulates your BaseUnitsPredictor
    """
    
    try:
        initialize_ml_model()
        
        # Validate ML model is available
        if not ml_model or not ml_model.get("is_trained", False):
            return func.HttpResponse(
                json.dumps({
                    "error": "ML model not available or not trained",
                    "status": "service_unavailable"
                }),
                status_code=503,
                mimetype="application/json"
            )
        
        # Parse request body
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON in request body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        if not req_body:
            return func.HttpResponse(
                json.dumps({"error": "Empty request body"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract scenario data
        scenario = req_body.get('scenario', req_body)
        
        # Validate required fields
        required_fields = ['Item', 'Customer', 'Week_Type', 'Base_Price', 'Actual_Price']
        missing_fields = [field for field in required_fields if field not in scenario]
        
        if missing_fields:
            return func.HttpResponse(
                json.dumps({
                    "error": f"Missing required fields: {missing_fields}",
                    "required_fields": required_fields
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        logging.info(f"Processing prediction for: {scenario.get('Item')} at {scenario.get('Customer')}")
        
        # DEMO PREDICTION LOGIC (replace with your actual ML model)
        base_price = float(scenario.get('Base_Price', 3.99))
        actual_price = float(scenario.get('Actual_Price', 2.99))
        week_type = scenario.get('Week_Type', 'Base')
        
        # Calculate discount percentage
        discount_pct = 0
        if base_price > 0:
            discount_pct = ((base_price - actual_price) / base_price) * 100
        
        # Demo prediction logic
        if week_type == 'Promo':
            # Simulate promotion lift based on discount
            base_units = 800 + np.random.normal(0, 50)  # Base demand
            lift_factor = 1 + (discount_pct / 100) * 1.5  # Higher discount = more lift
            units_predicted = base_units * lift_factor
        else:
            # Base week
            base_units = 800 + np.random.normal(0, 50)
            units_predicted = base_units
        
        # Ensure positive values
        units_predicted = max(0, units_predicted)
        base_units = max(0, base_units)
        
        # Calculate lift percentage
        lift_pct = 0
        if base_units > 0 and week_type == 'Promo':
            lift_pct = ((units_predicted - base_units) / base_units) * 100
        
        # Demo business metrics
        business_metrics = {
            "roi": round(np.random.uniform(0.5, 3.0), 2),  # Demo ROI
            "incremental_profit": round((units_predicted - base_units) * 0.50, 2),  # Demo profit
            "trade_rate": round(discount_pct * 0.8, 2)  # Demo trade rate
        }
        
        # Prepare response
        result = {
            "prediction_id": f"pred_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            "units_predicted": round(units_predicted, 0),
            "base_units": round(base_units, 0),
            "lift_pct": round(lift_pct, 2),
            "discount_pct": round(discount_pct, 2),
            "business_metrics": business_metrics,
            "input_scenario": scenario,
            "model_info": {
                "type": "Demo Model",
                "confidence": "high",
                "note": "This is a demo prediction. Replace with your actual ML model."
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        logging.info(f"✅ Demo prediction completed: {units_predicted:.0f} units, {lift_pct:.1f}% lift")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="model-info", methods=["GET"])
def model_info(req: func.HttpRequest) -> func.HttpResponse:
    """Get ML model information"""
    
    try:
        initialize_ml_model()
        
        info = {
            "model_available": ml_model is not None,
            "model_trained": ml_model.get("is_trained", False) if ml_model else False,
            "model_type": ml_model.get("model_type", "Unknown") if ml_model else "Not loaded",
            "version": ml_model.get("version", "Unknown") if ml_model else "Unknown",
            "supported_features": [
                "Units prediction",
                "Base_Units calculation", 
                "Lift percentage calculation",
                "Business metrics calculation"
            ],
            "demo_note": "This is a demonstration version. Replace with your actual BaseUnitsPredictor for production use."
        }
        
        return func.HttpResponse(
            json.dumps(info, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Model info failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )