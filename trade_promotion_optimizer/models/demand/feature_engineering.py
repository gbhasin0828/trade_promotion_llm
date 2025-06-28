"""
Universal Demand Prediction Engine for Trade Promotion Optimization System

This module provides ML models that can predict demand (units sold) for ANY promotion scenario,
including scenarios that have never been tested before. It's designed to work with LLM-parsed
queries and dynamic scenario generation.

Key Features:
1. Predicts demand for any promotion type (2for$5, BOGO, 20% off, etc.)
2. Handles both historical data training and new scenario prediction
3. Feature engineering that generalizes across promotion types
4. Confidence intervals and uncertainty quantification
5. Integration-ready for LLM semantic parsing

Architecture:
- Scenario-based feature engineering
- Ensemble models (XGBoost + Random Forest)
- Price elasticity modeling
- Promotion type encoding
- Customer/product embeddings

Author: Trade Promotion Optimizer
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import joblib
import warnings
from datetime import datetime, timedelta
import re

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Import our configuration
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.data_config import DATA_CONFIG, WeekType, PromoType, MerchType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class ScenarioFeatureEngineer:
    """
    EXPLANATION: Feature engineering for any promotion scenario.
    
    This class converts promotion scenarios (from LLM or historical data) into
    ML-ready features that capture the essential drivers of demand:
    
    1. Price elasticity (discount depth effect)
    2. Promotion type mechanics (BOGO vs % off vs 2for$X)
    3. Customer characteristics (retailer behavior patterns)
    4. Product characteristics (category, historical performance)
    5. Temporal factors (seasonality, trends)
    
    Key Innovation: Works with both historical data AND hypothetical scenarios.
    """
    
    def __init__(self):
        """Initialize the feature engineer with encoders and scalers."""
        self.customer_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder() 
        self.week_type_encoder = LabelEncoder() 
        self.promo_type_encoder = LabelEncoder()
        self.merch_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()


    
    def fit(self, df: pd.DataFrame):
        """
        EXPLANATION: Fit encoders on historical data.
        
        This learns the mapping between categorical values and encoded numbers.
        Must be called once on historical data before making predictions.
        """
        logger.info(f"Fitting feature engineer on {len(df)} historical records")
        
        features = []
        
        # Fit scaler on numerical features
        numerical_features = self._extract_numerical_features(df)



        features.append(self.scaler.fit_transform(numerical_features))

        return features


    def _extract_numerical_features(self, df : pd.DataFrame ) -> np.ndarray:
        """
        EXPLANATION: Extract numerical features for scaler fitting.
        
        This function extracts the exact same numerical features that will be used
        during prediction, so the StandardScaler can learn proper scaling parameters.
        
        Business Model Context:
        - List_Price: Manufacturer wholesale price to retailer
        - Base_Price: Retailer's regular consumer price  
        - Actual_Price: Retailer's promotional consumer price
        """
        numerical_features = []
        
        for _, row in df.iterrows():
            
            # 1. PRICE FEATURES (extract same as _extract_price_features)
            actual_price = float(row.get('Actual_Price', row.get('actual_price', 0)))
            base_price = float(row.get('Base_Price', row.get('base_price', actual_price)))
            list_price = float(row.get('List_Price', row.get('list_price', base_price)))
            
            # Calculate price-related features with correct business logic
            if actual_price < base_price*.95 :
                consumer_discount_pct = (base_price - actual_price) / base_price
                promotion_intensity = actual_price / base_price
            else:
                consumer_discount_pct = 0
                promotion_intensity = 0
            

            price_features = [
                actual_price,                    # Consumer price level
                base_price,                      # Regular consumer price level  
                list_price,                      # Wholesale price level
                consumer_discount_pct,           # Consumer discount (Base - Actual) / Base
                promotion_intensity,             # Promotion intensity: Actual / Base (discount depth)
            ]
            
            numerical_features.append(price_features)

        return np.array(numerical_features) 



if __name__ == "__main__":

    sample_data = pd.DataFrame({
        'Units': [1000, 800, 1200, 600, 1500],
        'Base_Price': [3.99, 3.99, 4.29, 3.99, 4.29], 
        'Actual_Price': [2.99, 3.99, 4.29, 2.49, 3.49],
        'List_Price': [3.49, 3.49, 3.49, 3.49, 3.49],
        'Week_Type': ['Promo', 'Base', 'Base', 'Promo', 'Promo'],
        'Promo_Type': ['Single', 'No_Promo', 'No_Promo', 'Single', 'Multiple'],
        'Merch': ['ISF_&_Flyer', 'No_Promo', 'No_Promo', 'ISF_&_Flyer_&_Ad', 'ISF_Only'],
        'Customer': ['Walmart', 'Walmart', 'Target', 'Walmart', 'Target'],
        'Item': ['Product_A', 'Product_A', 'Product_B', 'Product_A', 'Product_B']
    })


    print(sample_data.head(2))

    output = ScenarioFeatureEngineer()
    result = output.fit(sample_data)
    print(result)





