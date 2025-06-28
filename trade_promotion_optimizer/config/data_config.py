"""
Data configuration and schema definitions for Trade Promotion Optimization System

This module defines the structure, validation rules, and configuration for all data
used throughout the system. It serves as the single source of truth for data schemas.

Author: Trade Promotion Optimizer
Date: 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeekType(Enum):
    """
    EXPLANATION: Enumeration for week types in promotion data.
    
    Your data shows two types:
    - 'Promo': Weeks when promotions are running
    - 'Base': Regular weeks without promotions
    
    Using Enum ensures type safety and prevents typos like 'promo' vs 'Promo'
    """
    PROMO = "Promo"
    BASE = "Base"


class PromoType(Enum):
    """
    EXPLANATION: Types of promotions available.
    
    From your actual data:
    - 'Single': Single product promotion 
    - 'No_Promo': No promotion running
    - 'Multiple': Multiple product promotion
    """
    SINGLE = "Single"
    NO_PROMO = "No_Promo"
    MULTIPLE = "Multiple"


class MerchType(Enum):
    """
    EXPLANATION: Merchandising support types.
    
    From your actual data in the "Merch" field:
    - 'ISF_&_Flyer': In-store feature and flyer support
    - 'No_Promo': No merchandising support
    - 'ISF_Only': In-store feature only
    - 'ISF_&_Ad': In-store feature and advertising
    - 'ISF_&_Flyer_&_Ad': Full merchandising support
    
    This defines how the promotion is executed in-store.
    """
    ISF_AND_FLYER = "ISF_&_Flyer"
    NO_PROMO = "No_Promo"
    ISF_ONLY = "ISF_Only"
    ISF_AND_AD = "ISF_&_Ad"
    ISF_AND_FLYER_AND_AD = "ISF_&_Flyer_&_Ad"


@dataclass
class DataSchema:
    """
    EXPLANATION: This defines the structure of your trade promotion data.
    
    Based on your specifications:
    - Input columns: What comes from your raw data
    - Calculated columns: Computed using deterministic formulas
    - ML predicted columns: Predicted by machine learning models
    - Business metrics: Calculated from ML predictions
    """
    
    # =============================================================================
    # CORE INPUT COLUMNS (from your actual data)
    # =============================================================================
    input_cols: List[str] = field(default_factory=lambda: [
        "Units",                # Actual units sold in the week
        "Base_Price",           # Regular selling price (non-promo)
        "Actual_Price",         # Actual selling price (promo or base)
        "Customer",             # Retailer/customer name
        "Item",                 # Product identifier  
        "Week_Type",            # Promo vs Base week
        "Promo_Type",           # Type of promotion (Single, No_Promo, Multiple)
        "Merch",                # Merchandising support type
        "List_Price",           # Manufacturer's list price
        "COGS_Unit",            # Cost of goods sold per unit
        "Retailer_Margin_%"     # Retailer margin percentage
    ])
    
    # =============================================================================
    # CALCULATED COLUMNS (deterministic formulas)
    # =============================================================================
    calculated_cols: List[str] = field(default_factory=lambda: [
        "Retailer_Base_$_Margin",   # Base margin in dollars
        "Base_Trade_$_Unit",        # Base trade allowance per unit
        "Retailer_$_Margin",        # Actual margin in dollars  
        "Var_Trade_$_Unit",         # Variable trade allowance per unit
        "$_Trade_Unit",             # Total trade allowance per unit
        "%_Trade_Rate"              # Trade rate as percentage of list price
    ])
    
    # =============================================================================
    # ML PREDICTED COLUMNS
    # =============================================================================
    ml_predicted_cols: List[str] = field(default_factory=lambda: [
        "Base_Units"                # Units that would be sold in base week
    ])
    
    # =============================================================================
    # BUSINESS METRICS (calculated from ML predictions)
    # =============================================================================
    business_metrics_cols: List[str] = field(default_factory=lambda: [
        "Lift_%",                   # (Units - Base_Units) / Base_Units
        "Inc_Profit",               # Incremental profit vs base scenario
        "ROI"                       # Return on investment
    ])
    
    # =============================================================================
    # CATEGORICAL vs NUMERICAL CLASSIFICATION
    # =============================================================================
    categorical_cols: List[str] = field(default_factory=lambda: [
        "Customer", "Item", "Week_Type", "Promo_Type", "Merch"
    ])
    
    numerical_cols: List[str] = field(default_factory=lambda: [
        "Units", "Base_Price", "Actual_Price", "List_Price", "COGS_Unit", "Retailer_Margin_%"
    ])
    
    # =============================================================================
    # TARGET VARIABLES (What we want to predict)
    # =============================================================================
    targets: Dict[str, str] = field(default_factory=lambda: {
        "base_demand": "Base_Units",        # Primary ML target: predict base units
        "actual_demand": "Units",           # Actual units sold
        "lift": "Lift_%",                   # Promotion lift percentage
        "incremental_profit": "Inc_Profit", # Incremental profit from promotion
        "roi": "ROI"                        # Return on investment
    })
    
    # =============================================================================
    # REQUIRED COLUMNS (Must be present in input data)
    # =============================================================================
    required_cols: List[str] = field(default_factory=lambda: [
        "Units",              # Must have actual sales volume
        "Base_Price",         # Must have regular price
        "Actual_Price",       # Must have actual selling price
        "Customer",           # Must know which retailer
        "Item",               # Must know which product
        "Week_Type",          # Must know if promo or base week
        "Promo_Type",         # Must know promotion type
        "Merch",              # Must know merchandising support
        "List_Price",         # Must have manufacturer price
        "COGS_Unit",          # Must have cost information
        "Retailer_Margin_%"   # Must have retailer margin
    ])
    
    # =============================================================================
    # DATA VALIDATION RULES
    # =============================================================================
    validation_rules: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # Volume constraints
        "Units": {
            "min": 0,           # Cannot sell negative units
            "max": 1000000,     # Reasonable upper bound
            "null_allowed": False
        },
        
        # Price constraints
        "Actual_Price": {
            "min": 0.01,        # Prices must be positive
            "max": 1000,        # Reasonable upper bound for most CPG
            "null_allowed": False
        },
        "Base_Price": {
            "min": 0.01,
            "max": 1000,
            "null_allowed": False
        },
        "List_Price": {
            "min": 0.01,
            "max": 1000,
            "null_allowed": False
        },
        
        # Percentage constraints
        "Retailer_Margin_%": {
            "min": -0.5,        # Can have negative margins (loss leaders)
            "max": 0.9,         # 90% margin is very high but possible
            "null_allowed": False
        },
        "%_Trade_Rate": {
            "min": 0,           # Trade rates are non-negative
            "max": 0.8,         # 80% trade rate would be extreme
            "null_allowed": False
        },
        "ROI": {
            "min": -5.0,        # Can have 500% loss
            "max": 10.0,        # 1000% ROI is possible but rare
            "null_allowed": True  # ROI might not always be calculated
        },
        
        # Cost constraints
        "COGS_Unit": {
            "min": 0.01,        # COGS must be positive
            "max": 500,         # Reasonable upper bound
            "null_allowed": False
        }
    })
    
    # =============================================================================
    # BUSINESS LOGIC RULES AND FORMULAS
    # =============================================================================
    business_rules: Dict[str, Any] = field(default_factory=lambda: {
        # Promotion logic validation
        "promo_week_rules": {
            "var_trade_should_be_positive": True,    # Promo weeks should have extra trade investment
            "actual_price_should_be_lower": True,    # Actual_Price should be ≤ Base_Price for promos
            "units_should_be_higher": True           # Promo weeks should drive more volume
        },
        
        # Price relationship rules
        "price_relationships": {
            "actual_price_le_base_price": True,      # Actual_Price ≤ Base_Price (no price increases during promos)
            "list_price_baseline": True,             # List_Price serves as cost baseline
        },
        
        # Margin rules
        "margin_rules": {
            "min_acceptable_margin": 0.10,          # 10% minimum margin for most products
            "margin_calculation_tolerance": 0.01     # 1% tolerance for rounding errors
        }
    })
    
    # =============================================================================
    # BUSINESS FORMULAS (from your formula_functions.txt - EXACT COPY)
    # =============================================================================
    formulas: Dict[str, str] = field(default_factory=lambda: {
        "Retailer_Margin_%": "lambda Base_Price, Actual_Price, List_Price, Base_Trade_$_Unit, Var_Trade_$_Unit, Week_Type: (Base_Price - List_Price + Base_Trade_$_Unit) / Base_Price if Week_Type == 'Base' else (Actual_Price - List_Price + Base_Trade_$_Unit + Var_Trade_$_Unit) / Actual_Price",
        
        "$_Trade_Unit": "lambda Base_Trade_$_Unit, Var_Trade_$_Unit, Week_Type: Base_Trade_$_Unit if Week_Type == 'Base' else Base_Trade_$_Unit + Var_Trade_$_Unit",
        
        "%_Trade_Rate": "lambda $_Trade_Unit, List_Price: $_Trade_Unit / List_Price",
        
        "Profit_Unit": "lambda List_Price, COGS_Unit, Base_Trade_$_Unit, Var_$_Trade_Unit, Week_Type: List_Price - COGS_Unit - Base_Trade_$_Unit if Week_Type == 'Base' else List_Price - COGS_Unit - (Base_Trade_$_Unit + Var_$_Trade_Unit)",
        
        "Profit_Unit_Percentage": "lambda Profit_Unit, List_Price: Profit_Unit / List_Price",
        
        "Lift_%": "lambda Promo_Units, Base_Units, Week_Type: (Promo_Units - Base_Units) / Base_Units if Week_Type == 'Promo' else 0",
        
        "Inc_Profit": "lambda Promo_Units, Base_Units, List_Price, COGS_Unit, Base_Trade_$_Unit, Var_$_Trade_Unit: (Promo_Units * (List_Price - COGS_Unit - Base_Trade_$_Unit - Var_$_Trade_Unit)) - (Base_Units * (List_Price - COGS_Unit - Base_Trade_$_Unit)) if Promo_Units > 0 else 0",
        
        "ROI": "lambda Inc_Profit, Var_$_Trade_Unit, Promo_Units: Inc_Profit / (Promo_Units * Var_$_Trade_Unit) if Var_$_Trade_Unit > 0 else 0",
        
        "Discount": "lambda Base_Price, Actual_Price: (Base_Price - Actual_Price) / Base_Price * 100 if Base_Price > 0 else 0"
    })


@dataclass
class FeatureConfig:
    """
    EXPLANATION: Configuration for feature engineering process.
    
    This controls what additional features we create from your raw data.
    Feature engineering is crucial for ML model performance.
    """
    
    # Time-based features
    create_seasonal_features: bool = True
    create_holiday_features: bool = True  
    create_trend_features: bool = True
    use_cyclical_encoding: bool = True
    
    # Promotion features
    create_discount_features: bool = True
    create_elasticity_features: bool = True
    create_competitive_features: bool = True
    
    # Feature selection
    max_features: Optional[int] = None
    feature_selection_method: str = "mutual_info"
    feature_selection_threshold: float = 0.01


@dataclass
class ModelConfig:
    """
    EXPLANATION: Configuration for machine learning models.
    
    This centralizes all ML model settings and hyperparameters.
    """
    
    # Model selection
    enabled_models: List[str] = field(default_factory=lambda: [
        "xgboost", "random_forest"
    ])
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    validation_method: str = "time_series_split"
    
    # Model hyperparameters
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    })
    
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
    })
    
    # Ensemble configuration
    use_ensemble: bool = True
    ensemble_method: str = "weighted_average"
    ensemble_weights: Optional[Dict[str, float]] = None


# =============================================================================
# GLOBAL CONFIGURATION INSTANCES
# =============================================================================

# Create global configuration instances
DATA_CONFIG = DataSchema()
FEATURE_CONFIG = FeatureConfig()
MODEL_CONFIG = ModelConfig()


# =============================================================================
# CONFIGURATION VALIDATION FUNCTIONS
# =============================================================================

def validate_config() -> bool:
    """
    Validates that all configurations are consistent and valid.
    """
    logger.info("Validating configuration...")
    
    try:
        # Check that required columns are defined
        all_defined_cols = set(DATA_CONFIG.input_cols)
        required_cols_set = set(DATA_CONFIG.required_cols)
        
        missing_required = required_cols_set - all_defined_cols
        if missing_required:
            logger.error(f"Required columns not defined in input_cols: {missing_required}")
            return False
        
        # Validate numerical ranges
        for col, rules in DATA_CONFIG.validation_rules.items():
            if 'min' in rules and 'max' in rules:
                if rules['min'] >= rules['max']:
                    logger.error(f"Invalid range for {col}: min ({rules['min']}) >= max ({rules['max']})")
                    return False
        
        # Validate model configurations
        for model_name in MODEL_CONFIG.enabled_models:
            if model_name not in ['xgboost', 'lightgbm', 'random_forest', 'neural_network']:
                logger.error(f"Unknown model type: {model_name}")
                return False
        
        logger.info("Configuration validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        return False


def get_column_info(column_name: str) -> Dict[str, Any]:
    """Get comprehensive information about a specific column."""
    info = {
        "column_name": column_name,
        "is_input": column_name in DATA_CONFIG.input_cols,
        "is_calculated": column_name in DATA_CONFIG.calculated_cols,
        "is_ml_predicted": column_name in DATA_CONFIG.ml_predicted_cols,
        "is_business_metric": column_name in DATA_CONFIG.business_metrics_cols,
        "is_categorical": column_name in DATA_CONFIG.categorical_cols,
        "is_numerical": column_name in DATA_CONFIG.numerical_cols,
        "is_required": column_name in DATA_CONFIG.required_cols,
        "is_target": column_name in DATA_CONFIG.targets.values(),
        "validation_rules": DATA_CONFIG.validation_rules.get(column_name, {}),
    }
    
    # Add target type if it's a target column
    for target_type, target_col in DATA_CONFIG.targets.items():
        if target_col == column_name:
            info["target_type"] = target_type
            break
    
    return info


def print_schema_summary():
    """Print a human-readable summary of the data schema."""
    print("=" * 80)
    print("TRADE PROMOTION DATA SCHEMA SUMMARY")
    print("=" * 80)
    
    print(f"\n📥 INPUT COLUMNS ({len(DATA_CONFIG.input_cols)}):")
    for col in DATA_CONFIG.input_cols:
        required = "✓" if col in DATA_CONFIG.required_cols else " "
        print(f"  [{required}] {col}")
    
    print(f"\n🔢 CALCULATED COLUMNS ({len(DATA_CONFIG.calculated_cols)}):")
    for col in DATA_CONFIG.calculated_cols:
        print(f"  • {col}")
    
    print(f"\n🤖 ML PREDICTED COLUMNS ({len(DATA_CONFIG.ml_predicted_cols)}):")
    for col in DATA_CONFIG.ml_predicted_cols:
        print(f"  • {col}")
    
    print(f"\n📊 BUSINESS METRICS ({len(DATA_CONFIG.business_metrics_cols)}):")
    for col in DATA_CONFIG.business_metrics_cols:
        print(f"  • {col}")
    
    print(f"\n🎯 TARGET VARIABLES ({len(DATA_CONFIG.targets)}):")
    for target_type, target_col in DATA_CONFIG.targets.items():
        print(f"  • {target_type}: {target_col}")
    
    print(f"\n⚙️ ENABLED MODELS ({len(MODEL_CONFIG.enabled_models)}):")
    for model in MODEL_CONFIG.enabled_models:
        print(f"  • {model}")
    
    print("=" * 80)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # If this file is run directly, validate configuration and print summary
    print("Initializing Trade Promotion Optimizer Configuration...")
    
    if validate_config():
        print("✅ Configuration is valid!")
        print_schema_summary()
    else:
        print("❌ Configuration validation failed!")
        exit(1)