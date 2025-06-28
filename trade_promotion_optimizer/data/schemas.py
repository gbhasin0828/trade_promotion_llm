"""
Pydantic schemas for data validation and serialization in Trade Promotion Optimization System

This module defines type-safe data models using Pydantic for:
1. Input data validation
2. API request/response schemas  
3. Database models
4. Inter-service communication

Author: Trade Promotion Optimizer
Date: 2024
"""

from pydantic import BaseModel, field_validator, Field, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Import our configuration enums
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.data_config import WeekType, PromoType, MerchType


class TradePromotionInputRecord(BaseModel):
    """
    EXPLANATION: Pydantic model for input trade promotion data.
    
    This represents the raw data that comes from your Excel/CSV files.
    It includes automatic validation, type checking, and business rule enforcement.
    
    Benefits:
    - Catches data type errors automatically
    - Validates business rules (e.g., positive prices, valid enums)
    - Converts data types automatically when possible
    - Provides clear error messages for data issues
    - Enables API documentation generation
    """
    
    # =============================================================================
    # CORE INPUT FIELDS (from your data specification)
    # =============================================================================
    
    # Volume and pricing data
    units: float = Field(..., ge=0, description="Units sold during the week")
    base_price: float = Field(..., gt=0, description="Regular selling price (non-promo)")
    actual_price: float = Field(..., gt=0, description="Actual selling price (promo or base)")
    list_price: float = Field(..., gt=0, description="Manufacturer's list price")
    
    # Identification fields
    customer: str = Field(..., description="Retailer/customer name")
    item: str = Field(..., description="Product/item identifier")
    
    # Categorical fields (using our enums for validation)
    week_type: WeekType = Field(..., description="Promotion or base week")
    promo_type: PromoType = Field(..., description="Type of promotion")
    merch: MerchType = Field(..., description="Merchandising support type")
    
    # Cost and margin data
    cogs_unit: float = Field(..., gt=0, description="Cost of goods sold per unit")
    retailer_margin_pct: float = Field(..., ge=-0.5, le=0.9, description="Retailer margin percentage", alias="retailer_margin_%")
    
    # =============================================================================
    # VALIDATORS (Business logic validation)
    # =============================================================================
    
    @field_validator('units')
    @classmethod
    def units_validation(cls, v):
        """Ensure units are non-negative"""
        if v < 0:
            raise ValueError('Units cannot be negative')
        return v
    
    @field_validator('retailer_margin_pct')
    @classmethod
    def margin_validation(cls, v):
        """Validate margin percentage is in reasonable range"""
        if not -0.5 <= v <= 0.9:
            raise ValueError('Retailer margin percentage must be between -50% and 90%')
        return v
    
    @model_validator(mode='after')
    def validate_business_consistency(self):
        """
        EXPLANATION: Validates business rules and consistency across fields.
        
        Business Logic:
        1. Price validation: Promo price ≤ base price, base week consistency
        2. Promo type consistency: Base weeks = No_Promo, Promo weeks ≠ No_Promo  
        3. All prices must be positive
        """
        
        # 1. Price validation
        if self.actual_price <= 0:
            raise ValueError('Actual price must be positive')
        if self.base_price <= 0:
            raise ValueError('Base price must be positive')
        if self.list_price <= 0:
            raise ValueError('List price must be positive')
        
        # 2. Week type and price consistency
        # Convert enum to string value for comparison
        week_type_value = self.week_type.value if hasattr(self.week_type, 'value') else str(self.week_type)
        promo_type_value = self.promo_type.value if hasattr(self.promo_type, 'value') else str(self.promo_type)
        
        if week_type_value == "Promo":
            if self.actual_price > self.base_price:
                raise ValueError(f'Promo actual price ({self.actual_price}) should not exceed base price ({self.base_price})')
        elif week_type_value == "Base":
            tolerance = 0.01  # 1 cent tolerance
            if abs(self.actual_price - self.base_price) > tolerance:
                raise ValueError(f'Base week actual price ({self.actual_price}) should equal base price ({self.base_price})')
        
        # 3. Promo type and week type consistency
        if week_type_value == "Base" and promo_type_value != "No_Promo":
            raise ValueError(f'Base weeks should have promo_type="No_Promo", got "{promo_type_value}"')
        
        if week_type_value == "Promo" and promo_type_value == "No_Promo":
            raise ValueError(f'Promo weeks should not have promo_type="No_Promo"')
        
        return self
    
    class Config:
        """Pydantic configuration"""
        validate_default = True  # Validate default values
        validate_assignment = True  # Validate when fields are assigned
        use_enum_values = True  # Use enum string values in JSON
        arbitrary_types_allowed = False  # Only allow predefined types


class TradePromotionCalculatedRecord(BaseModel):
    """
    EXPLANATION: Model for data after applying business formulas.
    
    This extends the input record with calculated fields from your formulas.
    Represents the complete dataset after formula calculations but before ML predictions.
    """
    
    # Include all input fields
    input_data: TradePromotionInputRecord
    
    # =============================================================================
    # CALCULATED FIELDS (from your formulas)
    # =============================================================================
    
    # Margin calculations
    retailer_base_margin_dollar: float = Field(..., description="Base margin in dollars")
    retailer_margin_dollar: float = Field(..., description="Actual margin in dollars")
    
    # Trade calculations
    base_trade_unit: float = Field(..., ge=0, description="Base trade allowance per unit")
    var_trade_unit: float = Field(..., ge=0, description="Variable trade allowance per unit")
    trade_unit_total: float = Field(..., ge=0, description="Total trade allowance per unit")
    trade_rate_pct: float = Field(..., ge=0, description="Trade rate as percentage of list price")
    
    # Additional calculated metrics
    discount_pct: Optional[float] = Field(None, ge=0, le=1, description="Discount percentage")
    profit_unit: Optional[float] = Field(None, description="Profit per unit")
    profit_unit_pct: Optional[float] = Field(None, description="Profit percentage")
    
    class Config:
        arbitrary_types_allowed = True  # Allow nested models


class TradePromotionMLRecord(BaseModel):
    """
    EXPLANATION: Complete record including ML predictions.
    
    This represents the final dataset with:
    1. Input data
    2. Calculated fields
    3. ML predictions (Base_Units)
    4. Business metrics (Lift_%, Inc_Profit, ROI)
    """
    
    # Include calculated data
    calculated_data: TradePromotionCalculatedRecord
    
    # =============================================================================
    # ML PREDICTIONS
    # =============================================================================
    
    base_units: float = Field(..., ge=0, description="Predicted units if week was base (ML prediction)")
    base_units_confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in base units prediction")
    
    # =============================================================================
    # BUSINESS METRICS (calculated from ML predictions)
    # =============================================================================
    
    lift_pct: float = Field(..., description="Lift percentage: (Units - Base_Units) / Base_Units")
    inc_profit: float = Field(..., description="Incremental profit vs base scenario")
    roi: Optional[float] = Field(None, description="Return on investment")
    
    # Additional business insights
    is_profitable: bool = Field(..., description="Whether the promotion is profitable")
    roi_category: str = Field(..., description="ROI category: Excellent/Good/Poor/Loss")
    
    @field_validator('lift_pct')
    @classmethod
    def lift_validation(cls, v):
        """Validate lift percentage makes business sense"""
        # Lift can be negative (promotion underperformed)
        # But extreme values might indicate data issues
        if v < -0.9:  # More than 90% decline is suspicious
            raise ValueError(f'Lift percentage of {v:.1%} seems unrealistic (>90% decline)')
        if v > 10.0:  # More than 1000% increase is suspicious
            raise ValueError(f'Lift percentage of {v:.1%} seems unrealistic (>1000% increase)')
        return v
    
    @model_validator(mode='after')
    def calculate_derived_fields(self):
        """Calculate ROI category and profitability automatically"""
        # Set ROI category
        if self.roi is not None:
            roi = self.roi
            if roi >= 2.0:  # 200%+ ROI
                self.roi_category = "Excellent"
            elif roi >= 1.0:  # 100-200% ROI
                self.roi_category = "Good"
            elif roi >= 0.0:  # 0-100% ROI
                self.roi_category = "Poor"
            else:  # Negative ROI
                self.roi_category = "Loss"
        else:
            self.roi_category = "Unknown"
        
        # Set profitability
        if self.inc_profit is not None:
            self.is_profitable = self.inc_profit > 0
        else:
            self.is_profitable = False
        
        return self


class DatasetSummary(BaseModel):
    """Summary statistics for a complete dataset"""
    
    total_records: int = Field(..., ge=0, description="Total number of records")
    date_range: Dict[str, str] = Field(..., description="Start and end dates")
    
    # Categorical summaries
    customers: List[str] = Field(..., description="List of unique customers")
    items: List[str] = Field(..., description="List of unique items")
    
    # Volume summaries
    promo_weeks: int = Field(..., ge=0, description="Number of promotion weeks")
    base_weeks: int = Field(..., ge=0, description="Number of base weeks")
    
    # Performance summaries
    avg_units: float = Field(..., ge=0, description="Average units sold")
    avg_lift_pct: Optional[float] = Field(None, description="Average lift percentage")
    avg_roi: Optional[float] = Field(None, description="Average ROI")
    profitable_promotions_pct: Optional[float] = Field(None, ge=0, le=1, description="Percentage of profitable promotions")


class ValidationResult(BaseModel):
    """Results of data validation process"""
    
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    
    # Summary statistics
    records_processed: int = Field(..., ge=0, description="Number of records processed")
    records_valid: int = Field(..., ge=0, description="Number of valid records")
    records_invalid: int = Field(..., ge=0, description="Number of invalid records")
    
    # Dataset summary (if validation passed)
    summary: Optional[DatasetSummary] = Field(None, description="Dataset summary if validation passed")


class OptimizationRequest(BaseModel):
    """
    EXPLANATION: Schema for optimization requests from users.
    
    This represents a user's request for promotion optimization,
    either through API or natural language interface.
    """
    
    # Target specification
    products: List[str] = Field(..., min_items=1, description="List of products to optimize")
    customers: List[str] = Field(..., min_items=1, description="List of customers/retailers")
    
    # Objective
    objective: str = Field(..., description="Optimization objective: profit, roi, volume, revenue")
    
    # Constraints
    budget_limit: Optional[float] = Field(None, gt=0, description="Total budget limit")
    max_discount_pct: Optional[float] = Field(None, ge=0, le=0.5, description="Maximum discount percentage")
    min_margin_pct: Optional[float] = Field(None, ge=0, le=0.9, description="Minimum margin requirement")
    max_weeks_promoted: Optional[int] = Field(None, ge=1, le=52, description="Maximum weeks to promote")
    
    # Time horizon
    planning_weeks: int = Field(4, ge=1, le=52, description="Planning horizon in weeks")
    
    @field_validator('objective')
    @classmethod
    def validate_objective(cls, v):
        """Ensure objective is valid"""
        valid_objectives = ['profit', 'roi', 'volume', 'revenue', 'margin']
        if v.lower() not in valid_objectives:
            raise ValueError(f'Objective must be one of: {valid_objectives}')
        return v.lower()


class OptimizationResult(BaseModel):
    """Results of promotion optimization"""
    
    # Request context
    request: OptimizationRequest
    
    # Optimal solution
    optimal_promotions: List[Dict[str, Any]] = Field(..., description="List of optimal promotion recommendations")
    
    # Performance metrics
    expected_total_profit: float = Field(..., description="Expected total incremental profit")
    expected_total_investment: float = Field(..., ge=0, description="Required total investment")
    expected_roi: float = Field(..., description="Expected overall ROI")
    
    # Solution quality
    optimization_status: str = Field(..., description="Optimization status: optimal, feasible, infeasible")
    convergence_time: float = Field(..., ge=0, description="Time to solve in seconds")
    
    # Business insights
    recommendations: List[str] = Field(..., description="Business recommendations and insights")
    risk_factors: List[str] = Field(default_factory=list, description="Potential risks to consider")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_trade_promotion_record(data: Dict[str, Any]) -> TradePromotionInputRecord:
    """
    EXPLANATION: Validates a single trade promotion record.
    
    Args:
        data: Dictionary with trade promotion data
        
    Returns:
        Validated TradePromotionInputRecord
        
    Raises:
        ValidationError: If data doesn't meet business rules
    """
    return TradePromotionInputRecord(**data)


def validate_dataset(data_list: List[Dict[str, Any]]) -> ValidationResult:
    """
    EXPLANATION: Validates an entire dataset of trade promotion records.
    
    Args:
        data_list: List of dictionaries with trade promotion data
        
    Returns:
        ValidationResult with summary of validation process
    """
    errors = []
    warnings = []
    valid_records = []
    
    for i, record_data in enumerate(data_list):
        try:
            validated_record = validate_trade_promotion_record(record_data)
            valid_records.append(validated_record)
        except Exception as e:
            errors.append(f"Record {i+1}: {str(e)}")
    
    # Create summary if we have valid records
    summary = None
    if valid_records:
        summary = create_dataset_summary(valid_records)
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        records_processed=len(data_list),
        records_valid=len(valid_records),
        records_invalid=len(errors),
        summary=summary
    )


def create_dataset_summary(records: List[TradePromotionInputRecord]) -> DatasetSummary:
    """Create summary statistics for a list of validated records"""
    
    if not records:
        raise ValueError("Cannot create summary for empty dataset")
    
    # Extract unique values
    customers = list(set(record.customer for record in records))
    items = list(set(record.item for record in records))
    
    # Count week types
    promo_weeks = sum(1 for record in records if record.week_type == WeekType.PROMO)
    base_weeks = sum(1 for record in records if record.week_type == WeekType.BASE)
    
    # Calculate averages
    avg_units = sum(record.units for record in records) / len(records)
    
    return DatasetSummary(
        total_records=len(records),
        date_range={"start": "Unknown", "end": "Unknown"},  # Would need time field to calculate
        customers=customers,
        items=items,
        promo_weeks=promo_weeks,
        base_weeks=base_weeks,
        avg_units=avg_units
    )


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

if __name__ == "__main__":
    # Test the schemas with sample data
    print("Testing Trade Promotion Schemas...")
    
    # Sample valid record
    sample_data = {
        "units": 1000,
        "base_price": 3.99,
        "actual_price": 2.99,  # 25% discount
        "list_price": 3.49,
        "customer": "Walmart",
        "item": "Product_A",
        "week_type": "Promo",
        "promo_type": "Single",
        "merch": "ISF_&_Flyer",
        "cogs_unit": 1.75,
        "retailer_margin_%": 0.25
    }
    
    try:
        record = validate_trade_promotion_record(sample_data)
        print("✅ Sample record validation passed!")
        print(f"   Customer: {record.customer}")
        print(f"   Units: {record.units}")
        print(f"   Week Type: {record.week_type}")
        print(f"   Discount: {((record.base_price - record.actual_price) / record.base_price * 100):.1f}%")
        
    except Exception as e:
        print(f"❌ Sample record validation failed: {e}")
    
    # Test invalid record
    invalid_data = sample_data.copy()
    invalid_data["actual_price"] = 5.00  # Higher than base price for promo week
    
    print(f"\nTesting invalid record with actual_price={invalid_data['actual_price']} > base_price={invalid_data['base_price']}...")
    
    try:
        invalid_record = validate_trade_promotion_record(invalid_data)
        print("❌ Invalid record should have failed validation!")
        print(f"   Created record: {invalid_record}")
    except Exception as e:
        print(f"✅ Invalid record correctly rejected: {e}")
    
    print("\nSchema testing completed!")