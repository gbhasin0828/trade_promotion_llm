"""
Fixed Pydantic schema to match actual DataFrame column names
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
    EXPLANATION: Fixed Pydantic model that matches your actual DataFrame column names.
    
    The key change: Using the exact column names from your DataFrame instead of snake_case.
    """
    
    # =============================================================================
    # CORE INPUT FIELDS (matching your DataFrame column names exactly)
    # =============================================================================
    
    # Volume and pricing data
    Units: float = Field(..., ge=0, description="Units sold during the week")
    Base_Price: float = Field(..., gt=0, description="Regular selling price (non-promo)")
    Actual_Price: float = Field(..., gt=0, description="Actual selling price (promo or base)")
    List_Price: float = Field(..., gt=0, description="Manufacturer's list price")
    
    # Identification fields
    Customer: str = Field(..., description="Retailer/customer name")
    Item: str = Field(..., description="Product/item identifier")
    
    # Categorical fields (using our enums for validation)
    Week_Type: WeekType = Field(..., description="Promotion or base week")
    Promo_Type: PromoType = Field(..., description="Type of promotion")
    Merch: MerchType = Field(..., description="Merchandising support type")
    
    # Cost and margin data
    COGS_Unit: float = Field(..., gt=0, description="Cost of goods sold per unit")
    Retailer_Margin_Percent: float = Field(..., ge=-0.5, le=0.9, description="Retailer margin percentage", alias="Retailer_Margin_%")
    
    # =============================================================================
    # VALIDATORS (Business logic validation)
    # =============================================================================
    
    @field_validator('Units')
    @classmethod
    def units_validation(cls, v):
        """Ensure units are non-negative"""
        if v < 0:
            raise ValueError('Units cannot be negative')
        return v
    
    @field_validator('Retailer_Margin_Percent')
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
        if self.Actual_Price <= 0:
            raise ValueError('Actual price must be positive')
        if self.Base_Price <= 0:
            raise ValueError('Base price must be positive')
        if self.List_Price <= 0:
            raise ValueError('List price must be positive')
        
        # 2. Week type and price consistency
        # Convert enum to string value for comparison
        week_type_value = self.Week_Type.value if hasattr(self.Week_Type, 'value') else str(self.Week_Type)
        promo_type_value = self.Promo_Type.value if hasattr(self.Promo_Type, 'value') else str(self.Promo_Type)
        
        if week_type_value == "Promo":
            if self.Actual_Price > self.Base_Price:
                raise ValueError(f'Promo actual price ({self.Actual_Price}) should not exceed base price ({self.Base_Price})')
        elif week_type_value == "Base":
            tolerance = 0.01  # 1 cent tolerance
            if abs(self.Actual_Price - self.Base_Price) > tolerance:
                raise ValueError(f'Base week actual price ({self.Actual_Price}) should equal base price ({self.Base_Price})')
        
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
        # Allow field aliases
        populate_by_name = True


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


class DatasetSummary(BaseModel):
    """Summary statistics for a complete dataset"""
    
    total_records: int = Field(..., ge=0, description="Total number of records")
    date_range: Dict[str, str] = Field(default_factory=lambda: {"start": "Unknown", "end": "Unknown"}, description="Start and end dates")
    
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
        customers = list(set(record.Customer for record in valid_records))
        items = list(set(record.Item for record in valid_records))
        
        promo_weeks = sum(1 for record in valid_records if record.Week_Type == WeekType.PROMO)
        base_weeks = sum(1 for record in valid_records if record.Week_Type == WeekType.BASE)
        
        avg_units = sum(record.Units for record in valid_records) / len(valid_records)
        
        summary = DatasetSummary(
            total_records=len(valid_records),
            customers=customers,
            items=items,
            promo_weeks=promo_weeks,
            base_weeks=base_weeks,
            avg_units=avg_units
        )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        records_processed=len(data_list),
        records_valid=len(valid_records),
        records_invalid=len(errors),
        summary=summary
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the fixed schema with sample data
    print("Testing Fixed Trade Promotion Schema...")
    
    # Sample valid record matching your DataFrame column names
    sample_data = {
        "Units": 1000,
        "Base_Price": 3.99,
        "Actual_Price": 2.99,  # 25% discount
        "List_Price": 3.49,
        "Customer": "Walmart",
        "Item": "Product_A",
        "Week_Type": "Promo",
        "Promo_Type": "Single", 
        "Merch": "ISF_&_Flyer",
        "COGS_Unit": 1.75,
        "Retailer_Margin_%": 0.25
    }
    
    try:
        record = validate_trade_promotion_record(sample_data)
        print("✅ Sample record validation passed!")
        print(f"   Customer: {record.Customer}")
        print(f"   Units: {record.Units}")
        print(f"   Week Type: {record.Week_Type}")
        print(f"   Margin: {record.Retailer_Margin_Percent:.1%}")
        
    except Exception as e:
        print(f"❌ Sample record validation failed: {e}")
    
    print("\nFixed schema testing completed!")