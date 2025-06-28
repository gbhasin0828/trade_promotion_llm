"""
Excel data loader with comprehensive validation for Trade Promotion Optimization System

This module handles loading Excel files containing trade promotion data and converts
them to validated, type-safe format using our Pydantic schemas.

Key Features:
1. Loads Excel files in your specific format
2. Validates data against business rules  
3. Converts to standardized internal format
4. Provides detailed error reporting and data quality metrics
5. Handles common data issues gracefully

Author: Trade Promotion Optimizer
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import warnings
from datetime import datetime

# Import our configuration and schemas
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.data_config import DATA_CONFIG, WeekType, PromoType, MerchType
from data.schemas import (
    TradePromotionInputRecord, 
    ValidationResult, 
    DatasetSummary,
    validate_dataset
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


class ExcelDataLoader:
    """
    EXPLANATION: Professional Excel data loader for trade promotion data.
    
    This class handles the complete pipeline from raw Excel files to validated,
    type-safe data ready for analysis and modeling.
    
    Key Responsibilities:
    1. Read Excel files with error handling
    2. Clean and standardize column names
    3. Convert data types appropriately
    4. Validate against business rules
    5. Provide comprehensive data quality reporting
    6. Handle missing values and data inconsistencies
    
    Design Philosophy:
    - Fail gracefully with detailed error messages
    - Provide data quality insights, not just pass/fail
    - Make it easy to diagnose and fix data issues
    - Support iterative data cleaning workflow
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize the Excel data loader.
        
        Args:
            strict_validation: If True, fails on any validation error.
                             If False, warns but continues with valid records.
        """
        self.strict_validation = strict_validation
        self.validation_errors = []
        self.validation_warnings = []
        self.data_quality_issues = []
        
        # Column mapping for common variations in Excel files
        self.column_mapping = self._create_column_mapping()
        
        logger.info(f"ExcelDataLoader initialized (strict_validation={strict_validation})")
    
    def _create_column_mapping(self) -> Dict[str, str]:
        """
        EXPLANATION: Create mapping for common column name variations.
        
        Real-world Excel files often have slight variations in column names:
        - Different capitalization
        - Spaces vs underscores
        - Abbreviations vs full names
        - Extra characters or typos
        
        This mapping helps handle these variations automatically.
        """
        mapping = {}
        
        # Standard mappings for each expected column
        expected_columns = DATA_CONFIG.input_cols
        
        for col in expected_columns:
            # Add the exact column name
            mapping[col] = col
            mapping[col.lower()] = col
            mapping[col.upper()] = col
            
            # Add common variations
            col_variations = [
                col.replace('_', ' '),  # Underscore to space
                col.replace(' ', '_'),  # Space to underscore
                col.replace('%', 'Pct'),  # % to Pct
                col.replace('%', 'Percent'),  # % to Percent
                col.replace('$', 'Dollar'),  # $ to Dollar
                col.replace('$', ''),  # Remove $
            ]
            
            for variation in col_variations:
                mapping[variation] = col
                mapping[variation.lower()] = col
                mapping[variation.upper()] = col
        
        # Specific mappings for your data format
        specific_mappings = {
            # Price variations
            'Price': 'Actual_Price',
            'Promo_Price': 'Actual_Price',
            'Selling_Price': 'Actual_Price',
            
            # Margin variations
            'Margin': 'Retailer_Margin_%',
            'Retailer_Margin': 'Retailer_Margin_%',
            'Margin_Percent': 'Retailer_Margin_%',
            'Margin_Pct': 'Retailer_Margin_%',
            
            # Customer variations
            'Retailer': 'Customer',
            'Account': 'Customer',
            'Banner': 'Customer',
            
            # Product variations
            'Product': 'Item',
            'SKU': 'Item',
            'Product_Name': 'Item',
            
            # COGS variations
            'COGS': 'COGS_Unit',
            'Cost': 'COGS_Unit',
            'Unit_Cost': 'COGS_Unit',
        }
        
        mapping.update(specific_mappings)
        
        return mapping
    
    def load_file(self, 
                  file_path: str, 
                  sheet_name: Optional[str] = None,
                  header_row: int = 0) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        EXPLANATION: Main method to load and validate Excel file.
        
        This is the primary interface for loading trade promotion data.
        It handles the complete pipeline from file reading to validated output.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (if None, uses first sheet)
            header_row: Row number containing headers (0-indexed)
            
        Returns:
            Tuple of (cleaned_dataframe, validation_result)
            
        Process:
        1. Read Excel file with error handling
        2. Clean and standardize column names
        3. Convert data types
        4. Validate business rules
        5. Generate quality report
        """
        logger.info(f"Loading Excel file: {file_path}")
        
        try:
            # Step 1: Read Excel file
            df_raw = self._read_excel_file(file_path, sheet_name, header_row)
            logger.info(f"Read {len(df_raw)} rows and {len(df_raw.columns)} columns")
            
            # Step 2: Clean column names and map to standard format
            df_cleaned = self._clean_and_map_columns(df_raw)
            
            # Step 3: Validate required columns are present
            missing_cols = self._check_required_columns(df_cleaned)
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                logger.error(error_msg)
                return pd.DataFrame(), ValidationResult(
                    is_valid=False,
                    errors=[error_msg],
                    records_processed=len(df_raw),
                    records_valid=0,
                    records_invalid=len(df_raw)
                )
            
            # Step 4: Convert data types and handle missing values
            df_typed = self._convert_data_types(df_cleaned)
            
            # Step 5: Apply business rule validation
            df_validated = self._apply_business_validation(df_typed)
            
            # Step 6: Convert to Pydantic models for final validation
            records_list = df_validated.to_dict('records')
            validation_result = validate_dataset(records_list)
            
            # Step 7: Generate data quality report
            self._generate_quality_report(df_raw, df_validated, validation_result)
            
            logger.info(f"Validation completed: {validation_result.records_valid}/{validation_result.records_processed} records valid")
            
            return df_validated, validation_result
            
        except Exception as e:
            error_msg = f"Failed to load Excel file: {str(e)}"
            logger.error(error_msg)
            return pd.DataFrame(), ValidationResult(
                is_valid=False,
                errors=[error_msg],
                records_processed=0,
                records_valid=0,
                records_invalid=0
            )
    
    def _read_excel_file(self, 
                        file_path: str, 
                        sheet_name: Optional[str] = None,
                        header_row: int = 0) -> pd.DataFrame:
        """
        EXPLANATION: Read Excel file with comprehensive error handling.
        
        This method handles common Excel file issues:
        - File not found
        - Corrupted files
        - Multiple sheets
        - Different header row positions
        - Various Excel formats (.xlsx, .xls)
        """
        file_path = Path(file_path)
        
        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        # Check file extension
        if file_path.suffix.lower() not in ['.xlsx', '.xls']:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .xlsx or .xls")
        
        try:
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
            else:
                # If no sheet specified, try to read first sheet
                df = pd.read_excel(file_path, header=header_row)
            
            # Check if DataFrame is empty
            if df.empty:
                raise ValueError("Excel file is empty or contains no data rows")
            
            # Log sheet information for debugging
            if sheet_name:
                logger.info(f"Successfully read sheet '{sheet_name}' from {file_path.name}")
            else:
                logger.info(f"Successfully read first sheet from {file_path.name}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")
    
    def _clean_and_map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EXPLANATION: Clean column names and map to standard format.
        
        This method handles the messy reality of Excel column names:
        - Extra spaces
        - Inconsistent capitalization  
        - Special characters
        - Different naming conventions
        
        It maps them to our standardized column names from DATA_CONFIG.
        """
        df = df.copy()
        
        # Clean column names
        df.columns = df.columns.astype(str)  # Ensure all are strings
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        
        # Map to standard column names
        column_mapping = {}
        unmapped_columns = []
        
        for col in df.columns:
            if col in self.column_mapping:
                standard_name = self.column_mapping[col]
                column_mapping[col] = standard_name
                logger.debug(f"Mapped column '{col}' → '{standard_name}'")
            else:
                unmapped_columns.append(col)
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Log unmapped columns (might be okay if they're not required)
        if unmapped_columns:
            logger.warning(f"Unmapped columns (will be ignored): {unmapped_columns}")
            # Drop unmapped columns to avoid confusion
            df = df.drop(columns=unmapped_columns)
        
        logger.info(f"Column mapping completed. Final columns: {list(df.columns)}")
        return df
    
    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Check if all required columns are present in the DataFrame."""
        required_cols = set(DATA_CONFIG.required_cols)
        present_cols = set(df.columns)
        missing_cols = list(required_cols - present_cols)
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info(f"Available columns: {list(present_cols)}")
        
        return missing_cols
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EXPLANATION: Convert columns to appropriate data types.
        
        This method handles type conversion for:
        - Numerical columns (handle text, missing values)
        - Categorical columns (convert to standard categories)
        - Boolean columns
        - Handle common Excel data issues (dates as numbers, etc.)
        """
        df = df.copy()
        
        # Convert categorical columns
        categorical_cols = [col for col in DATA_CONFIG.categorical_cols if col in df.columns]
        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()  # Clean text
            df[col] = df[col].replace('nan', None)  # Handle pandas 'nan' strings
            df[col] = df[col].replace('', None)  # Handle empty strings
        
        # Convert numerical columns
        numerical_cols = [col for col in DATA_CONFIG.numerical_cols if col in df.columns]
        for col in numerical_cols:
            try:
                # Remove any non-numeric characters (like $ or %)
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
                    df[col] = df[col].replace('nan', np.nan)
                    df[col] = df[col].replace('', np.nan)
                
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Log conversion results
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Column '{col}': {null_count} values couldn't be converted to numeric")
                    
            except Exception as e:
                logger.error(f"Error converting column '{col}' to numeric: {str(e)}")
        
        logger.info("Data type conversion completed")
        return df
    
    def _apply_business_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EXPLANATION: Apply business rule validation and data cleaning.
        
        This method implements domain-specific validation:
        - Price relationships (actual price ≤ base price for promos)
        - Unit constraints (non-negative)
        - Margin reasonableness
        - Enum value validation
        
        It can either fix issues automatically or flag them for review.
        """
        df = df.copy()
        issues_found = []
        
        # Validate and clean categorical columns
        df = self._validate_categorical_columns(df, issues_found)
        
        # Validate numerical constraints
        df = self._validate_numerical_constraints(df, issues_found)
        
        # Validate business logic relationships
        df = self._validate_business_relationships(df, issues_found)
        
        # Handle missing values
        df = self._handle_missing_values(df, issues_found)
        
        # Store issues for reporting
        self.data_quality_issues.extend(issues_found)
        
        if issues_found:
            logger.warning(f"Found {len(issues_found)} data quality issues")
            for issue in issues_found[:5]:  # Show first 5 issues
                logger.warning(f"  - {issue}")
            if len(issues_found) > 5:
                logger.warning(f"  ... and {len(issues_found) - 5} more issues")
        
        return df
    
    def _validate_categorical_columns(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """Validate categorical columns against expected enum values."""
        
        # Week_Type validation
        if 'Week_Type' in df.columns:
            valid_week_types = [e.value for e in WeekType]
            invalid_mask = ~df['Week_Type'].isin(valid_week_types)
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, 'Week_Type'].unique()
                issues.append(f"Invalid Week_Type values: {list(invalid_values)}. Valid: {valid_week_types}")
                # Set invalid values to None (will be caught in Pydantic validation)
                df.loc[invalid_mask, 'Week_Type'] = None
        
        # Promo_Type validation
        if 'Promo_Type' in df.columns:
            valid_promo_types = [e.value for e in PromoType]
            invalid_mask = ~df['Promo_Type'].isin(valid_promo_types)
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, 'Promo_Type'].unique()
                issues.append(f"Invalid Promo_Type values: {list(invalid_values)}. Valid: {valid_promo_types}")
                df.loc[invalid_mask, 'Promo_Type'] = None
        
        # Merch validation
        if 'Merch' in df.columns:
            valid_merch_types = [e.value for e in MerchType]
            invalid_mask = ~df['Merch'].isin(valid_merch_types)
            if invalid_mask.any():
                invalid_values = df.loc[invalid_mask, 'Merch'].unique()
                issues.append(f"Invalid Merch values: {list(invalid_values)}. Valid: {valid_merch_types}")
                df.loc[invalid_mask, 'Merch'] = None
        
        return df
    
    def _validate_numerical_constraints(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """Validate numerical columns against business constraints."""
        
        # Check validation rules from config
        for col, rules in DATA_CONFIG.validation_rules.items():
            if col not in df.columns:
                continue
                
            # Min value validation
            if 'min' in rules:
                min_val = rules['min']
                below_min = df[col] < min_val
                if below_min.any():
                    count = below_min.sum()
                    issues.append(f"Column '{col}': {count} values below minimum ({min_val})")
                    # Optionally fix by setting to minimum
                    # df.loc[below_min, col] = min_val
            
            # Max value validation  
            if 'max' in rules:
                max_val = rules['max']
                above_max = df[col] > max_val
                if above_max.any():
                    count = above_max.sum()
                    issues.append(f"Column '{col}': {count} values above maximum ({max_val})")
                    # Optionally fix by setting to maximum
                    # df.loc[above_max, col] = max_val
        
        return df
    
    def _validate_business_relationships(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """Validate business logic relationships between columns."""
        
        # Check if we have the required columns for relationship validation
        required_for_validation = ['Week_Type', 'Promo_Type', 'Base_Price', 'Actual_Price']
        if not all(col in df.columns for col in required_for_validation):
            return df
        
        # For promo weeks, actual price should be ≤ base price
        promo_mask = df['Week_Type'] == 'Promo'
        if promo_mask.any():
            price_violation = promo_mask & (df['Actual_Price'] > df['Base_Price'])
            if price_violation.any():
                count = price_violation.sum()
                issues.append(f"Promo weeks with actual price > base price: {count} records")
        
        # For base weeks, actual price should equal base price (with tolerance)
        base_mask = df['Week_Type'] == 'Base'
        if base_mask.any():
            tolerance = 0.01
            price_diff = abs(df['Actual_Price'] - df['Base_Price'])
            price_violation = base_mask & (price_diff > tolerance)
            if price_violation.any():
                count = price_violation.sum()
                issues.append(f"Base weeks with actual price ≠ base price: {count} records")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """Handle missing values according to business rules."""
        
        # Check for missing values in required columns
        for col in DATA_CONFIG.required_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    issues.append(f"Missing values in required column '{col}': {missing_count} records")
        
        # For numerical columns, we might fill with median or specific business values
        # For now, we'll just report missing values
        
        return df
    
    def _generate_quality_report(self, 
                                df_raw: pd.DataFrame, 
                                df_final: pd.DataFrame, 
                                validation_result: ValidationResult):
        """Generate comprehensive data quality report."""
        
        logger.info("=" * 60)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 60)
        
        # Basic statistics
        logger.info(f"Raw records loaded: {len(df_raw)}")
        logger.info(f"Records after cleaning: {len(df_final)}")
        logger.info(f"Valid records: {validation_result.records_valid}")
        logger.info(f"Invalid records: {validation_result.records_invalid}")
        logger.info(f"Success rate: {validation_result.records_valid/len(df_raw)*100:.1f}%")
        
        # Data quality issues
        if self.data_quality_issues:
            logger.info(f"\nData Quality Issues ({len(self.data_quality_issues)}):")
            for issue in self.data_quality_issues:
                logger.info(f"  • {issue}")
        
        # Validation errors
        if validation_result.errors:
            logger.info(f"\nValidation Errors ({len(validation_result.errors)}):")
            for error in validation_result.errors[:10]:  # Show first 10
                logger.info(f"  • {error}")
            if len(validation_result.errors) > 10:
                logger.info(f"  ... and {len(validation_result.errors) - 10} more errors")
        
        # Summary statistics if we have a valid dataset
        if validation_result.summary:
            summary = validation_result.summary
            logger.info(f"\nDataset Summary:")
            logger.info(f"  Customers: {len(summary.customers)} ({', '.join(summary.customers[:3])}...)")
            logger.info(f"  Items: {len(summary.items)} ({', '.join(summary.items[:3])}...)")
            logger.info(f"  Promo weeks: {summary.promo_weeks}")
            logger.info(f"  Base weeks: {summary.base_weeks}")
            logger.info(f"  Average units: {summary.avg_units:.0f}")
        
        logger.info("=" * 60)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_excel_data(file_path: str, 
                   sheet_name: Optional[str] = None,
                   strict_validation: bool = True) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    EXPLANATION: Convenience function to load Excel data.
    
    This is the main entry point for loading trade promotion data from Excel files.
    It provides a simple interface while handling all the complexity internally.
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name (optional)
        strict_validation: Whether to enforce strict validation
        
    Returns:
        Tuple of (dataframe, validation_result)
        
    Example:
        df, result = load_excel_data('data.xlsx')
        if result.is_valid:
            print(f"Loaded {len(df)} valid records")
        else:
            print(f"Validation failed: {result.errors}")
    """
    loader = ExcelDataLoader(strict_validation=strict_validation)
    return loader.load_file(file_path, sheet_name)


def validate_excel_file(file_path: str, 
                       sheet_name: Optional[str] = None) -> ValidationResult:
    """
    EXPLANATION: Quick validation of Excel file without loading full data.
    
    Useful for checking data quality before committing to full processing.
    """
    df, result = load_excel_data(file_path, sheet_name, strict_validation=False)
    return result


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    print("Trade Promotion Excel Loader - Testing Mode")
    print("=" * 50)
    
    # You can test with your actual Excel file
    # test_file = "path/to/your/Raw_Input_Sample_Data.xlsx"
    
    # For demo, we'll show how to use the loader
    print("ExcelDataLoader Usage Example:")
    print("""
    # Basic usage:
    from data.loaders.excel_loader import load_excel_data
    
    df, result = load_excel_data('your_file.xlsx')
    
    if result.is_valid:
        print(f"Successfully loaded {len(df)} records")
        print(f"Customers: {result.summary.customers}")
        print(f"Items: {result.summary.items}")
    else:
        print("Validation failed:")
        for error in result.errors:
            print(f"  - {error}")
    
    # Advanced usage with specific sheet:
    df, result = load_excel_data(
        file_path='data.xlsx',
        sheet_name='Trade_Data',
        strict_validation=False  # Allow some validation errors
    )
    """)
    
    # Test column mapping
    loader = ExcelDataLoader()
    print(f"\nColumn mapping examples:")
    test_columns = ['Price', 'Margin', 'Retailer', 'Product', 'COGS']
    for col in test_columns:
        mapped = loader.column_mapping.get(col, 'Not found')
        print(f"  '{col}' → '{mapped}'")
    
    print("\nExcel loader ready for use!")