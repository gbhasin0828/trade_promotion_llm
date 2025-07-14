"""
Data Service to read Excel file and provide dropdown options

File: trade_llm/api/data_service.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class DataService:
    """Service to read Excel data and provide dropdown options"""
    
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path
        self.df = None
        self.last_loaded = None
        
    def load_excel_data(self) -> bool:
        """Load Excel data from file"""
        try:
            if not os.path.exists(self.excel_file_path):
                logger.error(f"Excel file not found: {self.excel_file_path}")
                return False
            
            logger.info(f"Loading Excel data from: {self.excel_file_path}")
            
            # Read Excel file
            self.df = pd.read_excel(self.excel_file_path)
            
            # Clean column names (remove extra spaces)
            self.df.columns = self.df.columns.str.strip()
            
            logger.info(f"Loaded {len(self.df)} rows with columns: {list(self.df.columns)}")
            logger.info(f"Available columns: {list(self.df.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            return False
    
    def get_dropdown_options(self) -> Dict[str, Any]:
        """Get all dropdown options from Excel data"""
        
        if self.df is None:
            if not self.load_excel_data():
                return self._get_fallback_options()
        
        try:
            options = {}
            
            # Get unique customers
            if 'Customer' in self.df.columns:
                customers = self.df['Customer'].dropna().unique().tolist()
                options['customers'] = sorted([str(c) for c in customers if str(c) != 'nan'])
                logger.info(f"Found {len(options['customers'])} unique customers")
            else:
                logger.warning("Customer column not found in Excel")
                options['customers'] = []
            
            # Get unique products/items
            item_columns = ['Item', 'Product', 'Product_Name']
            products = []
            for col in item_columns:
                if col in self.df.columns:
                    items = self.df[col].dropna().unique().tolist()
                    products.extend([str(i) for i in items if str(i) != 'nan'])
                    break
            options['products'] = sorted(list(set(products)))
            logger.info(f"Found {len(options['products'])} unique products")
            
            # Get unique week types
            if 'Week_Type' in self.df.columns:
                week_types = self.df['Week_Type'].dropna().unique().tolist()
                options['week_types'] = sorted([str(w) for w in week_types if str(w) != 'nan'])
                logger.info(f"Found week types: {options['week_types']}")
            else:
                logger.warning("Week_Type column not found in Excel")
                options['week_types'] = ['Base', 'Promo']  # Default fallback
            
            # Get unique promo types
            if 'Promo_Type' in self.df.columns:
                promo_types = self.df['Promo_Type'].dropna().unique().tolist()
                options['promo_types'] = sorted([str(p) for p in promo_types if str(p) != 'nan'])
                logger.info(f"Found promo types: {options['promo_types']}")
            else:
                logger.warning("Promo_Type column not found in Excel")
                options['promo_types'] = ['Single', 'Multiple', 'No_Promo']  # Default fallback
            
            # Get unique merchandising types
            if 'Merch' in self.df.columns:
                merch_types = self.df['Merch'].dropna().unique().tolist()
                options['merch_types'] = sorted([str(m) for m in merch_types if str(m) != 'nan'])
                logger.info(f"Found merch types: {options['merch_types']}")
            else:
                logger.warning("Merch column not found in Excel")
                options['merch_types'] = ['ISF_&_Flyer', 'ISF_Only', 'No_Promo']  # Default fallback
            
            # Get price ranges for reference
            price_columns = ['Base_Price', 'Actual_Price', 'List_Price']
            price_ranges = {}
            for col in price_columns:
                if col in self.df.columns:
                    prices = self.df[col].dropna()
                    if len(prices) > 0:
                        price_ranges[col] = {
                            'min': float(prices.min()),
                            'max': float(prices.max()),
                            'avg': float(prices.mean())
                        }
            options['price_ranges'] = price_ranges
            
            # Get sample data for validation
            options['sample_count'] = len(self.df)
            options['columns_found'] = list(self.df.columns)
            
            return {
                'success': True,
                'result': options,
                'message': f'Loaded data from {len(self.df)} records'
            }
            
        except Exception as e:
            logger.error(f"Error getting dropdown options: {str(e)}")
            return self._get_fallback_options()
    
    def _get_fallback_options(self) -> Dict[str, Any]:
        """Fallback options if Excel loading fails"""
        return {
            'success': False,
            'result': {
                'customers': ['Unknown_Customer'],
                'products': ['Product_A', 'Product_B'],
                'week_types': ['Base', 'Promo'],
                'promo_types': ['Single', 'Multiple', 'No_Promo'],
                'merch_types': ['ISF_&_Flyer', 'ISF_Only', 'No_Promo'],
                'price_ranges': {},
                'sample_count': 0,
                'columns_found': []
            },
            'message': 'Using fallback data - Excel file could not be loaded'
        }
    
    def get_sample_data(self, limit: int = 5) -> Dict[str, Any]:
        """Get sample rows from Excel for testing"""
        
        if self.df is None:
            if not self.load_excel_data():
                return {'success': False, 'message': 'Could not load Excel data'}
        
        try:
            sample = self.df.head(limit).to_dict('records')
            
            # Convert any numpy/pandas types to Python types for JSON serialization
            cleaned_sample = []
            for row in sample:
                cleaned_row = {}
                for key, value in row.items():
                    if pd.isna(value):
                        cleaned_row[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        cleaned_row[key] = float(value)
                    else:
                        cleaned_row[key] = str(value)
                cleaned_sample.append(cleaned_row)
            
            return {
                'success': True,
                'result': {
                    'sample_data': cleaned_sample,
                    'total_rows': len(self.df),
                    'columns': list(self.df.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting sample data: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against Excel schema"""
        
        if self.df is None:
            if not self.load_excel_data():
                return {'success': False, 'message': 'Could not load Excel data for validation'}
        
        try:
            validation_results = {
                'valid': True,
                'warnings': [],
                'errors': []
            }
            
            # Check if customer exists
            if 'Customer' in self.df.columns and 'Customer' in input_data:
                valid_customers = self.df['Customer'].dropna().unique()
                if input_data['Customer'] not in valid_customers:
                    validation_results['warnings'].append(
                        f"Customer '{input_data['Customer']}' not found in historical data"
                    )
            
            # Check if product exists
            item_columns = ['Item', 'Product', 'Product_Name']
            if 'Item' in input_data:
                product_found = False
                for col in item_columns:
                    if col in self.df.columns:
                        valid_products = self.df[col].dropna().unique()
                        if input_data['Item'] in valid_products:
                            product_found = True
                            break
                
                if not product_found:
                    validation_results['warnings'].append(
                        f"Product '{input_data['Item']}' not found in historical data"
                    )
            
            # Check price ranges
            price_columns = ['Base_Price', 'Actual_Price', 'List_Price']
            for col in price_columns:
                if col in input_data and col in self.df.columns:
                    price_range = self.df[col].dropna()
                    if len(price_range) > 0:
                        min_price, max_price = price_range.min(), price_range.max()
                        input_price = input_data[col]
                        
                        if input_price < min_price or input_price > max_price:
                            validation_results['warnings'].append(
                                f"{col} ${input_price:.2f} is outside historical range (${min_price:.2f} - ${max_price:.2f})"
                            )
            
            return {
                'success': True,
                'result': validation_results
            }
            
        except Exception as e:
            logger.error(f"Error validating input data: {str(e)}")
            return {'success': False, 'message': str(e)}


# Global data service instance
data_service = None

def get_data_service() -> DataService:
    """Get or create global data service instance"""
    global data_service
    
    if data_service is None:
        # Path to your Excel file
        excel_path = r"C:\Users\User\OneDrive\Desktop\trade_llm\Raw_Input_Data.xlsx"
        data_service = DataService(excel_path)
    
    return data_service


# Test the service
if __name__ == "__main__":
    # Test loading data
    service = get_data_service()
    
    print("Testing DataService...")
    print("="*50)
    
    # Test dropdown options
    options = service.get_dropdown_options()
    print(f"Success: {options['success']}")
    print(f"Message: {options['message']}")
    
    if options['success']:
        result = options['result']
        print(f"\nCustomers ({len(result['customers'])}): {result['customers'][:5]}...")
        print(f"Products ({len(result['products'])}): {result['products'][:5]}...")
        print(f"Week Types: {result['week_types']}")
        print(f"Promo Types: {result['promo_types']}")
        print(f"Merch Types: {result['merch_types']}")
        print(f"Total Records: {result['sample_count']}")
        print(f"Columns Found: {result['columns_found']}")
    
    # Test sample data
    print(f"\n{'='*50}")
    sample = service.get_sample_data(3)
    if sample['success']:
        print("Sample Data:")
        for i, row in enumerate(sample['result']['sample_data']):
            print(f"Row {i+1}: {row}")