"""
Simple Business Formula Engine for Trade Promotion Optimization

Applies your exact business formulas to calculate derived metrics.
Clean, simple implementation without unnecessary complexity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class FormulaEngine:
    """Simple formula engine that applies your business calculations."""
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all business formulas to input data."""
        df = df.copy()
        
        self._calculate_base_trade(df)
        self._calculate_var_trade(df)
        self._calculate_total_trade(df)
        self._calculate_trade_rate(df)
        self._calculate_margins(df)
        self._calculate_profit(df)
        self._calculate_discount(df)
        self._calculate_lift(df)
        self._calculate_inc_profit(df) 
        self._calculate_roi(df)
        
        return df
    
    def _calculate_base_trade(self, df):
         
         """Base_Trade_$_Unit = Base_Price * (Retailer_Margin_% - 1) + List_Price"""

         df['Base_Trade_$_Unit'] = df['Base_Price'] * (df['Retailer_Margin_%'] - 1) + df['List_Price']

         return df
    
    def _calculate_var_trade(self, df):

        df['Var_Trade_$_Unit'] = 0.0

        promo_mask = df['Week_Type'] == 'Promo'

        df.loc[promo_mask, 'Var_Trade_$_Unit'] = (
            (df.loc[promo_mask, 'Retailer_Margin_%'] * df.loc[promo_mask, 'Actual_Price']) -
            (df.loc[promo_mask, 'Actual_Price'] - df.loc[promo_mask, 'List_Price'] + 
             df.loc[promo_mask, 'Base_Trade_$_Unit'])
        )
        return df

    def _calculate_total_trade(self, df):
        """$_Trade_Unit: Base_Trade for Base weeks, Base_Trade + Var_Trade for Promo weeks"""
        df['$_Trade_Unit'] = df['Base_Trade_$_Unit'].copy()
        
        promo_mask = df['Week_Type'] == 'Promo'
        df.loc[promo_mask, '$_Trade_Unit'] = (
            df.loc[promo_mask, 'Base_Trade_$_Unit'] + df.loc[promo_mask, 'Var_Trade_$_Unit']
        )
        return df
    

    def _calculate_trade_rate(self, df):
        """%_Trade_Rate = $_Trade_Unit / List_Price"""
        df['%_Trade_Rate'] = df['$_Trade_Unit'] / df['List_Price']
        df['%_Trade_Rate'] = df['%_Trade_Rate'].fillna(0)
        return df
    
    
    def _calculate_margins(self, df):
        """Calculate margin fields"""
        # Retailer_Base_$_Margin
        df['Retailer_Base_$_Margin'] = df['Base_Price'] - df['List_Price'] + df['Base_Trade_$_Unit']
        
        # Retailer_$_Margin (depends on week type)
        base_mask = df['Week_Type'] == 'Base'
        promo_mask = df['Week_Type'] == 'Promo'
        
        df['Retailer_$_Margin'] = 0.0
        df.loc[base_mask, 'Retailer_$_Margin'] = df.loc[base_mask, 'Retailer_Base_$_Margin']
        df.loc[promo_mask, 'Retailer_$_Margin'] = (
            df.loc[promo_mask, 'Actual_Price'] - df.loc[promo_mask, 'List_Price'] + 
            df.loc[promo_mask, '$_Trade_Unit']
        )
        return df
    
    def _calculate_profit(self, df):
        """Profit_Unit = List_Price - COGS_Unit - Trade_Unit"""
        base_mask = df['Week_Type'] == 'Base'
        promo_mask = df['Week_Type'] == 'Promo'
        
        df['Profit_Unit'] = 0.0
        df.loc[base_mask, 'Profit_Unit'] = (
            df.loc[base_mask, 'List_Price'] - df.loc[base_mask, 'COGS_Unit'] - 
            df.loc[base_mask, 'Base_Trade_$_Unit']
        )
        df.loc[promo_mask, 'Profit_Unit'] = (
            df.loc[promo_mask, 'List_Price'] - df.loc[promo_mask, 'COGS_Unit'] - 
            df.loc[promo_mask, '$_Trade_Unit']
        )
        
        # Profit percentage
        df['Profit_Unit_Percentage'] = df['Profit_Unit'] / df['List_Price']
        df['Profit_Unit_Percentage'] = df['Profit_Unit_Percentage'].fillna(0)
        
        return df
    
    def _calculate_discount(self, df):
        """Discount = (Base_Price - Actual_Price) / Base_Price * 100"""
        df['Discount_Percentage'] = np.where(
            df['Base_Price'] > 0,
            (df['Base_Price'] - df['Actual_Price']) / df['Base_Price'] * 100,
            0
        )
        df['Discount_Dollar'] = df['Base_Price'] - df['Actual_Price']
        return df
    
    def _calculate_lift(self, df):
        """Lift_% = (Units - Base_Units) / Base_Units for Promo weeks"""
        promo_mask = df['Week_Type'] == 'Promo'
        df['Lift_%'] = 0.0
        df.loc[promo_mask, 'Lift_%'] = (
            (df.loc[promo_mask, 'Units'] - df.loc[promo_mask, 'Base_Units']) / 
            df.loc[promo_mask, 'Base_Units']
        )
        return df
    
    def _calculate_inc_profit(self, df):
        """Inc_Profit = (Promo_Units * Profit_Unit_Promo) - (Base_Units * Profit_Unit_Base)"""
        promo_mask = df['Week_Type'] == 'Promo'
        df['Inc_Profit'] = 0.0
        
        if promo_mask.any():
            # For promo weeks, calculate incremental profit
            promo_profit = df.loc[promo_mask, 'Units'] * df.loc[promo_mask, 'Profit_Unit']
            base_profit = df.loc[promo_mask, 'Base_Units'] * (
                df.loc[promo_mask, 'List_Price'] - df.loc[promo_mask, 'COGS_Unit'] - 
                df.loc[promo_mask, 'Base_Trade_$_Unit']
            )
            df.loc[promo_mask, 'Inc_Profit'] = promo_profit - base_profit
        
        return df
    
    def _calculate_roi(self, df):
        """ROI = Inc_Profit / (Promo_Units * Var_Trade_$_Unit)"""
        promo_mask = df['Week_Type'] == 'Promo'
        df['ROI'] = 0.0
        
        roi_mask = promo_mask & (df['Var_Trade_$_Unit'] > 0)
        if roi_mask.any():
            df.loc[roi_mask, 'ROI'] = (
                df.loc[roi_mask, 'Inc_Profit'] / 
                (df.loc[roi_mask, 'Units'] * df.loc[roi_mask, 'Var_Trade_$_Unit'])
            )
        
        return df


# Testing
if __name__ == "__main__":
    # Simple test
    sample_data = pd.DataFrame({
        'Base_Price': [3.99, 4.29],
        'Actual_Price': [2.99, 4.29],
        'List_Price': [3.49, 3.49],
        'COGS_Unit': [1.75, 1.75],
        'Week_Type': ['Promo', 'Base'],
        'Retailer_Margin_%': [0.25, 0.25],
        'Units': [1500, 800],
        'Base_Units' : [900, 800]
    })

    print(sample_data[['Week_Type', 'Base_Price', 'Actual_Price']])

    engine = FormulaEngine()

    result = engine.calculate_all(sample_data)
    print(result)

