"""
ML Model for predicting Units and Base_Units in Trade Promotion Optimization

CORRECT APPROACH:
1. Train ONE model with Units as dependent variable
2. For prediction:
   - Base weeks: Base_Units = Units 
   - Promo weeks: Predict Units normally, predict Base_Units by changing Week_Type to Base

SINGLE MARGIN APPROACH:
- Uses only 'Retailer_Margin_%' as the unified margin field
- No separate base/promo margins - simplified business logic

Author: Trade Promotion Optimizer  
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our configurations
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.data_config import DATA_CONFIG, MODEL_CONFIG
from data.loaders.excel_loader import load_excel_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseUnitsPredictor:
    """
    EXPLANATION: ML model for predicting Units with counterfactual Base_Units.
    
    CORRECT APPROACH:
    1. Train ONE model to predict Units (dependent variable)
    2. Use same model to predict Base_Units by modifying input features
    
    SINGLE MARGIN LOGIC:
    - Uses only 'Retailer_Margin_%' field
    - Simplified feature engineering without base/promo margin complexity
    
    Business Logic:
    - Base weeks: Base_Units = Units (no prediction needed)
    - Promo weeks: 
      * Units = model.predict(actual_features)
      * Base_Units = model.predict(features_with_base_week_and_base_price)
    """

    def __init__(self):
        self.model_xgb = None
        self.model_rf = None
        self.feature_columns = []
        self.label_encoders = {}
        self.is_trained = False
        self.feature_importance = {}
        
        logger.info("BaseUnitsPredictor initialized")

    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data and prepare for ML training.
        
        Args:
            file_path: Path to Excel file with training data
            
        Returns:
            Prepared DataFrame ready for ML training
        """
        logger.info(f"Loading training data from: {file_path}")
        
        # Load data using our Excel loader
        df, validation_result = load_excel_data(file_path, strict_validation=False)
        
        if not validation_result.is_valid:
            logger.warning(f"Data validation issues found: {len(validation_result.errors)} errors")
            for error in validation_result.errors[:5]:
                logger.warning(f"  - {error}")
        
        logger.info(f"Loaded {len(df)} records")

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("No data loaded from Excel file. Please check the file and column mappings.")

        # Apply feature engineering
        df_features = self._engineer_features(df)

        logger.info("Data preparation completed")
        return df_features
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML features from raw trade promotion data.
        
        SIMPLIFIED MARGIN LOGIC:
        - Uses only 'Retailer_Margin_%' field
        - Calculates margin if not provided
        - No separate base/promo margin handling
        """
        if df.empty:
            raise ValueError("Cannot engineer features on empty DataFrame")
        
        logger.info(f"Engineering features for {len(df)} records")
        logger.info(f"Available columns: {list(df.columns)}")
        
        df = df.copy()
        
        # SINGLE MARGIN HANDLING - Only use 'Retailer_Margin_%'
        if 'Retailer_Margin_%' not in df.columns:
            # Check if we have the alternative margin columns and create unified margin
            if 'Retailer_Base_Margin_%' in df.columns:
                logger.info("Using Retailer_Base_Margin_% as unified Retailer_Margin_%...")
                df['Retailer_Margin_%'] = df['Retailer_Base_Margin_%']
                
            elif 'Retailer_Promo_Margin_%' in df.columns:
                logger.info("Using Retailer_Promo_Margin_% as unified Retailer_Margin_%...")
                df['Retailer_Margin_%'] = df['Retailer_Promo_Margin_%']
                
            else:
                logger.info("No margin columns found, calculating basic margin...")
                # Check if we have required columns for calculation
                if 'Actual_Price' in df.columns and 'COGS_Unit' in df.columns:
                    df['Retailer_Margin_%'] = np.where(
                        df['Actual_Price'] > 0,
                        (df['Actual_Price'] - df['COGS_Unit']) / df['Actual_Price'],
                        0.25  # Default 25% margin
                    )
                else:
                    logger.error("Cannot calculate margin - missing required columns")
                    raise ValueError("Missing both margin columns and required fields to calculate margin")
        
        logger.info("Using unified Retailer_Margin_% for all calculations")
        
        # 1. Price-based features
        df['Discount_Percentage'] = np.where(
            df['Week_Type'] == 'Promo', 
            (df['Base_Price'] - df['Actual_Price']) / df['Base_Price'], 
            0
        )
        
        # 2. Promotion features
        df['Is_Promo'] = (df['Week_Type'] == 'Promo').astype(int)
        
        # 3. Merchandising intensity
        merch_intensity = {
            'No_Promo': 0,
            'ISF_Only': 1,
            'ISF_&_Ad': 2,
            'ISF_&_Flyer': 2,
            'ISF_&_Flyer_&_Ad': 3
        }
        df['Merch_Intensity'] = df['Merch'].map(merch_intensity).fillna(0)
        
        # 4. Additional price features
        df['Price_Index'] = df['Actual_Price'] / df['List_Price']
        df['COGS_Ratio'] = df['COGS_Unit'] / df['List_Price']
        
        # 5. Margin features (simplified - using only the unified margin)
        df['Margin_Dollar'] = df['Retailer_Margin_%'] * df['Actual_Price']
        
        # 6. Encode categorical variables
        categorical_features = ['Customer', 'Item', 'Week_Type', 'Promo_Type', 'Merch']
        
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                known_categories = set(self.label_encoders[col].classes_)
                df[col] = df[col].astype(str)
                unknown_mask = ~df[col].isin(known_categories)
                df.loc[unknown_mask, col] = 'Unknown'
                if 'Unknown' not in known_categories:
                    self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                df[f'{col}_Encoded'] = self.label_encoders[col].transform(df[col])
        
        logger.info("Feature engineering completed")
        return df

    def prepare_features_for_training(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and target for ML training.
        
        Returns:
        - X: Feature matrix 
        - y: Units target (dependent variable)
        """
        # Define feature columns for ML (simplified - no separate margin columns)
        feature_cols = [
            # Price features
            'Discount_Percentage', 'Price_Index', 'COGS_Ratio',
            # Margin features (unified)
            'Retailer_Margin_%', 'Margin_Dollar',
            # Promotion features
            'Is_Promo', 'Merch_Intensity',
            # Encoded categorical features
            'Customer_Encoded', 'Item_Encoded', 'Week_Type_Encoded', 
            'Promo_Type_Encoded', 'Merch_Encoded'
        ]
        
        # Filter to existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        # Create feature matrix
        X = df[available_cols].fillna(0).values
        
        # Target is always Units
        y = df['Units'].values
        
        logger.info(f"Prepared features: {X.shape[1]} features, {len(y)} samples")
        logger.info(f"Feature columns: {self.feature_columns}")
        
        return X, y

    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ML models for Units prediction.
        
        Training Strategy:
        1. Train ONE model with Units as dependent variable
        2. Use XGBoost + Random Forest ensemble
        3. This model will be used for both Units and Base_Units prediction
        
        Returns:
            Training results with performance metrics
        """
        logger.info("Starting model training...")
        
        # Prepare training data
        X, y = self.prepare_features_for_training(df)
        
        # Check if we have enough data
        if len(X) < 5:
            logger.warning(f"Very small dataset ({len(X)} samples). Results may not be reliable.")
        
        # Train ensemble models
        results = self._train_model_ensemble(X, y)
        
        self.is_trained = True
        
        logger.info("Model training completed successfully")
        return results
    
    def _train_model_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train ensemble of XGBoost and Random Forest models."""
        
        # For very small datasets, use all data for training
        if len(X) < 10:
            logger.warning("Small dataset detected. Using all data for training (no test split).")
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        results = {}
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_params = MODEL_CONFIG.xgboost_params.copy()
        # Adjust for small datasets
        if len(X_train) < 50:
            xgb_params['n_estimators'] = min(50, xgb_params['n_estimators'])
            xgb_params['max_depth'] = min(3, xgb_params['max_depth'])
        
        self.model_xgb = xgb.XGBRegressor(**xgb_params)
        self.model_xgb.fit(X_train, y_train)
        
        xgb_predict = self.model_xgb.predict(X_test)
        xgb_metrics = self._calculate_metrics(y_test, xgb_predict)
        results['xgboost'] = xgb_metrics
        
        # Store feature importance
        if len(self.feature_columns) == len(self.model_xgb.feature_importances_):
            feature_importance = dict(zip(self.feature_columns, self.model_xgb.feature_importances_))
            self.feature_importance['xgboost'] = feature_importance
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_params = MODEL_CONFIG.random_forest_params.copy()
        # Adjust for small datasets
        if len(X_train) < 50:
            rf_params['n_estimators'] = min(50, rf_params['n_estimators'])
            rf_params['max_depth'] = min(5, rf_params['max_depth'])
        
        self.model_rf = RandomForestRegressor(**rf_params)
        self.model_rf.fit(X_train, y_train)
        
        rf_predict = self.model_rf.predict(X_test)
        rf_metrics = self._calculate_metrics(y_test, rf_predict)
        results['random_forest'] = rf_metrics
        
        # Store feature importance
        if len(self.feature_columns) == len(self.model_rf.feature_importances_):
            feature_importance = dict(zip(self.feature_columns, self.model_rf.feature_importances_))
            self.feature_importance['random_forest'] = feature_importance
        
        # Create ensemble
        ensemble_predict = 0.6 * xgb_predict + 0.4 * rf_predict
        ensemble_metrics = self._calculate_metrics(y_test, ensemble_predict)
        results['ensemble'] = ensemble_metrics
        
        # Log results
        logger.info("Model Training Results:")
        logger.info(f"  XGBoost RMSE: {xgb_metrics['rmse']:.2f}, R²: {xgb_metrics['r2']:.3f}")
        logger.info(f"  Random Forest RMSE: {rf_metrics['rmse']:.2f}, R²: {rf_metrics['r2']:.3f}")
        logger.info(f"  Ensemble RMSE: {ensemble_metrics['rmse']:.2f}, R²: {ensemble_metrics['r2']:.3f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.001))) * 100
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for Units and Base_Units.
        
        CORRECT LOGIC:
        1. For all rows: Predict Units using actual features
        2. For Base weeks: Base_Units = Units (no additional prediction)
        3. For Promo weeks: Predict Base_Units by changing Week_Type to Base and Price to Base_Price
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with Units and Base_Units predictions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        df = df.copy()
        
        # Apply feature engineering
        df_features = self._engineer_features(df)
        
        # Prepare feature matrix for current data (actual scenario)
        X_actual = df_features[self.feature_columns].fillna(0).values
        
        # STEP 1: Predict Units for actual scenario
        units_pred_xgb = self.model_xgb.predict(X_actual)
        units_pred_rf = self.model_rf.predict(X_actual)
        df['Units_Predicted'] = 0.6 * units_pred_xgb + 0.4 * units_pred_rf
        
        # STEP 2: Handle Base_Units prediction
        base_mask = df['Week_Type'] == 'Base'
        promo_mask = df['Week_Type'] == 'Promo'
        
        # Initialize Base_Units
        df['Base_Units'] = 0.0
        
        # For Base weeks: Base_Units = Units (no prediction needed)
        df.loc[base_mask, 'Base_Units'] = df.loc[base_mask, 'Units']
        
        # For Promo weeks: Predict Base_Units using modified features
        if promo_mask.any():
            logger.info(f"Predicting Base_Units for {promo_mask.sum()} promo weeks...")
            
            # Create modified dataframe for base scenario prediction
            df_base_scenario = df_features.copy()
            
            # CRITICAL: Change promo weeks to look like base weeks
            df_base_scenario.loc[promo_mask, 'Week_Type'] = 'Base'
            df_base_scenario.loc[promo_mask, 'Actual_Price'] = df_base_scenario.loc[promo_mask, 'Base_Price']
            df_base_scenario.loc[promo_mask, 'Is_Promo'] = 0
            df_base_scenario.loc[promo_mask, 'Discount_Percentage'] = 0
            df_base_scenario.loc[promo_mask, 'Promo_Type'] = 'No_Promo'
            df_base_scenario.loc[promo_mask, 'Merch'] = 'No_Promo'
            
            # Re-encode categorical features for the modified scenario
            df_base_scenario.loc[promo_mask, 'Week_Type_Encoded'] = self.label_encoders['Week_Type'].transform(['Base'] * promo_mask.sum())
            df_base_scenario.loc[promo_mask, 'Promo_Type_Encoded'] = self.label_encoders['Promo_Type'].transform(['No_Promo'] * promo_mask.sum())
            df_base_scenario.loc[promo_mask, 'Merch_Encoded'] = self.label_encoders['Merch'].transform(['No_Promo'] * promo_mask.sum())
            
            # Recalculate derived features (using unified margin)
            df_base_scenario.loc[promo_mask, 'Price_Index'] = df_base_scenario.loc[promo_mask, 'Base_Price'] / df_base_scenario.loc[promo_mask, 'List_Price']
            df_base_scenario.loc[promo_mask, 'Margin_Dollar'] = df_base_scenario.loc[promo_mask, 'Retailer_Margin_%'] * df_base_scenario.loc[promo_mask, 'Base_Price']
            df_base_scenario.loc[promo_mask, 'Merch_Intensity'] = 0
            
            # Prepare feature matrix for base scenario
            X_base_scenario = df_base_scenario[self.feature_columns].fillna(0).values
            
            # Predict Base_Units using the same model
            base_units_pred_xgb = self.model_xgb.predict(X_base_scenario)
            base_units_pred_rf = self.model_rf.predict(X_base_scenario)
            base_units_predicted = 0.6 * base_units_pred_xgb + 0.4 * base_units_pred_rf
            
            # Assign Base_Units for promo weeks (fixed indexing)
            df.loc[promo_mask, 'Base_Units'] = base_units_predicted[promo_mask]
        
        # Business constraint: Base_Units should not exceed Units for promo weeks
        violation_mask = promo_mask & (df['Base_Units'] > df['Units'])
        if violation_mask.any():
            logger.warning(f"Adjusting {violation_mask.sum()} predictions where Base_Units > Units")
            df.loc[violation_mask, 'Base_Units'] = df.loc[violation_mask, 'Units'] * 0.95
        
        # Ensure non-negative predictions
        df['Units_Predicted'] = np.maximum(df['Units_Predicted'], 0)
        df['Base_Units'] = np.maximum(df['Base_Units'], 0)
        
        logger.info("Predictions completed")
        return df

    def save_models(self, model_dir: str):
        """Save trained models to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        joblib.dump(self.model_xgb, model_dir / "model_xgb.joblib")
        joblib.dump(self.model_rf, model_dir / "model_rf.joblib")
        
        # Save encoders and metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance
        }
        joblib.dump(metadata, model_dir / "metadata.joblib")
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str):
        """Load trained models from disk."""
        model_dir = Path(model_dir)
        
        # Load metadata
        metadata = joblib.load(model_dir / "metadata.joblib")
        self.feature_columns = metadata['feature_columns']
        self.label_encoders = metadata['label_encoders']
        self.feature_importance = metadata['feature_importance']
        
        # Load models
        self.model_xgb = joblib.load(model_dir / "model_xgb.joblib")
        self.model_rf = joblib.load(model_dir / "model_rf.joblib")
        
        self.is_trained = True
        logger.info(f"Models loaded from {model_dir}")


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_base_units_model(data_file: str, save_dir: Optional[str] = None) -> BaseUnitsPredictor:
    """
    Complete training pipeline for Units prediction model.
    
    This function:
    1. Loads and prepares training data
    2. Trains ML models with Units as dependent variable
    3. Evaluates performance
    4. Saves models for production use
    """
    logger.info("=" * 60)
    logger.info("TRAINING UNITS PREDICTION MODEL")
    logger.info("=" * 60)
    
    # Initialize predictor
    predictor = BaseUnitsPredictor()
    
    # Load and prepare data
    logger.info("Step 1: Loading and preparing data...")
    df = predictor.load_and_prepare_data(data_file)
    
    # Check if we have data
    if df.empty:
        logger.error("No data available for training. Please check your Excel file and column mappings.")
        return predictor
    
    # Train models
    logger.info("Step 2: Training ML models...")
    results = predictor.train_models(df)
    
    # Generate predictions on training data for validation
    logger.info("Step 3: Generating validation predictions...")
    df_with_predictions = predictor.predict(df)
    
    # Evaluate performance
    logger.info("Step 4: Evaluating model performance...")
    evaluate_model_performance(df_with_predictions)
    
    # Save models if requested
    if save_dir:
        logger.info(f"Step 5: Saving models to {save_dir}...")
        predictor.save_models(save_dir)
    
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 60)
    
    return predictor


def evaluate_model_performance(df: pd.DataFrame):
    """Evaluate model performance and business logic."""
    
    logger.info("MODEL PERFORMANCE EVALUATION")
    logger.info("-" * 40)
    
    # Evaluate Units prediction accuracy
    if 'Units' in df.columns and 'Units_Predicted' in df.columns:
        actual_units = df['Units'].values
        pred_units = df['Units_Predicted'].values
        
        units_mae = mean_absolute_error(actual_units, pred_units)
        units_rmse = np.sqrt(mean_squared_error(actual_units, pred_units))
        units_r2 = r2_score(actual_units, pred_units)
        units_mape = np.mean(np.abs((actual_units - pred_units) / np.maximum(actual_units, 0.001))) * 100
        
        logger.info(f"Units Prediction Performance:")
        logger.info(f"  MAE: {units_mae:.2f}")
        logger.info(f"  RMSE: {units_rmse:.2f}")
        logger.info(f"  R²: {units_r2:.3f}")
        logger.info(f"  MAPE: {units_mape:.1f}%")
    
    # Validate business logic
    base_mask = df['Week_Type'] == 'Base'
    promo_mask = df['Week_Type'] == 'Promo'
    
    if base_mask.any():
        # For base weeks, Base_Units should equal Units
        base_weeks_diff = abs(df.loc[base_mask, 'Units'] - df.loc[base_mask, 'Base_Units']).max()
        logger.info(f"Base weeks validation - Max difference between Units and Base_Units: {base_weeks_diff:.2f}")
    
    if promo_mask.any():
        # For promo weeks, show average lift
        promo_df = df[promo_mask]
        if 'Base_Units' in promo_df.columns:
            avg_lift = ((promo_df['Units'] - promo_df['Base_Units']) / promo_df['Base_Units'] * 100).mean()
            logger.info(f"Average Lift % (Promo weeks): {avg_lift:.1f}%")
    
    logger.info("-" * 40)


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Example usage with your data file
    data_file = r"C:\Users\User\OneDrive\Desktop\trade_llm\Raw_Input_Data.xlsx"
    
    print("Trade Promotion Units Predictor - Training")
    print("=" * 50)
    
    try:
        # Train the model
        model = train_base_units_model(
            data_file=data_file,
            save_dir="models/saved"
        )
        
        print("\n✅ Model training completed successfully!")
        
        # Show feature importance
        if model.feature_importance:
            print("\nFeature Importance (Top 10):")
            all_importance = {}
            for model_type, importance_dict in model.feature_importance.items():
                for feature, importance in importance_dict.items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)
            
            # Average importance across models
            avg_importance = {k: np.mean(v) for k, v in all_importance.items()}
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(sorted_importance[:10]):
                print(f"  {i+1:2d}. {feature}: {importance:.3f}")
        
        print(f"\nModel ready for production use!")
        
        # Test prediction logic
        print("\n" + "="*50)
        print("TESTING PREDICTION LOGIC")
        print("="*50)
        
        # Load test data
        df_test = model.load_and_prepare_data(data_file)
        predictions = model.predict(df_test)
        
        print("\nPrediction Results:")
        print(predictions[['Week_Type', 'Units', 'Units_Predicted', 'Base_Units']].round(2))
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()