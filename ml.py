

import json
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class FeatureInteractionGenerator(BaseEstimator, TransformerMixin):
    """Custom transformer to generate interaction features between specified pairs"""
    
    def __init__(self, interaction_pairs=None):
        """
        Initialize with optional predefined interaction pairs
        
        Args:
            interaction_pairs (list): List of strings in 'feature1/feature2' format
        """
        self.interaction_pairs = interaction_pairs or []
        self.valid_pairs = []
        
    def _validate_interaction_terms(self, X):
        """Internal method to validate and prepare interaction terms"""
        validated_pairs = []
        
        for pair in self.interaction_pairs:
            if not isinstance(pair, str):
                print(f"Skipping invalid interaction format: {pair}")
                continue
                
            features = pair.split("/")
            if len(features) != 2:
                print(f"Skipping malformed interaction term: {pair}")
                continue
                
            feature1, feature2 = features
            if feature1 in X.columns and feature2 in X.columns:
                validated_pairs.append((feature1, feature2))
            else:
                print(f"Features {feature1} or {feature2} not found in dataset")
                
        return validated_pairs
        
    def fit(self, X, y=None):
        """Validate and store feature pairs for interaction"""
        self.valid_pairs = self._validate_interaction_terms(X)
        return self
        
    def transform(self, X):
        """Generate interaction features for validated pairs"""
        if not self.valid_pairs:
            return X
            
        X_transformed = X.copy()
        for feat1, feat2 in self.valid_pairs:
            new_feature_name = f"{feat1}_interact_{feat2}"
            X_transformed[new_feature_name] = X[feat1] * X[feat2]
            
        return X_transformed

def load_pipeline_configuration(config_file_path):
    """Load and validate the pipeline configuration JSON file"""
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
            print(f"Configuration loaded from: {config_file_path}")
            return config.get('design_state_data', {})
    except Exception as error:
        raise ValueError(f"Configuration loading failed: {str(error)}")

def verify_dataset_integrity(dataframe):
    """Perform basic validation checks on the input dataframe"""
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if dataframe.empty:
        raise ValueError("Dataset contains no records")
        
    print(f"Dataset verified - Rows: {dataframe.shape[0]}, Columns: {dataframe.shape[1]}")
    print("Available features:", dataframe.columns.tolist())
    return dataframe

def configure_data_preprocessor(feature_data, processing_config):
    """Create preprocessing pipeline based on configuration"""
    if 'feature_handling' not in processing_config:
        raise ValueError("Missing feature handling configuration")
        
    feature_specs = processing_config['feature_handling']
    numeric_features = []
    categorical_features = []

    print("\nConfiguring data preprocessing:")
    
    # Classify features based on configuration
    for feature in feature_data.columns:
        if feature not in feature_specs:
            continue
            
        properties = feature_specs[feature]
        if not properties.get('is_selected', False):
            continue
            
        feature_type = properties.get('feature_variable_type', 'unknown')
        
        if feature_type == 'numerical':
            numeric_features.append(feature)
        elif feature_type == 'text':
            categorical_features.append(feature)

    print("Numerical features selected:", numeric_features)
    print("Categorical features selected:", categorical_features)

    # Construct preprocessing transformers
    processing_steps = []
    
    if numeric_features:
        numeric_pipeline = Pipeline([
            ('missing_value_handler', SimpleImputer(strategy='mean')),
            ('normalizer', StandardScaler())
        ])
        processing_steps.append(('numeric', numeric_pipeline, numeric_features))

    if categorical_features:
        processing_steps.append(('categorical', 
                               OneHotEncoder(handle_unknown='ignore'), 
                               categorical_features))

    preprocessor = ColumnTransformer(
        processing_steps,
        remainder='drop'  # Exclude unselected features
    )
    
    return preprocessor, numeric_features + categorical_features

def execute_ml_pipeline(config_path, dataset_path):
    """Main pipeline execution function"""
    print("\n" + "="*50)
    print("Custom ML Pipeline Execution")
    print("="*50)
    
    # Phase 1: Configuration Loading
    try:
        print("\n[Phase 1] Loading pipeline configuration...")
        pipeline_config = load_pipeline_configuration(config_path)
        if not pipeline_config:
            raise ValueError("Invalid configuration structure")
    except Exception as error:
        print(f"Configuration error: {str(error)}")
        return None

    # Phase 2: Data Loading and Validation
    try:
        print("\n[Phase 2] Loading dataset...")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file missing: {dataset_path}")
            
        raw_data = pd.read_csv(dataset_path)
        raw_data = verify_dataset_integrity(raw_data)
        raw_data.columns = raw_data.columns.str.strip()  # Clean column names
    except Exception as error:
        print(f"Data loading error: {str(error)}")
        return None

    # Phase 3: Target and Feature Preparation
    try:
        print("\n[Phase 3] Preparing features and target...")
        if 'target' not in pipeline_config or 'target' not in pipeline_config['target']:
            raise ValueError("Target specification missing in configuration")
            
        target_feature = pipeline_config['target']['target']
        if target_feature not in raw_data.columns:
            raise ValueError(f"Target feature '{target_feature}' not found in dataset")
            
        print(f"Target feature identified: {target_feature}")
        features = raw_data.drop(columns=[target_feature])
        target = raw_data[target_feature]
    except Exception as error:
        print(f"Feature preparation error: {str(error)}")
        return None

    # Phase 4: Feature Engineering
    try:
        print("\n[Phase 4] Applying feature engineering...")
        interaction_config = pipeline_config.get('feature_generation', {}).get(
            'explicit_pairwise_interactions', [])
            
        if interaction_config:
            print(f"Generating {len(interaction_config)} interaction features")
            feature_engineer = FeatureInteractionGenerator(interaction_config)
            features = feature_engineer.fit_transform(features)
            print("Updated feature set:", features.columns.tolist())
    except Exception as error:
        print(f"Feature engineering error: {str(error)}")
        return None

    # Phase 5: Data Preprocessing
    try:
        print("\n[Phase 5] Configuring data preprocessing...")
        data_preprocessor, selected_features = configure_data_preprocessor(
            features, pipeline_config)
        print("Preprocessing pipeline configured successfully")
    except Exception as error:
        print(f"Preprocessing error: {str(error)}")
        return None

    # Phase 6: Model Training and Evaluation
    try:
        print("\n[Phase 6] Building and assessing model...")
        
        # Split dataset into training and testing subsets
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Construct complete pipeline
        ml_pipeline = Pipeline([
            ('preprocessing', data_preprocessor),
            ('regressor', LinearRegression())  # Baseline model
        ])
        
        # Train model
        ml_pipeline.fit(X_train, y_train)
        
        # Generate the predictions
        predictions = ml_pipeline.predict(X_test)
        
        # Calculate and display performance of metrics
        print("\nModel Performance Assessment:")
        print("-"*40)
        print(f"R² Coefficient: {r2_score(y_test, predictions):.4f}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.4f}")
        print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}")
        
        return ml_pipeline
        
    except Exception as error:
        print(f"Modeling error: {str(error)}")
        return None

if __name__ == "__main__":
    # Configure OF  file paths
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, "algoparams_from_ui.json")
    data_file = os.path.join(current_directory, "iris.csv")
    
    print(f"\nConfiguration file path: {config_file}")
    print(f"Dataset file path: {data_file}")
    
    # Executing pipeline
    trained_model = execute_ml_pipeline(config_file, data_file)
    
    if trained_model:
        print("\nPipeline execution completed successfully!")
    else:
        print("\nPipeline execution encountered errors")