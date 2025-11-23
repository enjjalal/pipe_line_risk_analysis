import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Set up directories
output_dir = 'ml_results'
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_prepare_data():
    """Load and prepare the dataset for modeling."""
    print("Loading and preparing data...")
    
    try:
        # Load the enriched sensor data
        sensors_gdf = gpd.read_file('integrate/enriched_sensors.geojson')
        print("\n=== Dataset Loaded Successfully ===")
        print(f"Number of records: {len(sensors_gdf)}")
        print("Available columns in the dataset:", list(sensors_gdf.columns))
        
        # Define base features that we'd like to use
        possible_features = [
            'temperature', 'vibration', 'strain',
            'soil_ph', 'soil_clay_pct', 'soil_sand_pct', 'soil_organic_carbon',
            'elevation', 'slope', 'land_cover_index'
        ]
        
        # Only keep features that exist in the dataframe
        feature_columns = [col for col in possible_features if col in sensors_gdf.columns]
        
        # Check if we have any features to work with
        if not feature_columns:
            raise ValueError("No valid features found in the dataset. Please check your data.")
            
        # Add any available pipeline attributes
        pipeline_attrs = [
            col for col in sensors_gdf.columns 
            if col.startswith('pipeline_') and col not in ['pipeline_geometry', 'geometry']
        ]
        
        all_features = feature_columns + pipeline_attrs
        print("\n=== Features Selected for Modeling ===")
        print("Sensor Features:", feature_columns)
        print("Pipeline Attributes:", pipeline_attrs)
        
        # Calculate risk_score if it doesn't exist
        if 'risk_score' not in sensors_gdf.columns:
            print("\nCalculating risk_score based on available features...")
            
            # Normalize features to 0-1 range
            def normalize(series):
                return (series - series.min()) / (series.max() - series.min())
            
            # Calculate weighted risk score (adjust weights as needed)
            weights = {
                'strain': 0.3,
                'vibration': 0.25,
                'soil_clay_pct': 0.15,
                'soil_organic_carbon': 0.15,
                'slope': 0.15
            }
            
            # Initialize risk score
            sensors_gdf['risk_score'] = 0
            
            # Calculate weighted sum of normalized features
            for feature, weight in weights.items():
                if feature in sensors_gdf.columns:
                    sensors_gdf['risk_score'] += normalize(sensors_gdf[feature]) * weight
                else:
                    print(f"Warning: Feature '{feature}' not found in dataset")
            
            # Ensure risk_score is between 0 and 1
            sensors_gdf['risk_score'] = sensors_gdf['risk_score'].clip(0, 1)
        
        # Create risk_class based on risk_score
        print("\nCreating risk classes based on risk_score percentiles...")
        sensors_gdf['risk_class'] = pd.qcut(
            sensors_gdf['risk_score'],
            q=[0, 0.33, 0.67, 1],
            labels=['low', 'medium', 'high']
        )
        
        # Prepare features and targets
        print("\nPreparing features and targets...")
        X = sensors_gdf[all_features].copy()
        y_reg = sensors_gdf['risk_score'].copy()
        y_cls = sensors_gdf['risk_class'].astype('category').cat.codes
        
        # Handle missing values
        print("Handling missing values...")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Convert categorical variables to numerical
        print("Encoding categorical variables...")
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.factorize(X[col])[0]
        
        print("\n=== Data Preparation Complete ===")
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Number of features: {len(all_features)}")
        
        return X, y_reg, y_cls, all_features
        
    except Exception as e:
        print(f"\nError during data preparation: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the file 'integrate/enriched_sensors.geojson' exists")
        print("2. Check that the file contains the required columns (at minimum: 'risk_score')")
        print("3. Verify that the data types are correct (numeric for features)")
        raise

def train_regression_model(X_train, y_train):
    """Train XGBoost regression model."""
    print("Training regression model...")
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def train_classification_model(X_train, y_train):
    """Train XGBoost classification model."""
    print("Training classification model...")
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,  # low, medium, high
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_regression(model, X_test, y_test):
    """Evaluate regression model and return metrics."""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Regression Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Risk Score')
    plt.ylabel('Predicted Risk Score')
    plt.title('Actual vs Predicted Risk Scores')
    plt.savefig(os.path.join(output_dir, 'regression_predictions.png'))
    plt.close()
    
    return {'rmse': rmse, 'r2': r2}

def evaluate_classification(model, X_test, y_test):
    """Evaluate classification model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\nClassification Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def plot_feature_importance(model, feature_names, model_type):
    """Plot and save feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance - {model_type}')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), 
              [feature_names[i] for i in indices], 
              rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_importance_{model_type.lower()}.png'))
    plt.close()

def save_model_report(metrics_reg, metrics_cls, filename='model_report.txt'):
    """Save model evaluation report."""
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write("=== Model Evaluation Report ===\n\n")
        
        f.write("=== Regression Model ===\n")
        for key, value in metrics_reg.items():
            f.write(f"{key}: {value:.4f}\n")
            
        f.write("\n=== Classification Model ===\n")
        for key, value in metrics_cls.items():
            f.write(f"{key}: {value:.4f}\n")

def main():
    print("=== Pipeline Risk Analysis - Machine Learning Model ===\n")
    
    try:
        # Load and prepare data
        print("Step 1/5: Loading and preparing data...")
        X, y_reg, y_cls, feature_names = load_and_prepare_data()
        
        # Split data
        print("\nStep 2/5: Splitting data into training and test sets...")
        X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
            X, y_reg, y_cls, 
            test_size=0.2, 
            random_state=RANDOM_STATE, 
            stratify=y_cls
        )
        
        print(f"\n=== Data Splitting Complete ===")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of features: {len(feature_names)}")
        
        # Train and evaluate regression model
        print("\nStep 3/5: Training regression model...")
        reg_model = train_regression_model(X_train, y_train_reg)
        print("\nEvaluating regression model...")
        metrics_reg = evaluate_regression(reg_model, X_test, y_test_reg)
        
        # Train and evaluate classification model
        print("\nStep 4/5: Training classification model...")
        cls_model = train_classification_model(X_train, y_train_cls)
        print("\nEvaluating classification model...")
        metrics_cls = evaluate_classification(cls_model, X_test, y_test_cls)
        
        # Save models and results
        print("\nStep 5/5: Saving models and results...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        joblib.dump(reg_model, os.path.join(output_dir, 'risk_score_regressor.pkl'))
        joblib.dump(cls_model, os.path.join(output_dir, 'risk_class_classifier.pkl'))
        
        # Save feature names for later use
        with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(feature_names))
        
        # Generate and save plots
        plot_feature_importance(reg_model, feature_names, 'Regression')
        plot_feature_importance(cls_model, feature_names, 'Classification')
        
        # Save evaluation report
        save_model_report(metrics_reg, metrics_cls)
        
        print("\n=== Model Training and Evaluation Complete ===")
        print(f"\nResults saved to: {os.path.abspath(output_dir)}")
        print("\nOutput files:")
        print(f"- Models: risk_score_regressor.pkl, risk_class_classifier.pkl")
        print(f"- Feature importance plots")
        print(f"- Evaluation report: model_report.txt")
        print(f"- Feature names: feature_names.txt")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nModel training failed. Please check the error message above.")
        if 'No such file or directory' in str(e):
            print("Make sure the input file 'integrate/enriched_sensors.geojson' exists.")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
