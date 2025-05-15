import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load('ml/model.joblib')

# Print basic model info
print(f"Model type: {type(model).__name__}")

# For a Random Forest model specifically
if hasattr(model, 'n_estimators'):
    print(f"Number of trees: {model.n_estimators}")

# Feature importances
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_names = [f"pos_{i}" for i in range(len(importances))]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Print top 10 most important features
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    # Print summary statistics
    print("\nImportance statistics:")
    print(importance_df['Importance'].describe())

# For classification, print class information
if hasattr(model, 'classes_'):
    print("\nClasses:")
    print(model.classes_)

# Print other interesting attributes
print("\nModel parameters:")
print(model.get_params())