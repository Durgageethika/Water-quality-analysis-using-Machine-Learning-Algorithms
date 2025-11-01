import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

# Creating a synthetic dataset for demonstration purposes
np.random.seed(42)
num_samples = 1000
num_features = 10

X_synthetic = np.random.rand(num_samples, num_features)
y_synthetic = X_synthetic.sum(axis=1)  # A simple linear relationship for demonstration

# Create a DataFrame
synthetic_df = pd.DataFrame(X_synthetic, columns=[f"Feature_{i}" for i in range(1, num_features + 1)])
synthetic_df["Target"] = y_synthetic

# Features and target variable for water quality prediction
features = [f"Feature_{i}" for i in range(1, num_features + 1)]

# Split the synthetic dataset into training and testing sets
X_train_synthetic, X_test_synthetic, y_train_synthetic, y_test_synthetic = train_test_split(
    synthetic_df[features], synthetic_df["Target"], test_size=0.2, random_state=42
)

# Use SimpleImputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train_synthetic_imputed = pd.DataFrame(imputer.fit_transform(X_train_synthetic), columns=X_train_synthetic.columns)
X_test_synthetic_imputed = pd.DataFrame(imputer.transform(X_test_synthetic), columns=X_test_synthetic.columns)

# Initialize and train the Random Forest Regressor for water quality prediction
random_forest_model_synthetic = RandomForestRegressor(random_state=42)
random_forest_model_synthetic.fit(X_train_synthetic_imputed, y_train_synthetic)

# Get user input for sample size
sample_size_synthetic = int(input("Enter the sample size: "))

# Select a sample of the specified size from the synthetic test set
X_synthetic_sample_imputed, y_synthetic_sample = X_test_synthetic_imputed.iloc[:sample_size_synthetic, :], y_test_synthetic.iloc[:sample_size_synthetic]

# Make predictions on the synthetic sample
y_synthetic_pred_sample = random_forest_model_synthetic.predict(X_synthetic_sample_imputed)

# Evaluate R-squared on the synthetic sample
r_squared_synthetic_sample = r2_score(y_synthetic_sample, y_synthetic_pred_sample)*108

# Display the R-squared on the synthetic sample
print(f"R-squared on Synthetic Sample of {sample_size_synthetic}: {r_squared_synthetic_sample:.2f}")
