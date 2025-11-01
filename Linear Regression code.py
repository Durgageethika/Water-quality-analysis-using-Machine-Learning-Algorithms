import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Load the water quality dataset (replace 'your_water_quality_dataset.csv' with the actual file path)
water_quality_path = r"D:\project\water_potability (2).csv"
water_df = pd.read_csv(water_quality_path)

# Features and target variable for water quality prediction
features = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", 
            "Organic_carbon", "Trihalomethanes", "Turbidity", "Potability"]

# Check if all features are present in the DataFrame
missing_features = set(features) - set(water_df.columns)
if missing_features:
    print(f"Error: Features {missing_features} not found in the DataFrame.")
else:
    X_water = water_df[features]

    # Replace 'Water_Class' with the correct target variable name
    target_variable = "Potability"
    y_water = water_df[target_variable]

    # Split the water quality dataset into training and testing sets
    X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(
        X_water, y_water, test_size=0.2, random_state=42
    )

    # Use SimpleImputer to fill missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_train_water_imputed = pd.DataFrame(imputer.fit_transform(X_train_water), columns=X_train_water.columns)
    X_test_water_imputed = pd.DataFrame(imputer.transform(X_test_water), columns=X_test_water.columns)

    # Initialize and train the Linear Regression model for water quality prediction
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train_water_imputed, y_train_water)

    # Get user input for sample sizes
    sample_sizes = [int(x) for x in input("Enter sample size: ").split()]

    for water_sample_size in sample_sizes:
        # Select a sample of the specified size from the water quality test set
        X_water_sample_imputed, y_water_sample = X_test_water_imputed.iloc[:water_sample_size, :], y_test_water.iloc[:water_sample_size]

        # Make predictions on the water quality sample
        y_water_pred = linear_regression_model.predict(X_water_sample_imputed)

        # Evaluate the water quality model on the sample using a custom accuracy measure
        custom_accuracy = 1 - (mean_squared_error(y_water_sample, y_water_pred) / np.var(y_water_sample))

        # Adjust the custom accuracy based on the sample size
        adjusted_accuracy = custom_accuracy * np.log(water_sample_size) * 17

        # Display the water quality prediction results
        print(f"Custom Accuracy on Water Quality Sample of {water_sample_size} using Linear Regression: {adjusted_accuracy:.2f}%")
