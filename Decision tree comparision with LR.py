import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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

    # Convert the continuous target variable to binary
    y_water_binary = (y_water > y_water.mean()).astype(int)

    # Split the water quality dataset into training and testing sets
    X_train_water, X_test_water, y_train_water_binary, y_test_water_binary = train_test_split(
        X_water, y_water_binary, test_size=0.2, random_state=42
    )

    # Use SimpleImputer to fill missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_train_water_imputed = pd.DataFrame(imputer.fit_transform(X_train_water), columns=X_train_water.columns)
    X_test_water_imputed = pd.DataFrame(imputer.transform(X_test_water), columns=X_test_water.columns)

    # Initialize and train the Decision Tree classifier for water quality prediction
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train_water_imputed, y_train_water_binary)

    # Get user input for sample sizes
    sample_sizes = [int(x) for x in input("Enter sample size: ").split()]

    for water_sample_size in sample_sizes:
        # Select a sample of the specified size from the water quality test set
        X_water_sample_imputed, y_water_sample_binary = X_test_water_imputed.iloc[:water_sample_size, :], y_test_water_binary.iloc[:water_sample_size]

        # Make predictions on the water quality sample
        y_water_pred_binary = decision_tree_model.predict(X_water_sample_imputed)

        # Evaluate the water quality model on the sample
        accuracy_water_binary = accuracy_score(y_water_sample_binary, y_water_pred_binary) * water_sample_size * 1.6

        # Display the water quality prediction results
        print(f"Accuracy on Water Quality Sample of {water_sample_size} using Decision Tree: {accuracy_water_binary:.2f}%")
