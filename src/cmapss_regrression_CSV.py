# CMAPSS Dataset - Regression Model for Remaining Useful Life (RUL)

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 2: Load and Prepare the Dataset
# Set dataset path (update for Windows compatibility if needed)
cmapss_train_path = r"E:\Northeastern Univetsity\Projects\Predictive Maintenance for IoT Devices\Datasets\CMAPSS Data\train_FD001.txt"

try:
    # Load the dataset
    print(f"Loading dataset from: {cmapss_train_path}")
    cmapss_train = pd.read_csv(cmapss_train_path, delim_whitespace=True, header=None)
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {cmapss_train.shape}")

    # Drop unnecessary columns (e.g., IDs)
    X = cmapss_train.iloc[:, 2:-1]  # Features
    y = cmapss_train.iloc[:, -1]  # Remaining Useful Life (RUL)
    print("Features and labels extracted.")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Features normalized.")

    # Step 3: Train the Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    print("Model training completed.")

    # Step 4: Evaluate the Model
    # Make predictions
    y_pred = linear_model.predict(X_test)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R²):", r2)

    # Step 5: Analyze Feature Importance
    # Extract feature coefficients
    feature_importance = linear_model.coef_
    feature_names = [f"Feature {i}" for i in range(len(feature_importance))]

    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance (Linear Regression - CMAPSS Dataset)")
    plt.xlabel("Coefficient Value (Magnitude)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # Step 6: Save Outputs
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        "Actual RUL": y_test,
        "Predicted RUL": y_pred
    })
    predictions_path = os.path.join(os.getcwd(), "rul_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # Save feature importance to CSV
    importance_path = os.path.join(os.getcwd(), "feature_importance_cmapss.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

    # Save evaluation metrics to a text file
    metrics_path = os.path.join(os.getcwd(), "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"Mean Squared Error (MSE): {mse}\n")
        f.write(f"R-squared (R²): {r2}\n")
    print(f"Evaluation metrics saved to {metrics_path}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please check the dataset path and ensure the file exists.")
except Exception as e:
    print(f"An error occurred: {e}")
