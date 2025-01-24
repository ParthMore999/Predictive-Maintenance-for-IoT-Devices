# CMAPSS Dataset - Regression Model for Remaining Useful Life (RUL)

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

    # Step 3: Train and Evaluate Models
    # Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    linear_mae = mean_absolute_error(y_test, y_pred_linear)
    linear_mse = mean_squared_error(y_test, y_pred_linear)
    linear_r2 = r2_score(y_test, y_pred_linear)

    print("Linear Regression Performance:")
    print(f"Mean Absolute Error (MAE): {linear_mae}")
    print(f"Mean Squared Error (MSE): {linear_mse}")
    print(f"R-squared (R²): {linear_r2}")

    # Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)

    print("\nRandom Forest Regressor Performance:")
    print(f"Mean Absolute Error (MAE): {rf_mae}")
    print(f"Mean Squared Error (MSE): {rf_mse}")
    print(f"R-squared (R²): {rf_r2}")

    # Step 4: Visualize Predictions
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred_linear, alpha=0.6, label="Linear Regression")
    plt.scatter(y_test, y_pred_rf, alpha=0.6, label="Random Forest", marker='x')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Fit")
    plt.title("Actual vs Predicted RUL")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 5: Save Outputs
    # Save predictions from both models
    predictions_df = pd.DataFrame({
        "Actual RUL": y_test,
        "Predicted RUL (Linear Regression)": y_pred_linear,
        "Predicted RUL (Random Forest)": y_pred_rf
    })
    predictions_path = os.path.join(os.getcwd(), "rul_predictions_comparison.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # Save evaluation metrics to a text file
    metrics_path = os.path.join(os.getcwd(), "evaluation_metrics_comparison.txt")
    with open(metrics_path, "w") as f:
        f.write("Linear Regression Performance:\n")
        f.write(f"Mean Absolute Error (MAE): {linear_mae}\n")
        f.write(f"Mean Squared Error (MSE): {linear_mse}\n")
        f.write(f"R-squared (R²): {linear_r2}\n\n")
        f.write("Random Forest Regressor Performance:\n")
        f.write(f"Mean Absolute Error (MAE): {rf_mae}\n")
        f.write(f"Mean Squared Error (MSE): {rf_mse}\n")
        f.write(f"R-squared (R²): {rf_r2}\n")
    print(f"Evaluation metrics saved to {metrics_path}")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please check the dataset path and ensure the file exists.")
except Exception as e:
    print(f"An error occurred: {e}")
