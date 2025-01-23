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
cmapss_train_path = r"E:\Northeastern Univetsity\Projects\Predictive Maintenance for IoT Devices\Datasets\CMAPSS Data\test_FD001.txt"  # Update this to the actual path on your Windows machine

try:
    # Load the dataset
    print(f"Loading dataset from: {cmapss_train_path}")
    cmapss_train = pd.read_csv(cmapss_train_path, sep='\s+', header=None)
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
    y_pred = linear_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")

except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
