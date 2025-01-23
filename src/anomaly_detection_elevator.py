# Anomaly Detection in Elevator Dataset using Isolation Forest

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Step 2: Load and Clean the Dataset
data_path = r"E:\Northeastern Univetsity\Projects\Predictive Maintenance for IoT Devices\Datasets\Elevator predictive maintenance data set\predictive-maintenance-dataset.csv"
try:
    # Load the dataset
    elevator_data = pd.read_csv(data_path)
    
    # Drop unnecessary columns (e.g., ID) and handle missing values
    elevator_data_cleaned = elevator_data.dropna().drop(columns=["ID"], errors="ignore")

    # Scale numeric features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(elevator_data_cleaned)

    # Convert scaled features back into a DataFrame
    feature_columns = elevator_data_cleaned.columns
    scaled_elevator_data = pd.DataFrame(scaled_features, columns=feature_columns)

    print("Data loaded and cleaned successfully.")

except Exception as e:
    print(f"Error loading dataset: {e}")

# Step 3: Train Isolation Forest Model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
scaled_elevator_data["anomaly"] = iso_forest.fit_predict(scaled_features)

# Map anomalies (-1) and normal (1) values to labels
scaled_elevator_data["anomaly"] = scaled_elevator_data["anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

# Step 4: Analyze Anomalies
anomalies = scaled_elevator_data[scaled_elevator_data["anomaly"] == "Anomaly"]
normal = scaled_elevator_data[scaled_elevator_data["anomaly"] == "Normal"]

# Display summary statistics for anomalies and normal data
print("Anomalies Summary:")
print(anomalies.describe())

print("\nNormal Data Summary:")
print(normal.describe())

# Step 5: Visualize Anomalies vs Normal Data (Key Features)
key_features = ["revolutions", "humidity", "vibration"]
plt.figure(figsize=(15, 5))
for i, feature in enumerate(key_features):
    if feature in scaled_elevator_data.columns:
        plt.subplot(1, len(key_features), i + 1)
        sns.kdeplot(normal[feature], label="Normal", fill=True, alpha=0.5)
        sns.kdeplot(anomalies[feature], label="Anomaly", fill=True, alpha=0.5)
        plt.title(f"Distribution: {feature}")
        plt.xlabel(feature)
        plt.legend()

plt.tight_layout()
plt.show()

# Step 6: Save Results
output_path = "anomaly_detection_results.csv"
scaled_elevator_data.to_csv(output_path, index=False)
print(f"Anomaly detection results saved to {output_path}")
