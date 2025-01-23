# Predictive Maintenance for IoT Devices

## Project Overview
This project implements a predictive maintenance system for IoT devices using machine learning models. It focuses on:
- **CMAPSS Dataset**: Predicting Remaining Useful Life (RUL) for industrial equipment.
- **Elevator Dataset**: Anomaly detection for elevator sensors.
- **Environmental Sensor Data**: Identifying unusual patterns in IoT telemetry data.
- **Smoke Detection Dataset**: Classifying fire alarm triggers based on sensor readings.

---

## Repository Structure
```
predictive-maintenance-iot/
├── data/                  # Placeholder for datasets (add download instructions below)
├── notebooks/             # Jupyter notebooks for exploratory work (if applicable)
├── src/                   # Source code for modeling and utilities
│   ├── cmapss_regression.py  # Linear regression model for CMAPSS dataset
│   ├── data_preprocessing.py # Code for cleaning/normalizing datasets (optional)
│   ├── advanced_models.py    # Advanced modeling scripts (if needed later)
├── outputs/               # Outputs like feature importance CSVs or saved models
├── README.md              # Documentation for the project
├── requirements.txt       # Python dependencies for the project
└── LICENSE                # License file (optional)
```

---

## Datasets
### 1. **CMAPSS Dataset**
- **Description**: Sensor data for industrial equipment, used for Remaining Useful Life (RUL) prediction.
- **Source**: NASA Prognostics Data Repository.
- **Download Link**: [CMAPSS Dataset](https://data.nasa.gov/).

### 2. **Elevator Dataset**
- **Description**: Elevator sensor data for predictive maintenance.
- **Source**: Kaggle or other repositories.
- **Download Instructions**: Add to `data/` folder.

### 3. **Environmental Sensor Data**
- **Description**: IoT telemetry data with features like CO levels, temperature, and humidity.
- **Download Link**: Public repositories or Kaggle datasets.

### 4. **Smoke Detection Dataset**
- **Description**: Sensor data for fire alarm classification.
- **Source**: Kaggle.
- **Download Instructions**: Place in `data/` folder.

---

## How to Run the Project

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ParthMore999/predictive-maintenance-iot.git
   cd predictive-maintenance-iot
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place datasets in the `data/` folder following the structure above.

### Running the Code

#### **Linear Regression for CMAPSS Dataset**
1. Navigate to the `src/` folder:
   ```bash
   cd src
   ```

2. Run the regression script:
   ```bash
   python cmapss_regression.py
   ```

---

## Results and Outputs
- **Model Performance**: Metrics like MAE, MSE, and R² are printed for the CMAPSS regression model.
- **Feature Importance**: A CSV file `feature_importance_cmapss.csv` is generated in the project directory.

---

## Contributions
Contributions are welcome! If you encounter issues or have suggestions, feel free to open an issue or submit a pull request.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact
For questions or support, contact:
- **Your Name**: [moreparth999.com]
- **GitHub**: [https://github.com/ParthMore999]

