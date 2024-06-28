import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

# Define paths for data and model
data_path = os.path.join(os.path.dirname(
    __file__), '../data/processed/preprocessed_spx_data.csv')
model_path = 'trained_model.keras'

# Load the preprocessed data
data = pd.read_csv(data_path)

# Ensure 'timestamp' column is parsed as datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Select features and target
X = data.drop(['close', 'timestamp'], axis=1)  # Features
y = data['close']  # Target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Reshape the testing data for LSTM [samples, time steps, features]
X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

# Load the trained model
model = load_model(model_path)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate mean squared error for evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print the first 10 predictions vs actual values for comparison
print("Predicted vs Actual:")
for i in range(10):
    print(f"Predicted: {y_pred[i][0]}, Actual: {y_test.iloc[i]}")
