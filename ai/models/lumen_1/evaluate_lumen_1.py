import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load preprocessed data
data_path = os.path.join(os.path.dirname(
    __file__), '../data/processed/preprocessed_spx_data.csv')
data = pd.read_csv(data_path)

# Features and labels
# Features (exclude target variable and date)
X = data.drop(['close', 'timestamp'], axis=1)
y = data['close']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Reshape input to be 3D [samples, timesteps, features]
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Load the trained model
model = load_model('trained_model.keras')

# Make predictions
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print the first 10 predicted vs actual values
print("Predicted vs Actual:")
for i in range(10):
    print(f"Predicted: {y_pred[i][0]}, Actual: {y_test.iloc[i]}")
