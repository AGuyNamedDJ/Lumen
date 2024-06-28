import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import os

# Define paths
data_path = os.path.join(os.path.dirname(
    __file__), '../data/processed/preprocessed_spx_data.csv')
model_save_path = 'trained_model.keras'

# Load the preprocessed data
data = pd.read_csv(data_path)

# Ensure 'timestamp' column is parsed as datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Select features and target
X = data.drop(['close', 'timestamp'], axis=1)  # Features
y = data['close']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Reshape the data for LSTM [samples, time steps, features]
X_train = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define the model architecture
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define the checkpoint to save the model
checkpoint = ModelCheckpoint(
    model_save_path, monitor='val_loss', save_best_only=True, mode='min')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test), callbacks=[checkpoint])

print("Model trained and saved successfully.")
