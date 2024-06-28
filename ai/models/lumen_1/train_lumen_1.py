import pandas as pd
from sklearn.model_selection import train_test_split
from model_definitions import create_lstm_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Load preprocessed data
data_path = 'data/processed/preprocessed_spx_data.csv'
data = pd.read_csv(data_path)

# Split into features and target
X = data.drop(['close', 'timestamp'], axis=1).values  # Features
y = data['close'].values  # Target

# Reshape for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the model with hyperparameters
num_layers = 3
num_neurons = 100
model = create_lstm_model(
    (X_train.shape[1], X_train.shape[2]), num_layers=num_layers, num_neurons=num_neurons)

# Set up model checkpointing
checkpoint = ModelCheckpoint(
    'trained_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test), callbacks=[checkpoint])

print("Model training complete.")
