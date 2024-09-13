import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the combined historical data
historical_df = pd.read_csv('combined_historical_data.csv')
real_time_df = pd.read_csv('combined_real_time_data.csv')

# Function to prepare data for LSTM


def prepare_data(df, target_column, expected_features, sequence_length=30):
    # Select numeric columns
    columns_to_keep = df.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure the target column exists
    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in the data!")

    # Remove the target column from features if present
    if target_column in columns_to_keep:
        columns_to_keep.remove(target_column)

    # Limit the number of features if necessary
    if len(columns_to_keep) > expected_features:
        columns_to_keep = columns_to_keep[:expected_features]

    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)  # Fill any remaining NaNs with 0

    # Extract features and target
    X = df[columns_to_keep].values
    y = df[target_column].values

    # Load the scaler used during training
    scaler_filename = f'scaler_{target_column}.joblib'
    scaler = joblib.load(scaler_filename)
    X = scaler.transform(X)

    # Create sequences of the specified length
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    return X_sequences, y_sequences


# Prepare historical and real-time data
X_hist, y_hist = prepare_data(
    historical_df, target_column='close', expected_features=41)
X_real_time, y_real_time = prepare_data(
    real_time_df, target_column='current_price', expected_features=36)

# Split data into train and test sets
X_train_hist, X_test_hist, y_train_hist, y_test_hist = train_test_split(
    X_hist, y_hist, test_size=0.2, random_state=42)
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real_time, y_real_time, test_size=0.2, random_state=42)

# Save the test sets as .npy files
np.save('models/X_test_hist.npy', X_test_hist)
np.save('models/y_test_hist.npy', y_test_hist)
np.save('models/X_test_real.npy', X_test_real)
np.save('models/y_test_real.npy', y_test_real)

print("Test data saved successfully!")
