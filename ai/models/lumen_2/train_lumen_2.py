import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from definitions_lumen_2 import create_hybrid_model
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model directory and name
MODEL_DIR = os.path.join(BASE_DIR, '..', '..', 'models', 'lumen_2')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_NAME = 'Lumen2'

# Correct path to the featured data directory
FEATURED_DATA_DIR = os.path.join(BASE_DIR, '../../data/lumen_2/featured')

# Ensure the directory exists
if not os.path.exists(FEATURED_DATA_DIR):
    raise FileNotFoundError(
        f"The directory {FEATURED_DATA_DIR} does not exist.")

# Debug: List the contents of the featured directory
print("Files in featured directory:", os.listdir(FEATURED_DATA_DIR))

# Load historical SPX, SPY, and VIX data from the 'featured' directory
historical_spx = pd.read_csv(os.path.join(
    FEATURED_DATA_DIR, 'featured_historical_spx_featured.csv'))
historical_spy = pd.read_csv(os.path.join(
    FEATURED_DATA_DIR, 'featured_historical_spy_featured.csv'))
historical_vix = pd.read_csv(os.path.join(
    FEATURED_DATA_DIR, 'featured_historical_vix_featured.csv'))

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Function to load data


def load_data():
    # Load economic indicator data with proper date parsing
    economic_data_files = [
        'featured_consumer_confidence_data_featured.csv',
        'featured_consumer_sentiment_data_featured.csv',
        'featured_core_inflation_data_featured.csv',
        'featured_cpi_data_featured.csv',
        'featured_gdp_data_featured.csv',
        'featured_industrial_production_data_featured.csv',
        'featured_interest_rate_data_featured.csv',
        'featured_labor_force_participation_rate_data_featured.csv',
        'featured_nonfarm_payroll_employment_data_featured.csv',
        'featured_personal_consumption_expenditures_data_featured.csv',
        'featured_ppi_data_featured.csv',
        'featured_unemployment_rate_data_featured.csv'
    ]

    economic_data = {}
    for filename in economic_data_files:
        name = filename.replace('featured_', '').replace('_featured.csv', '')
        df = pd.read_csv(
            os.path.join(FEATURED_DATA_DIR, filename),
            parse_dates=['date']
        )
        # Clean DataFrame
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        df.dropna(subset=['date'], inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        df.reset_index(drop=True, inplace=True)
        economic_data[name] = df

    # Load real-time data with proper date parsing
    real_time_data_files = [
        'featured_real_time_spx_featured.csv',
        'featured_real_time_spy_featured.csv',
        'featured_real_time_vix_featured.csv'
    ]

    real_time_data = {}
    for filename in real_time_data_files:
        name = filename.replace('featured_', '').replace('_featured.csv', '')
        df = pd.read_csv(
            os.path.join(FEATURED_DATA_DIR, filename),
            parse_dates=['timestamp']
        )
        # Clean DataFrame
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        df.dropna(subset=['timestamp'], inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        df.reset_index(drop=True, inplace=True)
        real_time_data[name] = df

    # Load historical data with proper date parsing
    historical_data_files = [
        'featured_historical_spx_featured.csv',
        'featured_historical_spy_featured.csv',
        'featured_historical_vix_featured.csv'
    ]

    historical_data = {}
    for filename in historical_data_files:
        name = filename.replace('featured_', '').replace('_featured.csv', '')
        df = pd.read_csv(
            os.path.join(FEATURED_DATA_DIR, filename),
            parse_dates=['date']
        )
        # Clean DataFrame
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        df.dropna(subset=['date'], inplace=True)
        df = df.loc[:, ~df.columns.duplicated()]
        df.reset_index(drop=True, inplace=True)
        historical_data[name] = df

    # Print the columns of historical DataFrames for verification
    print("historical_spx columns:", historical_data['historical_spx'].columns)
    print("historical_spy columns:", historical_data['historical_spy'].columns)
    print("historical_vix columns:", historical_data['historical_vix'].columns)

    # Combine all data into a single dictionary
    data_dict = {**economic_data, **real_time_data, **historical_data}

    return data_dict

# Function to load scalers


def load_scalers():
    scaler_files = [
        'consumer_confidence_data_feature_scaler.joblib',
        'consumer_sentiment_data_feature_scaler.joblib',
        'core_inflation_data_feature_scaler.joblib',
        'cpi_data_feature_scaler.joblib',
        'gdp_data_feature_scaler.joblib',
        'industrial_production_data_feature_scaler.joblib',
        'interest_rate_data_feature_scaler.joblib',
        'labor_force_participation_rate_data_feature_scaler.joblib',
        'nonfarm_payroll_employment_data_feature_scaler.joblib',
        'personal_consumption_expenditures_data_feature_scaler.joblib',
        'ppi_data_feature_scaler.joblib',
        'unemployment_rate_data_feature_scaler.joblib',
        'historical_spx_feature_scaler.joblib',
        'real_time_spx_feature_scaler.joblib'
    ]

    scalers = {}
    for filename in scaler_files:
        key = filename.replace('_feature_scaler.joblib', '')
        path = os.path.join('models', 'lumen_2', filename)
        if os.path.exists(path):
            scalers[key] = joblib.load(path)
        else:
            print(f"Scaler file {path} does not exist!")

    return scalers

# Function to apply scalers to data


def apply_scalers(data_dict, scalers):
    for key, df in data_dict.items():
        # Identify datetime and target columns
        datetime_columns = [col for col in [
            'timestamp', 'date'] if col in df.columns]
        target_columns = []
        if 'close' in df.columns:
            target_columns.append('close')
        if 'current_price' in df.columns:
            target_columns.append('current_price')

        # Preserve target values
        target_values = df[target_columns] if target_columns else pd.DataFrame(
            index=df.index)

        # Drop datetime and target columns
        df_features = df.drop(columns=datetime_columns +
                              target_columns, errors='ignore')

        # Proceed if scaler exists for the key
        scaler_key = key.replace('featured_', '').replace('_featured', '')
        if scaler_key in scalers:
            print(f"Scaling data for: {key}")
            print(f"DataFrame columns before scaling for {
                  key}: {df_features.columns}")

            # Get expected columns from scaler
            expected_columns = scalers[scaler_key].feature_names_in_

            # Align DataFrame columns with scaler's expected columns
            df_features = df_features[expected_columns]

            # Attempt to scale
            try:
                df_scaled = pd.DataFrame(scalers[scaler_key].transform(
                    df_features), columns=df_features.columns)
                # Combine datetime, scaled features, and target values
                df_final = pd.concat([df[datetime_columns].reset_index(drop=True),
                                      df_scaled.reset_index(drop=True),
                                      target_values.reset_index(drop=True)], axis=1)
                data_dict[key] = df_final
            except Exception as e:
                print(f"Error scaling {key}: {e}")
                continue

# Function to check for NaN and Inf values


def check_for_nan_inf(X, y, dataset_name):
    nan_in_X = np.isnan(X).any()
    inf_in_X = np.isinf(X).any()
    nan_in_y = np.isnan(y).any()
    inf_in_y = np.isinf(y).any()

    if nan_in_X:
        print(f"NaN values found in X of {dataset_name}")
    if inf_in_X:
        print(f"Infinite values found in X of {dataset_name}")
    if nan_in_y:
        print(f"NaN values found in y of {dataset_name}")
    if inf_in_y:
        print(f"Infinite values found in y of {dataset_name}")

# Function to summarize data


def summarize_data(X, y, dataset_name):
    print(f"Summary of {dataset_name}:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X min: {np.min(X)}, X max: {np.max(X)}")
    print(f"y min: {np.min(y)}, y max: {np.max(y)}")
    print(f"X mean: {np.mean(X)}, X std: {np.std(X)}")
    print(f"y mean: {np.mean(y)}, y std: {np.std(y)}\n")

# Function to prepare data for training


def prepare_data(df, target_column='close', sequence_length=30):
    """
    Prepares the data for training by creating sequences of a specified length,
    removing unnecessary columns, and standardizing the features.
    """
    df = df.drop(columns=['date', 'timestamp'], errors='ignore')

    # Ensure the target column exists
    if target_column in df.columns:
        target_values = df[target_column]
        df = df.drop(columns=[target_column])
    else:
        print(f"Target column '{target_column}' not found in DataFrame.")
        return None, None

    # Select only numeric columns
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feature_columns].values
    y = target_values.values

    # Create sequences
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    return np.array(X_sequences), np.array(y_sequences)

# Function to merge data


def merge_data(data_dict, key_column='date', is_real_time=False):
    combined_df = None

    for key, df in data_dict.items():
        # Print the first few rows of each DataFrame to inspect before merging
        print(f"First few rows of {key} before merging:")
        print(df.head())

        if key_column in df.columns:
            # Convert key column to datetime
            df[key_column] = pd.to_datetime(df[key_column], errors='coerce')

            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(
                    combined_df,
                    df,
                    on=key_column,
                    how='outer',  # Use outer join to include all data points
                    suffixes=('', f'_{key}')
                )
                print(f"Combined DataFrame shape after merging {
                      key}: {combined_df.shape}")
        else:
            print(f"Key column '{key_column}' not found in {key}, skipping.")

    if combined_df is not None:
        # Handle missing data
        combined_df.sort_values(by=key_column, inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        combined_df.fillna(method='ffill', inplace=True)
        combined_df.fillna(method='bfill', inplace=True)

        # Save the combined DataFrame to CSV
        if is_real_time:
            filename = os.path.join(
                FEATURED_DATA_DIR, 'combined_real_time_data.csv')
        else:
            filename = os.path.join(
                FEATURED_DATA_DIR, 'combined_historical_data.csv')
        combined_df.to_csv(filename, index=False)
        print(f"Final combined data saved to '{filename}' for inspection.")

        # Also save the combined data as 'final_combined_data.csv' for overall inspection
        combined_df.to_csv(os.path.join(FEATURED_DATA_DIR,
                           'final_combined_data.csv'), index=False)
        print("Final combined data saved to 'final_combined_data.csv' for inspection.")

    return combined_df

# Function to train the model


def train_on_data(X_train, X_test, y_train, y_test, input_shape, model_name_suffix):
    """
    Train on the given data (historical or real-time) and return the trained model.
    """
    model = create_hybrid_model(input_shape=input_shape)

    # Set up optimizer with a smaller learning rate and gradient clipping
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Set up model checkpointing and early stopping
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, f'{MODEL_NAME}_{model_name_suffix}.keras'),
                                 save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stopping],
              verbose=1)

    return model

# Main function


def main():
    # Load the data
    data_dict = load_data()

    # Load the scalers
    scalers = load_scalers()

    # Apply scalers
    apply_scalers(data_dict, scalers)

    # Separate historical and real-time data keys
    historical_keys = [k for k in data_dict.keys()
                       if 'historical' in k]
    real_time_keys = [k for k in data_dict.keys()
                      if 'real_time' in k]

    # Create dictionaries for historical and real-time data
    historical_data_dict = {k: data_dict[k] for k in historical_keys}
    real_time_data_dict = {k: data_dict[k] for k in real_time_keys}

    # Merge historical data on 'date' column
    combined_historical_df = merge_data(
        historical_data_dict, key_column='date', is_real_time=False)
    if combined_historical_df is None or combined_historical_df.empty:
        print("No historical data available after merging.")
        return

    # Check for NaNs in combined historical data
    nan_counts = combined_historical_df.isna().sum()
    if nan_counts.any():
        print("NaN counts in combined historical DataFrame:")
        print(nan_counts[nan_counts > 0])

    # Handle missing values if necessary
    combined_historical_df.dropna(inplace=True)

    # Prepare data for training (X, y for historical) - use 'close' for historical data
    X_hist, y_hist = prepare_data(
        combined_historical_df, target_column='close', sequence_length=30)
    if X_hist is None or y_hist is None:
        print("Historical data preparation failed.")
        return

    # Check for NaNs in X_hist and y_hist
    check_for_nan_inf(X_hist, y_hist, "Historical Data")

    # Summarize historical data
    summarize_data(X_hist, y_hist, "Historical Data")

    # Convert data types
    X_hist = X_hist.astype('float32')
    y_hist = y_hist.astype('float32')

    # Split into train and test sets for historical data
    X_train_hist, X_test_hist, y_train_hist, y_test_hist = train_test_split(
        X_hist, y_hist, test_size=0.2, random_state=42)

    # Verify Input Shapes
    print(f"X_train_hist shape: {X_train_hist.shape}")
    print(f"X_test_hist shape: {X_test_hist.shape}")

    # Train on historical data
    print("Training on historical data...")
    historical_model = train_on_data(X_train_hist, X_test_hist, y_train_hist, y_test_hist,
                                     input_shape=(X_train_hist.shape[1], X_train_hist.shape[2]), model_name_suffix='historical')

    # Merge real-time data on 'timestamp' column
    combined_real_time_df = merge_data(
        real_time_data_dict, key_column='timestamp', is_real_time=True)
    if combined_real_time_df is None or combined_real_time_df.empty:
        print("No real-time data available after merging.")
        return

    # Handle missing values if necessary
    combined_real_time_df.dropna(inplace=True)

    # Prepare data for training (X, y for real-time) - use 'current_price' for real-time data
    X_real_time, y_real_time = prepare_data(
        combined_real_time_df, target_column='current_price', sequence_length=30)
    if X_real_time is None or y_real_time is None:
        print("Real-time data preparation failed.")
        return

    # Check for NaNs in X_real_time and y_real_time
    check_for_nan_inf(X_real_time, y_real_time, "Real-Time Data")

    # Summarize real-time data
    summarize_data(X_real_time, y_real_time, "Real-Time Data")

    # Convert data types
    X_real_time = X_real_time.astype('float32')
    y_real_time = y_real_time.astype('float32')

    # Split into train and test sets for real-time data
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real_time, y_real_time, test_size=0.2, random_state=42)

    # Verify Input Shapes
    print(f"X_train_real shape: {X_train_real.shape}")
    print(f"X_test_real shape: {X_test_real.shape}")

    # Train on real-time data
    print("Training on real-time data...")
    real_time_model = train_on_data(X_train_real, X_test_real, y_train_real, y_test_real,
                                    input_shape=(X_train_real.shape[1], X_train_real.shape[2]), model_name_suffix='real_time')

    print("Both models trained successfully.")


if __name__ == "__main__":
    main()
