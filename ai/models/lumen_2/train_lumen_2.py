import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Layer
from definitions_lumen_2 import create_hybrid_model
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model directory and name
MODEL_DIR = os.path.join(BASE_DIR, 'models')
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

# Load preprocessed data from the 'featured' directory


def load_data():
    consumer_confidence_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_consumer_confidence_data_featured.csv'))
    consumer_sentiment_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_consumer_sentiment_data_featured.csv'))
    core_inflation_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_core_inflation_data_featured.csv'))
    cpi_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_cpi_data_featured.csv'))
    gdp_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_gdp_data_featured.csv'))
    industrial_production_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_industrial_production_data_featured.csv'))
    interest_rate_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_interest_rate_data_featured.csv'))
    labor_force_participation_rate_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_labor_force_participation_rate_data_featured.csv'))
    nonfarm_payroll_employment_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_nonfarm_payroll_employment_data_featured.csv'))
    personal_consumption_expenditures_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_personal_consumption_expenditures_data_featured.csv'))
    ppi_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_ppi_data_featured.csv'))
    real_time_spx = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_real_time_spx_featured.csv'))
    real_time_spy = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_real_time_spy_featured.csv'))
    real_time_vix = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_real_time_vix_featured.csv'))
    unemployment_rate_data = pd.read_csv(os.path.join(
        FEATURED_DATA_DIR, 'featured_unemployment_rate_data_featured.csv'))

    return {
        'consumer_confidence_data': consumer_confidence_data,
        'consumer_sentiment_data': consumer_sentiment_data,
        'core_inflation_data': core_inflation_data,
        'cpi_data': cpi_data,
        'gdp_data': gdp_data,
        'historical_spx': historical_spx,
        'historical_spy': historical_spy,
        'historical_vix': historical_vix,
        'industrial_production_data': industrial_production_data,
        'interest_rate_data': interest_rate_data,
        'labor_force_participation_rate_data': labor_force_participation_rate_data,
        'nonfarm_payroll_employment_data': nonfarm_payroll_employment_data,
        'personal_consumption_expenditures_data': personal_consumption_expenditures_data,
        'ppi_data': ppi_data,
        'real_time_spx': real_time_spx,
        'real_time_spy': real_time_spy,
        'real_time_vix': real_time_vix,
        'unemployment_rate_data': unemployment_rate_data,
    }


def prepare_data(df, expected_features, target_column='close', sequence_length=30):
    """
    Prepares the data for training by creating sequences of a specified length,
    removing unnecessary columns, and standardizing the features.
    """
    # Remove unwanted columns and keep only numeric columns
    columns_to_keep = df.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure the target column exists
    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in the data!")

    # Remove the target column from features
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

    # Standardize features (fit scaler on training data)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Save the scaler for later use (e.g., in testing)
    scaler_filename = f'scaler_{target_column}.joblib'
    joblib.dump(scaler, scaler_filename)

    # Create sequences
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Log shapes of X and y to debug
    print(f"X shape: {X_sequences.shape}, y shape: {y_sequences.shape}")

    return X_sequences, y_sequences


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
            filename = 'combined_real_time_data.csv'
        else:
            filename = 'combined_historical_data.csv'
        combined_df.to_csv(filename, index=False)
        print(f"Final combined data saved to '{filename}' for inspection.")

        # Also save the combined data as 'final_combined_data.csv' for overall inspection
        combined_df.to_csv('final_combined_data.csv', index=False)
        print("Final combined data saved to 'final_combined_data.csv' for inspection.")

    return combined_df


def train_on_data(X_train, X_test, y_train, y_test, input_shape, model_name_suffix):
    """
    Train on the given data (historical or real-time) and return the trained model.
    """
    model = create_hybrid_model(input_shape=input_shape)

    # Set up model checkpointing and early stopping
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, f'{MODEL_NAME}_{model_name_suffix}.keras'),
                                 save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stopping])

    return model


def main():
    # Load the data
    data_dict = load_data()

    # Separate historical and real-time data keys
    historical_keys = {k: v for k, v in data_dict.items()
                       if 'historical' in k or k.startswith('historical')}
    real_time_keys = {k: v for k, v in data_dict.items()
                      if 'real_time' in k or k.startswith('real_time')}

    # Merge historical data on 'date' column
    combined_historical_df = merge_data(
        historical_keys, key_column='date', is_real_time=False)
    if combined_historical_df is None or combined_historical_df.empty:
        print("No historical data available after merging.")
        return

    # Prepare data for training (X, y for historical) - use 'close' for historical data
    X_hist, y_hist = prepare_data(
        combined_historical_df, expected_features=41, target_column='close', sequence_length=30)
    if X_hist is None or y_hist is None:
        print("Historical data preparation failed.")
        return

    # Merge real-time data on 'timestamp' column
    combined_real_time_df = merge_data(
        real_time_keys, key_column='timestamp', is_real_time=True)
    if combined_real_time_df is None or combined_real_time_df.empty:
        print("No real-time data available after merging.")
        return

    # Prepare data for training (X, y for real-time) - use 'current_price' for real-time data
    X_real_time, y_real_time = prepare_data(
        combined_real_time_df, expected_features=36, target_column='current_price', sequence_length=30)
    if X_real_time is None or y_real_time is None:
        print("Real-time data preparation failed.")
        return

    # Split into train and test sets for historical data
    X_train_hist, X_test_hist, y_train_hist, y_test_hist = train_test_split(
        X_hist, y_hist, test_size=0.2, random_state=42)

    # Split into train and test sets for real-time data
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real_time, y_real_time, test_size=0.2, random_state=42)

    # Train on historical data
    print("Training on historical data...")
    historical_model = train_on_data(X_train_hist, X_test_hist, y_train_hist, y_test_hist,
                                     input_shape=(X_train_hist.shape[1], X_train_hist.shape[2]), model_name_suffix='historical')

    # Train on real-time data
    print("Training on real-time data...")
    real_time_model = train_on_data(X_train_real, X_test_real, y_train_real, y_test_real,
                                    input_shape=(X_train_real.shape[1], X_train_real.shape[2]), model_name_suffix='real_time')

    print("Both models trained successfully.")


if __name__ == "__main__":
    main()
