import os
import numpy as np
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
    raise FileNotFoundError(f"The directory {FEATURED_DATA_DIR} does not exist.")

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
    cpi_data = pd.read_csv(os.path.join(FEATURED_DATA_DIR, 'featured_cpi_data_featured.csv'))
    gdp_data = pd.read_csv(os.path.join(FEATURED_DATA_DIR, 'featured_gdp_data_featured.csv'))
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
    ppi_data = pd.read_csv(os.path.join(FEATURED_DATA_DIR, 'featured_ppi_data_featured.csv'))
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

def prepare_data(df):
    """
    Prepares the data for training by removing unnecessary columns and 
    standardizing the features for LSTM compatibility.
    """
    # Remove unwanted columns like 'id', 'timestamp', and keep only numeric columns
    columns_to_keep = df.select_dtypes(include=[np.number]).columns.tolist()  # Keep only numeric columns
    
    # Drop the datetime/timestamp column if it exists
    if 'date' in df.columns or 'timestamp' in df.columns:
        df = df.drop(columns=['date', 'timestamp'], errors='ignore')

    # Fill missing values in X and y with forward fill and drop any rows that still contain NaNs
    df.fillna(method='ffill', inplace=True)  # Fill NaN values forward
    df.dropna(inplace=True)  # Ensure no NaN values remain after forward fill
    
    # Prepare features (X) and target (y)
    X = df[columns_to_keep].values  # Features
    if 'close' in df.columns:
        y = df['close'].values  # Target (price column)
    else:
        y = df[df.columns[0]].values  # Fallback target column

    # Ensure X and y have the same length
    if len(X) != len(y):
        min_length = min(len(X), len(y))
        X, y = X[:min_length], y[:min_length]

    # Standardize features (X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape for LSTM, with dimensions (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y

def merge_data(data_dict, key_column='date', is_real_time=False):
    combined_df = None
    
    for key, df in data_dict.items():
        if key_column in df.columns:
            # Check if there's overlap in the key column
            print(f"First few {key_column} values in {key}: {df[key_column].head()}")  # Debug
            
            # Convert to datetime if needed
            if df[key_column].dtype != 'datetime64[ns]':
                df[key_column] = pd.to_datetime(df[key_column], errors='coerce')
            
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(
                    combined_df, 
                    df, 
                    on=key_column, 
                    how='left', 
                    suffixes=('', f'_{key}')
                )
                print(f"Combined DataFrame shape after merging {key}: {combined_df.shape}")  # Debug
    
    if combined_df is not None:
        # Check missing data in each column
        missing_data = combined_df.isnull().mean() * 100
        print("Percentage of missing values per column before dropping columns:")
        print(missing_data)
        
        # Impute missing data (forward-fill and backward-fill)
        combined_df.fillna(method='ffill', inplace=True)
        combined_df.fillna(method='bfill', inplace=True)
        
        # Check remaining columns that still have missing data
        remaining_missing_data = combined_df.isnull().mean() * 100
        print("Remaining missing values after forward/backward fill:")
        print(remaining_missing_data)
        
        # Drop columns with excessive missing data
        threshold = 0.7  # Keep columns with at least 70% non-NaN values
        combined_df.dropna(axis=1, thresh=int(threshold * combined_df.shape[0]), inplace=True)
        
        print(f"Final Combined DataFrame shape after handling NaNs: {combined_df.shape}")
    
        # Optionally save the combined DataFrame to CSV for further inspection
        combined_df.to_csv('final_combined_data.csv', index=False)
        print("Final combined data saved to 'final_combined_data.csv' for inspection.")
    
    return combined_df
    

def train_model(X_train, X_test, y_train, y_test):
    """
    Trains the hybrid model on the given training data.
    """
    logging.debug("Starting model training...")

    # Create the hybrid model (CNN + LSTM + Transformer)
    model = create_hybrid_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary() 

    # Debugging: Print model summary and inspect the shapes
    model.summary()

    # Set up model checkpointing (save the best model)
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_NAME + '.keras'),
                                 save_best_only=True, monitor='val_loss', mode='min')

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stopping])

    logging.debug("Model training complete.")
    return model

def main():
    # Load the data
    data_dict = load_data()

    # Merge historical data on 'date' column
    combined_historical_df = merge_data(data_dict, key_column='date', is_real_time=False)
    if combined_historical_df.empty:
        print("No historical data available after merging.")
        return

    # **Add this to check the first few rows of the historical data**
    print("First few rows of combined historical data:")
    print(combined_historical_df.head())  # Check the first few rows of historical data
    combined_historical_df.to_csv('combined_historical_data.csv', index=False)  # Save to CSV for further inspection

    # Prepare data for training (X, y for historical)
    X_hist, y_hist = prepare_data(combined_historical_df)
    if X_hist is None or y_hist is None:
        print("Historical data preparation failed.")
        return

    # Merge real-time data on 'timestamp' column
    combined_real_time_df = merge_data(data_dict, key_column='timestamp', is_real_time=True)
    if combined_real_time_df.empty:
        print("No real-time data available after merging.")
        return

    # **Add this to check the first few rows of the real-time data**
    print("First few rows of combined real-time data:")
    print(combined_real_time_df.head())  # Check the first few rows of real-time data
    combined_real_time_df.to_csv('combined_real_time_data.csv', index=False)  # Save to CSV for further inspection

    # Prepare data for training (X, y for real-time)
    X_real_time, y_real_time = prepare_data(combined_real_time_df)
    if X_real_time is None or y_real_time is None:
        print("Real-time data preparation failed.")
        return

    # Split into train and test sets for historical data
    X_train_hist, X_test_hist, y_train_hist, y_test_hist = train_test_split(X_hist, y_hist, test_size=0.2, random_state=42)

    # Save the test data for evaluation later
    np.save(os.path.join(MODEL_DIR, 'X_test_hist.npy'), X_test_hist)
    np.save(os.path.join(MODEL_DIR, 'y_test_hist.npy'), y_test_hist)

    # Split into train and test sets for real-time data
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real_time, y_real_time, test_size=0.2, random_state=42)

    # Save the real-time test data for evaluation later
    np.save(os.path.join(MODEL_DIR, 'X_test_real.npy'), X_test_real)
    np.save(os.path.join(MODEL_DIR, 'y_test_real.npy'), y_test_real)

    logging.debug("Models trained successfully on both historical and real-time data.")

if __name__ == "__main__":
    main()