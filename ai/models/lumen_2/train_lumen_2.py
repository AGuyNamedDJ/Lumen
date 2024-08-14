import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from definitions_lumen_2 import create_hybrid_model
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths from environment variables
DATA_DIR = os.getenv('DATA_DIR', 'data/lumen_2/processed')
MODEL_DIR = os.getenv('MODEL_DIR', 'models/lumen_2')
MODEL_NAME = 'lumen_2'

# Load historical SPX, SPY, and VIX data
historical_spx = pd.read_csv(os.path.join(
    DATA_DIR, 'processed_historical_spx.csv'))
historical_spy = pd.read_csv(os.path.join(
    DATA_DIR, 'processed_historical_spy.csv'))
historical_vix = pd.read_csv(os.path.join(
    DATA_DIR, 'processed_historical_vix.csv'))

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load preprocessed data


def load_data():
    average_hourly_earnings_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_average_hourly_earnings_data.csv'))
    consumer_confidence_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_consumer_confidence_data.csv'))
    consumer_sentiment_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_consumer_sentiment_data.csv'))
    core_inflation_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_core_inflation_data.csv'))
    cpi_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_cpi_data.csv'))
    gdp_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_gdp_data.csv'))
    industrial_production_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_industrial_production_data.csv'))
    interest_rate_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_interest_rate_data.csv'))
    labor_force_participation_rate_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_labor_force_participation_rate_data.csv'))
    nonfarm_payroll_employment_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_nonfarm_payroll_employment_data.csv'))
    personal_consumption_expenditures_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_personal_consumption_expenditures.csv'))
    ppi_data = pd.read_csv(os.path.join(DATA_DIR, 'processed_ppi_data.csv'))
    real_time_spx = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_real_time_spx.csv'))
    real_time_spy = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_real_time_spy.csv'))
    real_time_vix = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_real_time_vix.csv'))
    unemployment_rate_data = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_unemployment_rate_data.csv'))

    return {
        'average_hourly_earnings_data': average_hourly_earnings_data,
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


# Feature Engineering
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def recompute_indicators(df, tf, has_close=True):
    # Timeframe dictionary to define rolling windows
    timeframe = {
        '1min': {'SMA_20': 20, 'SMA_50': 50, 'SMA_100': 100, 'EMA_12': 12, 'EMA_26': 26, 'RSI': 14},
        '5min': {'SMA_20': 100, 'SMA_50': 250, 'SMA_100': 500, 'EMA_12': 60, 'EMA_26': 130, 'RSI': 14},
        '30min': {'SMA_20': 600, 'SMA_50': 1500, 'SMA_100': 3000, 'EMA_12': 360, 'EMA_26': 780, 'RSI': 14},
        '1hour': {'SMA_20': 1200, 'SMA_50': 3000, 'SMA_100': 6000, 'EMA_12': 720, 'EMA_26': 1560, 'RSI': 14},
        'daily': {'SMA_20': 20, 'SMA_50': 50, 'SMA_100': 100, 'EMA_12': 12, 'EMA_26': 26, 'RSI': 14},
        'weekly': {'SMA_20': 104, 'SMA_50': 260, 'SMA_100': 520, 'EMA_12': 52, 'EMA_26': 104, 'RSI': 14},
        'monthly': {'SMA_20': 240, 'SMA_50': 600, 'SMA_100': 1200, 'EMA_12': 240, 'EMA_26': 520, 'RSI': 14}
    }

    # Get the appropriate settings for the timeframe
    settings = timeframe.get(tf, timeframe['daily'])

    # List of indicators to calculate
    indicators = []

    # Choose the appropriate price column based on whether 'close' exists
    price_column = 'close' if has_close else 'current_price'

    # Debug: print before processing
    print("Before indicator calculation:")
    print(df.head())

    # Ensure rolling operations have enough data
    if len(df) < settings['SMA_20']:
        print("Not enough data to calculate indicators for this timeframe.")
        return df

    # Calculate indicators and append to the list using the timeframe settings
    indicators.append(df[price_column].rolling(
        window=settings['SMA_20']).mean().rename('SMA_20'))
    indicators.append(df[price_column].rolling(
        window=settings['SMA_50']).mean().rename('SMA_50'))
    indicators.append(df[price_column].rolling(
        window=settings['SMA_100']).mean().rename('SMA_100'))
    indicators.append(df[price_column].ewm(
        span=settings['EMA_12'], adjust=False).mean().rename('EMA_12'))
    indicators.append(df[price_column].ewm(
        span=settings['EMA_26'], adjust=False).mean().rename('EMA_26'))
    indicators.append((df[price_column].rolling(window=settings['SMA_20']).mean(
    ) + 2 * df[price_column].rolling(window=settings['SMA_20']).std()).rename('Bollinger_Upper'))
    indicators.append((df[price_column].rolling(window=settings['SMA_20']).mean(
    ) - 2 * df[price_column].rolling(window=settings['SMA_20']).std()).rename('Bollinger_Lower'))
    indicators.append(calculate_rsi(
        df[price_column], period=settings['RSI']).rename('RSI'))

    # Concatenate all indicators at once using pd.concat with batching
    batch_size = 4  # Number of indicators to concatenate in each batch
    for i in range(0, len(indicators), batch_size):
        df = pd.concat([df] + indicators[i:i + batch_size], axis=1)

    # Debug: print after processing
    print("After indicator calculation:")
    print(df.head())

    return df.dropna()  # Drop rows with NaNs after recalculating indicators


def engineer_features(df, tf='daily', has_close=True):
    logging.debug("Starting feature engineering...")

    # Adjust the price column based on whether 'close' exists or not
    price_column = 'close' if has_close else 'current_price'

    # Ensure missing columns exist and are initialized to default values
    if 'volume' not in df.columns:
        df['volume'] = 0.0  # Default value for volume
    if 'conditions' not in df.columns:
        df['conditions'] = ''  # Default value for conditions (empty string)

    # Print the DataFrame before recomputing indicators for debugging
    print("DataFrame before recomputing indicators:")
    print(df.head())  # Debugging step

    # Check for NaNs in the key column before processing
    if df[price_column].isna().all():
        print(f"All values in {price_column} are NaN, skipping recompute.")
        return df  # Skip recomputation if there's no valid data

    # Recompute indicators based on the timeframe, ensuring that we have enough data
    df = recompute_indicators(df, tf, has_close=has_close)

    # Print the DataFrame after recomputing indicators for debugging
    print("DataFrame after recomputing indicators, before ffill:")
    print(df.head())  # Debugging step

    # Forward fill any remaining NaNs and drop rows that are still NaN
    df = df.ffill().dropna(subset=[price_column])

    # Print the DataFrame after filling and dropping NaNs for debugging
    print("DataFrame after ffill and dropna:")
    print(df.head())  # Debugging step

    # Log shape of data after feature engineering
    logging.debug(f"Feature engineering complete. Data shape: {df.shape}")

    return df


def prepare_data(df):
    # Only drop columns if they exist in the DataFrame
    # Dropping 'id' and 'volume' for training purposes
    columns_to_drop = ['id', 'volume']

    X = df.drop(columns_to_drop, axis=1).fillna(
        method='ffill').dropna().values  # Features
    y = df['close'].values  # Target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape for LSTM
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y


def train_model(X_train, X_test, y_train, y_test):
    logging.debug("Starting model training...")

    # Create hybrid model
    model = create_hybrid_model(input_shape=(
        X_train.shape[1], X_train.shape[2]))

    # Set up model checkpointing
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_NAME + '.h5'),
                                 save_best_only=True, monitor='val_loss', mode='min')

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stopping])

    logging.debug("Model training complete.")
    return model


def main():
    data_dict = load_data()

    # Process historical SPX, SPY, and VIX data
    df_spx = engineer_features(data_dict['historical_spx'])
    df_spy = engineer_features(data_dict['historical_spy'])
    df_vix = engineer_features(data_dict['historical_vix'])

    # Process real-time SPX, SPY, and VIX data
    real_time_spx = engineer_features(
        data_dict['real_time_spx'], tf='1min', has_close=False)
    real_time_spy = engineer_features(
        data_dict['real_time_spy'], tf='1min', has_close=False)
    real_time_vix = engineer_features(
        data_dict['real_time_vix'], tf='1min', has_close=True)

    # Optionally combine historical and real-time data
    combined_spx = pd.concat([df_spx, real_time_spx], ignore_index=True)
    combined_spy = pd.concat([df_spy, real_time_spy], ignore_index=True)
    combined_vix = pd.concat([df_vix, real_time_vix], ignore_index=True)

    # Prepare data for the model
    X_spx, y_spx = prepare_data(combined_spx)
    X_spy, y_spy = prepare_data(combined_spy)
    X_vix, y_vix = prepare_data(combined_vix)

    # Split into train and test sets for SPX
    X_train_spx, X_test_spx, y_train_spx, y_test_spx = train_test_split(
        X_spx, y_spx, test_size=0.2, random_state=42)

    # Split into train and test sets for SPY
    X_train_spy, X_test_spy, y_train_spy, y_test_spy = train_test_split(
        X_spy, y_spy, test_size=0.2, random_state=42)

    # Split into train and test sets for VIX
    X_train_vix, X_test_vix, y_train_vix, y_test_vix = train_test_split(
        X_vix, y_vix, test_size=0.2, random_state=42)

    # Train the models
    trained_model_spx = train_model(
        X_train_spx, X_test_spx, y_train_spx, y_test_spx)
    trained_model_spy = train_model(
        X_train_spy, X_test_spy, y_train_spy, y_test_spy)
    trained_model_vix = train_model(
        X_train_vix, X_test_vix, y_train_vix, y_test_vix)

    # Save model details or additional information if needed


def test_engineer_features():
    data_dict = load_data()

    # Access the specific DataFrames for SPX, SPY, and VIX data
    df_spx = data_dict['historical_spx']
    df_spy = data_dict['historical_spy']
    df_vix = data_dict['historical_vix']
    df_real_time_spx = data_dict['real_time_spx']
    df_real_time_spy = data_dict['real_time_spy']
    df_real_time_vix = data_dict['real_time_vix']

    # Print the columns to see what is available in historical data
    print("Historical SPX Columns in the DataFrame:", df_spx.columns)
    print("Historical SPY Columns in the DataFrame:", df_spy.columns)
    print("Historical VIX Columns in the DataFrame:", df_vix.columns)

    # Print the columns to see what is available in real-time data
    print("Real-Time SPX Columns in the DataFrame:", df_real_time_spx.columns)
    print("Real-Time SPY Columns in the DataFrame:", df_real_time_spy.columns)
    print("Real-Time VIX Columns in the DataFrame:", df_real_time_vix.columns)

    # Apply feature engineering separately on historical data
    df_spx = engineer_features(df_spx)
    df_spy = engineer_features(df_spy)
    df_vix = engineer_features(df_vix)

    # Apply feature engineering to real-time data
    df_real_time_spx = engineer_features(
        df_real_time_spx, tf='1min', has_close=False)
    df_real_time_spy = engineer_features(
        df_real_time_spy, tf='1min', has_close=False)
    df_real_time_vix = engineer_features(
        df_real_time_vix, tf='1min', has_close=True)

    # Display the first few rows to verify the feature engineering
    print("histroric spx", df_spx.head())
    print("historic spy", df_spy.head())
    print("historic vix", df_vix.head())
    print("real time spx", df_real_time_spx.head())
    print("real time spy", df_real_time_spy.head())
    print("real time vix", df_real_time_vix.head())


if __name__ == "__main__":
    test_engineer_features()
