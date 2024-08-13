import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from definitions_lumen_2 import create_hybrid_model
import logging
from preprocess import create_features_for_average_hourly_earnings


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Paths
DATA_DIR = '/Users/agndj/Desktop/Coding/Lumen/ai/data/lumen_2/processed'
MODEL_DIR = 'models'
MODEL_NAME = 'lumen_2'


# Load preprocessed data


def load_data():
    print(f"Loading data from directory: {DATA_DIR}")  # Debugging output
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
    historical_spx = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_historical_spx.csv'))
    historical_spy = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_historical_spy.csv'))
    historical_vix = pd.read_csv(os.path.join(
        DATA_DIR, 'processed_historical_vix.csv'))
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

    # No merging is performed here; data is returned as individual DataFrames
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


# Feature Engineering

def engineer_features(df):
    logging.debug("Starting feature engineering...")

    # Calculate additional features specific to average hourly earnings
    df = create_features_for_average_hourly_earnings(df)

    # Drop rows with missing values (if any)
    df = df.dropna()

    # Log shape of data
    logging.debug(f"Feature engineering complete. Data shape: {df.shape}")

    return df


# Technical Indicators
# Statistics
# Lagged Features
# Economic FRED
# Volume & Volatility
# Price scaling
# Time based
# Sentiment
# real time
# custom


# Prepare data for LSTM


def prepare_data(df):
    X = df.drop(['close_spx', 'timestamp'], axis=1).values  # Features
    y = df['close_spx'].values  # Target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reshape for LSTM
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    return X, y

# Train the model


def train_model(X_train, X_test, y_train, y_test):
    logging.debug("Starting model training...")

    # Create LSTM model
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

# Main training script


def main():
    df = load_data()
    df = engineer_features(df)
    X, y = prepare_data(df)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train the model
    trained_model = train_model(X_train, X_test, y_train, y_test)

    # Save model details or additional information if needed


def test_engineer_features():
    df = load_data()

    # Focus on the average hourly earnings data for testing
    df_ahe = df[['date', 'value']]  # assuming these columns exist in your data

    # Apply feature engineering
    df_ahe = engineer_features(df_ahe)

    # Display the first few rows to verify the feature engineering
    print(df_ahe.head())


if __name__ == "__main__":
    test_engineer_features()
