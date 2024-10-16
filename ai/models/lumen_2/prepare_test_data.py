import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the correct paths to your files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURED_DIR = os.path.join(BASE_DIR, '../../data/lumen_2/featured')

# Define the path to the model directory (for scalers)
MODEL_DIR = os.path.join(BASE_DIR, '../../models/lumen_2')

# Define paths to all datasets
data_paths = {
    'consumer_confidence': os.path.join(FEATURED_DIR, 'featured_consumer_confidence_data_featured.csv'),
    'consumer_sentiment': os.path.join(FEATURED_DIR, 'featured_consumer_sentiment_data_featured.csv'),
    'core_inflation': os.path.join(FEATURED_DIR, 'featured_core_inflation_data_featured.csv'),
    'cpi': os.path.join(FEATURED_DIR, 'featured_cpi_data_featured.csv'),
    'gdp': os.path.join(FEATURED_DIR, 'featured_gdp_data_featured.csv'),
    'industrial_production': os.path.join(FEATURED_DIR, 'featured_industrial_production_data_featured.csv'),
    'interest_rate': os.path.join(FEATURED_DIR, 'featured_interest_rate_data_featured.csv'),
    'labor_force': os.path.join(FEATURED_DIR, 'featured_labor_force_participation_rate_data_featured.csv'),
    'nonfarm_payroll': os.path.join(FEATURED_DIR, 'featured_nonfarm_payroll_employment_data_featured.csv'),
    'personal_consumption': os.path.join(FEATURED_DIR, 'featured_personal_consumption_expenditures_data_featured.csv'),
    'ppi': os.path.join(FEATURED_DIR, 'featured_ppi_data_featured.csv'),
    'unemployment_rate': os.path.join(FEATURED_DIR, 'featured_unemployment_rate_data_featured.csv'),
    'historical_spx': os.path.join(FEATURED_DIR, 'combined_historical_data.csv'),
    'real_time_spx': os.path.join(FEATURED_DIR, 'combined_real_time_data.csv')
}

# Load the corresponding scalers
scaler_names = {
    'consumer_confidence': 'consumer_confidence_data_feature_scaler',
    'consumer_sentiment': 'consumer_sentiment_data_feature_scaler',
    'core_inflation': 'core_inflation_data_feature_scaler',
    'cpi': 'cpi_data_feature_scaler',
    'gdp': 'gdp_data_feature_scaler',
    'industrial_production': 'industrial_production_data_feature_scaler',
    'interest_rate': 'interest_rate_data_feature_scaler',
    'labor_force': 'labor_force_participation_rate_data_feature_scaler',
    'nonfarm_payroll': 'nonfarm_payroll_employment_data_feature_scaler',
    'personal_consumption': 'personal_consumption_expenditures_data_feature_scaler',
    'ppi': 'ppi_data_feature_scaler',
    'unemployment_rate': 'unemployment_rate_data_feature_scaler',
    'historical_spx': 'historical_spx_feature_scaler',
    'real_time_spx': 'real_time_spx_feature_scaler'
}

# Load each dataset
dataframes = {name: pd.read_csv(path) for name, path in data_paths.items()}

# Function to prepare data for LSTM


# Function to prepare data for LSTM
def prepare_data(df, target_column, expected_features, scaler_name, sequence_length=30):
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
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)  # Fill any remaining NaNs with 0

    # Extract features and target
    X = df[columns_to_keep].values
    y = df[target_column].values

    # Load the specific scaler used for this dataset
    scaler_filename = os.path.join(MODEL_DIR, f'{scaler_name}.joblib')
    scaler = joblib.load(scaler_filename)

    # Add debugging information
    print(f"Processing {scaler_name}: X has {
          X.shape[1]} features, scaler expects {scaler.n_features_in_}")

    # Apply the scaler
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


# Prepare data for all datasets
prepared_data = {}
for name, df in dataframes.items():
    if name == 'historical_spx':
        target_column = 'close'
        expected_features = 41  # Update to 41
    elif name == 'real_time_spx':
        target_column = 'current_price'
        expected_features = 36  # Update to 36
    elif name == 'consumer_confidence':
        target_column = 'value'
        expected_features = 31
    elif name == 'consumer_sentiment':
        target_column = 'value'
        expected_features = 10
    elif name == 'core_inflation':
        target_column = 'value'
        expected_features = 22
    elif name == 'cpi':
        target_column = 'value'
        expected_features = 23
    elif name == 'gdp':
        target_column = 'value'
        expected_features = 35
    elif name == 'industrial_production':
        target_column = 'value'
        expected_features = 42
    elif name == 'interest_rate':
        target_column = 'value'
        expected_features = 41
    elif name == 'labor_force':
        target_column = 'value'
        expected_features = 39
    elif name == 'nonfarm_payroll':
        target_column = 'value'
        expected_features = 39
    elif name == 'personal_consumption':
        target_column = 'value'
        expected_features = 41
    elif name == 'ppi':
        target_column = 'value'
        expected_features = 41
    elif name == 'unemployment_rate':
        target_column = 'value'
        expected_features = 28
    else:
        raise ValueError(f"Unknown dataset: {name}")

    prepared_data[name] = prepare_data(
        df, target_column=target_column, expected_features=expected_features, scaler_name=scaler_names[name])

# Split data into train and test sets (example for historical data)
X_train_hist, X_test_hist, y_train_hist, y_test_hist = train_test_split(
    prepared_data['historical_spx'][0], prepared_data['historical_spx'][1], test_size=0.2, random_state=42)

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    prepared_data['real_time_spx'][0], prepared_data['real_time_spx'][1], test_size=0.2, random_state=42)

print("Test data saved successfully!")
