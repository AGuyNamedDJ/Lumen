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
    # Economic indicators
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
    # Historical data
    'historical_spx': os.path.join(FEATURED_DIR, 'featured_historical_spx_featured.csv'),
    'historical_spy': os.path.join(FEATURED_DIR, 'featured_historical_spy_featured.csv'),
    'historical_vix': os.path.join(FEATURED_DIR, 'featured_historical_vix_featured.csv'),
    # Real-time data
    'real_time_spx': os.path.join(FEATURED_DIR, 'featured_real_time_spx_featured.csv'),
    'real_time_spy': os.path.join(FEATURED_DIR, 'featured_real_time_spy_featured.csv'),
    'real_time_vix': os.path.join(FEATURED_DIR, 'featured_real_time_vix_featured.csv'),
}

# Load the corresponding scalers
scaler_names = {
    # Economic indicators
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
    # Historical data
    'historical_spx': 'historical_spx_feature_scaler',
    'historical_spy': 'historical_spy_feature_scaler',
    'historical_vix': 'historical_vix_feature_scaler',
    # Real-time data
    'real_time_spx': 'real_time_spx_feature_scaler',
    'real_time_spy': 'real_time_spy_feature_scaler',
    'real_time_vix': 'real_time_vix_feature_scaler',
}

# Load each dataset into a dictionary
dataframes = {}
for name, path in data_paths.items():
    # Determine if the file uses 'date' or 'timestamp' as the datetime column
    datetime_col = 'timestamp' if 'real_time' in name else 'date'
    df = pd.read_csv(path, parse_dates=[datetime_col])
    dataframes[name] = df

# Load scalers into a dictionary
scalers = {}
for key, scaler_name in scaler_names.items():
    scaler_path = os.path.join(MODEL_DIR, f'{scaler_name}.joblib')
    if os.path.exists(scaler_path):
        scalers[key] = joblib.load(scaler_path)
    else:
        print(f"Scaler file {scaler_path} does not exist!")

# Function to apply scalers and prepare dataframes


def apply_scalers(data_dict, scalers_dict):
    for key, df in data_dict.items():
        # Identify datetime and target columns
        datetime_columns = ['timestamp'] if 'real_time' in key else ['date']
        target_columns = []
        if 'close' in df.columns:
            target_columns.append('close')
        if 'current_price' in df.columns:
            target_columns.append('current_price')
        # Do NOT include 'value' in target_columns here

        # Preserve target values
        target_values = df[target_columns] if target_columns else pd.DataFrame(
            index=df.index)

        # Drop datetime and target columns
        df_features = df.drop(columns=datetime_columns +
                              target_columns, errors='ignore')

        # Proceed if scaler exists for the key
        scaler_key = key
        if scaler_key in scalers_dict:
            scaler = scalers_dict[scaler_key]
            # Align DataFrame columns with scaler's expected columns
            expected_columns = scaler.feature_names_in_
            df_features = df_features[expected_columns]

            # Apply scaling
            df_scaled = pd.DataFrame(scaler.transform(
                df_features), columns=df_features.columns)

            # Prefix feature column names with the dataset key
            df_scaled.columns = [f"{key}_{col}" for col in df_scaled.columns]

            # **Add this block to prefix target column names**
            if not target_values.empty:
                target_values.columns = [
                    f"{key}_{col}" for col in target_values.columns]

            # Combine datetime, scaled features, and target values
            df_final = pd.concat(
                [
                    df[datetime_columns].reset_index(drop=True),
                    df_scaled.reset_index(drop=True),
                    target_values.reset_index(drop=True),
                ],
                axis=1,
            )
            # Update the data_dict
            data_dict[key] = df_final
        else:
            print(f"Scaler for {key} not found.")
    return data_dict


# Apply scalers to the dataframes
dataframes = apply_scalers(dataframes, scalers)

# Function to merge dataframes on 'date' or 'timestamp'


def merge_dataframes(dataframes_dict, key_column='date'):
    from functools import reduce
    dataframes_list = list(dataframes_dict.values())
    combined_df = reduce(lambda left, right: pd.merge(
        left, right, on=key_column, how='outer'), dataframes_list)

    # Handle missing values
    combined_df.sort_values(by=key_column, inplace=True)
    combined_df.ffill(inplace=True)
    combined_df.bfill(inplace=True)
    return combined_df


# Separate historical and real-time dataframes
historical_keys = [k for k in dataframes.keys()
                   if 'historical' in k or 'consumer' in k or 'core_inflation' in k or 'cpi' in k or 'gdp' in k or 'industrial' in k or 'interest_rate' in k or 'labor_force' in k or 'nonfarm' in k or 'personal_consumption' in k or 'ppi' in k or 'unemployment_rate' in k]
real_time_keys = [k for k in dataframes.keys() if 'real_time' in k]

historical_dataframes = {k: dataframes[k] for k in historical_keys}
real_time_dataframes = {k: dataframes[k] for k in real_time_keys}

# Merge historical dataframes
combined_historical_df = merge_dataframes(
    historical_dataframes, key_column='date')

# Print columns to verify target column name
print("Columns in combined_historical_df:",
      combined_historical_df.columns.tolist())
print("Number of rows in combined_historical_df:", len(combined_historical_df))

# Merge real-time dataframes
combined_real_time_df = merge_dataframes(
    real_time_dataframes, key_column='timestamp')

# Print columns to verify target column name
print("Columns in combined_real_time_df:",
      combined_real_time_df.columns.tolist())


# Function to prepare data for LSTM
def prepare_data(df, target_column, sequence_length, feature_names):
    # Drop datetime columns
    datetime_columns = [col for col in [
        'timestamp', 'date'] if col in df.columns]
    df = df.drop(columns=datetime_columns, errors='ignore')

    # Ensure the target column exists
    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in the data!")

    # Extract features and target
    df_features = df.drop(columns=[target_column])

    # Align features with those used during training
    if feature_names is not None:
        # Ensure all feature names are present
        missing_features = set(feature_names) - set(df_features.columns)
        if missing_features:
            print(f"The following features are missing from the data and will be removed from the feature list: {
                  missing_features}")
            # Remove missing features from feature_names
            feature_names = [
                feature for feature in feature_names if feature not in missing_features]
        df_features = df_features[feature_names]
    else:
        # If feature_names is not provided, select all numeric columns
        df_features = df_features.select_dtypes(include=[np.number])

    # Convert to numpy arrays
    X = df_features.values
    # Use the original target column with prefixes
    y = df[target_column].values

    # Check for NaN or infinite values
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input features contain NaN or infinite values.")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("Target variable contains NaN or infinite values.")

    # Create sequences
    X_sequences = []
    y_sequences = []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    return X_sequences, y_sequences


# Load the feature names used during training for historical data
try:
    feature_names_hist = np.load(os.path.join(
        MODEL_DIR, 'feature_names_hist.npy'), allow_pickle=True)
except FileNotFoundError:
    feature_names_hist = None
    print("Feature names for historical data not found, proceeding without feature alignment.")

# Adjust feature names to include prefixes for historical data


def add_prefix(name):
    # List of known prefixes
    prefixes = ['historical_spy_', 'historical_spx_', 'historical_vix_',
                'consumer_confidence_', 'consumer_sentiment_', 'core_inflation_',
                'cpi_', 'gdp_', 'industrial_production_', 'interest_rate_',
                'labor_force_', 'nonfarm_payroll_', 'personal_consumption_',
                'ppi_', 'unemployment_rate_']
    # If the name already starts with a known prefix, return it as is
    if any(name.startswith(prefix) for prefix in prefixes):
        return name
    # If the name ends with '_historical_spy' or similar, adjust it
    elif name.endswith('_historical_spy'):
        return 'historical_spy_' + name.replace('_historical_spy', '')
    elif name.endswith('_historical_spx'):
        return 'historical_spx_' + name.replace('_historical_spx', '')
    elif name.endswith('_historical_vix'):
        return 'historical_vix_' + name.replace('_historical_vix', '')
    else:
        # Default to 'historical_spy_' prefix
        return 'historical_spy_' + name


if feature_names_hist is not None:
    feature_names_hist_adjusted = [
        add_prefix(name) for name in feature_names_hist]
    print("Adjusted historical feature names:", feature_names_hist_adjusted)
else:
    feature_names_hist_adjusted = None


# Prepare historical test data
X_test_hist, y_test_hist = prepare_data(
    combined_historical_df,
    target_column='historical_spx_close',
    sequence_length=30,
    feature_names=feature_names_hist_adjusted
)

# Load the feature names used during training for real-time data (if available)
try:
    feature_names_real = np.load(os.path.join(
        MODEL_DIR, 'feature_names_real.npy'), allow_pickle=True)
except FileNotFoundError:
    feature_names_real = None
    print("Feature names for real-time data not found, proceeding without feature alignment.")

# Adjust feature names to include prefixes for real-time data


def add_real_time_prefix(name):
    prefixes = ['real_time_spy_', 'real_time_spx_', 'real_time_vix_']
    if any(name.startswith(prefix) for prefix in prefixes):
        return name
    elif name.endswith('_real_time_spy'):
        return 'real_time_spy_' + name.replace('_real_time_spy', '')
    elif name.endswith('_real_time_spx'):
        return 'real_time_spx_' + name.replace('_real_time_spx', '')
    elif name.endswith('_real_time_vix'):
        return 'real_time_vix_' + name.replace('_real_time_vix', '')
    else:
        # Default to 'real_time_spy_' prefix
        return 'real_time_spy_' + name


if feature_names_real is not None:
    feature_names_real_adjusted = [
        add_real_time_prefix(name) for name in feature_names_real]
    print("Adjusted real-time feature names:", feature_names_real_adjusted)
else:
    feature_names_real_adjusted = None

# Identify missing features in combined_real_time_df
datetime_columns = ['timestamp', 'date']
data_columns = [
    col for col in combined_real_time_df.columns if col not in datetime_columns]

# Find missing features and remove them from the feature list
if feature_names_real_adjusted is not None:
    missing_features = set(feature_names_real_adjusted) - set(data_columns)
    if missing_features:
        print(f"The following features are missing from the data and will be removed from the feature list: {
              missing_features}")
        # Remove missing features from feature_names_real_adjusted
        feature_names_real_adjusted = [
            feature for feature in feature_names_real_adjusted if feature not in missing_features]

# Prepare real-time test data
X_test_real, y_test_real = prepare_data(
    combined_real_time_df,
    target_column='real_time_spx_current_price',
    sequence_length=30,
    feature_names=feature_names_real_adjusted
)

# Save test data
np.save(os.path.join(MODEL_DIR, 'X_test_hist.npy'), X_test_hist)
np.save(os.path.join(MODEL_DIR, 'y_test_hist.npy'), y_test_hist)

np.save(os.path.join(MODEL_DIR, 'X_test_real.npy'), X_test_real)
np.save(os.path.join(MODEL_DIR, 'y_test_real.npy'), y_test_real)

print("Test data prepared and saved successfully!")
