import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from dotenv import load_dotenv

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))  
sys.path.append(project_root)


def get_db_connection():
    db_url = os.getenv('DB_URL')
    print(f"Connecting to DB with URL: {db_url}")
    try:
        engine = create_engine(db_url)
        print("Engine created, attempting to connect...")
        connection = engine.connect()
        print("Connection successful!")
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise e


def load_data(query):
    try:
        connection = get_db_connection()
        print(f"Executing query: {query}")
        df = pd.read_sql(query, connection)
        print("Data loaded from the database. DataFrame head:")
        print(df.head())  
        connection.close()
        print("Connection closed successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e


# Data Cleaning
# Data Cleaning: average_hourly_earnings_data
def clean_average_hourly_earnings_data(df):
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[['id', 'date', 'value']]
    df = df.dropna(subset=['value'])
    df = df.drop_duplicates(subset=['date'])
    # Outlier removal:
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    return df


def clean_consumer_confidence_data(df):
    print("Initial DataFrame:")
    print(df.head())

    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    if 'updated_at' in df.columns:
        df = df.drop(columns=['updated_at'])

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    if 'id' in df.columns and 'value' in df.columns:
        df = df[['id', 'date', 'value']]
    else:
        raise KeyError(
            "Required columns ('id', 'date', 'value') are missing from the data.")

    # Handle missing values by dropping rows where 'value' is NaN
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])
        print("Dropped rows with missing 'value'.")

    before_duplicates = df.shape[0]
    df = df.drop_duplicates(subset=['date'])
    after_duplicates = df.shape[0]
    print(f"Removed {before_duplicates - after_duplicates} duplicate rows.")

    # Handle outliers in the 'value' column
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    print(f"Removed {outliers.shape[0]} outliers from 'value' column.")

    print("DataFrame after cleaning:")
    print(df.head())

    return df


# Clean Consumer Sentiment
def clean_consumer_sentiment_data(df):
    print("Initial DataFrame loaded from DB:")
    print(df.head(), df.info())

    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    # Ensure the 'date' column exists; if not, rename 'timestamp' to 'date' for consistency
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    # If neither 'date' nor 'timestamp' exists, raise an error
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    # Check for duplicate columns and drop them
    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]

    # Handle missing values in 'value' column
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])
        print("Dropped rows with missing 'value'.")

    # Remove duplicates based on 'date'
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=['date'])
    after_duplicates = len(df)
    print(f"Removed {before_duplicates - after_duplicates} duplicate rows.")

    # Handle outliers in 'value' column
    mean_value = df['value'].mean()
    std_value = df['value'].std()
    upper_bound = mean_value + 3 * std_value
    lower_bound = mean_value - 3 * std_value

    before_outliers = len(df)
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    after_outliers = len(df)
    print(f"Removed {before_outliers - after_outliers} outlier rows.")
    print(f"DataFrame after outlier removal: {df.head()}")

    # Check if the DataFrame is empty after cleaning
    if df.empty:
        print("DataFrame is empty after cleaning. Skipping further processing.")
        return df

    # Ensure 'id' is set as index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    # Debug: Final check
    print("Final cleaned DataFrame:")
    print(df.head())

    return df


def clean_core_inflation_data(df):
    # Check if 'created_at' or 'updated_at' columns exist and drop them
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    if 'updated_at' in df.columns:
        df = df.drop(columns=['updated_at'])

    # Ensure the 'date' column exists; if not, rename 'timestamp' to 'date' for consistency
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    # If neither 'date' nor 'timestamp' exists, raise an error
    if 'date' not in df.columns:
        raise KeyError(
            "The 'date' column is missing from the core inflation data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]

    # Handle missing values by dropping rows where 'value' is NaN
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Remove duplicates based on the 'date' column
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers in the 'value' column
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df

# Data Cleaning: CPI Data


def clean_cpi_data(df):
    # Drop the 'created_at' and 'updated_at' columns if they exist
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    # Ensure the 'date' column exists; if not, raise an error
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    # If neither 'date' nor 'timestamp' exists, raise an error
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Check for duplicate columns and drop them
    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]

    # Handle missing values by dropping rows where 'value' is NaN
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Remove duplicates based on the 'date' column
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers in the 'value' column
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df

# Data Cleaning: GDP Data


def clean_gdp_data(df):
    # Ensure 'id' and 'date' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Drop the 'updated_at' column if it exists
    df = df.drop(columns=['updated_at'], errors='ignore')

    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]

    # Remove duplicates based on 'date' column
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values by dropping rows where 'value' is NaN
    df = df.dropna(subset=['value'])

    # Check for duplicate columns and drop them
    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    return df

# Data Cleaning: Industrial PRoduction Data


def clean_industrial_production_data(df):
    # Check for 'created_at' and 'updated_at' columns and drop them if they exist
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    if 'updated_at' in df.columns:
        df = df.drop(columns=['updated_at'])

    # Ensure the 'date' column exists; if not, rename 'timestamp' to 'date' for consistency
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    # If neither 'date' nor 'timestamp' exists, raise an error
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Check for duplicate columns and drop them
    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]

    # Convert 'id' to integer if necessary (optional, depending on your data)
    df['id'] = df['id'].astype(int)

    # Handle missing values by dropping rows where 'value' is NaN
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Remove duplicates based on the 'date' column
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers (using Z-score method)
    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    df = df[(z_scores > -3) & (z_scores < 3)]

    return df


# Data Cleaning: Interest Rate Data


def clean_interest_rate_data(df):
    # Check for 'created_at' and 'updated_at' columns and drop them if they exist
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    if 'updated_at' in df.columns:
        df = df.drop(columns=['updated_at'])

    # Ensure the 'date' column exists; if not, rename 'timestamp' to 'date' for consistency
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    # If neither 'date' nor 'timestamp' exists, raise an error
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Check for duplicate columns and drop them
    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    # Keep only relevant columns: 'id', 'date', 'series_id' and 'value'
    df = df[['id', 'date', 'series_id', 'value']]

    # Convert 'id' to integer if necessary (optional, depending on your data)
    df['id'] = df['id'].astype(int)

    # Handle missing values by dropping rows where 'value' is NaN
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Remove duplicates based on the 'date' column
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers (using Z-score method)
    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    df = df[(z_scores > -3) & (z_scores < 3)]

    return df

# Data Cleaning: Labor Force Participation Rate Data


def clean_labor_force_participation_rate_data(df):
    print("Beginning to clean labor force participation rate data.")

    # Drop 'created_at' and 'updated_at' columns if they exist
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    # Ensure 'date' column exists and convert to datetime
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    # Handle missing values in 'value' column
    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df

# Data Cleaning: Nonfarm Payroll Employment Data


def clean_nonfarm_payroll_employment_data(df):
    print("Beginning to Nonfarm Payroll Employment Data.")

    # Drop 'created_at' and 'updated_at' columns if they exist
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    # Ensure 'date' column exists and convert to datetime
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    # Handle missing values in 'value' column
    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df


# Data Cleaning: Personal Consumption Expenditures Data


def clean_personal_consumption_expenditures_data(df):
    print("Beginning to clean Personal Consumption Expenditures Data.")

    # Drop 'created_at' and 'updated_at' columns if they exist
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    # Ensure 'date' column exists and convert to datetime
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    # Handle missing values in 'value' column
    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df


# Data Cleaning: PPI Data


def clean_ppi_data(df):
    print("Beginning to clean PPI Data.")

    # Drop 'created_at' and 'updated_at' columns if they exist
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    # Ensure 'date' column exists and convert to datetime
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    # Handle missing values in 'value' column
    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df


# Data Cleaning: Unemployment Rate Data


def clean_unemployment_rate_data(df):
    print("Beginning to clean Unemployment Rate Data.")

    # Drop 'created_at' and 'updated_at' columns if they exist
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    # Ensure 'date' column exists and convert to datetime
    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    # Handle missing values in 'value' column
    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df


# Data Cleaning: Historical SPX

def clean_historical_spx_data(df):
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")

    df['id'] = df['id'].astype(int)
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows with NaNs in critical price columns
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], how='any', inplace=True)

    # Handle outliers
    for column in ['open', 'high', 'low', 'close']:
        upper_bound = df[column].mean() + 3 * df[column].std()
        lower_bound = df[column].mean() - 3 * df[column].std()
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df = df[df['volume'] > 0]

    # No dropping duplicates on timestamp here.
    # df.drop_duplicates(subset=['timestamp'], inplace=True) # Removed

    df.set_index('id', inplace=True)
    return df


# Data Cleaning: Historical SPY

def clean_historical_spy_data(df):
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")

    df['id'] = df['id'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows with NaNs in critical columns
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], how='any', inplace=True)

    # If needed, handle outliers similar to SPX
    for column in ['open', 'high', 'low', 'close']:
        upper_bound = df[column].mean() + 3 * df[column].std()
        lower_bound = df[column].mean() - 3 * df[column].std()
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df = df[df['volume'] > 0]

    # No dropping duplicates on timestamp here.
    # df.drop_duplicates(subset=['timestamp'], inplace=True) # Removed

    df.set_index('id', inplace=True)
    return df

# Data Cleaning: Historical VIX

def clean_historical_vix_data(df):
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")

    df['id'] = df['id'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows where open, high, low, close are all NaN
    df.dropna(subset=['open', 'high', 'low', 'close'], how='all', inplace=True)

    # Forward fill missing values if necessary
    df.fillna(method='ffill', inplace=True)

    # If desired, handle outliers for 'close':
    upper_bound = df['close'].mean() + 3 * df['close'].std()
    lower_bound = df['close'].mean() - 3 * df['close'].std()
    df = df[(df['close'] >= lower_bound) & (df['close'] <= upper_bound)]

    # No dropping duplicates on timestamp here.
    # df.drop_duplicates(subset=['timestamp'], inplace=True) # Removed

    df.set_index('id', inplace=True)
    return df

# Data Cleaning: Real Time SPX
def clean_real_time_spx_data(df):
    if 'id' not in df.columns:
        raise KeyError("Missing 'id'.")
    if 'timestamp' not in df.columns:
        raise KeyError("Missing 'timestamp'.")

    # Convert timestamp to datetime
    df['id'] = df['id'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Set index once
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # Resample once
    df = df.resample('3T').last()

    # Forward-fill so 'current_price' isn't dropped
    df['current_price'] = df['current_price'].ffill()

    # Drop any rows still missing crucial columns
    df.dropna(subset=['current_price'], inplace=True)

    # Convert index back to normal column
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'timestamp'}, errors='ignore')  # If needed

    return df
# Data Cleaning: Real Time SPY
def clean_real_time_spy_data(df):
    if 'id' not in df.columns:
        raise KeyError("Missing 'id'.")
    if 'timestamp' not in df.columns:
        raise KeyError("Missing 'timestamp'.")

    # Convert timestamp to datetime
    df['id'] = df['id'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Set index once
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    # Resample once
    df = df.resample('3T').last()

    # Forward-fill so 'current_price' isn't dropped
    df['current_price'] = df['current_price'].ffill()

    # Drop any rows still missing crucial columns
    df.dropna(subset=['current_price'], inplace=True)

    # Convert index back to normal column
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'timestamp'}, errors='ignore')  # If needed

    return df

# Data Cleaning: Real Time VIX
def clean_real_time_vix_data(df):
    # Ensure 'id' and 'timestamp' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")

    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Set 'timestamp' as the index but keep it as a column
    if df.index.name != 'timestamp':
        df.set_index('timestamp', inplace=True, drop=False)
        print("'timestamp' column used as index.")

    # Remove duplicates based on index (timestamp)
    df = df[~df.index.duplicated(keep='first')]

    # Sort by index (timestamp) to ensure chronological order
    df.sort_index(inplace=True)
    print("DataFrame sorted by timestamp index.")

    # Reset index to keep 'timestamp' in the DataFrame as a column
    if 'timestamp' not in df.columns:
        df.reset_index(drop=False, inplace=True)
        print("Index reset, 'timestamp' kept as a column.")
    else:
        # If 'timestamp' is already a column, reset without keeping it as a column again
        df.reset_index(drop=True, inplace=True)
        print("Index reset without duplicating 'timestamp' column.")

    # Now set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index.")

    # Debugging: Print the final state of the DataFrame
    print("Final DataFrame state:\n", df.head())

    return df


# Normalize Data

def normalize_data(df):
    if df.empty:
        print("Warning: DataFrame is empty. Skipping normalization.")
        return df, None

    datetime_columns = []
    if 'timestamp' in df.columns:
        datetime_columns.append('timestamp')
    elif 'date' in df.columns:
        datetime_columns.append('date')

    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_columns = [col for col in non_numeric_columns if col not in datetime_columns]

    numeric_df = df.drop(columns=datetime_columns + non_numeric_columns, errors='ignore')
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_df.fillna(numeric_df.mean(), inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_numeric_df = pd.DataFrame(
        scaler.fit_transform(numeric_df),
        columns=numeric_df.columns,
        index=numeric_df.index
    )

    final_df = pd.concat([df[datetime_columns], df[non_numeric_columns], scaled_numeric_df], axis=1)
    return final_df, scaler

def create_sequences(data, seq_len=60):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        seq = data[i:i+seq_len]
        sequences.append(seq)
    return np.array(sequences)

def preprocess_data(query, table_name):
    df = load_data(query)

    if 'id' not in df.columns:
        raise KeyError(f"The 'id' column must be present in the {table_name} data.")

    timestamp_tables = [
        'historical_spx', 'historical_spy', 'historical_vix',
        'real_time_spx', 'real_time_spy', 'real_time_vix'
    ]
    if table_name in timestamp_tables:
        if 'timestamp' not in df.columns:
            raise KeyError(f"{table_name} requires a 'timestamp' column.")
    else:
        if 'date' not in df.columns:
            raise KeyError(f"{table_name} requires a 'date' column.")

    # CHANGED: call cleaning only
    cleaning_function = TABLE_CLEANING_FUNCTIONS.get(table_name)
    if cleaning_function:
        df = cleaning_function(df)
        print(f"Cleaned {table_name} data.")
    else:
        print(f"No cleaning function found for {table_name}; skipping...")

    if table_name in ['real_time_spx', 'real_time_spy']:
        if df.index.name != 'timestamp':
            df.set_index('timestamp', drop=False, inplace=True)
        # Resample again if you want:
        df = df.resample('3min').last()
        df.dropna(subset=['current_price'], how='any', inplace=True)
        df.index.name = 'tmp_index'
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={'tmp_index': 'timestamp'}, inplace=True)
        # Single horizon target
        df['target_1h'] = df['current_price'].shift(-20)
        print(f"Added single horizon (1h) to {table_name}.")

    # CHANGED: remove references to feature_creation_function, etc.

    df = df.loc[:, ~df.columns.duplicated()]
    print(f"Dropped duplicate columns, if any, for {table_name}.")

    df, _ = normalize_data(df)
    print(f"Data normalized for {table_name}.")

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_columns].values
    X_3D = create_sequences(X, seq_len=60)

    return df, X_3D

# Dictionary mapping table names to their respective cleaning functions
TABLE_CLEANING_FUNCTIONS = {
    "average_hourly_earnings_data": clean_average_hourly_earnings_data,
    "consumer_confidence_data": clean_consumer_confidence_data,
    "consumer_sentiment_data": clean_consumer_sentiment_data,
    "core_inflation_data": clean_core_inflation_data,
    "cpi_data": clean_cpi_data,
    "gdp_data": clean_gdp_data,
    "industrial_production_data": clean_industrial_production_data,
    "interest_rate_data": clean_interest_rate_data,
    "labor_force_participation_rate_data": clean_labor_force_participation_rate_data,
    "nonfarm_payroll_employment_data": clean_nonfarm_payroll_employment_data,
    "personal_consumption_expenditures": clean_personal_consumption_expenditures_data,
    "ppi_data": clean_ppi_data,
    "unemployment_rate_data": clean_unemployment_rate_data,
    "historical_spx": clean_historical_spx_data,
    "historical_spy": clean_historical_spy_data,
    "historical_vix": clean_historical_vix_data,
    "real_time_spx": clean_real_time_spx_data,
    "real_time_spy": clean_real_time_spy_data,
    "real_time_vix": clean_real_time_vix_data,
}

if __name__ == "__main__":
    processed_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'data',
        'lumen_2',
        'processed'
    )
    os.makedirs(processed_dir, exist_ok=True)

    from ai.utils.aws_s3_utils import auto_upload_file_to_s3 

    for table_name, cleaning_function in TABLE_CLEANING_FUNCTIONS.items():
        print(f"Processing table: {table_name}")
        query = f"SELECT * FROM {table_name}"
        processed_df, processed_array = preprocess_data(query, table_name)

        if processed_df.empty:
            print(f"Warning: The DataFrame for {table_name} is empty. Skipping saving.")
            continue

        output_path = os.path.join(processed_dir, f"processed_{table_name}.csv")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Deleted existing file: {output_path}")

        print(f"Saving file to {output_path}")
        processed_df.to_csv(output_path, index=False)
        print(f"{table_name} processing completed and saved to {output_path}")

        auto_upload_file_to_s3(
            local_path=output_path,
            s3_subfolder="data/lumen2/processed"
        )
        print(f"Uploaded {output_path} to S3 => data/lumen2/processed\n")