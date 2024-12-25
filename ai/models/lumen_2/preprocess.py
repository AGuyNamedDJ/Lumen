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
    print("Dropped duplicate columns, if any existed.")

    df = df[['id', 'date', 'value']]

    df = df.dropna(subset=['value'])

    df = df.drop_duplicates(subset=['date'])

    # Handle outliers in the 'value' column
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


# Features
# Features: Average Hourly Earnings Data

def create_features_for_average_hourly_earnings(df):
    # Log the beginning of the function
    print("Beginning create features for average hourly earnings data.")

    # Ensure 'date' is retained and is a column
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Set 'id' as the index
    if 'id' in df.columns:
        df.set_index('id', inplace=True)
        print("ID column set as index.")
    else:
        raise KeyError("The 'id' column is missing from the data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Annualized Growth Rate
    df['CAGR_12'] = (df['value'] / df['value'].shift(12)) ** (1/1) - 1

    # Monthly and Annual Percentage Change
    df['Monthly_Percentage_Change'] = df['value'].pct_change()
    df['Annual_Percentage_Change'] = df['value'].pct_change(periods=12)

    # MACD
    short_ema = df['value'].ewm(span=12, adjust=False).mean()
    long_ema = df['value'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Cumulative Sum of Changes
    df['Cumulative_Sum'] = df['value'].cumsum()

    # Rolling Growth Rates
    df['Rolling_3M_Growth'] = df['value'].pct_change(periods=3)
    df['Rolling_6M_Growth'] = df['value'].pct_change(periods=6)
    df['Rolling_12M_Growth'] = df['value'].pct_change(periods=12)

    # Average Hourly Earnings Ratio
    df['AHE_Ratio_12M'] = df['value'] / df['value'].rolling(window=12).mean()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax())
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin())

    # Instead of using loc, handle duplicates and use .at or .iat for safe access
    peak_idx_timestamps = peak_idx.dropna().map(
        lambda x: df.at[x, 'date'] if x in df.index else pd.NaT)
    trough_idx_timestamps = trough_idx.dropna().map(
        lambda x: df.at[x, 'date'] if x in df.index else pd.NaT)

    # Calculate days since peak/trough and handle NaN values
    df['Days_Since_Peak'] = (df['date'].map(pd.Timestamp.timestamp) - peak_idx_timestamps.map(pd.Timestamp.timestamp)).apply(
        lambda x: pd.Timedelta(seconds=x).days if pd.notnull(x) else None)
    df['Days_Since_Trough'] = (df['date'].map(pd.Timestamp.timestamp) - trough_idx_timestamps.map(pd.Timestamp.timestamp)).apply(
        lambda x: pd.Timedelta(seconds=x).days if pd.notnull(x) else None)

    # RSI Calculation
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # No need to reset the index, keep 'id' as index
    print("Finished processing average hourly earnings data.")
    return df


# Features: Consumer Confidence Data````


def create_features_for_consumer_confidence_data(df):
    # Ensure the 'date' column exists and set it as the index
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format if not already converted
    df['date'] = pd.to_datetime(df['date'])

    # Set 'date' as the index for easier time series operations
    # Keep 'date' in the DataFrame but use it as an index
    df = df.set_index('date', drop=False)
    print("Set 'date' column as index for time series operations.")

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    if len(df) >= 24:  # Ensure there's at least 24 months of data
        decomposition = seasonal_decompose(
            df['value'], model='multiplicative', period=12)
        df['Trend'] = decomposition.trend
        df['Seasonal'] = decomposition.seasonal
        df['Residual'] = decomposition.resid
    else:
        print("Not enough data points for seasonal decomposition. Skipping this feature.")
        df['Trend'] = np.nan
        df['Seasonal'] = np.nan
        df['Residual'] = np.nan

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from index to regular column.")

    return df


# Features: Create Features for Consumer Sentiment Data

def create_features_for_consumer_sentiment_data(df):
    print("Beginning create features for consumer sentiment data.")

    # Ensure 'date' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # Set 'date' as index but keep as a column
        df = df.set_index('date', drop=False)

    # Monthly and Annual Percentage Change
    df['Monthly_Percentage_Change'] = df['value'].pct_change()
    df['Annual_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Rolling Average
    df['Rolling_3M_Average'] = df['value'].rolling(window=3).mean()
    df['Rolling_6M_Average'] = df['value'].rolling(window=6).mean()
    df['Rolling_12M_Average'] = df['value'].rolling(window=12).mean()

    # MACD
    short_ema = df['value'].ewm(span=12, adjust=False).mean()
    long_ema = df['value'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # RSI
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Reset the index to make 'date' a column again, if needed
    df.reset_index(drop=True, inplace=True)
    print("Features created and 'date' reset from index to regular column.")

    return df


# Features: Core Inflation Data


def create_features_for_core_inflation_data(df):
    # Ensure 'date' is retained
    if 'date' not in df.columns:
        raise KeyError(
            "The 'date' column is missing from the core inflation data.")

    # Set 'date' as the index
    df = df.set_index('date', drop=False)  # Keep 'date' column in the data

    # Ensure the index is a datetime type
    df.index = pd.to_datetime(df.index)

    # Annualized Growth Rate
    df['CAGR_12'] = (df['value'] / df['value'].shift(12)) ** (1/1) - 1

    # Monthly and Annual Percentage Change
    df['Monthly_Percentage_Change'] = df['value'].pct_change()
    df['Annual_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()

    # Cumulative Sum of Changes
    df['Cumulative_Sum'] = df['value'].cumsum()

    # Rolling Growth Rates
    df['Rolling_3M_Growth'] = df['value'].pct_change(periods=3)
    df['Rolling_6M_Growth'] = df['value'].pct_change(periods=6)
    df['Rolling_12M_Growth'] = df['value'].pct_change(periods=12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Seasonal Decomposition
    if len(df) >= 24:  # Ensure there's at least 24 months of data
        decomposition = seasonal_decompose(
            df['value'], model='multiplicative', period=12)
        df['Trend'] = decomposition.trend
        df['Seasonal'] = decomposition.seasonal
        df['Residual'] = decomposition.resid
    else:
        print("Not enough data points for seasonal decomposition. Skipping this feature.")
        df['Trend'] = np.nan
        df['Seasonal'] = np.nan
        df['Residual'] = np.nan

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)

    return df


def create_features_for_cpi_data(df):
    # Ensure 'date' is set as index for easier time series operations
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Set 'date' as the index
    df = df.set_index('date', drop=False)

    # Ensure the index is a datetime type
    df.index = pd.to_datetime(df.index)

    # Monthly and Annual Percentage Change
    df['Monthly_Percentage_Change'] = df['value'].pct_change()
    df['Annual_Percentage_Change'] = df['value'].pct_change(periods=12)

    # MACD
    short_ema = df['value'].ewm(span=12, adjust=False).mean()
    long_ema = df['value'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # Seasonal Decomposition (Add Error Handling)
    period = 12  # Define the period for monthly data
    if len(df) >= 2 * period:  # Ensure we have enough data for decomposition
        try:
            decomposition = seasonal_decompose(
                df['value'], model='multiplicative', period=period)
            df['Trend'] = decomposition.trend
            df['Seasonal'] = decomposition.seasonal
            df['Residual'] = decomposition.resid
        except ValueError as e:
            print(f"Error during seasonal decomposition: {e}")
            # Use rolling mean as a fallback
            df['Trend'] = df['value'].rolling(window=period).mean()
            df['Seasonal'] = 0
            df['Residual'] = 0
    else:
        print(f"Not enough data points for seasonal decomposition. Minimum required: {
              2 * period}, found: {len(df)}")
        df['Trend'] = df['value'].rolling(window=period).mean()  # Fallback
        df['Seasonal'] = 0
        df['Residual'] = 0

    # Cumulative Sum of Changes
    df['Cumulative_Sum'] = df['value'].cumsum()

    # Rolling Growth Rates
    df['Rolling_3M_Growth'] = df['value'].pct_change(periods=3)
    df['Rolling_6M_Growth'] = df['value'].pct_change(periods=6)
    df['Rolling_12M_Growth'] = df['value'].pct_change(periods=12)

    # CPI Ratio
    df['CPI_Ratio_12M'] = df['value'] / df['value'].rolling(window=12).mean()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)

    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # RSI
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Reset the index to keep 'date' as a column
    df.reset_index(drop=True, inplace=True)

    return df


# Features: GDP Data


def create_features_for_gdp_data(df):
    # Print starting message
    print("Beginning create features for GDP data.")

    # Ensure 'date' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date', drop=False)  # Keep 'date' column in the data

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_2'] = df['value'].shift(2)
    df['Lag_4'] = df['value'].shift(4)
    df['Lag_8'] = df['value'].shift(8)

    # Rolling Statistics
    df['Rolling_Mean_4Q'] = df['value'].rolling(window=4).mean()
    df['Rolling_Mean_8Q'] = df['value'].rolling(window=8).mean()
    df['Rolling_Std_4Q'] = df['value'].rolling(window=4).std()
    df['Rolling_Std_8Q'] = df['value'].rolling(window=8).std()

    # Percentage Change
    df['QoQ_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=4)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=4)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Exponential Moving Average (EMA)
    df['EMA_4'] = df['value'].ewm(span=4, adjust=False).mean()
    df['EMA_8'] = df['value'].ewm(span=8, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(4) / df['value'].shift(4)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Reset the index to make 'date' a column again, if necessary
    df.reset_index(drop=True, inplace=True)

    return df


# Features: Industrial Production Data


def create_features_for_industrial_production_data(df):
    # Ensure 'date' is set as index for easier time series operations
    if 'date' not in df.index.names:
        # Use 'date' as the index but keep it as a column
        df = df.set_index('date', drop=False)

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax(
    ).timestamp() if pd.notnull(x.idxmax()) else None, raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin(
    ).timestamp() if pd.notnull(x.idxmin()) else None, raw=False)

    df['Days_Since_Peak'] = df.index.map(pd.Timestamp.timestamp) - peak_idx
    df['Days_Since_Trough'] = df.index.map(pd.Timestamp.timestamp) - trough_idx

    # Reset index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)

    return df

    # Features: Interest Rate Data


def create_features_for_interest_rate_data(df):
    # Ensure 'date' is set as index for easier time series operations
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date')

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset index to remove 'date' from being an index
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from being an index to a regular column.")

    return df

# Feature: Labor Force Participation Data


def create_features_for_labor_force_participation_rate_data(df):
    # Ensure 'date' is set as index for easier time series operations
    if 'date' not in df.index.names:
        # Use 'date' as the index but keep it as a column
        df = df.set_index('date', drop=False)

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax(
    ).timestamp() if pd.notnull(x.idxmax()) else None, raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin(
    ).timestamp() if pd.notnull(x.idxmin()) else None, raw=False)

    df['Days_Since_Peak'] = df.index.map(pd.Timestamp.timestamp) - peak_idx
    df['Days_Since_Trough'] = df.index.map(pd.Timestamp.timestamp) - trough_idx

    # Reset index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)

    return df


# Features: Nonfarm Payroll Employment Data


def create_features_for_nonfarm_payroll_employment_data(df):
    # Ensure the 'date' column exists and set it as the index
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format if not already converted
    df['date'] = pd.to_datetime(df['date'])

    # Set 'date' as the index for easier time series operations
    # Keep 'date' in the DataFrame but use it as an index
    df = df.set_index('date', drop=False)
    print("Set 'date' column as index for time series operations.")

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    if len(df) >= 24:  # Ensure there's at least 24 months of data
        decomposition = seasonal_decompose(
            df['value'], model='multiplicative', period=12)
        df['Trend'] = decomposition.trend
        df['Seasonal'] = decomposition.seasonal
        df['Residual'] = decomposition.resid
    else:
        print("Not enough data points for seasonal decomposition. Skipping this feature.")
        df['Trend'] = np.nan
        df['Seasonal'] = np.nan
        df['Residual'] = np.nan

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from index to regular column.")

    return df


# Features: Personal Consumption Expenditures Data


def create_features_for_personal_consumption_expenditures(df):
    # Ensure the 'date' column exists and set it as the index
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format if not already converted
    df['date'] = pd.to_datetime(df['date'])

    # Set 'date' as the index for easier time series operations
    # Keep 'date' in the DataFrame but use it as an index
    df = df.set_index('date', drop=False)
    print("Set 'date' column as index for time series operations.")

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    if len(df) >= 24:  # Ensure there's at least 24 months of data
        decomposition = seasonal_decompose(
            df['value'], model='multiplicative', period=12)
        df['Trend'] = decomposition.trend
        df['Seasonal'] = decomposition.seasonal
        df['Residual'] = decomposition.resid
    else:
        print("Not enough data points for seasonal decomposition. Skipping this feature.")
        df['Trend'] = np.nan
        df['Seasonal'] = np.nan
        df['Residual'] = np.nan

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from index to regular column.")

    return df

# Features: PPI Data


def create_features_for_ppi_data(df):
    # Ensure the 'date' column exists and set it as the index
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format if not already converted
    df['date'] = pd.to_datetime(df['date'])

    # Set 'date' as the index for easier time series operations
    # Keep 'date' in the DataFrame but use it as an index
    df = df.set_index('date', drop=False)
    print("Set 'date' column as index for time series operations.")

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    if len(df) >= 24:  # Ensure there's at least 24 months of data
        decomposition = seasonal_decompose(
            df['value'], model='multiplicative', period=12)
        df['Trend'] = decomposition.trend
        df['Seasonal'] = decomposition.seasonal
        df['Residual'] = decomposition.resid
    else:
        print("Not enough data points for seasonal decomposition. Skipping this feature.")
        df['Trend'] = np.nan
        df['Seasonal'] = np.nan
        df['Residual'] = np.nan

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from index to regular column.")

    return df

# Features: Unemployment Rate Data


def create_features_for_unemployment_rate_data(df):
    # Ensure the 'date' column exists and set it as the index
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format if not already converted
    df['date'] = pd.to_datetime(df['date'])

    # Set 'date' as the index for easier time series operations
    # Keep 'date' in the DataFrame but use it as an index
    df = df.set_index('date', drop=False)
    print("Set 'date' column as index for time series operations.")

    # Lag Features
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Seasonal Decomposition
    if len(df) >= 24:  # Ensure there's at least 24 months of data
        decomposition = seasonal_decompose(
            df['value'], model='multiplicative', period=12)
        df['Trend'] = decomposition.trend
        df['Seasonal'] = decomposition.seasonal
        df['Residual'] = decomposition.resid
    else:
        print("Not enough data points for seasonal decomposition. Skipping this feature.")
        df['Trend'] = np.nan
        df['Seasonal'] = np.nan
        df['Residual'] = np.nan

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from index to regular column.")

    return df

# Features: Historical SPX
def create_features_for_historical_spx(df):
    # Ensure 'timestamp' column exists and is of datetime type
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column must be present in historical_spx data.")
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Set 'timestamp' as index temporarily (but keep it as a column)
    df.set_index('timestamp', drop=False, inplace=True)

    # Simple Moving Averages (SMA)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['close'].rolling(window=20).std()

    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR)
    df['ATR'] = df['high'].combine(df['low'], max) - df['low'].combine(df['close'].shift(1), min)
    df['ATR_14'] = df['ATR'].rolling(window=14).mean()

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['Stochastic_Oscillator'] = 100 * ((df['close'] - low_min) / (high_max - low_min))

    # Reset index so that 'timestamp' is a normal column again
    df.reset_index(drop=True, inplace=True)
    return df


def create_features_for_historical_spy(df):
    # Ensure 'timestamp' column exists and is of datetime type
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column must be present in historical_spy data.")
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Set 'timestamp' as index temporarily (but keep it as a column)
    df.set_index('timestamp', drop=False, inplace=True)

    # Simple Moving Averages (SMA)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['close'].rolling(window=20).std()

    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR (Average True Range)
    df['ATR'] = df['high'].combine(df['low'], max) - df['low'].combine(df['close'].shift(1), min)
    df['ATR_14'] = df['ATR'].rolling(window=14).mean()

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['Stochastic_Oscillator'] = 100 * ((df['close'] - low_min) / (high_max - low_min))

    # Reset index so that 'timestamp' is a normal column again
    df.reset_index(drop=True, inplace=True)
    return df


def create_features_for_historical_vix(df):
    # Ensure 'timestamp' column exists and is of datetime type
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column must be present in historical_vix data.")
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Set 'timestamp' as index temporarily (but keep it as a column)
    df.set_index('timestamp', drop=False, inplace=True)

    # Drop rows with NaN in 'close' to ensure valid data for indicators
    df = df.dropna(subset=['close'])

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    df['ATR'] = high_low.combine(high_close, max).combine(low_close, max).rolling(window=14).mean()

    # Drop columns no longer needed
    df.drop(columns=['open', 'high', 'low', 'volume'], errors='ignore', inplace=True)

    # Forward fill any remaining NaNs
    df.ffill(inplace=True)

    # Reset index so that 'timestamp' is a normal column again
    df.reset_index(drop=True, inplace=True)
    return df


def create_features_for_real_time_spx(df):
    # Ensure the 'timestamp' column exists
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the SPX data.")

    # Ensure the index is a datetime type
    df.index = pd.to_datetime(df.index)

    # Feature Creation
    df['Lag_1'] = df['current_price'].shift(1)
    df['SMA_20'] = df['current_price'].rolling(window=20).mean()
    df['SMA_50'] = df['current_price'].rolling(window=50).mean()
    df['EMA_12'] = df['current_price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['current_price'].ewm(span=26, adjust=False).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['current_price'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['current_price'].rolling(window=20).std()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta = df['current_price'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['ATR'] = df['current_price'].rolling(window=14).std()

    # Check DataFrame after feature creation
    print("DataFrame after feature creation:")
    print(df.isna().sum())

    # Only drop rows with missing data if it won't empty the DataFrame
    # Allow keeping some rows with NaNs to maintain sufficient data
    if len(df.dropna()) > 20:  # Assuming 20 is a minimum threshold to consider data sufficient
        df.dropna(inplace=True)
        print("Dropped rows with NaN values.")

    return df

# Features: Real Time SPY


def create_features_for_real_time_spy(df):
    # Ensure the 'timestamp' column exists
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the SPX data.")

    # Ensure the index is a datetime type
    df.index = pd.to_datetime(df.index)

    # Feature Creation
    df['Lag_1'] = df['current_price'].shift(1)
    df['SMA_20'] = df['current_price'].rolling(window=20).mean()
    df['SMA_50'] = df['current_price'].rolling(window=50).mean()
    df['EMA_12'] = df['current_price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['current_price'].ewm(span=26, adjust=False).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * \
        df['current_price'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * \
        df['current_price'].rolling(window=20).std()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['current_price'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['ATR'] = df['current_price'].rolling(window=14).std()

    # Check DataFrame after feature creation
    print("DataFrame after feature creation:")
    print(df.isna().sum())

    # Only drop rows with missing data if it won't empty the DataFrame
    # Allow keeping some rows with NaNs to maintain sufficient data
    if len(df.dropna()) > 20:  # Assuming 20 is a minimum threshold to consider data sufficient
        df.dropna(inplace=True)
        print("Dropped rows with NaN values.")

    return df

# Features: Real Time VIX


def create_features_for_real_time_vix(df):
    # Ensure the 'timestamp' column exists
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the VIX data.")

    # Convert 'timestamp' to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set 'timestamp' as the index and drop it from columns
    df = df.set_index('timestamp', drop=True)

    # Feature 1: Daily percentage change
    df['Daily_Percentage_Change'] = df['current_price'].pct_change()

    # Feature 2: Rolling mean and standard deviation (volatility indicators)
    df['Rolling_Mean_10D'] = df['current_price'].rolling(window=10).mean()
    df['Rolling_Std_10D'] = df['current_price'].rolling(window=10).std()
    df['Rolling_Mean_20D'] = df['current_price'].rolling(window=20).mean()
    df['Rolling_Std_20D'] = df['current_price'].rolling(window=20).std()

    # Feature 3: MACD (Moving Average Convergence Divergence)
    short_ema = df['current_price'].ewm(span=12, adjust=False).mean()
    long_ema = df['current_price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # Feature 4: RSI (Relative Strength Index)
    delta = df['current_price'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Feature 5: Lagged values
    df['Lag_1'] = df['current_price'].shift(1)
    df['Lag_5'] = df['current_price'].shift(5)
    df['Lag_10'] = df['current_price'].shift(10)

    # Feature 6: Exponential Moving Averages
    df['EMA_10'] = df['current_price'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['current_price'].ewm(span=20, adjust=False).mean()

    # Feature 7: Cumulative sum of changes
    df['Cumulative_Sum'] = df['current_price'].cumsum()

    # Feature 8: Z-Score normalization for volatility spikes
    df['Z_Score'] = (df['current_price'] -
                     df['current_price'].mean()) / df['current_price'].std()

    # Feature 9: Bollinger Bands (upper and lower)
    rolling_mean = df['current_price'].rolling(window=20).mean()
    rolling_std = df['current_price'].rolling(window=20).std()
    df['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    df['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

    # Feature 10: Rate of Change (ROC)
    df['Rate_Of_Change'] = df['current_price'].diff(
        5) / df['current_price'].shift(5)

    # Reset the index to keep 'timestamp' as a column (if necessary for other operations)
    df.reset_index(drop=False, inplace=True)

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
    scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df),
                                     columns=numeric_df.columns,
                                     index=numeric_df.index)

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

    cleaning_function = TABLE_CLEANING_FUNCTIONS.get(table_name)
    if cleaning_function:
        df = cleaning_function(df)
        print(f"Cleaned {table_name} data.")

    if table_name in ['real_time_spx', 'real_time_spy']:
        if df.index.name != 'timestamp':
            df.set_index('timestamp', drop=False, inplace=True)
        df = df.resample('3T').last()
        df.dropna(subset=['current_price'], how='any', inplace=True)
        df.index.name = 'tmp_index'
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={'tmp_index': 'timestamp'}, inplace=True)
        df['target_1h'] = df['current_price'].shift(-20)
        print(f"Added single horizon (1h) to {table_name}.")

    feature_creation_function = TABLE_FEATURE_FUNCTIONS.get(table_name)
    if feature_creation_function:
        df = feature_creation_function(df)
        print(f"Features created for {table_name} data.")

    df = df.loc[:, ~df.columns.duplicated()]
    print(f"Dropped duplicate columns, if any, for {table_name}.")

    df, _ = normalize_data(df)
    print(f"Data normalized for {table_name}.")

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_columns].values
    X_3D = create_sequences(X, seq_len=60)

    return df, X_3D

# Add a mapping for feature creation functions similar to the cleaning functions
TABLE_FEATURE_FUNCTIONS = {
    "average_hourly_earnings_data": create_features_for_average_hourly_earnings,
    "consumer_confidence_data": create_features_for_consumer_confidence_data,
    "consumer_sentiment_data": create_features_for_consumer_sentiment_data,
    "core_inflation_data": create_features_for_core_inflation_data,
    "cpi_data": create_features_for_cpi_data,
    "gdp_data": create_features_for_gdp_data,
    "industrial_production_data": create_features_for_industrial_production_data,
    "interest_rate_data": create_features_for_interest_rate_data,
    "labor_force_participation_rate_data": create_features_for_labor_force_participation_rate_data,
    "nonfarm_payroll_employment_data": create_features_for_nonfarm_payroll_employment_data,
    "personal_consumption_expenditures": create_features_for_personal_consumption_expenditures,
    "ppi_data": create_features_for_ppi_data,
    "unemployment_rate_data": create_features_for_unemployment_rate_data,
    "historical_spx": create_features_for_historical_spx,
    "historical_spy": create_features_for_historical_spy,
    "historical_vix": create_features_for_historical_vix,
    "real_time_spx": create_features_for_real_time_spx,
    "real_time_spy": create_features_for_real_time_spy,
    "real_time_vix": create_features_for_real_time_vix,
}

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

    from ai.utils.aws_s3_utils import auto_upload_file_to_s3 

    for table_name, cleaning_function in TABLE_CLEANING_FUNCTIONS.items():
        print(f"Processing table: {table_name}")
        query = f"SELECT * FROM {table_name}"
        processed_df, processed_array = preprocess_data(query, table_name)

        if processed_df.empty:
            print(f"Warning: The feature DataFrame for {table_name} is empty. Skipping saving.")
            continue

        # 1) Save locally:
        output_path = os.path.join(processed_dir, f"processed_{table_name}.csv")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Deleted existing file: {output_path}")

        print(f"Saving file to {output_path}")
        processed_df.to_csv(output_path, index=False)
        print(f"{table_name} processing completed and saved to {output_path}")

        # 2) Now automatically upload to S3
        auto_upload_file_to_s3(
            local_path=output_path,
            s3_subfolder="data/lumen2/processed"
        )
        print(f"Uploaded {output_path} to S3 under data/lumen2/processed\n")