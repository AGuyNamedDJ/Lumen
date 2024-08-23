import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from dotenv import load_dotenv

load_dotenv()

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
        print(df.head())  # Print the first few rows of the DataFrame
        connection.close()
        print("Connection closed successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e


# Data Cleaning
# Data Cleaning: average_hourly_earnings_data
def clean_average_hourly_earnings_data(df):
    # Drop unnecessary columns 'created_at' and 'updated_at'
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    # Ensure the 'date' column exists; if not, raise an error
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
    df = df.dropna(subset=['value'])

    # Remove duplicates based on the 'date' column
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers in the 'value' column
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df

def clean_consumer_confidence_data(df):
    # Debug: Print the DataFrame before any operation
    print("Initial DataFrame:")
    print(df.head())

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
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    # Check for duplicate columns and drop them
    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    # Keep only relevant columns: 'id', 'date', and 'value'
    if 'id' in df.columns and 'value' in df.columns:
        df = df[['id', 'date', 'value']]
    else:
        raise KeyError("Required columns ('id', 'date', 'value') are missing from the data.")

    # Handle missing values by dropping rows where 'value' is NaN
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])
        print("Dropped rows with missing 'value'.")

    # Remove duplicates based on the 'date' column
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

    # Debug: Print the DataFrame after cleaning
    print("DataFrame after cleaning:")
    print(df.head())

    return df


# Clean Consumer Sentiment
def clean_consumer_sentiment_data(df):
    print("Initial DataFrame loaded from DB:")
    print(df.head(), df.info())  # Print initial state of the DataFrame

    # Drop unnecessary columns if they exist
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
        raise KeyError("The 'date' column is missing from the core inflation data.")

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
    # Ensure 'id' and 'timestamp' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")
    
    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)

    # Handle missing values
    df.dropna(inplace=True)

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add a 'date' column based on 'timestamp'
    df['date'] = df['timestamp'].dt.date

    # Remove duplicates based on 'timestamp'
    df.drop_duplicates(subset=['timestamp'], inplace=True)

    # Handle outliers in price data
    for column in ['open', 'high', 'low', 'close']:
        upper_bound = df[column].mean() + 3 * df[column].std()
        lower_bound = df[column].mean() - 3 * df[column].std()
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Ensure volume is a positive integer
    df = df[df['volume'] > 0]

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df

# Data Cleaning: Historical SPY


def clean_historical_spy_data(df):
    # Ensure 'id' and 'timestamp' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")
    
    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add a 'date' column based on 'timestamp'
    df['date'] = df['timestamp'].dt.date

    # Remove duplicates based on 'timestamp'
    df = df.drop_duplicates(subset=['timestamp'])

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df

# Data Cleaning: Historical VIX


def clean_historical_vix_data(df):
    # Ensure 'id' and 'timestamp' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")
    
    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add a 'date' column based on 'timestamp'
    df['date'] = df['timestamp'].dt.date

    # Remove duplicates based on 'timestamp'
    df = df.drop_duplicates(subset=['timestamp'])

    # Handle missing values more carefully
    # If the entire row is NaN for critical columns, drop it
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], how='all')

    # Forward fill NaN values to handle missing data points in critical columns
    df = df.fillna(method='ffill')

    # Re-check for NaNs after filling
    if df.isnull().sum().sum() > 0:
        print("Warning: Some NaN values remain after filling. They may affect feature generation.")

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df


# Data Cleaning: Real Time SPX


def clean_real_time_spx_data(df):
    # Ensure 'id' and 'timestamp' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")

    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)

    # Drop columns with no data (keeping only id, timestamp, and current_price)
    df = df[['id', 'timestamp', 'current_price']].copy()

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add a 'date' column based on 'timestamp'
    df['date'] = df['timestamp'].dt.date

    # Remove duplicates based on timestamp
    df = df.drop_duplicates(subset=['timestamp'])

    # Sort by timestamp to ensure chronological order
    df = df.sort_values(by='timestamp')

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df

# Data Cleaning: Real Time SPY


def clean_real_time_spy_data(df):
    # Ensure 'id' and 'timestamp' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")

    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)

    # Drop columns with no data (keeping only id, timestamp, and current_price)
    df = df[['id', 'timestamp', 'current_price', 'volume']].copy()

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add a 'date' column based on 'timestamp'
    df['date'] = df['timestamp'].dt.date

    # Remove duplicates based on timestamp
    df = df.drop_duplicates(subset=['timestamp'])

    # Sort by timestamp to ensure chronological order
    df = df.sort_values(by='timestamp')

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

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

    # Drop columns with no data (keeping only id, timestamp, and current_price)
    df = df[['id', 'timestamp', 'current_price']].copy()

    # Rename 'current_price' to 'close' for consistency with other tables
    df = df.rename(columns={'current_price': 'close'})

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Add a 'date' column based on 'timestamp'
    df['date'] = df['timestamp'].dt.date

    # Remove duplicates based on timestamp
    df = df.drop_duplicates(subset=['timestamp'])

    # Sort by timestamp to ensure chronological order
    df = df.sort_values(by='timestamp')

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

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
    decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx_timestamps = peak_idx.dropna().map(lambda x: df.at[x, 'date'] if x in df.index else pd.NaT)
    trough_idx_timestamps = trough_idx.dropna().map(lambda x: df.at[x, 'date'] if x in df.index else pd.NaT)

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
    df = df.set_index('date', drop=False)  # Keep 'date' in the DataFrame but use it as an index
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
        decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from index to regular column.")

    return df


# Features: Create Features for Consumer Sentiment Data

def create_features_for_consumer_sentiment_data(df):
    print("Beginning create features for consumer sentiment data.")
    
    # Ensure 'date' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date', drop=False)  # Set 'date' as index but keep as a column

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
        raise KeyError("The 'date' column is missing from the core inflation data.")

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
        decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    
    
# Features: CPI Data


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

    # Seasonal Decomposition
    decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Cumulative Sum of Changes
    df['Cumulative_Sum'] = df['value'].cumsum()

    # Rolling Growth Rates
    df['Rolling_3M_Growth'] = df['value'].pct_change(periods=3)
    df['Rolling_6M_Growth'] = df['value'].pct_change(periods=6)
    df['Rolling_12M_Growth'] = df['value'].pct_change(periods=12)

    # CPI Ratio
    df['CPI_Ratio_12M'] = df['value'] / df['value'].rolling(window=12).mean()

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp(), raw=False)

    df['Days_Since_Peak'] = (df.index.map(pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

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
    decomposition = seasonal_decompose(df['value'], model='multiplicative', period=4)
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
        df = df.set_index('date', drop=False)  # Use 'date' as the index but keep it as a column

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
    decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp() if pd.notnull(x.idxmax()) else None, raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp() if pd.notnull(x.idxmin()) else None, raw=False)
    
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
    decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset index to remove 'date' from being an index
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from being an index to a regular column.")

    return df

# Feature: Labor Force Participation Data
def create_features_for_labor_force_participation_rate_data(df):
    # Ensure 'date' is set as index for easier time series operations
    if 'date' not in df.index.names:
        df = df.set_index('date', drop=False)  # Use 'date' as the index but keep it as a column

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
    decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp() if pd.notnull(x.idxmax()) else None, raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp() if pd.notnull(x.idxmin()) else None, raw=False)
    
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
    df = df.set_index('date', drop=False)  # Keep 'date' in the DataFrame but use it as an index
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
        decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

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
    df = df.set_index('date', drop=False)  # Keep 'date' in the DataFrame but use it as an index
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
        decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

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
    df = df.set_index('date', drop=False)  # Keep 'date' in the DataFrame but use it as an index
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
        decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

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
    df = df.set_index('date', drop=False)  # Keep 'date' in the DataFrame but use it as an index
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
        decomposition = seasonal_decompose(df['value'], model='multiplicative', period=12)
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
    peak_idx = df['value'].expanding().apply(lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)
    print("Reset 'date' from index to regular column.")

    return df

# Features: Historical SPX


def create_features_for_historical_spx(df):
    # Ensure 'timestamp' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('timestamp')

    # Simple Moving Averages (SMA)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * \
        df['close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * \
        df['close'].rolling(window=20).std()

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
    df['ATR'] = df['high'].combine(
        df['low'], max) - df['low'].combine(df['close'].shift(1), min)
    df['ATR_14'] = df['ATR'].rolling(window=14).mean()

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['Stochastic_Oscillator'] = 100 * \
        ((df['close'] - low_min) / (high_max - low_min))

    return df

# Features: Historical SPY


def create_features_for_historical_spy(df):
    # Ensure 'timestamp' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('timestamp')

    # Simple Moving Averages (SMA)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * \
        df['close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * \
        df['close'].rolling(window=20).std()

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
    df['ATR'] = df['high'].combine(
        df['low'], max) - df['low'].combine(df['close'].shift(1), min)
    df['ATR_14'] = df['ATR'].rolling(window=14).mean()

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['Stochastic_Oscillator'] = 100 * \
        ((df['close'] - low_min) / (high_max - low_min))

    return df

# Features: Historical VIX


def create_features_for_historical_vix(df):
    # Ensure 'timestamp' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('timestamp')

    # Debug: Print the DataFrame after loading
    print("Loaded DataFrame from DB:")
    print(df.head())

    # Drop rows with NaN in 'close' to ensure valid data for indicators
    df = df.dropna(subset=['close'])
    print(f"DataFrame after dropping NaNs in 'close' column: {
          df.shape[0]} rows")

    if df.empty:
        print("Warning: DataFrame is empty after dropping NaNs. Exiting function.")
        return df  # Exit if DataFrame is empty

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
    df['ATR'] = high_low.combine(high_close, max).combine(
        low_close, max).rolling(window=14).mean()

    # Drop columns we no longer need
    df = df.drop(columns=['open', 'high', 'low', 'volume'], errors='ignore')

    # Forward fill to handle any NaN values that could be present
    df = df.ffill()

    # Debug: Print final DataFrame
    print(f"Final DataFrame after processing: {df.head()}")

    return df

# Features: Real Time SPX


def create_features_for_real_time_spx(df):
    # Ensure 'timestamp' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('timestamp')

    # Print initial DataFrame
    print("Initial DataFrame:")
    print(df.head())

    # Lag Features
    df['Lag_1'] = df['current_price'].shift(1)

    # Simple Moving Averages (SMA)
    df['SMA_20'] = df['current_price'].rolling(window=20).mean()
    df['SMA_50'] = df['current_price'].rolling(window=50).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['current_price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['current_price'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * \
        df['current_price'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * \
        df['current_price'].rolling(window=20).std()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df['current_price'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    df['ATR'] = df['current_price'].rolling(window=14).std()

    # Check DataFrame after feature creation
    print("DataFrame after feature creation:")
    print(df.isna().sum())

    # Drop NaNs generated by feature creation
    df = df.dropna()

    return df

# Features: Real Time SPY


def create_features_for_real_time_spy(df):
    # Ensure the dataframe is sorted by timestamp
    df = df.sort_values('timestamp')

    # Lag feature (Previous price)
    df['Lag_1'] = df['current_price'].shift(1)

    # Simple Moving Averages (SMA)
    df['SMA_20'] = df['current_price'].rolling(window=20).mean()
    df['SMA_50'] = df['current_price'].rolling(window=50).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_12'] = df['current_price'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['current_price'].ewm(span=26, adjust=False).mean()

    # Bollinger Bands
    df['Bollinger_Upper'] = df['SMA_20'] + 2 * \
        df['current_price'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['SMA_20'] - 2 * \
        df['current_price'].rolling(window=20).std()

    # Relative Strength Index (RSI)
    delta = df['current_price'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR)
    high_low = df['current_price'] - df['current_price'].shift(1)
    df['ATR'] = high_low.rolling(window=14).mean()

    # Drop rows with NaN values resulting from rolling calculations
    df = df.dropna()

    return df


# Features: Real Time VIX

def create_features_for_real_time_vix(df):
    # Ensure the 'date' column exists; if not, raise an error
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the VIX data.")

    # Set 'date' as the index
    df = df.set_index('date', drop=False)  # Keep 'date' column in the data

    # Ensure the index is a datetime type
    df.index = pd.to_datetime(df.index)

    # Feature 1: Daily percentage change
    df['Daily_Percentage_Change'] = df['close'].pct_change()

    # Feature 2: Rolling mean and standard deviation (volatility indicators)
    df['Rolling_Mean_10D'] = df['close'].rolling(window=10).mean()
    df['Rolling_Std_10D'] = df['close'].rolling(window=10).std()
    df['Rolling_Mean_20D'] = df['close'].rolling(window=20).mean()
    df['Rolling_Std_20D'] = df['close'].rolling(window=20).std()

    # Feature 3: MACD (Moving Average Convergence Divergence)
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # Feature 4: RSI (Relative Strength Index)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Feature 5: Lagged values
    df['Lag_1'] = df['close'].shift(1)
    df['Lag_5'] = df['close'].shift(5)
    df['Lag_10'] = df['close'].shift(10)

    # Feature 6: Exponential Moving Averages
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Feature 7: Cumulative sum of changes
    df['Cumulative_Sum'] = df['close'].cumsum()

    # Feature 8: Z-Score normalization for volatility spikes
    df['Z_Score'] = (df['close'] - df['close'].mean()) / df['close'].std()

    # Feature 9: Bollinger Bands (upper and lower)
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    df['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

    # Feature 10: Rate of Change (ROC)
    df['Rate_Of_Change'] = df['close'].diff(5) / df['close'].shift(5)

    # Reset the index to make 'date' a column again
    df.reset_index(drop=True, inplace=True)

    return df  

# Normalize Data

def normalize_data(df):
    # Identify datetime columns
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Check for 'date' or 'timestamp' and print which one exists, or raise an error
    if 'date' in df.columns and 'timestamp' in df.columns:
        print("Both 'date' and 'timestamp' columns exist.")
    elif 'date' in df.columns:
        print("'date' column exists.")
        datetime_columns.append('date')  # Ensure 'date' is treated as a datetime column
    elif 'timestamp' in df.columns:
        print("'timestamp' column exists.")
        datetime_columns.append('timestamp')  # Ensure 'timestamp' is treated as a datetime column
    else:
        print("Error: No 'date' or 'timestamp' column found.")
        return df, None

    # Identify non-numeric columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Remove overlaps and treat 'created_at' and 'updated_at' as datetime only
    non_numeric_columns = [col for col in non_numeric_columns if col not in datetime_columns]

    # Separate numeric data for scaling
    numeric_df = df.drop(columns=datetime_columns + non_numeric_columns, errors='ignore')

    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the numeric data
    scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), 
                                     columns=numeric_df.columns, 
                                     index=numeric_df.index)

    # Reattach the datetime and non-numeric columns without scaling
    final_df = pd.concat(
        [df[datetime_columns], df[non_numeric_columns], scaled_numeric_df], axis=1)

    return final_df, scaler


def preprocess_data(query, table_name):
    df = load_data(query)

    # Step 1: Check if 'date' or 'timestamp' column exists
    if 'id' not in df.columns:
        raise KeyError(f"The 'id' column must be present in the {table_name} data.")
    
    if 'date' not in df.columns and 'timestamp' not in df.columns:
        raise KeyError(f"Either 'date' or 'timestamp' column must be present in the {table_name} data.")

    # Step 2: Convert 'date' or 'timestamp' to datetime and ensure consistency
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp']  # Create 'date' column from 'timestamp' if not present
        print("'timestamp' column converted to datetime and 'date' column created.")
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print("'date' column converted to datetime.")

    # Step 3: Set 'date' as index
    df = df.set_index('date', drop=False)  # Keep 'date' in the DataFrame but use it as an index
    print("Set 'date' column as index.")

    # Step 4: Perform table-specific cleaning and feature creation
    cleaning_function = TABLE_CLEANING_FUNCTIONS.get(table_name)
    if cleaning_function:
        df = cleaning_function(df)
        print(f"Cleaned {table_name} data.")

    feature_creation_function = TABLE_FEATURE_FUNCTIONS.get(table_name)
    if feature_creation_function:
        df = feature_creation_function(df)
        print(f"Features created for {table_name} data.")

    # Normalize the data if the DataFrame is not empty
    if not df.empty:
        df, scaler = normalize_data(df)

    # Step 5: Check for duplicate columns and drop them
    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    # Reset index to remove 'date' from being an index
    df.reset_index(drop=True, inplace=True)
    print("'date' column reset from index to regular column.")

    return df


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

# Main Execution Block
if __name__ == "__main__":
    # Set the correct path for saving the processed data
    processed_dir = os.path.join(os.getcwd(), 'data', 'lumen_2', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    for table_name, cleaning_function in TABLE_CLEANING_FUNCTIONS.items():
        print(f"Processing table: {table_name}")

        # Construct the query to fetch the data for the current table
        query = f"SELECT * FROM {table_name}"

        # Load, clean, and process the data
        processed_df = preprocess_data(query, table_name)

        if processed_df.empty:
            print(f"Warning: The feature DataFrame for {table_name} is empty. Skipping normalization and saving.")
            continue

        # Save the cleaned data to the processed directory
        output_path = os.path.join(processed_dir, f"processed_{table_name}.csv")

        # Delete the file if it already exists
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Deleted existing file: {output_path}")

        print(f"Saving file to {output_path}")
        processed_df.to_csv(output_path, index=False)
        print(f"File exists: {os.path.exists(output_path)}")
        print(f"{table_name} processing completed and saved to {output_path}")