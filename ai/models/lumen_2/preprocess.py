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
    # Ensure the 'date' column exists and is named correctly
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Keep only the relevant columns
    df = df[['id', 'date', 'value']]

    # Handle missing values
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df


def clean_consumer_confidence_data(df):
    # Debug: Print the DataFrame before any operation
    print("Initial DataFrame:")
    print(df.head())

    # Handle missing values in 'value' column
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])
        print("Dropped rows with missing 'value'.")
    
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")
    
    # Remove duplicates based on 'date' column
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

    # Ensure 'id' column is still present
    if 'id' not in df.columns:
        raise KeyError("'id' column is missing after cleaning for consumer_confidence_data. Skipping setting index.")

    return df

# Data Cleaning: consumer_sentiment_data


def clean_consumer_sentiment_data(df):
    print("Initial DataFrame loaded from DB:")
    print(df.head(), df.info())  # Print initial state of the DataFrame

    # Handle missing values in 'value' column
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])
        print("Dropped rows with missing 'value'.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

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

    # Temporarily comment out the outlier removal to test
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
    if 'id' in df.columns:
        df.set_index('id', inplace=True)
        print("ID column set as index")
    else:
        print("'id' column is missing after cleaning. Cannot set index.")

    # Debug: Final check
    print("Final cleaned DataFrame:")
    print(df.head())

    return df
    
def clean_core_inflation_data(df):
    # Ensure 'id' and 'date' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")
    
    # Convert 'id' to integer if necessary (optional, depending on your data)
    df['id'] = df['id'].astype(int)
    
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")
    
    # Drop 'created_at' column since it's not needed
    df = df.drop(columns=['created_at'])
    print("Dropped 'created_at' column.")
    
    # Print columns and data types to debug
    print("DataFrame columns after conversion:", df.columns)
    print("DataFrame dtypes after conversion:", df.dtypes)
    
    # Remove duplicates based on the 'date' column
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['date'])
    after_dedup = len(df)
    print(f"Removed {before_dedup - after_dedup} duplicate rows.")
    
    # Handle missing values by dropping rows where 'value' is NaN
    missing_values = df['value'].isnull().sum()
    if missing_values > 0:
        df = df.dropna(subset=['value'])
        print(f"Removed {missing_values} rows with missing 'value'.")
    
    # Handle outliers by filtering out values outside 3 standard deviations
    mean_value = df['value'].mean()
    std_value = df['value'].std()
    upper_bound = mean_value + 3 * std_value
    lower_bound = mean_value - 3 * std_value
    before_outlier_removal = len(df)
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    after_outlier_removal = len(df)
    print(f"Removed {before_outlier_removal - after_outlier_removal} outlier rows.")
    
    # Print the columns after cleaning
    print("Columns after cleaning:", df.columns)
    
    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")
    
    return df
    

# Data Cleaning: CPI Data

def clean_cpi_data(df):
    # Ensure 'id' and 'date' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Drop the 'created_at' and 'updated_at' columns if they exist
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')
    
    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)
    
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove duplicates based on 'date' column
    df = df.drop_duplicates(subset=['date'])
    
    # Handle missing values by dropping rows where 'value' is NaN
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])
    
    # Handle outliers in the 'value' column
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    
    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

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

    # Remove duplicates based on 'date' column
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values by dropping rows where 'value' is NaN
    df = df.dropna(subset=['value'])

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df


# Data Cleaning: Industrial PRoduction Data


def clean_industrial_production_data(df):
    # Ensure 'id' and 'date' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'id' to integer if necessary (optional, depending on your data)
    df['id'] = df['id'].astype(int)

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates based on 'date' column
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values by dropping rows where 'value' is NaN
    df = df.dropna(subset=['value'])

    # Handle outliers (using Z-score method)
    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    df = df[(z_scores > -3) & (z_scores < 3)]

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df

# Data Cleaning: Interest Rate Data


def clean_interest_rate_data(df):
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

    # Remove duplicates based on 'date' and 'series_id' columns
    df = df.drop_duplicates(subset=['date', 'series_id'])

    # Handle missing values by dropping rows where 'value' is NaN
    df = df.dropna(subset=['value'])

    # Handle outliers using the Z-score method
    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    df = df[(z_scores > -3) & (z_scores < 3)]

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df


# Data Cleaning: Labor Force Participation Rate Data


def clean_labor_force_participation_rate_data(df):
    # Ensure 'id' and 'date' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Drop the 'updated_at' column if it exists
    df = df.drop(columns=['updated_at'], errors='ignore')

    # Convert 'id' to integer if necessary
    df['id'] = df['id'].astype(int)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove any duplicates based on 'date'
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values if any
    df = df.dropna(subset=['value'])

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df


# Data Cleaning: Nonfarm Payroll Employment Data

def clean_nonfarm_payroll_employment_data(df):
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

    # Remove any duplicates based on 'date' and 'value'
    df = df.drop_duplicates(subset=['date', 'value'])

    # Handle missing values
    df = df.dropna(subset=['value'])

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df


# Data Cleaning: Personal Consumption Expenditures Data


def clean_personal_consumption_expenditures_data(df):
    # Ensure 'id' and 'date' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")
    
    # Convert 'id' to integer if necessary (optional, depending on your data)
    df['id'] = df['id'].astype(int)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates based on 'date'
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values in the 'value' column
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df

# Data Cleaning: PPI Data


def clean_ppi_data(df):
    # Ensure 'id' and 'date' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")
    
    # Convert 'id' to integer if necessary (optional, depending on your data)
    df['id'] = df['id'].astype(int)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates based on 'date'
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values in the 'value' column
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    return df

# Data Cleaning: Unemployment Rate Data


def clean_unemployment_rate_data(df):
    # Ensure 'id' and 'date' columns exist
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")
    
    # Convert 'id' to integer if necessary (optional, depending on your data)
    df['id'] = df['id'].astype(int)

    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates based on 'date'
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values in the 'value' column
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

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

    # Convert 'id' to integer if necessary (optional, depending on your data)
    df['id'] = df['id'].astype(int)

    # Drop the 'conditions' column and keep only the columns with data
    df = df[['id', 'timestamp', 'current_price', 'volume']].copy()

    # Convert 'timestamp' to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

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

    # Fill forward for any missing data
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Convert 'timestamp' to datetime format using .loc to avoid the warning
    df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'])

    # Remove duplicates based on the 'timestamp' and 'current_price'
    df = df.drop_duplicates(subset=['timestamp', 'current_price'])

    # Set 'id' as the index
    df.set_index('id', inplace=True)
    print("ID column set as index")

    print("Final cleaned DataFrame:")
    print(df.head(), df.info())

    return df


# Features
# Features: Average Hourly Earnings Data


def create_features_for_average_hourly_earnings(df):
    # Set 'date' as the index
    if 'date' in df.columns:
        df = df.set_index('date')

    # Ensure the index is a datetime type
    df.index = pd.to_datetime(df.index)

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

    return df


# Features: Consumer Confidence Data


def create_features_for_consumer_confidence_data(df):
    # Ensure the 'date' column exists
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Ensure 'date' is set as index for easier time series operations
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

    return df


# Features: Create Features for Cosumer Sentiment Date

def create_features_for_consumer_sentiment_data(df):
    # Convert index to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date')

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

    return df

# Features: Core Inflation Data


def create_features_for_core_inflation_data(df):
    # Ensure 'date' is in datetime format and set it as index
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Check for missing values
    if df['value'].isnull().sum() > 0:
        print("Warning: Missing values found in 'value' column. Filling missing values.")
        df['value'].fillna(method='ffill', inplace=True)

    # Monthly and Annual Percentage Change
    df['Monthly_Percentage_Change'] = df['value'].pct_change()
    df['Annual_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum of Core Inflation
    df['Cumulative_Sum'] = df['value'].cumsum()

    # Rolling Averages
    df['Rolling_3M_Average'] = df['value'].rolling(window=3).mean()
    df['Rolling_6M_Average'] = df['value'].rolling(window=6).mean()
    df['Rolling_12M_Average'] = df['value'].rolling(window=12).mean()

    # Moving Average Convergence Divergence (MACD)
    short_ema = df['value'].ewm(span=12, adjust=False).mean()
    long_ema = df['value'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema

    # Seasonal Decomposition
    try:
        decomposition = seasonal_decompose(
            df['value'], model='additive', period=12)
        df['Trend'] = decomposition.trend
        df['Seasonal'] = decomposition.seasonal
        df['Residual'] = decomposition.resid
    except Exception as e:
        print(f"Error during seasonal decomposition: {e}")
        df['Trend'] = None
        df['Seasonal'] = None
        df['Residual'] = None

    # Relative Strength Index (RSI)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Rolling Standard Deviation (Volatility)
    df['Rolling_3M_StdDev'] = df['value'].rolling(window=3).std()
    df['Rolling_6M_StdDev'] = df['value'].rolling(window=6).std()
    df['Rolling_12M_StdDev'] = df['value'].rolling(window=12).std()

    # Inflation Momentum
    df['Momentum_3M'] = df['value'] - df['value'].shift(3)
    df['Momentum_6M'] = df['value'] - df['value'].shift(6)
    df['Momentum_12M'] = df['value'] - df['value'].shift(12)

    # Z-Scores
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

    return df
# Features: CPI Data


def create_features_for_cpi_data(df):
    # Convert index to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date')

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

    return df

# Features: GDP Data


def create_features_for_gdp_data(df):
    # Ensure 'date' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date')

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

    return df

# Features: Industrial Production Data


def create_features_for_industrial_production_data(df):
    # Ensure 'date' is set as index for easier time series operations
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

    return df

    # Features: Interest Rate Data


def create_features_for_interest_rate_data(df):
    # Ensure 'date' is set as index for easier time series operations
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

    return df

# Feature: Labor Force Participation Data


def create_features_for_labor_force_participation_rate_data(df):
    # Ensure 'date' is set as index for easier time series operations
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

    return df

# Features: Nonfarm Payroll Employment Data


def create_features_for_nonfarm_payroll_employment_data(df):
    # Ensure 'date' is set as index for easier time series operations
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date')

    # Lag Features: Useful to understand changes over time
    df['Lag_1'] = df['value'].shift(1)
    df['Lag_3'] = df['value'].shift(3)
    df['Lag_12'] = df['value'].shift(12)

    # Rolling Statistics: Moving averages and standard deviations
    df['Rolling_Mean_3M'] = df['value'].rolling(window=3).mean()
    df['Rolling_Mean_6M'] = df['value'].rolling(window=6).mean()
    df['Rolling_Mean_12M'] = df['value'].rolling(window=12).mean()
    df['Rolling_Std_3M'] = df['value'].rolling(window=3).std()
    df['Rolling_Std_6M'] = df['value'].rolling(window=6).std()
    df['Rolling_Std_12M'] = df['value'].rolling(window=12).std()

    # Percentage Change: Month-over-Month and Year-over-Year
    df['MoM_Percentage_Change'] = df['value'].pct_change()
    df['YoY_Percentage_Change'] = df['value'].pct_change(periods=12)

    # Cumulative Sum and Product
    df['Cumulative_Sum'] = df['value'].cumsum()
    df['Cumulative_Product'] = (1 + df['value'].pct_change()).cumprod()

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Z-Score: Normalizing the values based on mean and standard deviation
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Seasonal Decomposition: Decompose the series into trend, seasonal, and residual components
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Days Since Last Peak/Trough: Time since the last highest/lowest value
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # RSI (Relative Strength Index)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

# Features: Personal Consumption Expenditures Data


def create_features_for_personal_consumption_expenditures(df):
    # Ensure 'date' is set as index for easier time series operations
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

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # RSI (Relative Strength Index)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

# Features: PPI Data


def create_features_for_ppi_data(df):
    # Ensure 'date' is set as index for easier time series operations
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

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # RSI (Relative Strength Index)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

# Features: Unemployment Rate Data


def create_features_for_unemployment_rate_data(df):
    # Ensure 'date' is set as index for easier time series operations
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

    # Exponential Moving Average (EMA)
    df['EMA_12'] = df['value'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['value'].ewm(span=50, adjust=False).mean()

    # Rate of Change (ROC)
    df['ROC'] = df['value'].diff(12) / df['value'].shift(12)

    # Z-Score
    df['Z_Score'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Seasonal Decomposition
    decomposition = seasonal_decompose(
        df['value'], model='multiplicative', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

    # Days Since Last Peak/Trough
    peak_idx = df['value'].expanding().apply(
        lambda x: x.idxmax().timestamp(), raw=False)
    trough_idx = df['value'].expanding().apply(
        lambda x: x.idxmin().timestamp(), raw=False)
    df['Days_Since_Peak'] = (df.index.map(
        pd.Timestamp.timestamp) - peak_idx).apply(lambda x: pd.Timedelta(seconds=x).days)
    df['Days_Since_Trough'] = (df.index.map(
        pd.Timestamp.timestamp) - trough_idx).apply(lambda x: pd.Timedelta(seconds=x).days)

    # RSI (Relative Strength Index)
    delta = df['value'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

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
    print("DataFrame before feature creation:")
    print(df.head(), df.info())

    if df.empty:
        print("Warning: The DataFrame is empty before feature creation.")
        return pd.DataFrame()

    # Create Lag feature
    df['Lag_1'] = df['close'].shift(1)

    # Create moving averages only if there are enough data points
    if len(df) >= 20:
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['close'].rolling(window=20).std()
        df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['close'].rolling(window=20).std()

    if len(df) >= 50:
        df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Create EMA and MACD features
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate RSI only if there are enough data points
    if len(df) >= 14:
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate ATR
        df['ATR'] = df[['high', 'low', 'close']].diff().abs().max(axis=1).rolling(window=14).mean()

    # Fill remaining NaN values forward to avoid dropping rows
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    print("DataFrame after feature creation:")
    print(df.head(), df.info())

    return df
    

# Normalize Data


def normalize_data(df):
    # Separate datetime columns and non-numeric columns from the rest
    datetime_columns = df.select_dtypes(include=['datetime64']).columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns

    # Remove overlaps by treating 'created_at' and 'updated_at' as datetime only
    non_numeric_columns = non_numeric_columns.difference(datetime_columns)

    numeric_df = df.drop(columns=datetime_columns.union(non_numeric_columns))

    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the numeric data
    scaled_numeric_df = pd.DataFrame(scaler.fit_transform(
        numeric_df), columns=numeric_df.columns, index=numeric_df.index)

    # Reattach the datetime and non-numeric columns without scaling
    final_df = pd.concat(
        [df[datetime_columns], df[non_numeric_columns], scaled_numeric_df], axis=1)

    return final_df, scaler

def preprocess_data(query, table_name):
    df = load_data(query)

    # Check if 'id' and either 'date' or 'timestamp' columns exist
    if 'id' not in df.columns:
        raise KeyError(f"The 'id' column must be present in the {table_name} data.")
    
    if 'date' not in df.columns and 'timestamp' not in df.columns:
        raise KeyError(f"Either 'date' or 'timestamp' column must be present in the {table_name} data.")

    # Clean the data based on the table
    cleaning_function = TABLE_CLEANING_FUNCTIONS.get(table_name)
    if cleaning_function:
        df = cleaning_function(df)

    # Convert 'date' or 'timestamp' column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print("Converted 'date' column to datetime format.")
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print("Converted 'timestamp' column to datetime format.")
        df['date'] = df['timestamp'].dt.date  # Create 'date' column from 'timestamp'

    # Check if the DataFrame is empty after cleaning
    if df.empty:
        print(f"DataFrame for {table_name} is empty after cleaning. Skipping further processing.")
        return df

    # Check if 'id' column exists before setting it as the index
    if 'id' in df.columns:
        df.set_index('id', inplace=True)
        print("ID column set as index")
    else:
        print(f"'id' column is missing after cleaning for {table_name}. Skipping setting index.")

    # Create features based on the table
    feature_creation_function = TABLE_FEATURE_FUNCTIONS.get(table_name)
    if feature_creation_function:
        df = feature_creation_function(df)
    else:
        print(f"No specific feature function found for table: {table_name}. Skipping feature creation.")

    # Normalize the data if the DataFrame is not empty
    if not df.empty:
        df, scaler = normalize_data(df)

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