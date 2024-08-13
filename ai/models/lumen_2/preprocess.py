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
    engine = create_engine(db_url)
    print("Engine created, attempting to connect...")
    connection = engine.connect()
    print("Connection successful!")
    return connection


def load_data(query):
    try:
        connection = get_db_connection()
        df = pd.read_sql(query, connection)
        connection.close()
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise e


# Data Cleaning
# Data Cleaning: average_hourly_earnings_data
def clean_average_hourly_earnings_data(df):
    # Handle missing values
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df

# Data Cleaning: consumer_confidence_data


def clean_consumer_confidence_data(df):
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

    return df

# Data Cleaning: consumer_confidence_data


def clean_consumer_sentiment_data(df):
    # Handle missing values
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df


# Data Cleaning: core_inflation_data


def clean_core_inflation_data(df):
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

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
    print(f"Removed {before_outlier_removal -
          after_outlier_removal} outlier rows.")

    return df

# Data Cleaning: CPI Data


def clean_cpi_data(df):
    # Handle missing values
    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])

    # Handle outliers (if necessary)
    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df


# Data Cleaning: GDP Data


def clean_gdp_data(df):
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values (optional, depending on the dataset)
    df = df.dropna(subset=['value'])

    return df

# Data Cleaning: Industrial PRoduction Data


def clean_industrial_production_data(df):
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values
    df = df.dropna(subset=['value'])

    # Handle outliers (using Z-score method)
    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    df = df[(z_scores > -3) & (z_scores < 3)]

    return df

# Data Cleaning: Interest Rate Data


def clean_interest_rate_data(df):
    # Convert 'date' to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'series_id'])

    # Handle missing values
    df = df.dropna(subset=['value'])

    # Handle outliers (using Z-score method)
    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    df = df[(z_scores > -3) & (z_scores < 3)]

    return df

# Data Cleaning: Labor Forcec Participation Rate Data


def clean_labor_force_participation_rate_data(df):
    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Remove any duplicates based on 'date'
    df = df.drop_duplicates(subset=['date'])

    # Handle missing values if any
    df = df.dropna(subset=['value'])

    return df


# Features
# Features: Average Hourly Earnings Data


def create_features_for_average_hourly_earnings(df):
    # Convert index to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date')

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

# Features: Average Hourly Earnings Data


def create_features_for_consumer_confidence_data(df):
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
    if len(df) >= 12:  # Ensure there's at least 12 months of data
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

    return df

# Features: Create Features for Cosumer Sentiment Data


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
    # Convert index to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df = df.set_index('date')

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
    decomposition = seasonal_decompose(
        df['value'], model='additive', period=12)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid

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

# Preprocess Data


def preprocess_data(query, table_name):
    df = load_data(query)

    # Clean the data based on the table
    cleaning_function = TABLE_CLEANING_FUNCTIONS.get(table_name)
    if cleaning_function:
        df = cleaning_function(df)

    # Create features based on the table
    if table_name == "average_hourly_earnings_data":
        df = create_features_for_average_hourly_earnings(df)
    # Add more conditions for other tables here...

    df, scaler = normalize_data(df)

    return df


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

}

# Main
if __name__ == "__main__":
    # Set the correct path for saving the processed data
    test_dir = os.path.join(os.getcwd(), 'data', 'lumen_2', 'test')
    os.makedirs(test_dir, exist_ok=True)

    for table_name, cleaning_function in TABLE_CLEANING_FUNCTIONS.items():
        print(f"Processing table: {table_name}")

        # Construct the query to fetch the data for the current table
        query = f"SELECT * FROM {table_name}"

        # Load, clean, and process the data
        df = load_data(query)
        cleaned_df = cleaning_function(df)

        # Create features based on the table
        if table_name == "average_hourly_earnings_data":
            feature_df = create_features_for_average_hourly_earnings(
                cleaned_df)
        elif table_name == "consumer_confidence_data":
            feature_df = create_features_for_consumer_confidence_data(
                cleaned_df)
        elif table_name == "consumer_sentiment_data":
            feature_df = create_features_for_consumer_sentiment_data(
                cleaned_df)
        elif table_name == "core_inflation_data":
            feature_df = create_features_for_core_inflation_data(cleaned_df)
        elif table_name == "cpi_data":
            feature_df = create_features_for_cpi_data(cleaned_df)
        elif table_name == "gdp_data":
            feature_df = create_features_for_gdp_data(cleaned_df)
        elif table_name == "industrial_production_data":
            feature_df = create_features_for_industrial_production_data(
                cleaned_df)
        elif table_name == "interest_rate_data":
            feature_df = create_features_for_interest_rate_data(cleaned_df)
        elif table_name == "labor_force_participation_rate_data":
            feature_df = create_features_for_labor_force_participation_rate_data(
                cleaned_df)

        else:
            feature_df = cleaned_df  # In case the table does not have a specific feature function

        # Log the DataFrame to check if it is empty
        print(f"Feature DataFrame for {table_name}:\n{feature_df.head()}")

        if feature_df.empty:
            print(f"Warning: The feature DataFrame for {
                  table_name} is empty. Skipping normalization and saving.")
            continue

        normalized_df, scaler = normalize_data(feature_df)

        # Save the cleaned data to the test directory
        output_path = os.path.join(
            test_dir, f"test_processed_{table_name}.csv")
        print(f"Saving file to {output_path}")  # Log file path
        normalized_df.to_csv(output_path, index=False)
        # Check if file exists
        print(f"File exists: {os.path.exists(output_path)}")
        print(f"{table_name} processing completed and saved to {output_path}")
