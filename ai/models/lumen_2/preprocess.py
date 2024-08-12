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


def normalize_data(df):
    # Separate datetime columns from the rest
    datetime_columns = df.select_dtypes(include=['datetime64']).columns
    numeric_df = df.drop(columns=datetime_columns)

    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the numeric data
    scaled_numeric_df = pd.DataFrame(scaler.fit_transform(
        numeric_df), columns=numeric_df.columns)

    # Reattach the datetime columns without scaling
    final_df = pd.concat([df[datetime_columns], scaled_numeric_df], axis=1)

    return final_df, scaler


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

    # You can save the scaler for later use (e.g., inverse transforming predictions)
    # with open(f'{table_name}_scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)

    return df


# Dictionary mapping table names to their respective cleaning functions
TABLE_CLEANING_FUNCTIONS = {
    "average_hourly_earnings_data": clean_average_hourly_earnings_data,
    # "consumer_confidence_data": clean_consumer_confidence_data,
    # Add additional table and function mappings here...
}


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

        normalized_df, scaler = normalize_data(feature_df)

        # Save the cleaned data to the test directory
        output_path = os.path.join(
            test_dir, f"test_processed_{table_name}.csv")
        print(f"Saving file to {output_path}")  # Log file path
        normalized_df.to_csv(output_path, index=False)
        # Check if file exists
        print(f"File exists: {os.path.exists(output_path)}")
        print(f"{table_name} processing completed and saved to {output_path}")
