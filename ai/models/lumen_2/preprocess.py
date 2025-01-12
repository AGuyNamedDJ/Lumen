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

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    df = df[['id', 'date', 'value']]

    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])
        print("Dropped rows with missing 'value'.")

    before_duplicates = len(df)
    df = df.drop_duplicates(subset=['date'])
    after_duplicates = len(df)
    print(f"Removed {before_duplicates - after_duplicates} duplicate rows.")

    mean_value = df['value'].mean()
    std_value = df['value'].std()
    upper_bound = mean_value + 3 * std_value
    lower_bound = mean_value - 3 * std_value

    before_outliers = len(df)
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    after_outliers = len(df)
    print(f"Removed {before_outliers - after_outliers} outlier rows.")
    print(f"DataFrame after outlier removal: {df.head()}")

    if df.empty:
        print("DataFrame is empty after cleaning. Skipping further processing.")
        return df

    df.set_index('id', inplace=True)
    print("ID column set as index")

    print("Final cleaned DataFrame:")
    print(df.head())

    return df


def clean_core_inflation_data(df):
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    if 'updated_at' in df.columns:
        df = df.drop(columns=['updated_at'])

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError(
            "The 'date' column is missing from the core inflation data.")

    df['date'] = pd.to_datetime(df['date'])

    df = df[['id', 'date', 'value']]

    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    df = df.drop_duplicates(subset=['date'])

    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df

# Data Cleaning: CPI Data
def clean_cpi_data(df):
    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])

    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    df = df[['id', 'date', 'value']]

    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    df = df.drop_duplicates(subset=['date'])

    upper_bound = df['value'].mean() + 3 * df['value'].std()
    lower_bound = df['value'].mean() - 3 * df['value'].std()
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

    return df

# Data Cleaning: GDP Data


def clean_gdp_data(df):
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df = df.drop(columns=['updated_at'], errors='ignore')

    df['id'] = df['id'].astype(int)

    df['date'] = pd.to_datetime(df['date'])

    df = df[['id', 'date', 'value']]

    df = df.drop_duplicates(subset=['date'])

    df = df.dropna(subset=['value'])

    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    return df

# Data Cleaning: Industrial PRoduction Data


def clean_industrial_production_data(df):
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    if 'updated_at' in df.columns:
        df = df.drop(columns=['updated_at'])

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])

    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    df = df[['id', 'date', 'value']]

    df['id'] = df['id'].astype(int)

    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    df = df.drop_duplicates(subset=['date'])

    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    df = df[(z_scores > -3) & (z_scores < 3)]

    return df


# Data Cleaning: Interest Rate Data
def clean_interest_rate_data(df):
    if 'created_at' in df.columns:
        df = df.drop(columns=['created_at'])
    if 'updated_at' in df.columns:
        df = df.drop(columns=['updated_at'])

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])

    df = df.loc[:, ~df.columns.duplicated()]
    print("Dropped duplicate columns, if any existed.")

    df = df[['id', 'date', 'series_id', 'value']]

    df['id'] = df['id'].astype(int)

    if df['value'].isnull().sum() > 0:
        df = df.dropna(subset=['value'])

    df = df.drop_duplicates(subset=['date'])

    z_scores = (df['value'] - df['value'].mean()) / df['value'].std()
    df = df[(z_scores > -3) & (z_scores < 3)]

    return df

# Data Cleaning: Labor Force Participation Rate Data


def clean_labor_force_participation_rate_data(df):
    print("Beginning to clean labor force participation rate data.")

    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df

# Data Cleaning: Nonfarm Payroll Employment Data


def clean_nonfarm_payroll_employment_data(df):
    print("Beginning to Nonfarm Payroll Employment Data.")

    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df


# Data Cleaning: Personal Consumption Expenditures Data
def clean_personal_consumption_expenditures_data(df):
    print("Beginning to clean Personal Consumption Expenditures Data.")

    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df


# Data Cleaning: PPI Data
def clean_ppi_data(df):
    print("Beginning to clean PPI Data.")

    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

    df.set_index('id', inplace=True)
    print("ID column set as index.")

    return df


# Data Cleaning: Unemployment Rate Data
def clean_unemployment_rate_data(df):
    print("Beginning to clean Unemployment Rate Data.")

    df = df.drop(columns=['created_at', 'updated_at'], errors='ignore')

    if 'date' not in df.columns and 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'date'})

    if 'date' not in df.columns:
        raise KeyError("The 'date' column is missing from the data.")

    df['date'] = pd.to_datetime(df['date'])
    print("Converted 'date' column to datetime format.")

    df = df.drop_duplicates(subset=['date'])
    print("Dropped duplicates based on 'date' column.")

    df = df.dropna(subset=['value'])
    print("Dropped rows with missing 'value'.")

    df = df[['id', 'date', 'value']]
    print("Kept only relevant columns: 'id', 'date', 'value'.")

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
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], how='any', inplace=True)

    for column in ['open', 'high', 'low', 'close']:
        upper_bound = df[column].mean() + 3 * df[column].std()
        lower_bound = df[column].mean() - 3 * df[column].std()
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df = df[df['volume'] > 0]

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

    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], how='any', inplace=True)

    for column in ['open', 'high', 'low', 'close']:
        upper_bound = df[column].mean() + 3 * df[column].std()
        lower_bound = df[column].mean() - 3 * df[column].std()
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df = df[df['volume'] > 0]
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

    df.dropna(subset=['open', 'high', 'low', 'close'], how='all', inplace=True)

    df.fillna(method='ffill', inplace=True)

    upper_bound = df['close'].mean() + 3 * df['close'].std()
    lower_bound = df['close'].mean() - 3 * df['close'].std()
    df = df[(df['close'] >= lower_bound) & (df['close'] <= upper_bound)]

    df.set_index('id', inplace=True)
    return df

# Data Cleaning: Real Time SPX
def clean_real_time_spx_data(df):
    if 'id' not in df.columns:
        raise KeyError("Missing 'id'.")
    if 'timestamp' not in df.columns:
        raise KeyError("Missing 'timestamp'.")

    df['id'] = df['id'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    df = df.resample('3T').last()

    df['current_price'] = df['current_price'].ffill()

    df.dropna(subset=['current_price'], inplace=True)

    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'timestamp'}, errors='ignore')

    return df

# Data Cleaning: Real Time SPY
def clean_real_time_spy_data(df):
    if 'id' not in df.columns:
        raise KeyError("Missing 'id'.")
    if 'timestamp' not in df.columns:
        raise KeyError("Missing 'timestamp'.")

    df['id'] = df['id'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    df = df.resample('3T').last()

    df['current_price'] = df['current_price'].ffill()

    df.dropna(subset=['current_price'], inplace=True)

    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'timestamp'}, errors='ignore')

    return df

# Data Cleaning: Real Time VIX
def clean_real_time_vix_data(df):
    if 'id' not in df.columns:
        raise KeyError("The 'id' column is missing from the data.")
    if 'timestamp' not in df.columns:
        raise KeyError("The 'timestamp' column is missing from the data.")

    df['id'] = df['id'].astype(int)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if df.index.name != 'timestamp':
        df.set_index('timestamp', inplace=True, drop=False)
        print("'timestamp' column used as index.")

    df = df[~df.index.duplicated(keep='first')]

    df.sort_index(inplace=True)
    print("DataFrame sorted by timestamp index.")

    if 'timestamp' not in df.columns:
        df.reset_index(drop=False, inplace=True)
        print("Index reset, 'timestamp' kept as a column.")
    else:
        df.reset_index(drop=True, inplace=True)
        print("Index reset without duplicating 'timestamp' column.")

    df.set_index('id', inplace=True)
    print("ID column set as index.")

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

    cleaning_function = TABLE_CLEANING_FUNCTIONS.get(table_name)
    if cleaning_function:
        df = cleaning_function(df)
        print(f"Cleaned {table_name} data.")
    else:
        print(f"No cleaning function found for {table_name}; skipping...")

    if table_name in ['real_time_spx', 'real_time_spy']:
        if df.index.name != 'timestamp':
            df.set_index('timestamp', drop=False, inplace=True)
        df = df.resample('3min').last()
        df.dropna(subset=['current_price'], how='any', inplace=True)
        df.index.name = 'tmp_index'
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={'tmp_index': 'timestamp'}, inplace=True)
        df['target_1h'] = df['current_price'].shift(-20)
        print(f"Added single horizon (1h) to {table_name}.")

    df = df.loc[:, ~df.columns.duplicated()]
    print(f"Dropped duplicate columns, if any, for {table_name}.")

    target_col = None
    if "target_1h" in df.columns:
        target_col = df["target_1h"].copy()
        df.drop(columns=["target_1h"], inplace=True)

    df, _ = normalize_data(df)
    print(f"Data normalized for {table_name}.")

    if target_col is not None:
        df["target_1h"] = target_col

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_columns].values
    X_3D = create_sequences(X, seq_len=60)
    return df, X_3D

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