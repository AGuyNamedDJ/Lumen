import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Environment variables
DB_URL = os.getenv('DB_URL')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Function to establish a database connection using SQLAlchemy


def get_db_connection():
    try:
        engine = create_engine(DB_URL)
        return engine
    except Exception as e:
        print(f"Error creating database connection: {e}")
        return None

# Function to fetch historical data from the database


def fetch_historical_data():
    try:
        engine = get_db_connection()
        if engine is None:
            raise Exception("Database connection failed")
        query = "SELECT * FROM historical_spx"
        data = pd.read_sql(query, engine)
        print("Historical data fetched successfully from the database.")
        return data
    except Exception as e:
        print(f"Error fetching historical data from database: {e}")
        return None

# Function to fetch live data from the database


def fetch_live_data():
    try:
        engine = get_db_connection()
        if engine is None:
            raise Exception("Database connection failed")
        query = "SELECT * FROM real_time_spx"
        data = pd.read_sql(query, engine)
        print("Live data fetched successfully from the database.")
        return data
    except Exception as e:
        print(f"Error fetching live data from database: {e}")
        return None

# Function to aggregate live data into daily OHLC format


def aggregate_live_data(live_data):
    live_data['date'] = live_data['timestamp'].dt.date
    ohlc_dict = {
        'current_price': ['first', 'max', 'min', 'last']
    }
    daily_ohlc = live_data.groupby('date').agg(ohlc_dict)
    daily_ohlc.columns = ['open', 'high', 'low', 'close']
    daily_ohlc.reset_index(inplace=True)
    return daily_ohlc

# Function to handle missing values in the dataset


def handle_missing_values(data):
    data.ffill(inplace=True)  # Forward fill missing values
    data.bfill(inplace=True)  # Backward fill missing values
    return data

# Function to normalize the data using MinMaxScaler


def normalize_data(data):
    scaler = MinMaxScaler()
    data[['open', 'high', 'low', 'close', 'volume', 'current_price', 'ema_10', 'ema_50']] = scaler.fit_transform(
        data[['open', 'high', 'low', 'close', 'volume', 'current_price', 'ema_10', 'ema_50']].fillna(0))
    return data

# Function to calculate EMA


def calculate_ema(data, span):
    return data['close'].ewm(span=span, adjust=False).mean()

# Function to preprocess the fetched data


def preprocess_data(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    data = handle_missing_values(data)

    # Calculate EMA
    data['ema_10'] = calculate_ema(data, 10)
    data['ema_50'] = calculate_ema(data, 50)

    data = normalize_data(data)
    return data

# Function to combine duplicate columns


def combine_columns(data):
    for col in ['open', 'high', 'low', 'close']:
        if f"{col}_x" in data.columns and f"{col}_y" in data.columns:
            data[col] = data[f"{col}_x"].combine_first(data[f"{col}_y"])
            data.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True)
    return data

# Function to create chat-formatted data


def create_chat_data(data):
    chat_data = []
    for i in range(1, len(data)):
        if pd.isna(data.index[i]):
            continue
        chat_data.append({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": f"Predict the closing price for {data.index[i].strftime('%Y-%m-%d %H:%M:%S')}."},
                {"role": "assistant",
                 "content": f"The closing price will be {data.iloc[i]['close']}."}
            ]
        })
    return chat_data

# Main function to orchestrate data fetching and preprocessing


def main():
    print("Starting data preprocessing...")

    historical_data = fetch_historical_data()
    live_data = fetch_live_data()

    if historical_data is not None and live_data is not None:
        daily_live_data = aggregate_live_data(live_data)

        # Combine the data
        combined_data = pd.concat(
            [historical_data, live_data, daily_live_data], ignore_index=True)
        combined_data.sort_values(by='timestamp', inplace=True)
        combined_data = combine_columns(combined_data)
        combined_data.to_csv('data/processed/combined_data.csv', index=False)
        print(
            f"Combined data shape before preprocessing: {combined_data.shape}")

        preprocessed_data = preprocess_data(combined_data)
        preprocessed_data.to_csv(
            'data/processed/preprocessed_combined_data.csv')
        print("Data preprocessed and saved successfully.")

        chat_data = create_chat_data(preprocessed_data)
        with open('data/processed/preprocessed_combined_chat_data.jsonl', 'w') as f:
            for entry in chat_data:
                f.write(f"{entry}\n")
        print("Chat data preprocessed and saved successfully.")
    else:
        print("Failed to preprocess data.")


# Entry point for the script
if __name__ == "__main__":
    main()
