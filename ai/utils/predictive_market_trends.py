import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Database connection


def get_db_connection():
    engine = create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return engine

# Fetch historical data


def fetch_historical_data(engine):
    query = "SELECT timestamp, open, high, low, close, volume FROM historical_spx ORDER BY timestamp ASC"
    df = pd.read_sql(query, engine)
    return df

# Train predictive model


def train_predictive_model():
    engine = get_db_connection()
    df = fetch_historical_data(engine)

    # Feature engineering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['price_change'] = df['close'].pct_change()
    df.dropna(inplace=True)

    # Define features and target
    X = df[['open', 'high', 'low', 'volume']]
    y = df['close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model


if __name__ == "__main__":
    model = train_predictive_model()
    print("Model trained successfully.")
