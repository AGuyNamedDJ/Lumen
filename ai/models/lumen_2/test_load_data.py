import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()


def get_db_connection():
    db_url = os.getenv('DB_URL')
    engine = create_engine(db_url)
    return engine.connect()


def load_data(query):
    connection = get_db_connection()
    df = pd.read_sql(query, connection)
    connection.close()
    return df


if __name__ == "__main__":
    # Example query to test loading data
    # Adjust the table name and query as per your DB schema
    query = "SELECT * FROM historical_spx LIMIT 10;"
    try:
        df = load_data(query)
        print("Data loaded successfully!")
        print(df.head())  # Print the first few rows of the DataFrame
    except Exception as e:
        print(f"Failed to load data: {e}")
