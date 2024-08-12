import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_db_connection():
    db_url = os.getenv('DB_URL')
    engine = create_engine(db_url)
    return engine.connect()


if __name__ == "__main__":
    try:
        connection = get_db_connection()
        print("Connection successful!")
        connection.close()
    except Exception as e:
        print(f"Connection failed: {e}")
