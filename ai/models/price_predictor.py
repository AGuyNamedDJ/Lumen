import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to connect to the database and fetch data


def fetch_data():
    # Placeholder connection string, replace with your actual database URL
    database_url = "postgresql://username:password@localhost:5432/yourdbname"
    engine = create_engine(database_url)
    query = "SELECT * FROM your_table_name"  # Adjust the query as needed
    df = pd.read_sql(query, engine)
    return df


# Load data from database
df = fetch_data()

# Process your SPX data
# Assuming the dataframe `df` has columns 'Open', 'High', 'Low', 'Close', 'Volume'
# Here you might need to preprocess or engineer features as needed
df['feature'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
df['target'] = df['Close']  # Predicting closing price

# Assuming 'target' is what you want to predict
X = df[['feature']]  # Features (input)
y = df['target']  # Target (output)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model Definition
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape features for LSTM Layer
X_train_reshaped = X_train.values.reshape(
    (X_train.shape[0], X_train.shape[1], 1))

# Model Training
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32)

# If needed, add code here to evaluate the model on test data
# X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
# y_pred = model.predict(X_test_reshaped)
# Compare y_pred with y_test to evaluate the model
