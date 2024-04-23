import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Dummy Data Preparation (Replace this with your actual SPX data loading logic)
data = {
    'date': pd.date_range(start='1/1/2020', periods=100),
    'feature': np.random.rand(100),
    'target': np.random.rand(100)
}
df = pd.DataFrame(data)

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

# Expansion
