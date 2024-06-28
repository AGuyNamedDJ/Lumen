from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_model(input_shape):
    """
    Creates and compiles an LSTM model for predicting SPX percentage change.

    Parameters:
    input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
    model (Sequential): A compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


if __name__ == "__main__":
    # Example input shape: 1 timestep, 5 features
    input_shape = (1, 5)
    model = create_model(input_shape)
    # Print model summary to verify the architecture
    model.summary()
