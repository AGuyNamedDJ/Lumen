from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_lstm_model(input_shape, num_layers=2, num_neurons=50):
    """
    Create an LSTM model with specified hyperparameters.

    :param input_shape: Shape of the input data.
    :param num_layers: Number of LSTM layers.
    :param num_neurons: Number of neurons in each LSTM layer.
    :return: Compiled LSTM model.
    """
    model = Sequential()

    # Add LSTM layers
    for i in range(num_layers):
        if i == 0:
            model.add(LSTM(num_neurons, return_sequences=(
                i < num_layers - 1), input_shape=input_shape))
        else:
            model.add(LSTM(num_neurons, return_sequences=(i < num_layers - 1)))

    # Add a Dense layer for output
    model.add(Dense(1))

    model.compile(optimizer='Adam', loss='mean_squared_error')

    return model


# Main function to test model creation
if __name__ == "__main__":
    input_shape = (1, 7)  # Example input shape (time steps, features)
    model = create_lstm_model(input_shape, num_layers=3, num_neurons=100)
    model.summary()
