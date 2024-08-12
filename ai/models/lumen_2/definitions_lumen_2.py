from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def create_hybrid_model(input_shape, num_lstm_layers=2, num_cnn_filters=64, num_transformer_heads=4, dropout_rate=0.2):
    """
    Create a hybrid model combining CNN, LSTM, and Transformer layers.

    :param input_shape: Shape of the input data.
    :param num_lstm_layers: Number of LSTM layers.
    :param num_cnn_filters: Number of CNN filters.
    :param num_transformer_heads: Number of heads for the Transformer.
    :param dropout_rate: Dropout rate to prevent overfitting.
    :return: Compiled hybrid model.
    """
    inputs = Input(shape=input_shape)

    # CNN Layer
    cnn_out = Conv1D(filters=num_cnn_filters, kernel_size=3,
                     padding='same', activation='relu')(inputs)

    # LSTM Layers
    lstm_out = cnn_out
    for _ in range(num_lstm_layers):
        lstm_out = LSTM(100, return_sequences=True)(lstm_out)
        lstm_out = Dropout(dropout_rate)(lstm_out)

    # Transformer Layer
    transformer_out = tf.keras.layers.MultiHeadAttention(
        num_heads=num_transformer_heads, key_dim=input_shape[-1])(lstm_out, lstm_out)
    # Aggregating across time steps
    transformer_out = tf.reduce_mean(transformer_out, axis=1)

    # Concatenate LSTM and Transformer outputs
    combined = Concatenate()(
        [tf.reduce_mean(lstm_out, axis=1), transformer_out])

    # Final Dense Layers
    dense_out = Dense(100, activation='relu')(combined)
    outputs = Dense(1)(dense_out)

    # Compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')

    return model


# Main function to test model creation
if __name__ == "__main__":
    # Example input shape (time steps, features)
    input_shape = (30, 10)  # 30 time steps, 10 features
    model = create_hybrid_model(input_shape, num_lstm_layers=2,
                                num_cnn_filters=64, num_transformer_heads=4, dropout_rate=0.3)
    model.summary()
