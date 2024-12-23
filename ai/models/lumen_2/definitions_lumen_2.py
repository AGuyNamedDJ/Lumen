import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout,
    Concatenate, Layer, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom", name="ReduceMeanLayer")
class ReduceMeanLayer(Layer):
    def __init__(self, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Average over the sequence dimension (axis=1)
        return tf.reduce_mean(inputs, axis=1)

    def get_config(self):
        base_config = super(ReduceMeanLayer, self).get_config()
        return base_config

def create_hybrid_model(input_shape,
                        num_lstm_layers=2,
                        num_cnn_filters=64,
                        num_transformer_heads=4,
                        dropout_rate=0.2):
    """
    Create a hybrid model combining CNN, LSTM, and Transformer layers,
    returning a single (1-dimensional) output for a single forecast horizon.

    Parameters:
    - input_shape (tuple): (sequence_length, num_features)
    - num_lstm_layers (int): Number of LSTM layers.
    - num_cnn_filters (int): Number of filters in the CNN layer.
    - num_transformer_heads (int): Number of heads for MultiHeadAttention.
    - dropout_rate (float): Dropout rate to apply.

    Returns:
    - model (tf.keras.Model): The compiled Keras model, producing 1 scalar output.
    """
    # Input layer expects data shaped like (batch_size, sequence_length, num_features)
    inputs = Input(shape=input_shape, name='Input')

    # CNN layer
    cnn_out = Conv1D(
        filters=num_cnn_filters,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='CNN_Conv1D'
    )(inputs)

    # LSTM stack
    lstm_out = cnn_out
    for i in range(num_lstm_layers):
        lstm_out = LSTM(100, return_sequences=True, name=f'LSTM_{i+1}')(lstm_out)
        lstm_out = Dropout(dropout_rate, name=f'Dropout_LSTM_{i+1}')(lstm_out)

    # Transformer attention
    transformer_out = MultiHeadAttention(
        num_heads=num_transformer_heads,
        key_dim=100,
        name='Transformer_MHA'
    )(lstm_out, lstm_out)

    # Pool (reduce) along the sequence dimension
    transformer_out = ReduceMeanLayer(name='Transformer_ReduceMean')(transformer_out)
    lstm_mean_out   = ReduceMeanLayer(name='LSTM_ReduceMean')(lstm_out)

    # Concatenate LSTM and Transformer outputs
    combined = Concatenate(name='Concatenate')([lstm_mean_out, transformer_out])

    # Dense + Dropout
    dense_out = Dense(100, activation='relu', name='Dense_1')(combined)
    dense_out = Dropout(dropout_rate, name='Dropout_Dense_1')(dense_out)

    # Final single scalar output
    outputs = Dense(1, activation='linear', name='Output')(dense_out)

    model = Model(inputs, outputs, name='Hybrid_CNN_LSTM_Transformer_Model')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model