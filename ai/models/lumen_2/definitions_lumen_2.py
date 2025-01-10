import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Concatenate, Layer, MultiHeadAttention, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="ReduceMeanLayer")
class ReduceMeanLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Averages over the time dimension => shape (batch, features)
        return tf.reduce_mean(inputs, axis=1)

    def get_config(self):
        base_config = super().get_config()
        return base_config


def create_hybrid_model(
    input_shape,
    num_lstm_layers=2,          # number of stacked LSTM layers
    lstm_hidden=128,            # hidden dimension for LSTM
    num_transformer_heads=2,    # fewer heads for smaller daily data
    dropout_rate=0.3,           # more dropout to reduce overfit
    learning_rate=3e-4          # lower LR can help stability
):
    # Input layer
    inputs = Input(shape=input_shape, name='Input')

    # 1) LSTM stack
    x = inputs
    for i in range(num_lstm_layers):
        x = LSTM(lstm_hidden, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)

    # 2) Optional Multi-Head Self-Attention
    #    Key_dim is set to the LSTM hidden size for consistency
    attn = MultiHeadAttention(num_heads=num_transformer_heads, key_dim=lstm_hidden)(
        x, x
    )
    attn = Dense(lstm_hidden, activation='relu')(attn)
    attn = Dropout(dropout_rate)(attn)

    # 3) Reduce both attention output & final LSTM output
    attn_mean = ReduceMeanLayer()(attn)  # shape (batch, lstm_hidden)
    lstm_mean = ReduceMeanLayer()(x)     # shape (batch, lstm_hidden)
    merged = Concatenate()([lstm_mean, attn_mean])  # shape (batch, 2*lstm_hidden)

    # 4) Dense layers on top
    #    You can add more than one hidden Dense if you have enough data
    x = Dense(128, activation='relu')(merged)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # 5) Final output (regression => 1 dimension)
    outputs = Dense(1, activation='linear')(x)

    # Build & compile
    model = Model(inputs, outputs, name='Daily_LSTM_Attn_Model')
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model