import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout,
    Concatenate, Layer, MultiHeadAttention, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom", name="ReduceMeanLayer")
class ReduceMeanLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        # Average over sequence (axis=1)
        return tf.reduce_mean(inputs, axis=1)
    def get_config(self):
        return super().get_config()

def create_hybrid_model(
    input_shape,
    num_lstm_layers=2,
    num_cnn_filters=64,
    num_transformer_heads=4,
    dropout_rate=0.2
):
    # Input
    inputs = Input(shape=input_shape, name='Input')
    # CNN + BN
    x = Conv1D(filters=num_cnn_filters, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    # LSTM stack
    for i in range(num_lstm_layers):
        x = LSTM(100, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
    # Transformer
    attn = MultiHeadAttention(num_heads=num_transformer_heads, key_dim=100)(x, x)
    attn = Dense(100, activation='relu')(attn)
    attn = Dropout(dropout_rate)(attn)
    # Reduce
    attn_mean = ReduceMeanLayer()(attn)
    lstm_mean = ReduceMeanLayer()(x)
    merged = Concatenate()([lstm_mean, attn_mean])
    # Dense
    dense_out = Dense(100, activation='relu')(merged)
    dense_out = Dropout(dropout_rate)(dense_out)
    outputs = Dense(1, activation='linear')(dense_out)
    # Build
    model = Model(inputs, outputs, name='Hybrid_CNN_LSTM_Transformer_Model')
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return model