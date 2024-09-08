from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf

# Custom Layer to wrap tf.reduce_mean


@register_keras_serializable()
class ReduceMeanLayer(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def get_config(self):
        config = super(ReduceMeanLayer, self).get_config()
        return config


def create_hybrid_model(input_shape, num_lstm_layers=2, num_cnn_filters=64, num_transformer_heads=4, dropout_rate=0.2):
    """
    Create a hybrid model combining CNN, LSTM, and Transformer layers.
    """
    inputs = Input(shape=input_shape)  # Modify to match your input shape

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

    # Use custom ReduceMeanLayer instead of tf.reduce_mean
    transformer_out = ReduceMeanLayer()(transformer_out)
    lstm_mean_out = ReduceMeanLayer()(lstm_out)

    # Concatenate LSTM and Transformer outputs
    combined = Concatenate()([lstm_mean_out, transformer_out])

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
    # Example input shapes (features)
    hist_input_shape = (1, 41)  # 41 features for historical data
    real_time_input_shape = (1, 36)  # 36 features for real-time data

    model = create_hybrid_model(hist_input_shape, real_time_input_shape, num_lstm_layers=2,
                                num_cnn_filters=64, num_transformer_heads=4, dropout_rate=0.3)

    # Save the model for testing
    model.save('Lumen2.keras')

    # Print the model summary
    model.summary()

    # Load the model back to test deserialization with custom layer
    from tensorflow.keras.models import load_model
    loaded_model = load_model('Lumen2.keras', custom_objects={
                              'ReduceMeanLayer': ReduceMeanLayer})

    # Check if the loaded model works
    loaded_model.summary()
