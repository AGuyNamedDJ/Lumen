from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Concatenate, Layer
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
import tensorflow as tf

# Custom Layer to wrap tf.reduce_mean


@register_keras_serializable(package="Custom", name="ReduceMeanLayer")
class ReduceMeanLayer(Layer):
    def __init__(self, **kwargs):
        super(ReduceMeanLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def get_config(self):
        base_config = super(ReduceMeanLayer, self).get_config()
        return base_config


def create_hybrid_model(input_shape, num_lstm_layers=2, num_cnn_filters=64,
                        num_transformer_heads=4, dropout_rate=0.2):
    """
    Create a hybrid model combining CNN, LSTM, and Transformer layers.
    """
    inputs = Input(
        shape=input_shape)  # input_shape now includes sequence_length

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
        num_heads=num_transformer_heads, key_dim=inputs.shape[-1])(lstm_out, lstm_out)

    # Use custom ReduceMeanLayer instead of tf.reduce_mean
    transformer_out = ReduceMeanLayer()(transformer_out)
    lstm_mean_out = ReduceMeanLayer()(lstm_out)

    # Concatenate LSTM and Transformer outputs
    combined = Concatenate()([lstm_mean_out, transformer_out])

    # Final Dense Layers
    dense_out = Dense(100, activation='relu')(combined)
    # Output layer for regression with linear activation
    outputs = Dense(1)(dense_out)

    # Compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')

    return model


# Main function to test model creation
if __name__ == "__main__":
    # Example input shapes (sequence_length, features)
    sequence_length = 30  # Adjust the sequence length as needed
    hist_input_shape = (sequence_length, 41)  # 41 features for historical data
    # 36 features for real-time data
    real_time_input_shape = (sequence_length, 36)

    # Create and save the historical model
    hist_model = create_hybrid_model(hist_input_shape, num_lstm_layers=2,
                                     num_cnn_filters=64, num_transformer_heads=4, dropout_rate=0.3)
    hist_model.save('Lumen2_historical.keras')

    # Create and save the real-time model
    real_time_model = create_hybrid_model(real_time_input_shape, num_lstm_layers=2,
                                          num_cnn_filters=64, num_transformer_heads=4, dropout_rate=0.3)
    real_time_model.save('Lumen2_real_time.keras')

    # Print the model summaries
    print("Historical Model Summary:")
    hist_model.summary()

    print("\nReal-Time Model Summary:")
    real_time_model.summary()

    # Load the models back to test deserialization with custom layer
    from tensorflow.keras.models import load_model
    loaded_hist_model = load_model('Lumen2_historical.keras', custom_objects={
                                   'ReduceMeanLayer': ReduceMeanLayer})
    loaded_real_time_model = load_model('Lumen2_real_time.keras', custom_objects={
                                        'ReduceMeanLayer': ReduceMeanLayer})

    # Check if the loaded models work
    print("\nLoaded Historical Model Summary:")
    loaded_hist_model.summary()

    print("\nLoaded Real-Time Model Summary:")
    loaded_real_time_model.summary()
