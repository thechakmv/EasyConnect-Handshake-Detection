from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load only handshake data
X, y = load_data('Python/data/handshake', label=1)  # No need for labels here

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.2)

# Autoencoder architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder on handshake data only
autoencoder.fit(X_train, X_train, 
                epochs=50, 
                batch_size=16, 
                validation_data=(X_test, X_test))

# Save model
autoencoder.save('handshake_autoencoder.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()

with open('./handshake_project/include/handshake_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("One-class TFLite model saved as 'handshake_model.tflite'")
