import os
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to load data from a directory
def load_data(directory):
    data = []
    for file in os.listdir(directory):
        df = pd.read_csv(os.path.join(directory, file))
        for i in range(len(df)):
            data.append(df.iloc[i].values)
    return np.array(data)

# Load only handshake data for training
X_handshake = load_data('Python/data/handshake')

# Load non-handshake data for evaluation
X_non_handshake = load_data('Python/data/non_handshake')

scaler = StandardScaler()
#X_handshake = scaler.fit_transform(X_handshake)
#X_non_handshake = scaler.transform(X_non_handshake)

# Train/test split on handshake data
X_train, X_val = train_test_split(X_handshake, test_size=0.2, random_state=42)

# Build autoencoder
input_dim = X_handshake.shape[1]


model = Sequential([
    Dense(64, input_shape=(input_dim,)),
    LeakyReLU(alpha=0.01),
    
    Dense(32),
    LeakyReLU(alpha=0.01),
    
    Dense(16),
    LeakyReLU(alpha=0.01),
    
    Dense(8),
    LeakyReLU(alpha=0.01),
    
    Dense(16),
    LeakyReLU(alpha=0.01),
    
    Dense(32),
    LeakyReLU(alpha=0.01),
    
    Dense(64),
    LeakyReLU(alpha=0.01),
    
    Dense(input_dim, activation='linear')  # Output layer for autoencoder (no activation or linear)
])

# Compile and train model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=100, validation_data=(X_val, X_val))

# Save model
model.save('handshake_autoencoder.h5')

# Convert to TFLite and save
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('./handshake_project/include/handshake_autoencoder.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite autoencoder model saved as 'handshake_autoencoder.tflite'")

# -------- Evaluation --------

# Function to compute reconstruction error
def get_reconstruction_errors(model, data):
    reconstructed = model.predict(data)
    mse = np.mean(np.square(data - reconstructed), axis=1)
    return mse

# Compute reconstruction errors
errors_val_handshake = get_reconstruction_errors(model, X_val)
errors_non_handshake = get_reconstruction_errors(model, X_non_handshake)

# Choose threshold (95th percentile of handshake validation errors)
threshold = np.percentile(errors_val_handshake, 85)

# Predict class: True if error < threshold (i.e., likely a handshake)
y_pred_handshake = errors_val_handshake < threshold
y_pred_non_handshake = errors_non_handshake < threshold

# Combine for evaluation
y_true = np.concatenate([np.ones_like(y_pred_handshake), np.zeros_like(y_pred_non_handshake)])
y_pred = np.concatenate([y_pred_handshake, y_pred_non_handshake])

# Show classification report
print("\nClassification Report (threshold = {:.5f}):".format(threshold))
print(classification_report(y_true, y_pred, target_names=["Non-Handshake", "Handshake"]))

plt.hist(errors_val_handshake, bins=50, alpha=0.6, label='Handshake')
plt.hist(errors_non_handshake, bins=50, alpha=0.6, label='Non-handshake')
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Reconstruction Error Distribution")
plt.show()