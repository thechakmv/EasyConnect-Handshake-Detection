import pandas as pd
import numpy as np
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data(directory, label):
    data = []
    labels = []
    for file in os.listdir(directory):
        df = pd.read_csv(os.path.join(directory, file))
        for i in range(len(df)):
            data.append(df.iloc[i].values)
            labels.append(label)
    return np.array(data), np.array(labels)

X1, y1 = load_data('Python/data/handshake', label=1)
X2, y2 = load_data('Python/data/non_handshake', label=0)

X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Dense(64, activation='relu', input_shape=(6,))
    Dense(32, activation='relu', input_shape=(6,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

model.save('handshake_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('./handshake_project/include/handshake_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as 'handshake_model.tflite'")