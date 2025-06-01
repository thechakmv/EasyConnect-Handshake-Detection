import pandas as pd
import numpy as np
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

#This code trains on handshakes and non handshakes

def load_labeled_data(directory, label):
    data = []
    for file in os.listdir(directory):
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            data.append((row.values, label))
    return data

handshake_data = load_labeled_data('Python/data/handshake', 1)
non_handshake_data = load_labeled_data('Python/data/non_handshake', 0)

# Combine and shuffle
all_data = handshake_data + non_handshake_data
np.random.shuffle(all_data)

# Split features and labels
X = np.array([x for x, _ in all_data])
y = np.array([y for _, y in all_data])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Dense(64, activation='relu', input_shape=(6,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

model.save('handshake_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('./handshake_project/include/handshake_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as 'handshake_model.tflite'")

# Predict on test set
y_pred_probs = model.predict(X_test).flatten()
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=["Non-Handshake", "Handshake"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Handshake", "Handshake"], yticklabels=["Non-Handshake", "Handshake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()