import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
import matplotlib.pyplot as plt

# 📥 Load Excel Data
df = pd.read_excel("healthcareDataset.xlsx")

# 🧹 Preprocessing
df = df.dropna()  # Remove missing rows
X = df.drop("Defects", axis=1).values
y = df["Defects"].values

# 🔄 Normalize Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🧩 Reshape for CNN (e.g., 8x8 if 64 features)
num_features = X_scaled.shape[1]
side = int(np.ceil(np.sqrt(num_features)))  # Make it square
pad_size = side**2 - num_features
X_padded = np.pad(X_scaled, ((0,0),(0,pad_size)), mode='constant')
X_reshaped = X_padded.reshape(-1, side, side, 1)

# 🧪 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# 🧠 CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(side, side, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🚀 Train Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 📈 Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# 🧪 Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.2f}")

# 🔍 Predict on New Sample
sample = X_test[0].reshape(1, side, side, 1)
prediction = model.predict(sample)
print("Prediction:", "Defective" if prediction[0][0] > 0.5 else "Non-defective")