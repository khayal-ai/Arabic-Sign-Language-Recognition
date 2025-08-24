import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ===== Load Dataset =====
data_dir = "dataset/"
categories = ["Aleff", "Baa", "Taa", "Thaa", "Jeem", "Haa", "Khaa"]
img_size = 64

data = []
for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)  
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (img_size, img_size))
            data.append([img_array, class_num])
        except Exception as e:
            pass

# ===== Prepare Features and Labels =====
X = []
y = []
for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
y = to_categorical(y, len(categories))

# ===== Split Dataset =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Build CNN Model =====
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== Train Model =====
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# ===== Evaluate =====
loss, acc = model.evaluate(X_test, y_test)
print(f"ACCURACY: {acc*100:.2f}%")

# ===== Save Model =====
model.save("sign_model.h5")

# ===== Plot Accuracy and Loss =====
plt.figure(figsize=(12,5))

# Accuracy Plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
