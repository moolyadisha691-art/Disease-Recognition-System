import os
import cv2
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# Dataset location
dataset_path = "dataset"

# Disease categories
categories = ["acne", "eczema", "psoriasis"]

# Image size
img_size = 224

X = []
y = []

# Read images
for label, category in enumerate(categories):

    folder_path = os.path.join(dataset_path, category)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path)

        if img is not None:

            img = cv2.resize(img, (img_size, img_size))

            img = img / 255.0

            X.append(img)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Total images:", len(X))
print("Data shape:", X.shape)
print("Labels:", y)

# Create CNN model
model = Sequential([

    Input(shape=(224, 224, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),

    Dense(3, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=2
)

# Save trained model
model.save("disease_model.keras")

print("Model saved successfully!")