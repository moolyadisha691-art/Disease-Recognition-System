import os
import cv2
import numpy as np

dataset_path = "dataset"

categories = ["acne", "eczema", "psoriasis"]

X = []
y = []

img_size = 224

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

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))
print("Data shape:", X.shape)
print("Labels:", y)