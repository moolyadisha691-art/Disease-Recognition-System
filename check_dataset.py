import os

dataset_path = "dataset"

for disease in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, disease)

    if os.path.isdir(folder_path):
        images = os.listdir(folder_path)

        print(disease, ":", len(images), "images")