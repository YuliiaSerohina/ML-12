import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from skimage import io, transform


data_folder = "pic"
file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
              if os.path.isfile(os.path.join(data_folder, f))]

X_resized = []
standard_size = (300, 300)
for file_path in file_paths:
    img = io.imread(file_path)
    img_resized = transform.resize(img, standard_size, anti_aliasing=True)
    img_resized_flat = img_resized.flatten()
    X_resized.append(img_resized_flat)

X_resized = np.array(X_resized)
nbrs_resized = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_resized)

test_images = ["car1.jpeg", "fb1.jpeg"]
test_X_resized = []

for img_path in test_images:
    img = io.imread(os.path.join(data_folder, img_path))
    img_resized = transform.resize(img, standard_size, anti_aliasing=True)
    img_resized_flat = img_resized.flatten()
    test_X_resized.append(img_resized_flat)

test_X_resized = np.array(test_X_resized)
distances_resized, indices_resized = nbrs_resized.kneighbors(test_X_resized)

for i, test_image in enumerate(test_images):
    print(f"Closest images to {test_image}:")
    for j, index in enumerate(indices_resized[i]):
        print(f"{j+1}. {file_paths[index]}")


