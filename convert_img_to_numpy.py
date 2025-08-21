import numpy as np
import os
import cv2


root_dir = "./Dataset"

X = []
y_freshness = []
y_fruit = []
fruit_set = set()
IMAGE_SIZE = (224, 224)

for folder in os.listdir(root_dir):
    if folder.endswith(".txt"):
        continue
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    parts = folder.lower().split("_")
    if len(parts) != 2:
        print("Skip", folder)
        continue

    freshness_str, fruit_str = parts
    fruit_set.add(fruit_str)

fruit_lst = sorted(list(fruit_set))
fruit_map = {fruit: idx for idx, fruit in enumerate(fruit_lst)}
print("Fruit mapping:", fruit_map)

for folder in os.listdir(root_dir):
    if folder.endswith(".txt"):
        continue
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    parts = folder.lower().split("_")
    if len(parts) != 2:
        continue

    freshness_str, fruit_str = parts
    freshness_label = 0 if freshness_str == 'fresh' else 1
    fruit_label = fruit_map[fruit_str]

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print("Không đọc được ảnh", img_path)
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        X.append(img)
        y_freshness.append(freshness_label)
        y_fruit.append(fruit_label)


X = np.array(X)
y_freshness = np.array(y_freshness)
y_fruit = np.array(y_fruit)

np.save('X.npy', X)
np.save('y_freshness.npy', y_freshness)
np.save('y_fruit.npy', y_fruit)


