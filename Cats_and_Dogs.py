import numpy as np
import matplotlib.pyplot as plt 
import os # iterate through directories 
import cv2 # for image operations 



DATADIR = "C:/Users\Ahmed Khaled\Desktop\Deeplearning\Cats and Dogs\PetImages"
CATEGORIES = ["Dog", "Cat"]


IMG_SIZE = 50
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs directory
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()

import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

np.save('features.npy', x)
np.save('label.npy', y)
