# Standard
import os
import random

# Third-party
import cv2
import numpy as np
from tqdm import tqdm # Progress bars

# Local


num_images = int(input("Enter the number of images per category to train on"))
print("Moving {num_images} from each category") # 900

# get all the directory paths
root_path = 'data\\asl_alphabet_train\\asl_alphabet_train'
dir_paths = os.listdir(root_path)
dir_paths.sort()

# move specified number of images from each category
for dir_path  in tqdm(dir_paths):
    all_images = os.listdir(f"{root_path}\\{dir_path}")
    os.makedirs(f"data\\training_images\\{dir_path}", exist_ok=True)
    for i in range(num_images): 
        # generate a random id between 0 and 2999
        rand_id = (random.randint(0, 2999))
        image = cv2.imread(f"{root_path}\\{dir_path}\\{all_images[rand_id]}")
        cv2.imwrite(f"data\\training_images\\{dir_path}\\{dir_path}{i}.jpg", image)
