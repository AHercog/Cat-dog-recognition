import os
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm

from project_files.image_parser_files.label_img import label_img


def create_train_data(train_dir, img_size):
    training_data = []
    training_data_labels = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        training_data.append(np.array(img))
        training_data_labels.append(np.array(label))
    return training_data, training_data_labels
