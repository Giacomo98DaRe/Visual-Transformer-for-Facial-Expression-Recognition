# IMPORT

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CelebA_dataset(Dataset):
    def __init__(self, images, labels, images_dir, transform):
        self.images = images
        self.labels = labels
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = os.path.join(self.images_dir, self.images[idx])
        img = Image.open(img_name)
        img = np.array(img)
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label


# Data path
file_path = 'data/Celeb-A/Labels/identity_CelebA.txt'


# Read and update labels

new_labels = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        image_name = parts[0]  
        label = int(parts[1]) - 1  
        new_labels.append(f"{image_name} {label}")

# New path for new updated labels
new_file_path = 'data/Celeb-A/Labels/identity_CelebA_New.txt'

# Save in new file
with open(new_file_path, 'w') as file:
    for label in new_labels:
        file.write(label + "\n")
