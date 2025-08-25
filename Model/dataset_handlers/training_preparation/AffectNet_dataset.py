# IMPORT

from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

from torchvision import transforms
from data_augmentation import landmarks_focus
########################################################################################################################

# DATASET DEFINITION
class MyAffectNet(Dataset):
    def __init__(self, image_dir, annotations_dir, transforms):
        self.image_paths = [os.path.join(image_dir, f) for f in
                          sorted(os.listdir(image_dir))]  # lista di path delle immagini
        self.labels = [np.load(os.path.join(annotations_dir, f)) for f in
                       sorted(os.listdir(annotations_dir))]  # lista di etichette
        self.labels = [str(element) for element in self.labels]
        self.image_transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # LABEL
        label = int((self.labels[index]))

        # IMAGE
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        img = self.image_transforms(img)

        return img, label

    # Print the number of associated samples for each label and return a zip of the two (in case you want to print them)
    def labels_cardinality(self):
            cardinality_dict = {
                "0": 0,
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0,
                "6": 0,
                "7": 0,
            }

            sample_label, sample_counts = np.unique(self.labels, return_counts=True)
            for label, counts in zip(sample_label, sample_counts):
                cardinality_dict[str(label)] = counts

            return cardinality_dict

########################################################################################################################

class MyAffectNet_two_streams(Dataset):
    def __init__(self, image_dir, annotations_dir, first_channel_transf, second_channel_transf):
        self.image_paths = [os.path.join(image_dir, f) for f in
                          sorted(os.listdir(image_dir))]  # lista di path delle immagini
        self.labels = [np.load(os.path.join(annotations_dir, f)) for f in
                       sorted(os.listdir(annotations_dir))]  # lista di etichette
        self.labels = [str(element) for element in self.labels]
        self.first_channel_transforms = first_channel_transf
        self.second_channel_transforms = second_channel_transf

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # LABEL
        label = int((self.labels[index]))

        # IMAGE
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        # FIRST CHANNEL
        first_channel_img = self.first_channel_transforms(img)

        # SECOND CHANNEL
        second_channel_img = self.second_channel_transforms(img)

        return first_channel_img, second_channel_img, label

    # Print the number of associated samples for each label and return a zip of the two (in case you want to print them)
    def labels_cardinality(self):
            cardinality_dict = {
                "0": 0,
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0,
                "6": 0,
                "7": 0,
            }

            sample_label, sample_counts = np.unique(self.labels, return_counts=True)
            for label, counts in zip(sample_label, sample_counts):
                cardinality_dict[str(label)] = counts

            return cardinality_dict