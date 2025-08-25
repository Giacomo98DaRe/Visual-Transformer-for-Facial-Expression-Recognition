# IMPORT

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from AffectNet_dataset import MyAffectNet
import data_augmentation

########################################################################################################################
# DATA PATH

base_folder_train_path = "/data/train"
base_folder_val_path = "/data/val"

train_images = os.path.join(base_folder_train_path, "images")
train_labels = os.path.join(base_folder_train_path, "exp_annotations")

val_images = os.path.join(base_folder_val_path, "images")
val_labels = os.path.join(base_folder_val_path, "exp_annotations")

########################################################################################################################
# DATASET

default_transformation = data_augmentation.default_resize_transformation()

train_AffectNet = MyAffectNet(train_images, train_labels, transforms=default_transformation)

########################################################################################################################
# DATALOADER

# SAMPLER

# sampler, sample_weights = sampler_creation(train_AffWild2)
# print("Sampler created!\n")
# train_AffWild2_dataloader = DataLoader(train_AffWild2, 32, shuffle=False, sampler=sampler)

# NO SAMPLER
train_AffectNet_dataloader = DataLoader(train_AffectNet, 32, shuffle=True, sampler=None)

########################################################################################################################
# PRINT AND SAVE HISTROGRAMS

def save_batch_distribution(dataloader, counter, out_dir):
    i = 0
    for images, labels in dataloader:
        if i >= counter:
            break

        unique_labels, label_counts = torch.unique(labels, return_counts=True)

        # Crea una lista di contatori inizializzati a 0 per tutte le etichette da 0 a 8
        all_labels = list(range(8))

        plt.figure(figsize=(15, 5))
        plt.bar(unique_labels, label_counts.tolist())
        plt.xlabel('Emotion Code')
        plt.ylabel('Counts')
        plt.title(f'Distribution of labels in batch {i + 1}')
        plt.xticks(range(len(all_labels)), all_labels)
        # plt.yticks(range(int(max(label_counts)) + 1))
        plt.yticks()
        plt.tight_layout()

        plt.savefig(os.path.join(out_dir, f"training_labels_distribution_ex_{i + 1}"))
        plt.close()

        i += 1

########################################################################################################################

out_dir = "pre-sampling"
save_batch_distribution(train_AffectNet_dataloader, 3, out_dir)