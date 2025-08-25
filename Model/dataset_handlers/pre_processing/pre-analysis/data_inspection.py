import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

from AffectNet_dataset import MyAffectNet

########################################################################################################################
# FOLDER PATHS

base_folder_train_path = "/data/train"

train_images = os.path.join(base_folder_train_path, "images")
train_labels = os.path.join(base_folder_train_path, "exp_annotations")

out_dir = ".."

########################################################################################################################
# LABEL MAPPING

emotion_labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

def label_to_emotion(code, emotion_labels):
    return emotion_labels[int(code)]

########################################################################################################################
# DATASET AND DATALOADER CREATION

train_AffectNet = MyAffectNet(train_images, train_labels, transforms=None)

train_AffectNet_dataloader = DataLoader(train_AffectNet, 64,  shuffle=True)
########################################################################################################################
# VALIDATION Display image and label -> ne genero un po' da tenere da parte

ligrid_path = os.path.join(out_dir, "label and image grid")

# """
# FIRST EXAMPLE
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(train_AffectNet.__len__(), size=(1,)).item()
    img, label = train_AffectNet.__getitem__(sample_idx)
    figure.add_subplot(rows, cols, i)
    plt.title(label_to_emotion(label, emotion_labels))
    plt.axis("off")
    img = img.permute(1, 2, 0)
    plt.imshow(img, cmap="gray")

figure.savefig(os.path.join(ligrid_path, "label and image grid 1"))
plt.close()

# SECOND EXAMPLE
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(train_AffectNet.__len__(), size=(1,)).item()
    img, label = train_AffectNet.__getitem__(sample_idx)
    figure.add_subplot(rows, cols, i)
    plt.title(label_to_emotion(label, emotion_labels))
    plt.axis("off")
    img = img.permute(1, 2, 0)
    plt.imshow(img, cmap="gray")

figure.savefig(os.path.join(ligrid_path, "label and image grid 2"))
plt.close()
# """

########################################################################################################################

labels_counts = train_AffectNet.labels_cardinality()
labels, counts = zip(*labels_counts)

figure, ax = plt.subplots()
ax.bar(labels, counts)
ax.set_xlabel('Emotion Labels')
ax.set_ylabel('Counts')
ax.set_title('Distribution of Emotion Labels')

plt.xticks(rotation=45)
plt.yticks()
plt.tight_layout()

figure.savefig(os.path.join(out_dir, "labels distribution"))
plt.close()