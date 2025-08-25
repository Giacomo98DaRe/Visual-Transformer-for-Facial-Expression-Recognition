########################################################################################################################
# LIBRARIES
import sys

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
import os

from torchvision.datasets import ImageFolder

# MODULES
from output_handlers import output_dir_creation, out_logger_creation, loss_acc_graph, my_confusion_matrix
from AffectNet_dataset import MyAffectNet, MyAffectNet_two_streams
from lfw_dataset import my_lfw_dataset
from CelebA_dataset import CelebA_dataset
import my_sampler
import model_training
import data_augmentation
import two_streams_training
import training_with_checkpoints

# MODELS
from models.Models.Vit.pure_models import vit_creation
# from models.Swin_Vit.pre_trained_model import ImageNet1k
# from models.DinoV2Encoder import dinoV2plusViT
from models.Models.Dino_MTCNNSwin_Vit import Dino_MTCNNSwin_Vit
from models.Dino_PosterV2 import Dino_Poster8cls
from models.PosterV2 import PosterV2_8cls

########################################################################################################################
# SEED

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

########################################################################################################################
# DEVICE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("I'm running on: " + str(device) + "\n")

########################################################################################################################
# CACHE CLEANING

torch.cuda.empty_cache()

########################################################################################################################
# MODEL

# LFW
# dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dino_extractor.head = nn.Linear(768, 5749)

# CELEB-A
dino_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dino_extractor.head = nn.Linear(768, 10177)

# FINE TUNING
for param in dino_extractor.parameters():
    param.requires_grad = True

# print(dino_extractor)
# sys.exit()

print("I downloaded the model and it is on the device!\n")

########################################################################################################################
# TRAINING PARAMETERS

# batch_size = 16

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001
# weight_decay = 0.001
optimizer = optim.SGD(dino_extractor.parameters(), lr=learning_rate)
#, weight_decay=weight_decay)

# Lo scheduler permette di cambiare lr durante il training. Ne esistono varie tipologie. Per ora teniamolo fisso.
# scheduler = optim.STEPLR(optimizer, step_size=1, gamma=gamma)

num_epochs = 2

# """
# 1 STREAM
loss_vect = {
    "train": [],
    "val": []
}
accuracy_vect = {
    "train": [],
    "val": []
}
# """


print("I set training parameters!\n")

########################################################################################################################
# DATA PATH

# lfw_path = '../data/lfw'

celebA_path = '../data/Celeb-A'

########################################################################################################################
# DATA AUGMENTATION

my_transformations = data_augmentation.my_transformations()

landmark_stream_transf = data_augmentation.landmark_stream_transformations()
features_stream_transf = data_augmentation.features_stream_transformations()

default_resize_transformation = data_augmentation.default_resize_transformation()

########################################################################################################################
# DATASET CREATION

"""
# LFW

lfw_dataset = ImageFolder(root=lfw_path) # , transform=my_transformations)
num_classes = len(lfw_dataset.classes)

# sys.exit()

val_size = int(0.20 * len(lfw_dataset))
train_size = len(lfw_dataset) - val_size

# Suddividi il dataset in training e validation sets
train_lfw, val_lfw = random_split(lfw_dataset, [train_size, val_size])

train_lfw = my_lfw_dataset(train_lfw, my_transformations)
val_lfw = my_lfw_dataset(val_lfw, default_resize_transformation)
"""


# CELEB-A

celebA_images_path = os.path.join(celebA_path, 'Images/img_celeba/img_celeba')
celebA_labels_path = os.path.join(celebA_path, 'Labels/identity_CelebA.txt')

with open(celebA_labels_path, 'r') as f:
    data = [line.strip().split() for line in f.readlines()]

images, labels = zip(*data)
labels = list(map(int, labels))

train_img_celebA, val_img_celebA, train_labels_celebA, val_labels_celebA = train_test_split(images, labels, test_size=0.2, random_state=seed)

train_celebA = CelebA_dataset(train_img_celebA, train_labels_celebA, celebA_images_path, my_transformations)
val_celebA = CelebA_dataset(val_img_celebA, val_labels_celebA, celebA_images_path, default_resize_transformation)

print("Dataset created!\n")

########################################################################################################################
# WEIGHTED SAMPLER

# sampler, sample_weights = my_sampler.sampler_creation(train_AffectNet8)

# sampler, weights = my_sampler.two_streams_sampler_creation(train_AffectNet8)

# print("Sampler created!\n")

########################################################################################################################
# DATALOADER

train_celebA_dataloader = DataLoader(train_celebA, batch_size=32, shuffle=True) #, sampler=sampler)
val_celebA_dataloader = DataLoader(val_celebA, batch_size=8, shuffle=False)

print("Dataloader created!\n")

# DATALOADER DICTIONARY -> TRAIN MODEL
dataloaders = {
    "train": train_celebA_dataloader,
    "val": val_celebA_dataloader
}

########################################################################################################################
# MODEL CALL

best_model_weights_path = "models/my_pretrainings/DINO_fine_tuning.pth"

model_train, val_predictions = model_training.train_model(dino_extractor, device, num_epochs, criterion, optimizer, dataloaders, loss_vect, accuracy_vect, best_model_weights_path)

print("Training finished!\n\n")



