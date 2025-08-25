Entry point. Here you can decide...

########################################################################################################################
# LIBRARIES
import sys
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os

# MODULES
from output_handlers import output_dir_creation, out_logger_creation, loss_acc_graph, my_confusion_matrix
from AffectNet_dataset import MyAffectNet, MyAffectNet_two_streams
import my_sampler
import model_training
import data_augmentation
import two_streams_training
import training_with_checkpoints

# MODELS
from models.Models.Vit.pure_models import vit_creation
from models.Models.Swin_Vit.pre_trained_model import ImageNet1k
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

#########################################################################################################################
# OUTPUT DIR

# run_path, run_counter = output_dir_creation.out_dir_creation()

########################################################################################################################
# LOGGING

# logger = out_logger_creation.logger_creation(run_path, run_counter)

########################################################################################################################
# MODEL CREATION: Here you can decide which model run

# GOOGLE STANDARD CONFIGURATION
# model = vit_creation.my_ViT()

# PRE TRAINED VIT IMAGE NET
# model = ImageNet1k.pre_trained_ImageNet()

# PRE TRAINED SWIN VIT IMAGE NET
# model = ImageNet1k.pre_trained_Swin_ViT()

# PRE TRAINED INCEPTION RESNET
#model = dinoV2plusViT.dino_and_vit()

# TWO STREAMS MODEL
# model = Dino_MTCNNSwin_Vit.dino_MTCNNSwin_Vit()

# POSTER
# model = PosterV2_8cls.pyramid_trans_expr2()

# POSTER AND DINO: My solution
model = Dino_Poster8cls.Dino_PosterV2()

checkpoint_path  = "checkpoint"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])

model.to(device)

print("I created the model and it is on the device!\n")

########################################################################################################################
# TRAINING PARAMETERS

# Using batch size in dataloaders
batch_size = 32

# Classification problem: Cross entropy should be okay
criterion = nn.CrossEntropyLoss()

# Optimizer: how to update weights
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(checkpoint['optim_state'])
# Optionally I could apply weight decay, ...

# Scheduler allows runtime changing of learning rate. Up to now will be fixed
# scheduler = optim.STEPLR(optimizer, step_size=1, gamma=gamma)

# Decide how many epochs (1 is default for local trials)
num_epochs = 1

# Initializing accuracy and loss vectors

"""
# 1 STREAM SOLUTION
loss_vect = {
    "train": [],
    "val": []
}
accuracy_vect = {
    "train": [],
    "val": []
}
"""


# 2 STREAMS SOLUTION
# Inizializza loss_vect con tutte le chiavi necessarie vuote
loss_vect = {
    "stream_1": [],
    "stream_2": [],
    "val": []
}

accuracy_vect = {
    "stream_1": [],
    "stream_2": [],
    "val": []
}

print("I set training parameters!\n")

########################################################################################################################
# DATA PATH

# Full dataset case
# base_folder_train_path = "data/AFFECT_NET_8_LABELS/data/train"
# base_folder_val_path = "data/AFFECT_NET_8_LABELS/data/val"

# Local trial case
base_folder_train_path = "data/AFFECT_NET_8_LABELS/trial/train"
base_folder_val_path = "data/AFFECT_NET_8_LABELS/trial/val"

train_images = os.path.join(base_folder_train_path, "images")
train_labels = os.path.join(base_folder_train_path, "annotations")

val_images = os.path.join(base_folder_val_path, "images")
val_labels = os.path.join(base_folder_val_path, "annotations")

########################################################################################################################
# LABELS DEFINITION

emotion_labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

def label_to_emotion(code, emotion_labels):
    return emotion_labels[int(code)]

########################################################################################################################
# DATA AUGMENTATION

my_transformation = data_augmentation.my_transformations()

landmark_stream_transf = data_augmentation.landmark_stream_transformations()
features_stream_transf = data_augmentation.features_stream_transformations()

default_resize_transformation = data_augmentation.default_resize_transformation()

########################################################################################################################
# DATASET CREATION

# ONE STREAM MODEL
# train_AffectNet8 = MyAffectNet(train_images, train_labels, transforms=my_transformation)
# val_AffectNet8 = MyAffectNet(val_images, val_labels, transforms=default_resize_transformation)

# TWO STREAMS MODEL
train_AffectNet8 = MyAffectNet_two_streams(train_images, train_labels, first_channel_transf=default_resize_transformation, second_channel_transf=features_stream_transf)
val_AffectNet8 = MyAffectNet_two_streams(val_images, val_labels, first_channel_transf=default_resize_transformation, second_channel_transf=default_resize_transformation)

print("Dataset created!\n")

########################################################################################################################
# WEIGHTED SAMPLER

# sampler, sample_weights = my_sampler.sampler_creation(train_AffectNet8)

sampler, sampler_weights = my_sampler.two_streams_sampler_creation(train_AffectNet8)

print("Sampler created!\n")

########################################################################################################################
# DATALOADER

train_AffectNet8_dataloader = DataLoader(train_AffectNet8, batch_size=32, shuffle=False, sampler=sampler)
val_AffectNet8_dataloader = DataLoader(val_AffectNet8, batch_size=8, shuffle=True)

print("Dataloader created!\n")

# DATALOADER DICTIONARY -> TRAIN MODEL
dataloaders = {
    "train": train_AffectNet8_dataloader,
    "val": val_AffectNet8_dataloader
}

import matplotlib.pyplot as plt

# Extract a batch from the dataloader
dataiter = iter(train_AffectNet8_dataloader)
images, labels = next(dataiter)

# Count each occurence of the emotion in the dataloader
unique, counts = labels.unique(return_counts=True)

# Plot an occurencies hystogram
plt.bar(unique, counts)
plt.xlabel('Emotion Code')
plt.ylabel('Counts')
plt.title('Distribution of labels after sampling')
plt.xticks(unique)  # Questo assicura che tutti gli emotion codes siano mostrati sull'asse x
plt.show()

# sys.exit()

########################################################################################################################
# MODEL CALL: Choose the correct training mode

# best_model_weights_path = os.path.join(run_path, f"best_model_{run_counter}.pth")

# 1 STREAM TRAINING
# model_train, val_predictions = model_training.train_model(model, device, num_epochs, criterion, optimizer, dataloaders, loss_vect, accuracy_vect) # , run_path) # , best_model_weights_path, logger1)

# 2 STREAMS TRAINING
model_train, val_predictions = two_streams_training.two_streams_train_model(model, device, num_epochs, criterion, optimizer, dataloaders, loss_vect, accuracy_vect)

print("Training finished!\n\n")

########################################################################################################################
# OUTPUT GRAPHS

# I move the train and loss vectors to CPU, as GPU is not able to convert tensors to np arrays
"""
loss_train = torch.tensor(loss_vect["train"]).cpu()
loss_val = torch.tensor(loss_vect["val"]).cpu()

accuracy_train = torch.tensor(accuracy_vect["train"]).cpu()
accuracy_val = torch.tensor(accuracy_vect["val"]).cpu()

loss_acc_graph.loss_history_graph(loss_train, loss_val, run_path, run_counter)
loss_acc_graph.acc_history_graph(accuracy_train, accuracy_val, run_path, run_counter)
"""

########################################################################################################################
# CONFUSION MATRIX

my_confusion_matrix.confusion_matrix_print(val_predictions, emotion_labels) #, run_path, run_counter)