# LIBRARIES
import sys

from PIL import Image
from torchvision import transforms
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from AffectNet_dataset import MyAffectNet
from torchvision.models.swin_transformer import SwinTransformer

class SwinTransformerWithHooks(SwinTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_hooks = []

    def add_feature_hook(self, layer):
        def hook(module, input, output):
            self.feature_hooks.append(output)
        layer.register_forward_hook(hook)

    def forward_with_hooks(self, x):
        # Reset the feature hooks
        self.feature_hooks = []
        # Forward pass con la raccolta degli output intermedi
        output = self.forward(x)
        return output, self.feature_hooks


patch_size = [2, 2]
depths = [2, 2, 6, 2]
num_heads = [3, 6, 12, 24]
window_size = [2, 2]

swin_configuration = {
    "patch_size": patch_size,
    "embed_dim": 96,  (Swin Vit tiny)
    "depths": depths,
    "num_heads": num_heads,
    "window_size": window_size, # -> attention window dimension
    "mlp_ratio": 4,
    "dropout": 0.4,
    "attention_dropout": 0.4,
    "stochastic_depth_prob": 0.1,
    "num_classes": 8
}

model = SwinTransformerWithHooks(**swin_configuration)
model.eval()  # Model in evaluation mode

# Load image
image_path = 'data/AFFECT_NET_8_LABELS/trial/train/images/23.jpg'
image = Image.open(image_path)

# Define the transformations
# Note: You must use the mean and standard deviation values ​​used during model training.
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Trasforma l'immagine e aggiungi una dimensione batch
input_tensor = transform(image).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)
model = model.to(device)

# Add hook to the desired layer
for block in model.features[1]:
    model.add_feature_hook(block)

# Forward pass with hook
with torch.no_grad():
    output, feature_hooks = model.forward_with_hooks(input_tensor)

# Should print the sizes of intermediate features captured by hooks
for i, feature_map in enumerate(feature_hooks):
    print(f"Feature Map at layer {i}: {feature_map.shape}")
