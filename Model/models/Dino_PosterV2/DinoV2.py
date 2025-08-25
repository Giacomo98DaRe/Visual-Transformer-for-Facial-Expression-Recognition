# IMPORT

import os
import torch
import torch.nn as nn

########################################################################################################################

class DINOExtractor(nn.Module):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    def forward(self, x):
        # Estrai le caratteristiche dal modello DINO
        features = self.dino_model(x)
        return features


model = DINOExtractor()
print(model)