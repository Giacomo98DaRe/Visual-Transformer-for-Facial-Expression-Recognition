import torch
import torch.nn as nn
from torchvision import transforms
from data_augmentation import landmarks_focus

from models.Models.Vit.pure_models.vit_encoder import MyVisionTransformer_no_projection
from models.Feature_extractor.LandmarkDetector.MTCNN import MTCNN_creation
from models.Models.Swin_Vit.pure_model.swin_encoder import MySwin_no_projection

########################################################################################################################

class dino_MTCNNSwin_Vit(nn.Module):
    def __init__(self):
        super().__init__()

        self.swin_encoder = MySwin_no_projection()

        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        self.middle_linear_proj = nn.Linear(1536, 512)

        self.last_linear_proj = nn.Linear(512, 256)

        self.vit_classifier = MyVisionTransformer_no_projection()


    def forward(self, first_channel_img, second_channel_img):

    # FIRST ENCODER

        encoded_eyes_mouth = self.swin_encoder(first_channel_img)

    # SECOND ENCODER

        # DINOV2
        encoded_repr = self.encoder(second_channel_img)


    # CLASSIFIER

        final_encoded_repr = torch.cat((encoded_eyes_mouth, encoded_repr), dim=1)

        final_encoded_repr = self.middle_linear_proj(final_encoded_repr)
        final_encoded_repr = self.last_linear_proj(final_encoded_repr)

        final_encoded_repr = final_encoded_repr.unsqueeze(1)  # Aggiungi una dimensione
        final_encoded_repr = final_encoded_repr.expand(-1, 256, 256)  # Espandi il tensore

        out = self.vit_classifier(final_encoded_repr)

        return out
