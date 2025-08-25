import sys

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torchvision.models.vision_transformer import VisionTransformer

import torch


def reshape_tensor(input_tensor, canali, altezza, larghezza):
    batch_size, dim = input_tensor.size()

    # Calcola la dimensione desiderata per l'ultimo asse del tensore
    dim_ultimo_asse = canali * altezza * larghezza

    # Verifica che la dimensione del tensore iniziale sia compatibile
    assert dim == dim_ultimo_asse, "Dimensione del tensore iniziale non compatibile con le dimensioni desiderate."

    # Reshape il tensore iniziale in [batch_size, canali, altezza * larghezza]
    reshaped_tensor = input_tensor.view(batch_size, canali, altezza * larghezza)

    # Ridimensiona il tensore nelle dimensioni desiderate [batch_size, canali, altezza, larghezza]
    final_tensor = reshaped_tensor.view(batch_size, canali, altezza, larghezza)

    return final_tensor


class inc_and_vit(nn.Module):

    def __init__(self):
        super().__init__()

        # ENCODER
        incResNet = InceptionResnetV1(pretrained='vggface2')
        for parameters in incResNet.parameters():
            parameters.requires_grad = False

        new_last_linear = nn.Linear(1792, 768, bias=False)
        incResNet.last_linear = new_last_linear

        new_last_bn = nn.BatchNorm1d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        incResNet.last_bn = new_last_bn

        incResNet.logits = nn.Linear(768, 768, bias=True)

        # incResNet = nn.Sequential(*list(incResNet.children())[:-2])

        self.embedder = incResNet

        # VIT
        vit_configuration = {
            "image_size": 16,
            "patch_size": 4,
            "num_layers": 4,
            "num_heads": 4,
            "hidden_dim": 768,
            "mlp_dim": 3072,
            "dropout": 0.4,
            "attention_dropout": 0.4,
            "num_classes": 8
        }

        self.vit = VisionTransformer(**vit_configuration)

    def forward(self, x):

        embedded_repr = self.embedder(x) # EMBEDDER_REPR SIZE: [BATCH_SIZE , LAST_LINEAR_OUT_FEATURES] -> EX: 8, 512
        # print(embedded_repr.size())
        # embedded_repr = reshape_tensor(embedded_repr, canali=3, altezza=16, larghezza=16)
        embedded_repr = reshape_tensor(embedded_repr, canali=3, altezza=16, larghezza=16)

        # THE IMPORTANT THUS IS THAT IN THE VIEW, THE FIRST PARAMETER IS BATCH_SIZE, THE SECOND 3, WHILE IMG_SIZE MUST GIVE (WHEN      	MULTIPLIED) THE REMAINING VALUE

        out = self.vit(embedded_repr)

        return out
