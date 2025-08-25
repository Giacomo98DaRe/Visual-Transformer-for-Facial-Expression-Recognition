# IMPORT

from torch import nn
from torchvision.models.swin_transformer import swin_b

########################################################################################################################

def pre_trained_Swin_ViT():
    model = swin_b(weights='Swin_B_Weights.DEFAULT', progress = True)

    for parameters in model.parameters():
        # print("Freezing parameters.\n")
        parameters.requires_grad = False

    model.head = nn.Linear(1024, 8, bias=True)

    return model