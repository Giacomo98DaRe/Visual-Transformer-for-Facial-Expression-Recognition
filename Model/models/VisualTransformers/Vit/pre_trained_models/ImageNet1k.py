# IMPORT

from torch import nn
from torchvision.models.vision_transformer import vit_b_16

########################################################################################################################

def pre_trained_ImageNet():

    model = vit_b_16(weights='ViT_B_16_Weights.DEFAULT', progress = True)

    for parameters in model.parameters():
        print("Freezing parameters.\n")
        parameters.requires_grad = False

    model.heads = nn.Linear(in_features=768, out_features=8, bias=True)

    print("ImageNet set.\n")

    return model