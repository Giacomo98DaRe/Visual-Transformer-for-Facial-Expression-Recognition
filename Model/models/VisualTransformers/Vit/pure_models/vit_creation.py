# IMPORT

from torchvision.models.vision_transformer import VisionTransformer

########################################################################################################################

def my_ViT():
    # Lista di parametri che passerò al VIT -> # DEFAULT: BASE_VIT BY GOOGLE
    vit_configuration = {
        "image_size": 224,
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "dropout": 0.5,
        "attention_dropout": 0.5,
        "num_classes": 8
    }

    model = VisionTransformer(**vit_configuration)

    return model