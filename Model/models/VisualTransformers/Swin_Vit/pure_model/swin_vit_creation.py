# IMPORT

from torchvision.models.swin_transformer import SwinTransformer

########################################################################################################################

def my_swin_ViT():
    # Parameter dictionary -> # DEFAULT: BASE_SWIN_VIT

    patch_size = [2, 2]
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    window_size = [2, 2]

    swin_configuration = {
        "patch_size": patch_size,
        "embed_dim": 96,  # -> dovrebbe essere = 96? (Swin tiny)
        "depths": depths,
        "num_heads": num_heads,
        "window_size": window_size, # -> dimensione della finestra di attenzione, ovvero con quanti "vicini" comunica ogni patch
        "mlp_ratio": 4,
        "dropout": 0.4,
        "attention_dropout": 0.4,
        "stochastic_depth_prob": 0.1,
        "num_classes": 8
    }

    model = SwinTransformer(**swin_configuration)

    return model