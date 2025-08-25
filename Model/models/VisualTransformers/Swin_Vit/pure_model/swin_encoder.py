import torch.nn as nn
from torchvision.models.vision_transformer import Encoder
from torchvision.models.swin_transformer import SwinTransformer

class MySwin_no_projection(nn.Module):
    def __init__(self):
        super(MySwin_no_projection, self).__init__()

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
            "dropout": 0.2,
            "attention_dropout": 0.2,
            "stochastic_depth_prob": 0.1,
        }

        self.swin_encoder = SwinTransformer(**swin_configuration)

        self.swin_encoder.head = nn.Linear(768, 768, bias=True)


    def forward(self, x):
        out = self.swin_encoder(x)
        out = self.swin_encoder.head(out)

        return out


