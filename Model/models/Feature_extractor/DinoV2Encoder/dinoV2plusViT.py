import torch
import torch.nn as nn
from torchvision.models.vision_transformer import Encoder

class MyVisionTransformer_no_projection(nn.Module):
    def __init__(self, embed_dim=256, num_classes=8, depth=2, num_heads = 16, hidden_dim = 256, mlp_dim = 3072, dropout = 0.2, attention_dropout = 0.2):
        super(MyVisionTransformer_no_projection, self).__init__()
        self.encoder = Encoder(embed_dim, depth, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)
        self.heads = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.heads(x)
        return x

########################################################################################################################

class dino_and_vit(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        for parameters in self.encoder.parameters():
            print("Freezing parameters.\n")
            parameters.requires_grad = False

        self.middle_linear_proj = nn.Linear(768, 512)

        self.last_linear_proj = nn.Linear(512, 256)

        self.vit_classifier = MyVisionTransformer_no_projection()

    def forward(self, x):

        encoded_repr = self.encoder(x)
        # encoded_repr = encoded_repr.unsqueeze(1)  # Aggiungi una dimensione (batch_size, 1, 768)
        # encoded_repr = encoded_repr.expand(-1, 768, 768)  # Espandi il tensore a (batch_size, 768, 768)

        encoded_repr = self.middle_linear_proj(encoded_repr)
        encoded_repr = self.last_linear_proj(encoded_repr)

        encoded_repr = encoded_repr.unsqueeze(1)  # Aggiungi una dimensione
        encoded_repr = encoded_repr.expand(-1, 256, 256)  # Espandi il tensore

        out = self.vit_classifier(encoded_repr)

        return out