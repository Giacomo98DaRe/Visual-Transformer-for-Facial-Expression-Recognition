import torch.nn as nn
from torchvision.models.vision_transformer import Encoder

class MyVisionTransformer_no_projection(nn.Module):
    def __init__(self, embed_dim=256, num_classes=8, depth=2, num_heads=16, hidden_dim=256, mlp_dim=3072, dropout=0.2,
                 attention_dropout=0.2):
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