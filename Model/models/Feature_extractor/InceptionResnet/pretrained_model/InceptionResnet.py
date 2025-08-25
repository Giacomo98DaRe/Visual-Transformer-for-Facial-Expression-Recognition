import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

def IncResNet_creation():

    model = InceptionResnetV1(pretrained='vggface2')

    for parameters in model.parameters():
        parameters.requires_grad = False

    model.last_linear = nn.Linear(1792, 512, bias=False)
    model.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    model.logits = nn.Linear(512, 8, bias=True)

    return model