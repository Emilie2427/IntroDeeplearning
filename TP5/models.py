import torch.nn as nn
from torchvision import models

# 1. Chargement de ResNet50 pré-entraîné
resnet_pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2. Suppression des deux dernières couches (AdaptiveAvgPool2d et Linear)
# On veut conserver les dimensions spatiales pour l'attention (pas un vecteur 1D global)
cnn_features = nn.Sequential(*list(resnet_pretrained.children())[:-2])

# 3. Gel des poids (Transfer Learning)
for param in cnn_features.parameters():
    param.requires_grad = False