import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

data_dir = "/mnt/ssd/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 1. Pipeline d'entraînement (avec Data Augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 2. Pipeline de validation (déterministe)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def get_dataloaders(batch_size=256, n_workers=8, subset_size=None):
    print("Chargement d'ImageNet...")
    train_dataset = ImageNet(root=data_dir, split="train", transform=train_transform)
    val_dataset   = ImageNet(root=data_dir, split="val",   transform=val_dataset)

    # Optionnel : sous-ensemble pour tester rapidement le code en séance
    if subset_size is not None:
         generator = torch.Generator().manual_seed(42)
         train_indices = torch.randperm(len(train_dataset), generator=generator)[:subset_size]
         val_indices   = torch.randperm(len(val_dataset), generator=generator)[:subset_size // 4]
         train_dataset = Subset(train_dataset, train_indices)
         val_dataset   = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)
    return train_loader, val_loader

class MLP(nn.Module):
    def __init__(self, input_size=3 * 227 * 227, hidden_size=256, num_classes=1000):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 1. Instanciation des deux modèles
mlp_model = MLP()
alexnet_model = AlexNet()

# 2. Appel de la fonction count_parameters pour compter les poids
nb_params_mlp = count_parameters(mlp_model)
nb_params_alexnet = count_parameters(alexnet_model)

# 3. Affichage des résultats 
# (Le :, permet d'afficher les milliers avec des séparateurs pour que ce soit plus lisible)
print("=== Vérification du nombre de paramètres ===")
print(f"Nombre total de paramètres pour le MLP : {nb_params_mlp:,}")
print(f"Nombre total de paramètres pour AlexNet : {nb_params_alexnet:,}")