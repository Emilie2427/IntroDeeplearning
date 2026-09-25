import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Importation du Dataset défini à l'exercice 1
from dataset import CardioDataset


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # Sortie binaire [0, 1]
        )

    def forward(self, x):
        return self.net(x)


# 1. Chargement des données et des DataLoaders
dataset = CardioDataset("data/cardio_train.csv")

generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(
    dataset, [0.8, 0.1, 0.1], generator=generator
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 2. Récupération d'un exemple pour obtenir la dimension des features dynamiquement
sample_batch = next(iter(train_loader))
input_dim = sample_batch["features"].shape[1]

# 3. Initialisation du modèle, du critère et de l'optimiseur
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=input_dim, hidden_size=128).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = 0.1
l2_lambda = 0

# 4. Boucle d'entraînement
for epoch in range(10):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        inputs, targets = batch["features"].to(device), batch["labels"].to(
            device
        )

        # Réinitialisation des gradients
        optimizer.zero_grad()

        # Pass avant (Forward pass)
        outputs = model(inputs)
        base_loss = criterion(outputs, targets)

        # Calcul de la pénalité L1 (somme des valeurs absolues des poids)
        l1_penalty = sum(p.abs().sum() for p in model.parameters())

        # Calcul de la pénalité L2 (somme des carrés des poids)
        l2_penalty = sum((p**2).sum() for p in model.parameters())

        # Loss totale
        loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

        # Rétropropagation
        loss.backward()

        # Mise à jour des poids
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Époque [{epoch+1}/10] - Loss moyenne : {running_loss / len(train_loader):.4f}"
    )
