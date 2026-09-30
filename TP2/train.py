import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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


def train_model(opt_name, learning_rate=0.001, epochs=30):
    model = MLP(input_size=input_dim, hidden_size=128).to(device)
    criterion = nn.BCELoss()
    
    # Choix de l'optimiseur
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif opt_name == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    writer = SummaryWriter(f'runs/cardio_{opt_name}_lr{learning_rate}')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Logging de la loss moyenne par epoch
        writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)
        
    writer.close()
    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_targets = []
    all_preds_probs = []
    
    with torch.no_grad(): # Désactiver le calcul des gradients
        for batch in test_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            
            # Stocker les probabilités et les cibles pour sklearn
            all_targets.extend(targets.cpu().numpy())
            all_preds_probs.extend(outputs.cpu().numpy())
            
    # Conversion en numpy array
    all_targets = np.array(all_targets)
    all_preds_probs = np.array(all_preds_probs)
    
    # Prédictions binaires (seuil à 0.5)
    all_preds_classes = (all_preds_probs > 0.5).astype(int)
    
    # Calcul des métriques
    precision = precision_score(all_targets, all_preds_classes)
    recall = recall_score(all_targets, all_preds_classes)
    f1 = f1_score(all_targets, all_preds_classes)
    auc = roc_auc_score(all_targets, all_preds_probs)
    
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Lancement des expériences
for opt in ["SGD", "Momentum", "RMSprop", "Adam"]:
    train_model(opt, learning_rate=0.001)

print("\n=== Évaluation du modèle final (Adam) sur le jeu de test ===")
best_model = train_model("Adam", learning_rate=0.001, epochs=30)

evaluate_model(best_model, test_loader)