import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

# 1. Imports depuis vos fichiers
from dataset import train_loader, test_loader, dataset
from models import ImageCaptioningModel, cnn_features

# 2. Reconstitution du vocabulaire et des embeddings dans train.py 
# (si ce n'est pas déjà fait dans dataset.py)
print("Construction du vocabulaire et des embeddings...")
vocabulary = set()
# Récupération des mots depuis le dataset Flickr30k
# (ou vous pouvez importer votre fonction create_embedding_layer ici)
for _, captions in dataset:
    for cap in captions:
        for word in cap.lower().split():
            vocabulary.add(word)

import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    
# Tokenizer simple local si non importé
def tokenize(sentence, word2idx, max_length=20):
    res = [0] 
    sentence = sentence.lower()
    for word in sentence.split():
        if word in word2idx:
            res.append(word2idx.get(word, 0))
    if len(res) > max_length:
        return res[:max_length]
    return res + [0] * (max_length - len(res))

# Pour faire simple avec un embedding aléatoire si Word2Vec n'est pas requis/dispo, 
# ou en reprenant votre fonction de creation d'embedding :
vocab_list = list(vocabulary)
word2idx = {word: idx + 1 for idx, word in enumerate(vocab_list)}
word2idx[""] = 0  # Padding / Start
idx2word = [""] + vocab_list

embedding_layer = nn.Embedding(len(idx2word), 300)

# 3. Définition du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil : {device}")

# 4. Instanciation du modèle global
lstm_hidden_dim = 512
model = ImageCaptioningModel(
    cnn_model=cnn_features, 
    embedding_model=embedding_layer, 
    lstm_hidden_dim=lstm_hidden_dim
).to(device)

# 5. Optimiseur, Critère et Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
writer = SummaryWriter("runs/captioning")

def untokenize(tokens, idx2word):
    return [idx2word[x] for x in tokens if x < len(idx2word)]

num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, (images, captions) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        images = images.to(device)
        
        # Tokenisation manuelle du batch (simplifiée)
        caps = torch.LongTensor([tokenize(x, word2idx) for x in captions[0]]).to(device)
        
        # On fournit toutes les séquences sauf le dernier mot en entrée
        outputs = model(images, caps[:, :-1])
        
        # On s'attend à prédire la séquence décalée d'un cran (captions[:, 1:])
        # Reshape pour la Loss : (batch_size * seq_len, vocab_size)
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), caps[:, 1:].reshape(-1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if (i + 1) % 10 == 0:
            writer.add_scalar('Loss/Train_Step', total_loss / 10, epoch * len(train_loader) + i)
            total_loss = 0

    # --- Évaluation et Génération ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, captions in test_loader:
            images = images.to(device)
            caps = torch.LongTensor([tokenize(x, word2idx) for x in captions[0]]).to(device)
            
            outputs = model(images, caps[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), caps[:, 1:].reshape(-1))
            val_loss += loss.item()

            # --- Visualisation d'une prédiction ---
            predictions = outputs[-1].argmax(dim=-1)
            pred_text = " ".join(untokenize(predictions.cpu().numpy(), idx2word))
            true_text = " ".join(untokenize(caps[-1][1:].cpu().numpy(), idx2word))
            
    print(f"Epoch {epoch + 1}")
    print(f"Prediction : {pred_text}")
    print(f"Target     : {true_text}")

    print(f"Epoch {epoch + 1}")
    print(f"Prediction : {pred_text}")
    print(f"Target     : {true_text}")
    

    # 📸 AJOUT POUR SAUVEGARDER L'IMAGE
    # On prend la dernière image du batch et on la ramène sur le CPU
    img_visu = images[-1].clone().cpu()
    
    # On "dé-normalise" l'image pour retrouver les vraies couleurs
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_visu = img_visu * std + mean
    
    # On sauvegarde l'image
    save_image(img_visu, f"image_epoch_{epoch + 1}.png")
    
    writer.add_scalar('Loss/Validation', val_loss / len(test_loader), epoch)
    
    scheduler.step()
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)