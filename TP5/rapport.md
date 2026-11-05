# CI5 : Image Captioning avec RNN et Attention sur ResNet

## Exercice 1 

### Question 1.a 

Pour regrouper les images dans un tenseur, il faut que toutes les images aient la même dimension. 

## Exercice 2 

### Question 1 

requires_grad s'il est à True, il enregistre toutes les opérations mathématiques faites. Ici on ne veut pas enregistrer toutes les opérations pour économiser de la mémoire (VRAM), par ailleurs on utilise les poids de ResNet50 pré-entrainé grâce à cette fonction weights=models.ResNet50_Weights.DEFAULT. 

### Question 2 

La shape exacte du tenseur de sortie sera (32, 2048, 7, 7). 

mport torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionModule, self).__init__()
        # Couche pour calculer le score d'attention
        self.attention_layer = nn.Conv2d(feature_dim + hidden_dim, 1, 1)

    def forward(self, features, hidden_state):
        # features shape: (batch_size, 2048, 7, 7)
        # hidden_state shape: (batch_size, hidden_dim)

        # On "étale" l'état caché sur la grille spatiale 7x7
        hidden_state_expanded = hidden_state.unsqueeze(2).unsqueeze(3).repeat(1, 1, features.size(2), features.size(3))
        
        # Concaténation des caractéristiques de l'image et de l'état du RNN (sur la dimension des canaux)
        concat_features = torch._______((features, hidden_state_expanded), dim=____)
        
        # Calcul des scores (batch_size, 1, 7, 7) -> on retire la dimension de taille 1
        attention_scores = self.attention_layer(concat_features).squeeze(1)
        
        # Application du Softmax sur les 49 régions (7x7)
        attention_weights = F._______(attention_scores.view(attention_scores.size(0), -1), dim=1)
        attention_weights = attention_weights.view(attention_scores.size(0), 1, attention_scores.size(1), attention_scores.size(2))
        
        # Calcul du vecteur de contexte par somme pondérée sur les dimensions H et W (dim 2 et 3)
        context_vector = torch._______(torch._______(features * attention_weights, dim=3), dim=2)
        
        return context_vector, attention_weights