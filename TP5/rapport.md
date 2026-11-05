# CI5 : Image Captioning avec RNN et Attention sur ResNet

## Exercice 1 

### Question 1.a 

Pour regrouper les images dans un tenseur, il faut que toutes les images aient la même dimension. 

## Exercice 2 

### Question 1 

requires_grad s'il est à True, il enregistre toutes les opérations mathématiques faites. Ici on ne veut pas enregistrer toutes les opérations pour économiser de la mémoire (VRAM), par ailleurs on utilise les poids de ResNet50 pré-entrainé grâce à cette fonction weights=models.ResNet50_Weights.DEFAULT. 

### Question 2 

La shape exacte du tenseur de sortie sera (32, 2048, 7, 7). 

## Exercice 3 

### Question 3.a

Le context_vector correspond à une synthèse visuelle dynamique et pondérée des zones de l'image sur lesquelles le modèle doit se concentrer pour générer le mot suivant. Le problème de Seq2Seq est qu'on réduit l'image, les nombreuses données associées à un vecteur, et donc on perd beaucoup trop d'information. conctext_vetcor permet de remplacer un résumé figé par un zoom dynamique qui se met à jour à chaque nouveau mot. 

