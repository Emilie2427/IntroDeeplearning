# CI2 : Régularisation, optimisation et métriques

## Exercice 1

1. StandardScaler() sert à normaliser les données d'un jeu de données. Si on le fait avant le split, on le fait sur la totalité des données, donc les informations du jeu de test se retrouvent  utilisées lors de l'entraînement, c'est une fuite de données.
2. torch.utils.data.IterableDataset.

