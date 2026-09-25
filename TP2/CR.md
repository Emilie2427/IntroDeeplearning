# CI2 : Régularisation, optimisation et métriques

## Exercice 1

1. StandardScaler() sert à normaliser les données d'un jeu de données. Si on le fait avant le split, on le fait sur la totalité des données, donc les informations du jeu de test se retrouvent  utilisées lors de l'entraînement, c'est une fuite de données.
2. torch.utils.data.IterableDataset.

## Exercice 2

résultat 2_1_1
résultat 2_1_2

1. Les lambdas sont petits, c'est-à-dire qu'il y a une petite régularisation. En augmentant les lambdas, on augmente la régularisation. 
Donc la Loss totale a des termes ajoutés beaucoup plus grand, d'où l'augmentation de la loss moyenne qu'on observe sur les captures 
2. weight_decay
3. L1, c'est Lasso, donc il effectue une sélection de variables en forçant plusieurs poids à atteindre exactement zéro. Alors que L2, qui est Ridge, réduit uniformément la valeur absolue de l'ensemble des poids vers zéro sans les annuler complètement, évitant ainsi qu'une seule variable prédomine excessivement. 

