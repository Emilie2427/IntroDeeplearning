# CI3 : Réseaux convolutifs

## Exercice 1 

### Question 1.1

La descente de gradient oscille fortement sans normalisation. En normalisant on uniformise les valeurs, donc la descente de gradient converge plus rapidement. Mathématiquement, normaliser des valeurs, c'est centrer les valeurs et les diviser par l'écart type. Si on calculait les statistiques sur l'ensemble des valeurs, il y a un risque de fuite des données.  

### Question 1.2

RandomResizedCrop et RandomHorizontalFlip permettent de rendre le modèle plus robuste pendant l'entrainement, mais comme on ne veut pas complexifier les calculs lors de la validation, on n'applique pas le random. 

### Question 1.3

Si on applique RandomHorizontalFlip sur des images avec du texte, le texte ne sera plus lisible. 

## Exercice 2 


