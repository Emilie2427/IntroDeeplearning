# CI3 : Réseaux convolutifs

## Exercice 1 

### Question 1.1

La descente de gradient oscille fortement sans normalisation. En normalisant on uniformise les valeurs, donc la descente de gradient converge plus rapidement. Mathématiquement, normaliser des valeurs, c'est centrer les valeurs et les diviser par l'écart type. Si on calculait les statistiques sur l'ensemble des valeurs, il y a un risque de fuite des données.  

### Question 1.2

RandomResizedCrop et RandomHorizontalFlip permettent de rendre le modèle plus robuste pendant l'entrainement, mais comme on ne veut pas complexifier les calculs lors de la validation, on n'applique pas le random. 

### Question 1.3

Si on applique RandomHorizontalFlip sur des images avec du texte, le texte ne sera plus lisible. 

## Exercice 2 

 ### Question 2.1 

 Notre modèle MLP a 3 couches. 

1. **Couche 1** 
    * Poids : $3 \times 227 \times 227 \times 256 = 39\,574\,272$
    * Biais : $1 \times 256$
    * **Total Couche 1** = $39\,574\,528$

2. **Couche 2** 
    * Poids : $256 \times 256 = 365\,536$
    * Biais : $1 \times 256$
    * **Total Couche 2** = $65\,792$

2. **Couche 3** 
    * Poids : $256 \times 1000 = 256\,000$
    * Biais : $1 \times 1000$
    * **Total Couche 3** = $257\,000$

Nombre total de paramètres avec le biais : $39\,897\,320$ </br>
Sans le biais : $39\,895\,808$  </br>

La couche 1 concentre 99% des paramètres 

### Question 2.2 

La taille du vecteur obtenu par le flatten est très grande. Par ailleurs on perd la relatiion de voisinage entre les pixels, le modeèle devient sensible aux translations. 





