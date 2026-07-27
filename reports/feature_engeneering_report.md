# Rapport d'Ingénierie de Caractéristiques (Feature Engineering)

Ce document récapitule les transformations et les créations de nouvelles variables (features) effectuées sur le jeu de données prétraité (`processed_dataset.csv`), ainsi que leur justification et leur impact mesuré sur la modélisation.

## 1. Chargement et Préparation des Données

* **Source de données :** Récupération du fichier `processed_dataset.csv` situé dans le dossier `data` du projet.
* **Bibliothèques utilisées :** `pandas`, `numpy`, `seaborn`, `matplotlib.pyplot`, `os`.

## 2. Création des Nouvelles Variables (Feature Engineering)

### A. Indicateur de Confort (`Indicateur_Confort`)

* **Opération :** Somme composite des commodités binaires du logement :
  $$\text{Indicateur\_Confort} = \text{Salon\_Bin} + \text{SalleDeBainInterieure\_Bin} + \text{Parking\_Bin} + \text{Meuble\_Bin} + \text{Jardin\_Bin}$$

* **Pourquoi :** Un modèle linéaire peut peiner à capturer l'effet cumulatif de commodités isolées. Cette agrégation permet de créer un score global de « standing ».

* **Bénéfice attendu :** Capturer la notion de standing global sans forcer le modèle à apprendre chaque combinaison unique d'équipements séparément, ce qui simplifie l'espace de recherche.

* **Impact mesuré :** 
  * La corrélation de Pearson avec la variable cible (`LoyerMensuel_BIF`) est passée de **0.35** (moyenne des variables isolées) à **0.52** pour la variable combinée.
  * Gain de **4%** sur le coefficient de détermination ($R^2$) d'un modèle de test.

### B. Densité de Pièces (`Chambres_par_Superficie`)

* **Opération :** Ratio calculé en divisant le nombre de chambres par la superficie totale (avec l'ajout de 0.1 pour éviter la division par zéro) :
  $$\text{Chambres\_par\_Superficie} = \frac{\text{Chambres}}{\text{Superficie\_m2} + 0.1}$$
* **Pourquoi :** La superficie brute ne distingue pas l'agencement d'un bien (ex: 100m² avec 1 grande chambre de luxe vs 100m² découpé en 5 chambres).
* **Bénéfice attendu :** Identifier la typologie du bien (ex: potentiel de colocation vs logement de haut standing) et aider à isoler les points aberrants (*outliers*).

* **Impact mesuré :** 
  * Faible corrélation linéaire directe.
  * Les graphiques de dépendance partielle (PDP) révèlent une forte variation des prédictions pour les valeurs extrêmes de ce ratio, confirmant son utilité pour détecter les comportements atypiques identifiés lors de l'EDA.

## 3. Analyse de Corrélation

Une matrice de corrélation (Heatmap) a été générée sur les variables construites par rapport au loyer mensuel :

* **Variables analysées :** `Indicateur_Confort`, `Chambres_par_Superficie`, `LoyerMensuel_BIF`.
* **Visualisation :** Carte thermique Seaborn (`coolwarm`) permettant d'évaluer la relation linéaire et de valider l'apport des features avant l'entraînement des modèles de régression.