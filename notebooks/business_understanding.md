# Business Understanding : Prédiction des Loyers à Bujumbura

## 1. Contexte & Problème Métier
Le marché immobilier locatif à Bujumbura souffre d'une forte opacité des prix, caractérisée par une fixation empirique des loyers et des écarts injustifiés selon les quartiers. 

**Problématique :** Comment estimer de manière objective et automatisée le loyer mensuel équitable d'un logement résidentiel à Bujumbura en se basant sur ses caractéristiques intrinsèques et sa localisation ?

**Public cible & Usage final :**
* **Agences immobilières et propriétaires :** Évaluer et fixer un loyer juste pour optimiser le taux d'occupation.
* **Locataires :** Vérifier la cohérence d'un loyer proposé par rapport au marché local.
* **Intégration technique :** Alimenter un outil d'estimation en ligne permettant une simulation rapide.

## 2. Dérisking & Cadrage Machine Learning

* **Variable Cible :** `LoyerMensuel_BIF` (Prix du loyer exprimé en Francs Burundais - BIF).
* **Type de Problème :** **Régression Apprise / Apprentissage Supervisé**.
* **Métrique Principale de Succès Métier :** 
  * **RMSE (Root Mean Squared Error) & MAE (Mean Absolute Error) :** Mesurer l'erreur moyenne de prédiction en BIF.
  * **R² (Coefficient de détermination) :** Atteindre au minimum **R² ≥ 0.75** sur l'ensemble de test pour valider l'expliabilité du modèle.

## 3. Périmètre & Contraintes Techniques

Axe | Description / Contrainte | Impact sur le Projet |
:--- | :--- | :--- |
**Volume de données** | Dataset restreint de **510 observations** | Risque élevé de surapprentissage (*overfitting*). Nécessite l'usage de modèles robustes (Régression régularisée, Arbres) et d'une validation croisée rigoureuse. |
**Qualité des données** | Lignes entièrement manquantes & anomalies (ex: `AgeMaison` aberrant) | Phase de nettoyage stricte obligatoire avant modélisation. |
**Variable Cible** | Plafonnement observé à **2 600 000 BIF** (effet de censuration) | Nécessite un traitement spécifique de la distribution (ex: transformation log) pour stabiliser la variance. |
**Granularité spatiale** | Hétérogénéité marquée selon les `Quartier` (ex: Rohero vs Bwiza) | Obligation de capturer la valeur foncière par un encodage adapté (*Target Encoding*). |

## 4. Synthèse des Critères de Succès

1. **Métier :** Fournir une estimation de loyer cohérente avec les réalités économiques des différents quartiers de Bujumbura.
2. **Technique :** Déployer un pipeline de prétraitement et de modélisation capable de prédire le loyer sans biais majeur sur les biens de standing moyen à élevé.