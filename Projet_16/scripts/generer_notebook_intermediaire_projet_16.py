import nbformat as nbf
import os

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    # --- Cellule 1 : Titre ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 🎬 PROJET 16 : PRÉDICTION DU BOX-OFFICE (Niveau Intermédiaire) 🍿

**Objectif :** Construire un modèle de régression capable de prédire les revenus d'un film en fonction de ses caractéristiques (Budget, Casting, Date, Genre).

---

## 📅 STRUCTURE DU PROJET

### 📋 SESSION 1 : Analyse Exploratoire & Nettoyage
- Chargement et inspection des types
- Gestion des valeurs manquantes et aberrantes
- Analyse des corrélations (Budget vs Revenu)

### 📋 SESSION 2 : Feature Engineering Avancé
- Extraction de features temporelles (Saisonnalité)
- Encodage des variables catégorielles (One-Hot)
- Transformation Logarithmique des variables financières

### 📋 SESSION 3 : Modélisation & Optimisation
- Entraînement d'un modèle de Régression (Random Forest / Gradient Boosting)
- Évaluation (MAE, RMSE, R²)
- Analyse de l'importance des features

### 🎁 SESSION 3 - PART 4 : Tâches Bonus
- Analyse du ROI (Return on Investment)
- Recommandation de stratégie de sortie (Mois optimal)

---
"""))

    # --- SESSION 1 ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : DATA EXPLORATION & CLEANING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.1 : Chargement et Inspection
**Objectif :** Charger `box_office.csv` et identifier les types de données.
**Livrables :**
- `df.head()`, `df.info()`
- Identification des colonnes cibles et features
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.2 : Nettoyage des Données
**Objectif :** Gérer les valeurs manquantes et incohérentes.
**Approches recommandées :**
- `Genre` manquant : Supprimer les lignes (car difficile à imputer)
- `Budget` ou `Revenus` <= 0 : Vérifier et nettoyer si nécessaire
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.3 : Analyse Exploratoire (EDA)
**Objectif :** Comprendre les facteurs de succès.
**Visualisations attendues :**
- Scatterplot : Budget vs Revenus (coloré par Genre)
- Barplot : Revenu moyen par Genre
- Distribution des Revenus (Histogramme)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # --- SESSION 2 ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : FEATURE ENGINEERING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.1 : Gestion des Dates (Recipe 1)
**Objectif :** Extraire des informations exploitables de `Date_Sortie`.
**Features à créer :**
- `Annee`, `Mois`
- `Jour_Semaine` (Le week-end est-il meilleur ?)
- `Trimestre` (Optionnel)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Encodage des Catégories (Recipe 2)
**Objectif :** Transformer `Genre` en format numérique.
**Méthode :** One-Hot Encoding (`pd.get_dummies`) car il n'y a pas d'ordre intrinsèque entre les genres.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.3 : Transformation Logarithmique (Recipe 4)
**Objectif :** Réduire l'asymétrie (skewness) des variables financières.
**Théorie :** Les revenus de films suivent souvent une loi de puissance (quelques blockbusters gagnent tout). Le logarithme normalise cette distribution.
**Action :** Créer `Log_Budget` et `Log_Revenus` (si utilisé comme cible, sinon juste Budget).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # --- SESSION 3 ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : MODELING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.1 : Préparation et Split
**Objectif :** Séparer Features (X) et Target (y).
**Target :** `Revenus`
**Features :** Log_Budget, Score_Acteurs, Annee, Mois, Genres_Encoded...
**Split :** 80% Train, 20% Test
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Entraînement (Régression)
**Modèle recommandé :** RandomForestRegressor
**Pourquoi ?** Gère bien les relations non-linéaires et les interactions entre variables.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.3 : Évaluation
**Métriques clés :**
- **MAE** (Erreur absolue moyenne) : Interprétable en $.
- **R²** : Pourcentage de variance expliquée.
- **Feature Importance** : Quelles variables pèsent le plus ? (Budget ? Genre ? Acteurs ?)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # --- PART 4 BONUS ---
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: ROI Analysis & Classification
**Goal:** Transformez ce problème de régression en classification.
1. Calculez le `ROI = Revenus / Budget`.
2. Créez une classe :
   - `Flop` (ROI < 1)
   - `Profitable` (ROI >= 1)
3. Visualisez la proportion de Flops par Genre.

### Bonus Task 2: Optimal Release Strategy
**Goal:** Déterminez le meilleur mois pour sortir un film.
1. Calculez le revenu moyen par `Mois`.
2. Affichez un graphique.
3. Recommandez une stratégie (ex: "Éviter Janvier, viser Juin").

**Deliverables:**
- Graphiques d'analyse
- Conclusions écrites
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici pour les bonus
"""))

    # Sauvegarde
    with open('Projet_16_Box_Office_Intermediaire.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_intermediaire()
