import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "11"
    PROJECT_TITLE = "Sous-évaluation Immobilière"
    DATASET_NAME = "immobilier.csv"
    TARGET_COL = "Price"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE} (Version Intermédiaire)

## 🏁 Objectif : Le Chasseur de Bonnes Affaires 🏠
Votre mission est de construire un modèle capable de prédire la **Juste Valeur Marchande** (`{TARGET_COL}`) d'une maison.
Ensuite, vous utiliserez ce modèle pour identifier les propriétés **sous-évaluées** (bonnes affaires).

---

## 📋 Programme

### 🕵️‍♀️ SESSION 1 : From Raw Data to Clean Insights
- Gestion des valeurs manquantes (Année, Localisation)
- Analyse de la relation Surface vs Prix

### 🏗️ SESSION 2 : The Art of Feature Engineering
- **Recipe Math :** Calcul de l'Âge de la maison
- **Recipe Categories :** Encodage de la Localisation

### 🤖 SESSION 3 : Building & Trusting Your Model
- Régression (RandomForestRegressor)
- Évaluation (MAE, R²)
- **Bonus :** Filtrer les maisons où `Prix Réel < Estimation - 10%`

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 1.1 : Chargement et Nettoyage 🧹

**Objectif :** Charger `{DATASET_NAME}` et traiter les valeurs manquantes.

**Points d'attention :**
- `Annee_Construction` et `Localisation` ont des manquants.
- **Stratégie :** Médiane pour l'année, Mode pour la localisation.

**Livrables attendus :**
- Un DataFrame propre.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 1.2 : Analyse Exploratoire (EDA) 🔍

**Objectif :** Comprendre la cible `{TARGET_COL}`.

**Questions :**
- Quelle est la distribution des prix ?
- Y a-t-il une corrélation linéaire entre Surface et Prix ?

**Livrables attendus :**
- Histogramme des prix.
- Scatterplot Surface vs Prix.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.1 : Recipe Math (L'Âge) ➗

**Contexte :** L'année de construction (ex: 1990) est moins parlante que l'âge (ex: 35 ans).

**Objectif :** Créer une feature `Age`.

**Formule :** `Année Actuelle - Annee_Construction`.

**Livrables attendus :**
- Nouvelle colonne `Age`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Recipe Categories 🏷️

**Contexte :** `Localisation` est du texte.

**Objectif :** Encoder cette variable.

**Approche recommandée :** One-Hot Encoding (`pd.get_dummies`).

**Livrables attendus :**
- DataFrame avec colonnes binaires (ex: `Localisation_Maarif`).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🤖 SESSION 3 : Building & Trusting Your Model
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 3.1 : Split et Entraînement 🏋️

**Objectif :** Entraîner un modèle de Régression.

**Modèle recommandé :** `RandomForestRegressor`

**Consigne :** Retirez `ID_Maison` et `{TARGET_COL}` des features.

**Livrables attendus :**
- Modèle entraîné.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Évaluation 📊

**Objectif :** Mesurer la précision en Dirhams.

**Métriques clés :**
- **MAE (Mean Absolute Error) :** Erreur moyenne en devise.
- **R² :** Qualité de l'ajustement.

**Livrables attendus :**
- Affichage des métriques.
- Graphique "Réalité vs Prédiction".
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- PART 4 : BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: Le Détecteur de Bonnes Affaires 💎
**Goal:** Identifier les maisons vendues moins cher que leur estimation.
**Approach:**
1. Créez un DataFrame comparant `Vrai_Prix` et `Estimation`.
2. Calculez la différence en %.
3. Filtrez les lignes où `Vrai_Prix` est inférieur à `Estimation` de plus de 10%.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Analyse par Quartier 🏙️
**Goal:** Visualiser le prix moyen par quartier.
**Approach:** Groupby sur le dataset original (avant encodage) et Barplot.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Immobilier_Intermediaire.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
