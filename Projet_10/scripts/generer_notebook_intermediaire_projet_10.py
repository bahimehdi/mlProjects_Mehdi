import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "10"
    PROJECT_TITLE = "Recommandation de Voyage Personnalisée"
    DATASET_NAME = "recommandation_voyage.csv"
    TARGET_COL = "Note_Destination"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE} (Version Intermédiaire)

## 🏁 Objectif : L'Agence de Voyage IA
Votre mission est de construire un moteur de recommandation capable de prédire la satisfaction (`{TARGET_COL}`) d'un utilisateur pour un voyage donné.
C'est un problème de **Régression** (prédire une note continue).

---

## 📋 Programme

### 🕵️‍♀️ SESSION 1 : From Raw Data to Clean Insights
- Gestion des valeurs manquantes (Budget, Age)
- Analyse de la distribution des notes et des corrélations

### 🏗️ SESSION 2 : The Art of Feature Engineering
- **Recipe Categories :** Encodage One-Hot
- **Recipe Math :** Log-transformation du Budget (pour gérer les écarts de richesse)

### 🤖 SESSION 3 : Building & Trusting Your Model
- Régression (RandomForestRegressor)
- Évaluation (MAE, RMSE, R²)
- **Bonus :** Créer une fonction de recommandation et segmenter les utilisateurs (Clustering)

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
- `Budget_Quotidien` a des valeurs manquantes.
- Stratégie recommandée : Remplacer par la **Médiane** (plus robuste que la moyenne face aux milliardaires !).

**Livrables attendus :**
- Un DataFrame propre sans NaNs.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 1.2 : Analyse Exploratoire (EDA) 🔍

**Objectif :** Comprendre la cible `{TARGET_COL}`.

**Questions :**
- Quelle est la note moyenne ?
- Y a-t-il une corrélation entre le Budget et la Note ?

**Livrables attendus :**
- Histogramme de la distribution des notes.
- Scatterplot Budget vs Note.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.1 : Recipe Categories 🏷️

**Contexte :** `Style_Voyage` et `Climat_Prefere` sont du texte.

**Objectif :** Les encoder pour le modèle.

**Approche recommandée :**
- **One-Hot Encoding** (`pd.get_dummies`) car pas d'ordre logique.

**Livrables attendus :**
- DataFrame avec colonnes binaires (ex: `Style_Voyage_Adventure`).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Recipe Math (Log Transformation) ➗

**Contexte :** Le `Budget_Quotidien` a une distribution très étalée (skewed).
Cela peut gêner certains modèles.

**Objectif :** Créer une nouvelle feature `Log_Budget`.

**Formule :** `np.log1p(Budget)` (le +1 évite log(0)).

**Livrables attendus :**
- Nouvelle colonne `Log_Budget`.
- Comparaison visuelle (Histogramme Budget vs Log_Budget).
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

**Consigne :** N'oubliez pas de retirer `ID_Utilisateur` (inutile) et `Budget_Quotidien` (remplacé par Log_Budget) des features.

**Livrables attendus :**
- Modèle entraîné sur le train set (80%).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Évaluation 📊

**Objectif :** Mesurer la précision des prédictions.

**Métriques clés :**
- **MAE (Mean Absolute Error) :** L'erreur moyenne en points de note.
- **RMSE (Root Mean Squared Error) :** Punit plus les grosses erreurs.
- **R² :** Pourcentage de variance expliquée.

**Livrables attendus :**
- Affichage des métriques.
- Graphique "Réalité vs Prédiction" (Scatterplot).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- PART 4 : BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: Moteur de Recommandation 🌟
**Goal:** Pour un utilisateur donné (Age, Budget), trouver le meilleur voyage.
**Approach:**
1. Créez un utilisateur fictif (ex: 30 ans, 500€).
2. Générez toutes les combinaisons possibles de Style et Climat.
3. Utilisez le modèle pour prédire la note de chaque combinaison.
4. Affichez le Top 3.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Tribus de Voyageurs (Clustering) 👥
**Goal:** Segmenter les utilisateurs en groupes homogènes.
**Approach:** Utilisez `KMeans` sur `Age` et `Budget_Quotidien`.
**Visualisation:** Scatterplot coloré par Cluster.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Recommandation_Intermediaire.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
