import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "09"
    PROJECT_TITLE = "Gravité des Accidents de la Route"
    DATASET_NAME = "accidents_route.csv"
    TARGET_COL = "Gravite"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE} (Version Intermédiaire)

## 🏁 Objectif : Vision Zéro
Votre mission est de construire un modèle de classification capable de prédire la **Gravité** (1 à 4) d'un accident.
Le défi principal sera de gérer le déséquilibre des classes (les accidents mortels sont heureusement plus rares) et d'exploiter les données géographiques.

---

## 📋 Programme

### 🕵️‍♀️ SESSION 1 : From Raw Data to Clean Insights
- Nettoyage des données (valeurs manquantes, coordonnées GPS invalides)
- Analyse Exploratoire (Distribution de la gravité, corrélations)

### 🏗️ SESSION 2 : The Art of Feature Engineering
- **Recipe Geography :** Extraction Lat/Lon
- **Recipe Categories :** Encodage des variables catégorielles

### 🤖 SESSION 3 : Building & Trusting Your Model
- Classification Multi-classe
- Gestion du déséquilibre (SMOTE ou Class Weights)
- Évaluation (Recall, F1-Score, Matrice de Confusion)

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 1.1 : Chargement et Nettoyage Initial

**Objectif :** Charger `{DATASET_NAME}` et traiter les valeurs manquantes.

**Points d'attention :**
- La colonne `Meteo` contient des valeurs manquantes. Quelle est la meilleure stratégie pour une variable catégorielle ? (Mode ? "Inconnu" ?)
- Vérifiez les types de données.

**Livrables attendus :**
- Un DataFrame propre (sans NaNs dans les colonnes critiques)
- Un aperçu des 5 premières lignes
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 1.2 : Analyse de la Cible (Target Analysis)

**Objectif :** Comprendre la distribution de la variable `{TARGET_COL}`.

**Question clé :** Les classes sont-elles équilibrées ? Si non, quelle classe est minoritaire ?

**Livrables attendus :**
- Un graphique (Countplot) montrant la fréquence de chaque niveau de gravité.
- Le calcul des pourcentages pour chaque classe.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.1 : Recipe Geography (Coordonnées GPS) 🗺️

**Contexte :** La colonne `Localisation` est une chaîne de caractères "Lat, Lon".

**Objectif :** Créer deux nouvelles colonnes numériques `Latitude` et `Longitude`.

**Challenge :**
- Certaines lignes contiennent "0.0, 0.0". Ce sont des erreurs (null island).
- **Action requise :** Identifiez et supprimez ces lignes aberrantes.

**Livrables attendus :**
- DataFrame avec `Latitude` et `Longitude` (float)
- Suppression de la colonne originale `Localisation`
- Suppression des lignes avec coordonnées (0,0)
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Recipe Categories 🏷️

**Contexte :** Les colonnes `Meteo`, `Type_Route`, `Type_Vehicule` sont du texte.

**Objectif :** Les transformer en nombres pour le modèle.

**Approche recommandée :**
- **One-Hot Encoding** (`pd.get_dummies`) car il n'y a pas d'ordre intrinsèque (Nominal).

**Livrables attendus :**
- Un DataFrame `df_encoded` prêt pour l'entraînement (toutes colonnes numériques).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🤖 SESSION 3 : Building & Trusting Your Model
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 3.1 : Split Train/Test

**Objectif :** Séparer les features (X) et la target (y), puis diviser en ensembles d'entraînement et de test.

**Conseil :** Utilisez `stratify=y` dans `train_test_split` pour conserver la même proportion de classes (surtout les rares) dans les deux ensembles.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Entraînement avec Gestion du Déséquilibre ⚖️

**Contexte Métier :**
La classe 4 (Accident Mortel) est minoritaire mais CRITIQUE.
Un modèle standard risque de l'ignorer pour maximiser l'Accuracy globale.

**Objectif :** Maximiser le **Recall** (Rappel) pour les classes graves (3 et 4).

**Stratégies (choisissez-en une) :**
1. **SMOTE (Recommandé) :** Génération de données synthétiques pour les classes minoritaires.
   - `from imblearn.over_sampling import SMOTE`
2. **Class Weights :** Dire au modèle de "payer plus cher" les erreurs sur les classes rares.
   - `RandomForestClassifier(class_weight='balanced')`

**Livrables attendus :**
- Un modèle entraîné (sur données rééquilibrées ou avec poids).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.3 : Évaluation Approfondie

**Objectif :** Valider la performance du modèle.

**Métriques clés :**
- **Confusion Matrix :** Pour voir les confusions entre classes adjacentes (ex: prédire 3 au lieu de 4).
- **Classification Report :** Regardez le F1-Score et le Recall par classe.

**Question de réflexion :**
Le modèle arrive-t-il à bien détecter la classe 4 ?
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- PART 4 : BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: Identifier les "Points Noirs" ⚫
**Goal:** Trouver les zones géographiques où les accidents sont fréquents ou graves.
**Approach:** Utilisez un Scatterplot (Longitude vs Latitude) coloré par Gravité.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Impact de la Pluie 🌧️
**Goal:** Déterminer si la pluie aggrave les accidents.
**Approach:** Comparez la distribution de la Gravité pour Meteo='Rain' vs 'Clear'.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Recommandation de Vitesse 🚀
**Goal:** Identifier sur quel `Type_Route` les accidents graves sont les plus fréquents pour recommander des radars.
**Approach:** Analysez le pourcentage d'accidents graves (3+4) par type de route.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Accidents_Intermediaire.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
