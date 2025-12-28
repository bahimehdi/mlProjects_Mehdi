import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "13"
    PROJECT_TITLE = "Prévision des Arrivées de Visiteurs"
    DATASET_NAME = "prevision_visiteurs.csv"
    TARGET_COL = "Visiteurs"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE} (Version Intermédiaire)

## 🏁 Objectif : Le Prophète du Tourisme 🔮
Votre mission est de construire un modèle capable de prédire le nombre de **Visiteurs** pour les prochains jours.
Cela permettra à la ville de gérer les flux et d'éviter la surfréquentation.

---

## 📋 Programme

### 🕵️‍♀️ SESSION 1 : From Raw Data to Clean Insights
- Gestion des valeurs manquantes (Prix, Événement)
- Analyse temporelle (Saisonnalité)

### 🏗️ SESSION 2 : The Art of Feature Engineering
- **Recipe Dates :** Extraction (Mois, Jour, Weekend)
- **Recipe Categories :** Encodage de la Ville

### 🤖 SESSION 3 : Building & Trusting Your Model
- Régression Temporelle (RandomForestRegressor)
- **Split Temporel :** Entraîner sur le passé, tester sur le futur
- **Bonus :** Détecter les jours de "Surtourisme"

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 1.1 : Chargement et Nettoyage 🧹

**Objectif :** Charger `{DATASET_NAME}` en parsant les dates.

**Points d'attention :**
- Utilisez `parse_dates=['Date']` dans `read_csv`.
- `Prix_Moyen_Hotel` -> Médiane.
- `Indicateur_Evenement` -> 0 (si NaN).

**Livrables attendus :**
- Un DataFrame propre avec des dates au bon format.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 1.2 : Analyse Exploratoire (EDA) 🔍

**Objectif :** Visualiser l'évolution temporelle.

**Questions :**
- Y a-t-il une tendance à la hausse ?
- Y a-t-il des pics récurrents (saisonnalité) ?

**Livrables attendus :**
- Lineplot `Date` vs `{TARGET_COL}`.
- Boxplot `Indicateur_Evenement` vs `{TARGET_COL}`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.1 : Recipe Dates 🕐

**Contexte :** Les modèles ne lisent pas les dates brutes.

**Objectif :** Extraire des features numériques.

**Features suggérées :**
- `Mois`, `Jour_Semaine`, `Jour_Mois`.
- `Est_Weekend` (0 ou 1).

**Livrables attendus :**
- Nouvelles colonnes dans le DataFrame.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Recipe Categories 🏷️

**Contexte :** La `City` est une variable catégorielle.

**Objectif :** Encoder cette variable.

**Approche recommandée :** One-Hot Encoding (`pd.get_dummies`).

**Livrables attendus :**
- DataFrame prêt pour l'IA (tout numérique).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🤖 SESSION 3 : Building & Trusting Your Model
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 3.1 : Split Temporel ✂️

**Objectif :** Séparer Passé (Train) et Futur (Test).

**Règle d'or :** Ne JAMAIS mélanger (`shuffle=False`) pour des séries temporelles !
Coupez les données à 80% (les 20% les plus récents sont le test).

**Livrables attendus :**
- `X_train`, `y_train` (Passé).
- `X_test`, `y_test` (Futur).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Entraînement et Évaluation 📊

**Objectif :** Entraîner un `RandomForestRegressor`.

**Métriques :**
- MAE (Erreur moyenne en nombre de visiteurs).
- R².

**Visualisation :**
- Tracez sur un même graphique la courbe Réelle (Test) et la courbe Prédite.

**Livrables attendus :**
- Scores MAE et R².
- Graphique de comparaison temporel.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- PART 4 : BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: Détecteur de Jours de Pointe 🚨
**Goal:** Identifier les jours futurs où la fréquentation dépassera un seuil critique (ex: 20 000).
**Approach:**
1. Créez un DataFrame avec `Date` et `Prediction`.
2. Filtrez les lignes où `Prediction > 20000`.
3. Affichez ces dates pour prévenir la mairie.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Visiteurs_Intermediaire.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
