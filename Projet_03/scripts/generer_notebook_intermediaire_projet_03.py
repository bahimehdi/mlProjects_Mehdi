import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Titre
    cells.append(nbf.v4.new_markdown_cell("""
# 😷 Projet 3 : Qualité de l'Air & Santé
## Version Intermédiaire - "Voici le chemin, marche seul"

---

### 🎯 L'Objectif
Construire un modèle de régression capable de **prédire le nombre d'admissions hospitalières** (`Admissions_Respiratoires`) en fonction de la qualité de l'air et des conditions environnementales.

**Contexte Métier :**
- **Cible** : `Admissions_Respiratoires` (Numérique)
- **Métrique Clé** : MAE (Mean Absolute Error) pour l'interprétabilité, R² pour la performance globale.
- **Impact** : Aider les hôpitaux à anticiper l'afflux de patients lors des pics de pollution.

---

### 📋 SESSION 1 : From Raw Data to Clean Insights

#### Étape 1.1 : Chargement et Inspection
**Objectif :** Charger `qualite_air.csv` et identifier les problèmes de qualité.

**Livrables attendus :**
- Dimensions du dataset
- Types des colonnes (attention à la colonne `Date` !)
- Nombre de valeurs manquantes par colonne

**Conseil :** Utilisez `df.info()` et `df.isnull().sum()`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 1.2 : Nettoyage des Données (Sanity Check)
**Objectif :** Traiter les valeurs manquantes et corriger les types.

**Approches recommandées :**
1. **Valeurs manquantes (`PM2_5`, `NO2`)** :
   - Remplacer par la **médiane** (plus robuste aux outliers que la moyenne).
2. **Dates (`Date`)** :
   - Convertir en objet `datetime` avec `pd.to_datetime()`.

**Livrables attendus :**
- Un dataset sans valeurs manquantes.
- La colonne `Date` en format datetime.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 1.3 : Analyse Exploratoire (EDA)
**Objectif :** Comprendre les facteurs de pollution.

**Analyses à réaliser :**
1. **Séries Temporelles** : Visualisez l'évolution de `PM2_5` et `Admissions_Respiratoires` dans le temps.
2. **Corrélation Trafic/Pollution** : Scatter plot entre `Volume_Trafic` et `NO2`.
3. **Distribution** : Histogramme des admissions.

**Conseil :** Utilisez `sns.lineplot` pour les séries temporelles.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 📋 SESSION 2 : The Art of Feature Engineering

#### Étape 2.1 : Feature Engineering Temporel (Recipe 1)
**Objectif :** Transformer la date en informations utiles pour le modèle.

**Features à créer :**
- `Mois` : Pour capturer la saisonnalité (hiver vs été).
- `Jour_Semaine` : Pour capturer l'effet week-end (moins de trafic ?).
- `Est_Weekend` : Binaire (1 si Samedi/Dimanche, 0 sinon).

**Pourquoi ?** La pollution varie fortement selon les saisons et l'activité humaine (semaine vs week-end).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.2 : Encodage des Catégories (Recipe 2)
**Objectif :** Transformer `Direction_Vent` (texte) en nombres.

**Approche :**
- Utilisez **One-Hot Encoding** (`pd.get_dummies`) car il n'y a pas d'ordre logique entre Nord, Sud, Est, Ouest.

**Livrables attendus :**
- Colonnes `Vent_N`, `Vent_S`, etc.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.3 : Feature Engineering Mathématique (Recipe 4)
**Objectif :** Créer un indicateur global de pollution.

**Idée :**
- Créez `Pollution_Index` = `PM2_5` + `NO2`.
- (Optionnel) Créez une interaction `Trafic_x_Vent` ?

**Conseil :** Vérifiez la corrélation de cette nouvelle feature avec la cible `Admissions_Respiratoires`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 📋 SESSION 3 : Building & Trusting Your Model

#### Étape 3.1 : Préparation et Split
**Objectif :** Diviser les données en Train/Test.

**Contraintes :**
- Supprimez la colonne `Date` originale (non gérée par le modèle).
- Split 80/20.
- `random_state=42` pour la reproductibilité.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 3.2 : Entraînement (Régression)
**Modèle recommandé :** `RandomForestRegressor`

**Pourquoi ?** Il gère bien les relations non-linéaires (ex: effet de seuil de la pollution sur la santé).

**Livrables attendus :**
- Un modèle entraîné sur X_train, y_train.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 3.3 : Évaluation Approfondie
**Objectif :** Valider la performance du modèle.

**Métriques à calculer :**
1. **MAE** : Erreur moyenne en nombre de patients.
2. **RMSE** : Sensibilité aux grosses erreurs.
3. **R²** : Qualité globale du modèle.

**Visualisation :**
- Scatter plot `y_test` vs `y_pred` avec une ligne diagonale parfaite.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 🎁 Part 4: Going Further (Bonus Tasks)

#### Bonus Task 1: Classification "Sain" vs "Dangereux"
**Goal:** Créer une alerte simple pour le public.
**Approach:**
1. Créez une colonne `Status` : "Dangereux" si PM2.5 > 100, sinon "Sain".
2. Affichez la distribution (combien de jours dangereux ?).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 2: Analyse Hebdomadaire
**Goal:** Identifier le meilleur jour pour faire du sport en extérieur.
**Approach:**
1. Groupez par `Jour_Semaine`.
2. Calculez la moyenne de `PM2_5`.
3. Visualisez avec un bar chart.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 3: Prédiction pour Demain (Lag)
**Goal:** Prédire les admissions de demain en utilisant les données d'aujourd'hui.
**Approach:**
1. Créez une feature `Admissions_Hier` avec `shift(1)`.
2. Ré-entraînez le modèle avec cette nouvelle feature puissante.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # Assign cells to notebook
    nb['cells'] = cells

    # Sauvegarde
    nbf.write(nb, 'donnees_fr/Projet_03/Projet_03_Intermediaire.ipynb')

if __name__ == "__main__":
    generer_notebook_intermediaire()
