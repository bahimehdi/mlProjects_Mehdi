import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    # Titre
    nb.cells.append(nbf.v4.new_markdown_cell("""
# PROJET 20 : PREDICTION D'EPIDEMIE (Niveau Intermediaire)

**Objectif :** Construire un modele de regression pour predire le nombre de cas hebdomadaires d epidemie.

---

## STRUCTURE DU PROJET

### SESSION 1 : Analyse Exploratoire & Nettoyage
- Chargement et inspection
- Gestion des valeurs manquantes
- Analyse des tendances temporelles

### SESSION 2 : Feature Engineering
- Extraction de features temporelles (Mois, Annee)
- Creation de lag features et moyennes mobiles
- Encodage des regions

### SESSION 3 : Modelisation (Regression)
- Entrainement RandomForestRegressor
- Evaluation (MAE, RMSE, R2)

### Part 4: Taches Bonus
- Classification de risque
- Analyse temps de latence (Pluie -> Epidemie)
- Allocation ressources medicales

---
"""))

    # SESSION 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 1 : DATA EXPLORATION & CLEANING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 1.1 : Chargement et Inspection
**Objectif :** Charger `epidemie.csv` et identifier la cible.
**Livrables :**
- `df.head()`, `df.info()`
- Variable cible : `Cases`
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 1.2 : Nettoyage
**Objectif :** Gerer les valeurs manquantes dans `Google_Trends` et `Precipitations`.
**Approches recommandees :**
- Imputation par mediane (recommande pour donnees continues)
- Suppression (si < 5%)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 1.3 : Analyse Exploratoire
**Objectif :** Comprendre la dynamique temporelle.
**Visualisations attendues :**
- Evolution des cas dans le temps (line plot)
- Cas moyens par region
- Correlation temperature vs cas

**Conseil :** Cherchez des pics epidemiques et des tendances saisonnieres.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # SESSION 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 2 : FEATURE ENGINEERING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 2.1 : Features Temporelles (Recipe 1)
**Objectif :** Extraire des informations de `Week`.
**Features a creer :**
- `Mois`, `Annee`, `Week_of_Year`

**Conseil :** Utilisez `pd.to_datetime()` puis `.dt` accessors.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 2.2 : Encodage des Categories (Recipe 2)
**Objectif :** Transformer `Region` en format numerique.
**Methode :** One-Hot Encoding (`pd.get_dummies`).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 2.3 : Features Series Temporelles (Recipe 6)
**Objectif :** Creer des features specifiques aux series temporelles.

**Features recommandees :**
1. **Lag 1** : Cas de la semaine precedente
   - Formule : `df['Cases'].shift(1)`
   - Pourquoi : Le passe recent influence le futur proche.

2. **Moyenne mobile (4 semaines)** : Tendance lissee
   - Formule : `df['Cases'].shift(1).rolling(4).mean()`
   - Pourquoi : Capture la tendance generale.

**IMPORTANT :** Utilisez `shift(1)` pour eviter le data leakage.

**Conseil :** Triez par date avant de creer les lag features.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # SESSION 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 3 : MODELING - REGRESSION
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 3.1 : Preparation et Split
**Objectif :** Separer Features (X) et Target (y).
**Target :** `Cases`
**Split :** 80% Train, 20% Test

**Important :** Supprimez `Week` et les lignes avec NaN (creees par lag).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 3.2 : Entrainement
**Modele recommande :** RandomForestRegressor
**Alternative :** GradientBoostingRegressor (meilleur pour series temporelles).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 3.3 : Evaluation
**Metriques cles :**
- **MAE** : Erreur moyenne en nombre de cas.
- **R2** : Pourcentage de variance expliquee.

**Feature Importance :** Identifiez les variables les plus influentes.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # PART 4 BONUS
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 4: Going Further (Bonus)

### Bonus Task 1: Classifier le Niveau de Risque

**Goal:** Creer une classification Faible/Moyen/Epidemique.

**Approach:**
1. Definir seuils : Faible < 200, Moyen 200-500, Epidemique > 500
2. Classifier les predictions
3. Compter la distribution

**Deliverable :** Distribution des niveaux de risque et alertes.

### Bonus Task 2: Temps de Latence (Pluie -> Epidemie)

**Goal:** Determiner combien de semaines apres la pluie l epidemie eclate.

**Approach:**
1. Tester lags de 0 a 8 semaines pour `Precipitations`
2. Calculer correlation avec `Cases` pour chaque lag
3. Identifier le lag avec correlation maximale

**Deliverable :** Temps de latence optimal (nombre de semaines).

### Bonus Task 3: Google Trends comme Indicateur Precoce

**Goal:** Verifier si Google Trends predit les cas officiels.

**Approach:**
1. Calculer correlation entre `Google_Trends` et `Cases`
2. Si correlation > 0.7 : Excellent indicateur
3. Si correlation > 0.5 : Utile
4. Sinon : Pas fiable

**Deliverable :** Score de correlation et recommandation.

### Bonus Task 4: Allocation des Ressources Medicales

**Goal:** Recommander l allocation de lits d hopital par region.

**Approach:**
1. Calculer cas moyens par region
2. Allouer 5% des cas comme nombre de lits necessaires
3. Creer table d allocation

**Deliverable :** Table Region -> Nombre de lits a preparer.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici pour les bonus
"""))

    # Sauvegarde
    with open('Projet_20_Epidemie_Intermediaire.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_intermediaire()
