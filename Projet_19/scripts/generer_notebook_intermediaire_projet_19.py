import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    # Titre
    nb.cells.append(nbf.v4.new_markdown_cell("""
# PROJET 19 : DETECTION DE FRAUDE CARTE DE CREDIT (Niveau Intermediaire)

**Objectif :** Construire un classifieur binaire pour detecter les fraudes avec un **Rappel Eleve** (la classe minoritaire).

---

## STRUCTURE DU PROJET

### SESSION 1 : Analyse Exploratoire & Nettoyage
- Chargement et inspection
- Gestion des valeurs manquantes
- Analyse du desequilibre de classe

### SESSION 2 : Feature Engineering
- Encodage des categories
- Creation de features metier (heure inhabituelle, z-score montant)

### SESSION 3 : Modelisation IMBALANCED Classification
- SMOTE pour equilibrer
- Entrainement avec RandomForest
- Evaluation avec **RECALL prioritaire**

### Part 4: Taches Bonus
- Analyse Cout-Benefice (FP vs FN)
- Systeme de scoring temps reel

---
"""))

    # SESSION 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 1 : DATA EXPLORATION & CLEANING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 1.1 : Chargement et Inspection
**Objectif :** Charger `fraude_carte_credit.csv` et identifier la cible.
**Livrables :**
- `df.head()`, `df.info()`
- Variable cible : `Class` (0=Legitime, 1=Fraude)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 1.2 : Nettoyage
**Objectif :** Gerer les valeurs manquantes dans `Location_Distance`.
**Approches recommandees :**
- Imputation par mediane (recommande pour donnees continues)
- Suppression (si < 5%)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 1.3 : Analyser le Desequilibre
**Objectif :** Calculer le ratio Fraude/Legitime.
**Visualisations attendues :**
- Countplot des classes
- Boxplot Amount par Class

**Conseil :** Le desequilibre (~97% legitime, ~3% fraude) necessite SMOTE.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # SESSION 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 2 : FEATURE ENGINEERING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 2.1 : Encodage des Categories (Recipe 2)
**Objectif :** Transformer `Transaction_Type` en format numerique.
**Methode :** One-Hot Encoding (`pd.get_dummies`).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 2.2 : Features Metier (Recipe 6)
**Objectif :** Creer des variables specifiques a la detection de fraude.

**Features recommandees :**
1. **Is_Night** : 1 si Time_Hour entre 0-5h (transactions suspectes)
2. **Amount_Zscore** : (Amount - mean) / std (deviations extremes)
3. **Has_Fraud_History** : 1 si Previous_Fraud_Attempts > 0

**Conseil :** Les fraudeurs operent souvent la nuit avec des montants inhabituels.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # SESSION 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 3 : MODELING - IMBALANCED CLASSIFICATION
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 3.1 : Preparation et Split
**Objectif :** Separer Features (X) et Target (y).
**Target :** `Class`
**Split :** 80% Train, 20% Test (avec `stratify=y` pour preserver le ratio)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Etape 3.2 : Entrainement avec Gestion du Desequilibre

**Contexte Metier :** 
La classe minoritaire (Fraude) represente ~3% des donnees.
Les faux negatifs (manquer une fraude) sont beaucoup plus couteux (500) que les faux positifs (10).

**Objectif :** Maximiser le **Recall** de la classe fraude (>= 85% recommande).

**Strategies de reequilibrage (choisissez une) :**

1. **SMOTE (Synthetic Minority Over-sampling)** Recommande
   - Genere des exemples synthetiques de fraudes
   - Librairie : `imblearn.over_sampling.SMOTE`
   - Avantage : Pas de perte de donnees
   - Inconvenient : Peut creer des exemples bruites

2. **Random Undersampling**
   - Supprime des exemples legitimes
   - Avantage : Rapide
   - Inconvenient : Perte d'information

3. **Class Weight Balancing**
   - Parametre `class_weight='balanced'` dans RandomForest
   - Avantage : Simple
   - Inconvenient : Moins efficace si desequilibre > 1:10

**Livrables attendus :**
- Distribution avant/apres reequilibrage
- Modele entraine sur donnees equilibrees
- **Recall >= 85%** pour la classe fraude
- Matrice de confusion avec analyse des Faux Negatifs
- ROC Curve et AUC

**Metriques a calculer (par ordre de priorite) :**
1. **Recall (classe fraude)** METRIQUE PRINCIPALE
2. ROC-AUC
3. F1-Score
4. Precision (secondaire)

**Conseil :** Si Recall < 80%, ajustez le threshold :
```python
# Lower threshold to catch more frauds (increase Recall)
threshold = 0.3  # Default is 0.5
y_pred_custom = (y_pred_proba >= threshold).astype(int)
```
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # PART 4 BONUS
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 4: Going Further (Bonus)

### Bonus Task 1: Analyse Cout-Benefice (Threshold Analysis)

**Goal:** Trouver le seuil de probabilite optimal qui minimise le cout total.

**Why it matters:** 
- Faux Positif (bloquer transaction legitime) = Perte de satisfaction client (10)
- Faux Negatif (manquer une fraude) = Perte financiere (500)

**Approach:**
1. Definir couts : `cost_FP = 10`, `cost_FN = 500`
2. Tester seuils de 0.1 a 0.9
3. Calculer cout total pour chaque seuil
4. Choisir seuil avec cout minimum

**Example:**
```python
# TODO: Loop through thresholds
# thresholds = np.arange(0.1, 0.9, 0.05)
# for threshold in thresholds:
#     y_pred_custom = (y_pred_proba >= threshold).astype(int)
#     FP = ((y_pred_custom == 1) & (y_test == 0)).sum()
#     FN = ((y_pred_custom == 0) & (y_test == 1)).sum()
#     total_cost = FP * 10 + FN * 500
#     print(f"Threshold {threshold:.2f}: Cost = ${total_cost}")
```

**Deliverable:** Seuil optimal et graphique de comparaison des couts.

### Bonus Task 2: Systeme de Scoring en Temps Reel

**Goal:** Creer une fonction qui score une nouvelle transaction en temps reel.

**Approach:**
1. Definir fonction `fraud_score(transaction_dict)`
2. Preparer les features (same pipeline que training)
3. Retourner probabilite de fraude
4. Comparer au seuil optimal

**Deliverable:** Fonction + test sur 5 transactions fictives.

### Bonus Task 3: Analyse des Patterns de Fraude

**Goal:** Identifier les caracteristiques communes des fraudes.

**Approach:**
1. Separer frauds_df et legit_df
2. Comparer moyennes : Amount, Time_Hour, Location_Distance
3. Calculer pourcentages : Is_Night, Is_Foreign, Pin_Entered

**Deliverable:** Tableau de comparaison et recommandations de prevention.

### Bonus Task 4: Visualisation t-SNE (Optionnel)

**Goal:** Visualiser la separation Fraude vs Legitime en 2D.

**Approach:**
1. Appliquer t-SNE sur X_test
2. Scatter plot colore par vraie classe
3. Identifier zones de chevauchement

**Deliverable:** Graphique t-SNE avec legende.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici pour les bonus
"""))

    # Sauvegarde
    with open('Projet_19_Fraude_Intermediaire.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_intermediaire()
