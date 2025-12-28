import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --- Titre ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🎓 Projet 4 : Système d'Alerte Précoce de Décrochage Scolaire
## Version Intermédiaire - "Voici le chemin, marche seul"

---

### 🎯 L'Objectif
Construire un modèle de classification capable de **détecter les élèves à risque de décrochage** (`A_Decroche` = 1).

**Contexte Métier :**
- **Cible** : `A_Decroche` (Binaire : 0 ou 1)
- **Problème** : Classification Déséquilibrée (Les décrocheurs sont minoritaires).
- **Priorité** : **Maximiser le Rappel (Recall)**. Il est plus grave de rater un élève en difficulté (Faux Négatif) que d'inquiéter un élève pour rien (Faux Positif).

---

### 📋 SESSION 1 : From Raw Data to Clean Insights

#### Étape 1.1 : Chargement et Inspection
**Objectif :** Charger `decrochage_scolaire.csv` et comprendre la structure.

**Livrables attendus :**
- Dimensions et types.
- Identification des colonnes catégorielles (`Education_Parents`) et numériques.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 1.2 : Nettoyage & Gestion des Manquants
**Objectif :** Remplir les trous.

**Approches recommandées :**
- `Temps_Trajet` (Numérique) : Médiane.
- `Education_Parents` (Catégorique) : Mode.

**Livrables attendus :**
- Dataset propre sans NaN.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 1.3 : Analyse du Déséquilibre (EDA)
**Objectif :** Quantifier le déséquilibre de classe.

**Analyses à réaliser :**
1. **Countplot** de la cible `A_Decroche`. Calculez le % de décrocheurs.
2. **Boxplot** : `Presence` vs `A_Decroche`. Les décrocheurs sont-ils moins présents ?
3. **Barplot** : `Education_Parents` vs `A_Decroche`.

**Question :** Si vous aviez un modèle qui prédit toujours "0" (Ne décroche pas), quelle serait son Accuracy ? Pourquoi est-ce trompeur ?
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 📋 SESSION 2 : The Art of Feature Engineering

#### Étape 2.1 : Encodage (Recipe 2)
**Objectif :** Traiter `Education_Parents`.

**Approche :**
- One-Hot Encoding (`pd.get_dummies`) est recommandé car il n'y a pas d'ordre strict linéaire évident (ou discutable).

**Livrables attendus :**
- Colonnes `Edu_HighSchool`, `Edu_Uni`, etc.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.2 : Feature Engineering Métier (Recipe 4)
**Objectif :** Créer des indicateurs de risque composites.

**Idées de Features :**
1. `Risk_Score` : Combinaison de `Presence` (faible) et `Notes` (faibles).
   - Ex: `(100 - Presence) + (20 - Notes) * Coeff`
2. `Long_Commute` : Binaire. Est-ce que le trajet > 45 min ?

**Conseil :** Vérifiez si ces nouvelles features sont corrélées avec la cible.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 📋 SESSION 3 : Building & Trusting Your Model

#### Étape 3.1 : Split Stratifié
**Objectif :** Diviser Train/Test en gardant la même proportion de décrocheurs.

**Contrainte :** Utilisez `stratify=y` dans `train_test_split`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 3.2 : Entraînement avec Poids
**Objectif :** Forcer le modèle à apprendre sur la classe minoritaire.

**Modèle :** `RandomForestClassifier`
**Paramètre Clé :** `class_weight='balanced'` (Indispensable !)

**Livrables attendus :**
- Modèle entraîné.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 3.3 : Évaluation (Focus Recall)
**Objectif :** Valider la capacité de détection.

**Métriques :**
1. **Recall (Rappel)** : PRIORITAIRE. Doit être élevé.
2. **Confusion Matrix** : Visualisez les Faux Négatifs (Élèves ratés).
3. **F1-Score** : Bon compromis.

**Question :** Combien d'élèves à risque avez-vous manqués (Faux Négatifs) ?
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 🎁 Part 4: Going Further (Bonus Tasks)

#### Bonus Task 1: Facteurs Clés
**Goal:** Quels sont les signes avant-coureurs ?
**Approach:** Affichez `model.feature_importances_`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 2: Segmentation (Clustering)
**Goal:** Identifier des profils types d'élèves.
**Approach:** KMeans sur `Presence` et `Notes`. Visualisez les clusters.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 3: Système de Recommandation
**Goal:** Suggérer une action pour chaque élève à risque.
**Approach:**
- Si `Risk_Score` > Seuil → "Tutorat"
- Si `Trajet` > 60 → "Internat/Transport"
Créez une fonction et appliquez-la.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    nb['cells'] = cells
    nbf.write(nb, 'donnees_fr/Projet_04/Projet_04_Intermediaire.ipynb')

if __name__ == "__main__":
    generer_notebook_intermediaire()
