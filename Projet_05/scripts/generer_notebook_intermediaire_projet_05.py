import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --- Titre ---
    cells.append(nbf.v4.new_markdown_cell("""
# 💸 Projet 5 : Score de Risque Micro-Crédit
## Version Intermédiaire - "Voici le chemin, marche seul"

---

### 🎯 L'Objectif
Construire un modèle capable de **prédire la probabilité de défaut** de paiement et de créer un **Score de Crédit** pour les entrepreneurs non-bancarisés.

**Contexte Métier :**
- **Cible** : `Defaillant` (Binaire : 0 ou 1)
- **Besoin** : Pas juste une prédiction binaire, mais une **Probabilité** (0 à 100%) pour créer un score.
- **Métrique Clé** : **ROC-AUC** (capacité à discriminer les risques).

---

### 📋 SESSION 1 : From Raw Data to Clean Insights

#### Étape 1.1 : Chargement et Inspection
**Objectif :** Charger `micro_credit.csv` et comprendre les demandeurs.

**Livrables attendus :**
- Distribution du taux de défaut global.
- Statistiques descriptives des montants de prêt.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 1.2 : Nettoyage
**Objectif :** Gérer les valeurs manquantes.

**Approches recommandées :**
- `Annees_Activite` (Numérique) : Médiane.
- `Usage_Mobile` (Numérique) : Médiane.

**Livrables attendus :**
- Dataset propre.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 1.3 : Analyse Exploratoire (EDA)
**Objectif :** Identifier les facteurs de risque.

**Analyses à réaliser :**
1. **Boxplot** : `Montant_Pret` vs `Defaillant`. Les gros prêts sont-ils plus risqués ?
2. **Barplot** : Taux de défaut par `Type_Entreprise`.
3. **Scatter** : `Usage_Mobile` vs `Montant_Pret` (coloré par défaut).

**Question :** Quel secteur devrait avoir des conditions de prêt plus strictes ?
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 📋 SESSION 2 : The Art of Feature Engineering

#### Étape 2.1 : Encodage (Recipe 2)
**Objectif :** Traiter `Type_Entreprise`.

**Approche :**
- One-Hot Encoding (`pd.get_dummies`).

**Livrables attendus :**
- Colonnes `Secteur_Retail`, `Secteur_Agri`, etc.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.2 : Feature Engineering Métier (Recipe 4)
**Objectif :** Créer des indicateurs de risque composites.

**Idées de Features :**
1. `Ratio_Pret_Mobile` : `Montant_Pret` / (`Usage_Mobile` + 1) → Proxy "dette/revenu".
2. `Nouveau_Business` : Binaire (< 2 ans d'activité).
3. `Gros_Pret` : Binaire (Montant > médiane).

**Conseil :** Testez la corrélation de ces features avec la cible.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 📋 SESSION 3 : Building & Trusting Your Model

#### Étape 3.1 : Split
**Objectif :** Diviser Train/Test (80/20).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 3.2 : Entraînement
**Objectif :** Créer un modèle capable de prédire des probabilités.

**Modèle :** `RandomForestClassifier`

**Livrables attendus :**
- Modèle entraîné.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 3.3 : Évaluation (Focus Probabilités)
**Objectif :** Mesurer la qualité de la discrimination.

**Métriques :**
1. **ROC-AUC Score** : PRIORITAIRE. Mesure la capacité à classer correctement.
2. **Confusion Matrix** : Pour voir les erreurs absolues.
3. **Classification Report** : Precision, Recall.

**IMPORTANT :** Utilisez `predict_proba` pour obtenir les probabilités, pas juste `predict`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
---

### 🎁 Part 4: Going Further (Bonus Tasks)

#### Bonus Task 1: Score de Crédit (300-850)
**Goal:** Transformer les probabilités en scores bancaires.
**Approach:** `Score = 850 - (Proba_Defaut * 550)`
Affichez les scores pour un échantillon de demandeurs.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 2: Segmentation de Risque (A, B, C)
**Goal:** Classifier les demandeurs en 3 catégories.
**Approach:**
- A (Score > 700) : Faible risque
- B (600-700) : Risque moyen
- C (< 600) : Risque élevé
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 3: Montant de Prêt Recommandé
**Goal:** Pour chaque segment de risque, définir un montant maximum sûr.
**Approach:** Règle métier basée sur le score.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    nb['cells'] = cells
    nbf.write(nb, 'donnees_fr/Projet_05/Projet_05_Intermediaire.ipynb')

if __name__ == "__main__":
    generer_notebook_intermediaire()
