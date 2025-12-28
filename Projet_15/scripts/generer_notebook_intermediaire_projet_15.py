import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CELLULES DU NOTEBOOK ---
    
    # TITRE ET INTRODUCTION
    cells = [
        nbf.v4.new_markdown_cell("""# 🎓 PROJET 15 : Optimiseur d'Annulation d'Hôtel (Version Intermédiaire)
## 🏁 Objectif du Projet
Développer un modèle prédictif pour estimer la probabilité d'annulation d'une réservation d'hôtel.
L'objectif final est de fournir une recommandation de **limite de surréservation** (Overbooking Limit) pour maximiser le taux d'occupation sans risque.

## 📂 Les Données
Fichier : `annulation_hotel.csv`
Cible : `Annule` (0 = Non, 1 = Oui)

---
# 📋 SESSION 1 : From Raw Data to Clean Insights (45 min)
"""),
        
        # PART 1: THE SETUP
        nbf.v4.new_markdown_cell("""## 🛠️ Part 1: The Setup
**Objectif :** Charger les données et préparer l'environnement.
**Livrables :** DataFrame chargé, aperçu des types et dimensions."""),
        
        nbf.v4.new_code_cell("""# Votre code ici (Imports : pandas, numpy, matplotlib, seaborn)
# Votre code ici (Chargement du CSV)
# Votre code ici (Info et Head)"""),
        
        # PART 2: THE SANITY CHECK
        nbf.v4.new_markdown_cell("""## 🧹 Part 2: The Sanity Check
**Objectif :** Nettoyer le dataset pour l'analyse.
**Approches recommandées :**
- **Valeurs manquantes :** Imputation (Médiane pour numériques, Mode pour catégorielles).
- **Doublons :** Suppression.
**Livrables :** Un dataset propre sans nulls ni doublons."""),
        
        nbf.v4.new_code_cell("""# Votre code ici (Analyse des manquants)
# Votre code ici (Traitement des manquants)
# Votre code ici (Suppression des doublons)"""),
        
        # PART 3: EDA
        nbf.v4.new_markdown_cell("""## 📊 Part 3: Exploratory Data Analysis
**Objectif :** Comprendre les facteurs d'annulation.
**Questions clés :**
1. Quel est le taux d'annulation global ? (Déséquilibre de classe ?)
2. Le délai de réservation (`Delai_Reservation`) influence-t-il l'annulation ?
3. Les clients "Corporate" annulent-ils moins que les "Online" ?
**Livrables :** Graphiques pertinents (Countplot, Boxplot, Barplot)."""),
        
        nbf.v4.new_code_cell("""# Votre code ici (Analyse de la cible 'Annule')
# Votre code ici (Analyse Delai_Reservation vs Annule)
# Votre code ici (Analyse Segment_Marche vs Annule)"""),
        
        nbf.v4.new_markdown_cell("""---
# 🧪 SESSION 2 : The Art of Feature Engineering (45 min)
"""),
        
        nbf.v4.new_markdown_cell("""### Étape 2.1 : Encodage des Catégories
**Objectif :** Transformer les variables textuelles en format numérique.
**Approche :** One-Hot Encoding pour `Segment_Marche`."""),
        
        nbf.v4.new_code_cell("""# Votre code ici (pd.get_dummies)"""),
        
        nbf.v4.new_markdown_cell("""### Étape 2.2 : Feature Engineering Métier (Recipe 6)
**Objectif :** Créer des variables explicatives plus puissantes.
**Suggestions :**
1. **Lead_Time_Category** : Discrétiser `Delai_Reservation` (ex: <7 jours, 7-30 jours, >30 jours).
2. **Has_Requests** : Binaire (1 si `Demandes_Speciales` > 0, sinon 0)."""),
        
        nbf.v4.new_code_cell("""# Votre code ici (Création de Lead_Time_Category)
# Votre code ici (Création de Has_Requests)"""),
        
        nbf.v4.new_markdown_cell("""---
# 🤖 SESSION 3 : Building & Trusting Your Model (45 min)
"""),
        
        nbf.v4.new_markdown_cell("""### 3.1 Split Train/Test
**Objectif :** Préparer les données pour l'entraînement.
**Conseil :** Utilisez `stratify=y` si les classes sont déséquilibrées."""),
        
        nbf.v4.new_code_cell("""# Votre code ici (train_test_split)"""),
        
        nbf.v4.new_markdown_cell("""### 3.3 Calibration et Analyse Coût-Bénéfice (CAS 3)

**Contexte Métier :**
L'objectif n'est pas seulement de classer correctement, mais d'obtenir des **probabilités fiables**
pour prendre des décisions business (ex: combien de chambres sur-réserver).

**Objectif :** Calibrer le modèle pour que `predict_proba` reflète les vraies probabilités.

**Approche recommandée :**
1. Entraîner un `RandomForestClassifier` classique.
2. Appliquer `CalibratedClassifierCV` avec méthode 'sigmoid' ou 'isotonic'.
3. Évaluer avec **ROC-AUC** et **Brier Score**.

**Livrables attendus :**
- Modèle calibré.
- Scores ROC-AUC et Brier.
- Histogramme des probabilités prédites."""),
        
        nbf.v4.new_code_cell("""# Votre code ici (Entraînement RandomForest)
# Votre code ici (Calibration avec CalibratedClassifierCV)
# Votre code ici (Prédiction des probabilités)
# Votre code ici (Évaluation AUC et Brier Score)"""),
        
        nbf.v4.new_markdown_cell("""## 🎁 Part 4: Going Further (Bonus)
"""),
        
        nbf.v4.new_markdown_cell("""### Bonus Task 1: Calculate Optimal Overbooking Limit

**Goal:** Recommend how many extra rooms the hotel can safely sell.

**Why it matters:** Hotels lose money if rooms stay empty, but overbooking too much causes customer complaints and costs.

**Approach:**
1. Get cancellation probabilities for all future bookings: `model.predict_proba()`
2. Calculate expected cancellations: `sum(probabilities) * 0.8` (80% confidence)
3. Recommend overbooking: `int(expected_cancellations)`

**Example:**
```python
# future_probabilities = model.predict_proba(X_test)[:, 1]
# expected_cancellations = future_probabilities.sum() * 0.8
# print(f"Safe to overbook by: {int(expected_cancellations)} rooms")
```"""),
        
        nbf.v4.new_code_cell("""# Votre code ici (Calcul de la limite de surréservation)"""),
        
        nbf.v4.new_markdown_cell("""### Bonus Task 2: Customer Segmentation
**Goal:** Segmenter les clients en "Fiable", "Incertain", "À Risque" selon leur probabilité d'annulation.
**Livrable :** Un graphique (Pie chart ou Bar plot) montrant la répartition des segments."""),
        
        nbf.v4.new_code_cell("""# Votre code ici (Segmentation et Visualisation)""")
    ]
    
    nb.cells = cells
    
    with open('notebook_intermediaire_projet_15.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook Intermédiaire généré avec succès !")

if __name__ == "__main__":
    create_notebook()
