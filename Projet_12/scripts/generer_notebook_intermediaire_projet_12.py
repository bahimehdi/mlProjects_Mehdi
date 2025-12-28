import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "12"
    PROJECT_TITLE = "Détection de Fraude E-commerce"
    DATASET_NAME = "fraude_ecommerce.csv"
    TARGET_COL = "Est_Frauduleux"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE} (Version Intermédiaire)

## 🏁 Objectif : Le Cyber-Détective 🕵️‍♂️
Votre mission est de construire un modèle capable de détecter les transactions frauduleuses (`{TARGET_COL} = 1`).
Attention : La fraude est rare, mais coûteuse. Ne laissez rien passer !

---

## 📋 Programme

### 🕵️‍♀️ SESSION 1 : From Raw Data to Clean Insights
- Gestion des valeurs manquantes (Pays, Temps)
- Analyse de la relation Pays vs Fraude

### 🏗️ SESSION 2 : The Art of Feature Engineering
- **Recipe Categories :** Encodage des Pays
- **Recipe Domain :** Comparaison Pays IP vs Pays Carte

### 🤖 SESSION 3 : Building & Trusting Your Model
- Classification (RandomForestClassifier)
- **Gestion du Déséquilibre :** SMOTE
- **Bonus :** Calcul du ROI (Retour sur Investissement) du modèle

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
- `Pays_IP` et `Pays_Carte` ont des manquants -> Remplacer par "Inconnu".
- `Temps_Depuis_Derniere` -> Remplacer par la Médiane.

**Livrables attendus :**
- Un DataFrame propre sans NaN.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 1.2 : Analyse Exploratoire (EDA) 🔍

**Objectif :** Comprendre le déséquilibre de classe.

**Questions :**
- Quel est le pourcentage de fraudes ?
- Les montants frauduleux sont-ils plus élevés ?

**Livrables attendus :**
- Countplot de la cible.
- Boxplot Montant vs Fraude.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.1 : Recipe Domain (Risque Pays) 🎯

**Contexte :** Si l'IP vient d'un pays et la carte d'un autre, c'est suspect.

**Objectif :** Créer une feature binaire `Pays_Different`.

**Livrables attendus :**
- Nouvelle colonne `Pays_Different` (0 ou 1).
- Visualisation du lien avec la fraude.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Recipe Categories 🏷️

**Contexte :** Les modèles ne lisent pas le texte ("France", "USA").

**Objectif :** Encoder `Pays_IP` et `Pays_Carte`.

**Approche recommandée :** One-Hot Encoding (`pd.get_dummies`).

**Livrables attendus :**
- DataFrame 100% numérique.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🤖 SESSION 3 : Building & Trusting Your Model
"""))

    cells.append(nbf.v4.new_markdown_cell(f"""
### Étape 3.1 : Split et SMOTE 🏋️

**Objectif :** Préparer les données pour l'entraînement.

**Problème :** La fraude est rare (Classe déséquilibrée).

**Solution :** Utiliser **SMOTE** sur le train set pour équilibrer les classes.

**Livrables attendus :**
- `X_train_balanced`, `y_train_balanced` avec autant de fraudes que de non-fraudes.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Entraînement et Évaluation 📊

**Objectif :** Entraîner un `RandomForestClassifier` et maximiser le **Recall**.

**Pourquoi le Recall ?** Car rater une fraude coûte cher.

**Livrables attendus :**
- Rapport de classification.
- Matrice de confusion.
- Score de Recall.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- PART 4 : BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: Calcul du ROI (Retour sur Investissement) 💰
**Scenario :**
- Fraude détectée (Vrai Positif) = Gain du montant de la transaction.
- Fausse Alerte (Faux Positif) = Perte de 10€ (frais dossier).
- Fraude ratée (Faux Négatif) = Perte du montant.

**Objectif :** Calculez le gain total de votre modèle sur le test set.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Fraude_Intermediaire.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
