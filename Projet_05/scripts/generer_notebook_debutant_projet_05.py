import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --- Titre et Introduction ---
    cells.append(nbf.v4.new_markdown_cell("""
# 💸 Projet 5 : Score de Risque Micro-Crédit
## Version Débutant - "Je te montre, puis tu répètes"

---

### 🎯 L'Objectif de ce Projet

Les institutions de micro-crédit prêtent aux personnes non-bancarisées pour lancer de petites entreprises. Votre mission est de **prédire le risque de défaut** (`Defaillant = 1`) et de créer un **Score de Crédit** pour ces entrepreneurs.

**Ce que vous allez apprendre :**
- 💰 Analyser des données de prêt alternatives (usage mobile, type d'entreprise)
- 📊 Prédire la **Probabilité de Défaut** (pas juste Oui/Non, mais un score de 0 à 1)
- 🎯 Convertir une probabilité en **Score de Crédit** (300-850, comme les banques)
- 📉 Évaluer avec **ROC-AUC** (capacité de discrimination)

---

> **💡 Comment utiliser ce notebook :**
> 1. **Les cellules avec du code complet** → Lisez et exécutez-les pour voir l'exemple
> 2. **Les cellules avec # TODO** → C'est votre tour ! Répétez la technique
> 3. **Les Questions ❓** → Réfléchissez avant de passer à la suite

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights (45 min)

## Part 1: The Setup (10 min)

### 📘 Theory: Les Bibliothèques
"""))

    cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

print("✅ Bibliothèques importées !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Étape 1.1 : Charger les Données
Le fichier est `micro_credit.csv`.
"""))

    cells.append(nbf.v4.new_code_cell("""
df = pd.read_csv('micro_credit.csv')

print("📊 Aperçu des données :")
display(df.head())
print(f"\\n✅ Dimensions : {df.shape[0]} demandes, {df.shape[1]} colonnes")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Sanity Check (15 min)

### 📘 Theory: Valeurs Manquantes
Vérifions la qualité des données.
"""))

    cells.append(nbf.v4.new_code_cell("""
print("🔍 Valeurs manquantes :")
print(df.isnull().sum())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Exemple : Remplir 'Annees_Activite'
Pour les années d'activité (numérique), on utilise la **médiane**.
"""))

    cells.append(nbf.v4.new_code_cell("""
if df['Annees_Activite'].isnull().sum() > 0:
    mediane_annees = df['Annees_Activite'].median()
    df['Annees_Activite'].fillna(mediane_annees, inplace=True)
    print(f"✅ Annees_Activite rempli avec : {mediane_annees}")
else:
    print("✅ Pas de valeurs manquantes pour Annees_Activite")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (20 min)

### 📊 Visualisation 1 : Risque par Montant
Les gros prêts sont-ils plus risqués ?
"""))

    cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Defaillant', y='Montant_Pret', palette=['green', 'red'])
plt.title('📊 Montant du Prêt vs Défaut')
plt.xlabel('Défaut (0=Non, 1=Oui)')
plt.ylabel('Montant du Prêt (MAD)')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez un **Barplot** pour comparer le taux de défaut par `Type_Entreprise` (Retail, Agri, Service).
Quel secteur est le plus risqué ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Barplot Type_Entreprise vs Defaillant

# defaut_par_type = df.groupby('Type_Entreprise')['Defaillant'].mean() * 100
# defaut_par_type.plot(kind='bar', color='coral')
# plt.title('Taux de Défaut par Secteur (%)')
# plt.ylabel('% Défaut')
# plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : The Art of Feature Engineering (45 min)

## Part 1: The Concept (10 min)
Transformons les données pour créer des indicateurs de risque.

## Part 2: The Lab - Choose Your Recipe (30 min)

### 🏷️ Recipe 2: Categories
`Type_Entreprise` est catégoriel. Encodons-le.
"""))

    cells.append(nbf.v4.new_code_cell("""
df = pd.get_dummies(df, columns=['Type_Entreprise'], prefix='Secteur')
print("✅ Encodage terminé !")
display(df.head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ➗ Recipe 4: Math Magic
Créons un ratio **Prêt / Usage Mobile** (proxy pour "dette vs revenus").
Si quelqu'un demande 50,000 MAD mais a une facture mobile de 20 MAD, c'est suspect !
"""))

    cells.append(nbf.v4.new_code_cell("""
# Éviter la division par zéro
df['Ratio_Pret_Mobile'] = df['Montant_Pret'] / (df['Usage_Mobile'] + 1)

print("✅ Feature Ratio_Pret_Mobile créée !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez une feature binaire `Nouveau_Business` :
- 1 si `Annees_Activite` < 2 (moins de 2 ans d'existence)
- 0 sinon
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer Nouveau_Business

# df['Nouveau_Business'] = (df['Annees_Activite'] < 2).astype(int)
# print("✅ Feature Nouveau_Business créée !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)
Préparons X et y.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Supprimer l'ID
if 'ID_Demandeur' in df.columns:
    df = df.drop(columns=['ID_Demandeur'])

X = df.drop(columns=['Defaillant'])
y = df['Defaillant']

print(f"✅ Prêt ! X shape: {X.shape}, y shape: {y.shape}")
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : Building & Trusting Your Model (45 min)

## Part 1: The Split (10 min)
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Split effectué !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training (15 min)

### 📘 Theory: RandomForest pour le Score de Crédit
Nous allons utiliser `RandomForestClassifier` qui peut nous donner des **probabilités** (via `predict_proba`).
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

print("🚀 Entraînement...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation (20 min)

### 🎯 Métrique Clé : ROC-AUC
**ROC-AUC** mesure la capacité du modèle à **discriminer** entre bons et mauvais payeurs.
- 0.5 = Hasard
- 1.0 = Parfait
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Prédire les probabilités
y_proba = model.predict_proba(X_test)[:, 1]  # Colonne 1 = Probabilité de Défaut

# Calculer ROC-AUC
auc = roc_auc_score(y_test, y_proba)
print(f"📊 ROC-AUC Score : {auc:.3f}")

if auc > 0.75:
    print("🌟 Excellent pouvoir discriminant !")
elif auc > 0.65:
    print("👍 Bon modèle")
else:
    print("⚠️ Le modèle peut être amélioré")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Affichez la **Matrice de Confusion** pour voir les erreurs.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Matrice de Confusion

# y_pred = model.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Prédit')
# plt.ylabel('Réel')
# plt.title('Matrice de Confusion')
# plt.show()
"""))

    # --- PART 4 BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Créer un Score de Crédit (300-850)
**Goal:** Transformer la probabilité (0-1) en un score bancaire classique.
**Formule :** `Score = 850 - (Proba_Defaut * 550)`
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer des scores de crédit pour les 10 premiers
# scores = 850 - (y_proba[:10] * 550)
# resultats = pd.DataFrame({
#     'Probabilite_Defaut': y_proba[:10],
#     'Score_Credit': scores.astype(int)
# })
# print(resultats)
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Montant de Prêt Sûr
**Goal:** Pour chaque demandeur, calculer le montant maximum "sûr".
**Logique Simple :**
- Si Score > 700 : Montant demandé OK
- Si Score 600-700 : 70% du montant
- Si Score < 600 : 50% du montant
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Calculer montant sûr
# def montant_sur(row):
#     if row['Score_Credit'] > 700:
#         return row['Montant_Pret']
#     elif row['Score_Credit'] > 600:
#         return row['Montant_Pret'] * 0.7
#     else:
#         return row['Montant_Pret'] * 0.5
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Feature Importance
**Goal:** Quels facteurs influencent le plus le risque ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Plot Feature Importance
# importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# importances.head(5).plot(kind='barh', color='teal')
# plt.title('Top 5 Facteurs de Risque')
# plt.xlabel('Importance')
# plt.show()
"""))

    nb['cells'] = cells
    nbf.write(nb, 'donnees_fr/Projet_05/Projet_05_Debutant.ipynb')

if __name__ == "__main__":
    generer_notebook_debutant()
