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
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE}

## 🏁 Objectif : Vision Zéro
Votre mission est d'analyser les accidents de la route pour comprendre ce qui rend un accident grave.
Nous allons construire un modèle capable de prédire la **Gravité** (1 à 4) d'un accident en fonction de la météo, de la route et du véhicule.

---

## 📋 Programme des 3 Sessions

### 🕵️‍♀️ SESSION 1 : Enquêteur de Données (45 min)
- **Part 1 :** Chargement et Nettoyage (Attention aux coordonnées GPS !)
- **Part 2 :** Analyse Exploratoire (Où sont les accidents ?)

### 🏗️ SESSION 2 : Architecte de Features (45 min)
- **Part 1 :** Feature Engineering (Géographie & Catégories)
- **Part 2 :** Préparation finale pour l'IA

### 🤖 SESSION 3 : Entraîneur d'IA (45 min)
- **Part 1 :** Entraînement du Modèle (Classification)
- **Part 2 :** Évaluation (On ne veut rater aucun accident grave !)
- **Part 3 :** Bonus (Points noirs & Recommandations)

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights
"""))

    # Part 1: Setup
    cells.append(nbf.v4.new_markdown_cell("""
## 🛠️ Part 1: The Setup
Commençons par charger nos outils et les données.
"""))
    
    cells.append(nbf.v4.new_code_cell(f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour voir toutes les colonnes
pd.set_option('display.max_columns', None)

# Chargement des données
df = pd.read_csv('{DATASET_NAME}')

print("✅ Données chargées avec succès !")
print(f"📊 Dimensions : {{df.shape[0]}} lignes, {{df.shape[1]}} colonnes")
df.head()
"""))

    # Part 2: Sanity Check
    cells.append(nbf.v4.new_markdown_cell("""
## 🧹 Part 2: The Sanity Check
Les données réelles sont rarement parfaites. Nettoyons-les !

### 2.1 Valeurs Manquantes
"""))

    cells.append(nbf.v4.new_code_cell("""
# Vérifions les valeurs manquantes
print(df.isnull().sum())

# Visualisons les manquants
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Carte des Valeurs Manquantes")
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
> **💡 Tip:** Pour la météo manquante, nous allons utiliser le "Mode" (la valeur la plus fréquente).
"""))

    cells.append(nbf.v4.new_code_cell("""
# Remplacer les valeurs manquantes dans 'Meteo' par la valeur la plus fréquente
mode_meteo = df['Meteo'].mode()[0]
df['Meteo'].fillna(mode_meteo, inplace=True)

# Vérification
print(f"Valeurs manquantes restantes dans Meteo : {df['Meteo'].isnull().sum()}")
"""))

    # Part 3: EDA
    cells.append(nbf.v4.new_markdown_cell("""
## 🔍 Part 3: Exploratory Data Analysis (EDA)
Analysons la cible : **Gravite**.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Distribution de la Gravité
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='{TARGET_COL}', palette='Reds')
plt.title("Distribution de la Gravité des Accidents")
plt.xlabel("Niveau de Gravité (1=Léger, 4=Mortel)")
plt.ylabel("Nombre d'accidents")
plt.show()

print("❓ Question : Y a-t-il un déséquilibre de classe ? (Une gravité beaucoup plus fréquente ?)")
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
Transformons nos données brutes en informations utiles pour le modèle.
"""))

    # Recipe 5: Geography
    cells.append(nbf.v4.new_markdown_cell("""
### 🗺️ Recipe: Geography (Coordonnées GPS)
La colonne `Localisation` contient "Latitude, Longitude" sous forme de texte. L'ordinateur a besoin de deux colonnes numériques séparées.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Exemple pour la première ligne
exemple = "34.2503, -6.8078"
lat, lon = exemple.split(', ')
print(f"Latitude: {lat}, Longitude: {lon}")

# Appliquons cela à tout le dataset
# Nous utilisons une fonction lambda pour séparer la chaîne
df[['Latitude', 'Longitude']] = df['Localisation'].str.split(', ', expand=True).astype(float)

# Supprimons l'ancienne colonne
df.drop('Localisation', axis=1, inplace=True)

print("✅ Coordonnées extraites !")
df[['Latitude', 'Longitude']].head()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
> **⚠️ Warning:** Certaines coordonnées sont "0.0, 0.0". Ce sont probablement des erreurs (accidents en plein océan !).
"""))

    cells.append(nbf.v4.new_code_cell("""
# Filtrons les coordonnées invalides (0, 0)
# On considère que le Maroc est environ entre Lat 21-36 et Lon -17 à -1
mask_valid = (df['Latitude'] != 0) & (df['Longitude'] != 0)
df_clean = df[mask_valid].copy()

print(f"Lignes avant nettoyage : {len(df)}")
print(f"Lignes après suppression des GPS 0.0 : {len(df_clean)}")
"""))

    # Recipe 2: Categories
    cells.append(nbf.v4.new_markdown_cell("""
### 🏷️ Recipe: Categories
Les modèles ne comprennent pas "Pluie" ou "Camion". Ils veulent des chiffres.
Utilisons le **Label Encoding** pour les variables ordonnées (si on en avait) ou **One-Hot Encoding** pour les catégories nominales.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Encodage One-Hot pour Meteo, Type_Route, Type_Vehicule
# pd.get_dummies crée des colonnes binaires (0/1)
categorical_cols = ['Meteo', 'Type_Route', 'Type_Vehicule']
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

print("✅ Encodage terminé !")
df_encoded.head()
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🤖 SESSION 3 : Building & Trusting Your Model
C'est le moment d'entraîner notre IA !
"""))

    # Part 1: Split
    cells.append(nbf.v4.new_markdown_cell("""
## ✂️ Part 1: The Split
Séparons les données pour l'entraînement et le test.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
from sklearn.model_selection import train_test_split

X = df_encoded.drop(['ID_Accident', '{TARGET_COL}'], axis=1)
y = df_encoded['{TARGET_COL}']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {{X_train.shape}}")
print(f"Test shape: {{X_test.shape}}")
"""))

    # Part 2: Training (Imbalanced)
    cells.append(nbf.v4.new_markdown_cell("""
## 🏋️ Part 2: Training (Gestion du Déséquilibre)
Les accidents très graves (4) sont heureusement plus rares. Mais pour l'IA, c'est un problème : elle risque de les ignorer.
Nous allons utiliser **SMOTE** pour créer des exemples synthétiques des classes rares.
"""))

    cells.append(nbf.v4.new_code_cell("""
from imblearn.over_sampling import SMOTE

print("Avant SMOTE :")
print(y_train.value_counts().sort_index())

# Application de SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\\nAprès SMOTE :")
print(y_train_balanced.value_counts().sort_index())
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestClassifier

# Entraînement sur données équilibrées
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

print("✅ Modèle entraîné !")
"""))

    # Part 3: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""
## 📊 Part 3: Evaluation
Regardons si notre modèle arrive à détecter les accidents graves.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Rapport de Classification :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de Confusion")
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.show()
"""))

    # Part 4: Bonus
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### 🌍 Bonus Task 1: Identifier les "Points Noirs"
Où sont concentrés les accidents ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# Visualisation géographique simple
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_clean, x='Longitude', y='Latitude', hue='Gravite', palette='viridis', alpha=0.6)
plt.title("Carte des Accidents par Gravité")
plt.show()

# TODO: Essayez d'identifier visuellement les zones denses (clusters)
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🌧️ Bonus Task 2: Impact de la Pluie
La pluie augmente-t-elle vraiment la gravité ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# Tableau croisé : Météo vs Gravité
# Note: On utilise df_clean avant encodage pour avoir les noms de météo
crosstab = pd.crosstab(df_clean['Meteo'], df_clean['Gravite'], normalize='index')

# Visualisation
crosstab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='RdYlGn_r')
plt.title("Proportion de Gravité par Météo")
plt.ylabel("Proportion")
plt.show()

print("❓ Question : La barre rouge (Gravité 4) est-elle plus grande quand il pleut ?")
"""))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Accidents_Debutant.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
