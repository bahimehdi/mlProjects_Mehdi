import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    # --- Cellules du Notebook ---
    
    cells = []
    
    # Titre et Introduction
    cells.append(nbf.v4.new_markdown_cell("""
# 😷 Projet 3 : Qualité de l'Air & Santé
## Version Débutant - "Je te montre, puis tu répètes"

---

### 🎯 L'Objectif de ce Projet

La pollution de l'air est un enjeu de santé publique majeur. Votre mission est de **prédire le nombre d'admissions à l'hôpital pour problèmes respiratoires** en fonction de la qualité de l'air et du trafic routier.

**Ce que vous allez apprendre :**
- 🧹 Nettoyer des données temporelles et gérer les valeurs manquantes
- 📅 Extraire des informations utiles à partir de dates (Feature Engineering)
- 🤖 Entraîner un modèle de **Régression** (prédire un nombre)
- 📉 Évaluer votre modèle avec des métriques adaptées (MAE, RMSE)

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
Nous allons utiliser :
- **pandas** : Pour manipuler le tableau de données
- **numpy** : Pour les calculs mathématiques
- **matplotlib/seaborn** : Pour les graphiques
"""))

    cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour de beaux graphiques
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

print("✅ Bibliothèques importées !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Étape 1.1 : Charger les Données
Le fichier s'appelle `qualite_air.csv`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Charger le dataset
df = pd.read_csv('qualite_air.csv')

# Afficher les premières lignes
print("📊 Aperçu des données :")
display(df.head())

print(f"\\n✅ Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Sanity Check (15 min)

### 📘 Theory: Valeurs Manquantes
Les capteurs de pollution tombent parfois en panne, créant des "trous" dans les données (NaN).
Nous devons les détecter et les remplir (imputation).
"""))

    cells.append(nbf.v4.new_code_cell("""
# Vérifier les valeurs manquantes
print("🔍 Valeurs manquantes par colonne :")
print(df.isnull().sum())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Exemple : Remplir les manquants pour PM2_5
Pour une valeur numérique continue comme `PM2_5` (particules fines), une bonne stratégie est de remplacer les trous par la **médiane** (valeur du milieu), car elle est moins sensible aux valeurs extrêmes que la moyenne.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Remplir PM2_5 avec la médiane
mediane_pm25 = df['PM2_5'].median()
df['PM2_5'].fillna(mediane_pm25, inplace=True)

print(f"✅ PM2_5 rempli avec la médiane : {mediane_pm25}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Faites la même chose pour la colonne `NO2` (Dioxyde d'Azote).
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Remplir les valeurs manquantes pour NO2 avec la médiane

# 1. Calculer la médiane
# mediane_no2 = df['NO2'].median()

# 2. Remplir les NaN
# df['NO2'].fillna(mediane_no2, inplace=True)

# print(f"✅ NO2 rempli avec la médiane : {mediane_no2}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📘 Theory: Conversion de Date
La colonne `Date` est lue comme du texte. Pour l'analyser, nous devons la convertir en format **datetime**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Convertir la colonne Date
df['Date'] = pd.to_datetime(df['Date'])

print("✅ Colonne Date convertie !")
print(df.dtypes)
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (20 min)

### 📊 Visualisation 1 : Évolution de la Pollution
Regardons comment les PM2.5 évoluent dans le temps.
"""))

    cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(12, 5))
sns.lineplot(data=df, x='Date', y='PM2_5', color='orange')
plt.title('📈 Évolution des Particules Fines (PM2.5) dans le temps')
plt.xlabel('Date')
plt.ylabel('Concentration PM2.5')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ❓ Question
Voyez-vous des pics de pollution ? À quoi pourraient-ils correspondre (saison, événement) ?
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez un **Scatter Plot** (nuage de points) pour voir la relation entre `Volume_Trafic` (axe X) et `NO2` (axe Y).
Le trafic routier augmente-t-il la pollution au NO2 ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer un scatter plot Trafic vs NO2

# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=df, x='Volume_Trafic', y='NO2', alpha=0.5)
# plt.title('🚗 Relation Trafic vs NO2')
# plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : The Art of Feature Engineering (45 min)

## Part 1: The Concept (10 min)
Les modèles de Machine Learning ne "comprennent" pas les dates comme nous.
"2022-01-01" ne veut rien dire pour eux.
Mais "Mois = 1" (Hiver) ou "Jour = Samedi" (Week-end), ça c'est utile !

## Part 2: The Lab - Choose Your Recipe (30 min)

### 🕐 Recipe 1: Dates & Time
Nous allons extraire des informations de la colonne `Date`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Extraire le mois et le jour de la semaine
df['Mois'] = df['Date'].dt.month
df['Jour_Semaine'] = df['Date'].dt.dayofweek  # 0=Lundi, 6=Dimanche

print("✅ Nouvelles features temporelles créées !")
display(df[['Date', 'Mois', 'Jour_Semaine']].head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez une feature binaire `Est_Weekend` :
- 1 si `Jour_Semaine` est 5 ou 6 (Samedi/Dimanche)
- 0 sinon
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer la feature Est_Weekend

# df['Est_Weekend'] = df['Jour_Semaine'].apply(lambda x: 1 if x >= 5 else 0)

# print("✅ Feature Est_Weekend créée !")
# display(df[['Date', 'Jour_Semaine', 'Est_Weekend']].head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🏷️ Recipe 2: Categories
La colonne `Direction_Vent` contient du texte (N, S, E, W).
Les modèles préfèrent les nombres. Nous allons utiliser le **One-Hot Encoding**.
Cela va créer une colonne par direction (ex: `Direction_Vent_N`, `Direction_Vent_S`, etc.).
"""))

    cells.append(nbf.v4.new_code_cell("""
# One-Hot Encoding pour la direction du vent
df = pd.get_dummies(df, columns=['Direction_Vent'], prefix='Vent')

print("✅ Encodage effectué !")
display(df.head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ➗ Recipe 4: Math Magic
Parfois, combiner deux variables donne plus d'information.
Imaginons un "Indice de Pollution Composite" qui combine PM2.5 et NO2.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Créer une feature combinée
df['Pollution_Composite'] = df['PM2_5'] + df['NO2']

print("✅ Feature Pollution_Composite créée !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)
Nous devons supprimer la colonne `Date` originale car le modèle ne sait pas la lire (nous avons déjà extrait le mois et le jour).
"""))

    cells.append(nbf.v4.new_code_cell("""
# Supprimer la colonne Date
df_model = df.drop(columns=['Date'])

# Définir X (features) et y (cible)
X = df_model.drop(columns=['Admissions_Respiratoires'])
y = df_model['Admissions_Respiratoires']

print(f"✅ Prêt pour le modèle ! X shape: {X.shape}, y shape: {y.shape}")
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : Building & Trusting Your Model (45 min)

## Part 1: The Split (10 min)
Nous divisons nos données en deux :
- **Train (80%)** : Pour que le modèle apprenne
- **Test (20%)** : Pour vérifier s'il a bien appris (examen final)
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Train set : {X_train.shape[0]} lignes")
print(f"✅ Test set  : {X_test.shape[0]} lignes")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training (15 min)
Nous allons utiliser un **RandomForestRegressor**.
C'est un modèle puissant qui utilise une multitude d'arbres de décision pour faire une moyenne.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestRegressor

# Créer le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner le modèle
print("🚀 Entraînement en cours...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation (20 min)
Pour une **Régression**, nous utilisons ces métriques :
- **MAE (Mean Absolute Error)** : L'erreur moyenne (en nombre d'admissions).
- **RMSE (Root Mean Squared Error)** : Pénalise plus les grosses erreurs.
- **R² Score** : À quel point notre modèle explique bien les variations (proche de 1 = parfait).
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Faire des prédictions
y_pred = model.predict(X_test)

# Calculer les métriques
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE  : {mae:.2f} admissions")
print(f"📊 RMSE : {rmse:.2f} admissions")
print(f"📊 R²   : {r2:.3f}")

if r2 > 0.7:
    print("🌟 Excellent modèle !")
elif r2 > 0.5:
    print("👍 Bon modèle")
else:
    print("⚠️ Le modèle peut être amélioré")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📊 Visualisation : Réel vs Prédit
"""))

    cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Vraies Admissions')
plt.ylabel('Admissions Prédites')
plt.title('🎯 Précision du Modèle')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Classifier les Jours "Sain" vs "Dangereux"
**Goal:** Créer une nouvelle catégorie basée sur les PM2.5.
**Seuil :** Si PM2.5 > 100, c'est "Dangereux", sinon "Sain".
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer la colonne 'Qualite_Air'
# df['Qualite_Air'] = df['PM2_5'].apply(lambda x: 'Dangereux' if x > 100 else 'Sain')
# print(df['Qualite_Air'].value_counts())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Le Jour le Plus Propre
**Goal:** Quel jour de la semaine a le moins de pollution (PM2.5) en moyenne ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Groupby Jour_Semaine et moyenne de PM2_5
# pollution_par_jour = df.groupby('Jour_Semaine')['PM2_5'].mean().sort_values()
# print("📅 Pollution moyenne par jour (0=Lundi) :")
# print(pollution_par_jour)
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Prévoir pour Demain (Lag Features)
**Goal:** Utiliser la pollution d'aujourd'hui pour prédire celle de demain.
C'est une technique avancée de Time Series !
"""))

    cells.append(nbf.v4.new_code_cell("""
# Exemple simple de création de lag (décalage)
# df['PM2_5_Hier'] = df['PM2_5'].shift(1)
# print(df[['Date', 'PM2_5', 'PM2_5_Hier']].head())
"""))

    # Assign cells to notebook
    nb['cells'] = cells

    # Sauvegarde
    nbf.write(nb, 'donnees_fr/Projet_03/Projet_03_Debutant.ipynb')

if __name__ == "__main__":
    generer_notebook_debutant()
