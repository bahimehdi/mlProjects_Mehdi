import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --- Titre et Introduction ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🥗 Projet 7 : Prévision du Gaspillage Alimentaire
## Version Débutant - "Je te montre, puis tu répètes"

---

### 🎯 L'Objectif de ce Projet

Le gaspillage alimentaire coûte cher aux supermarchés et nuit à l'environnement. Votre mission est de **prédire les unités vendues** pour optimiser les commandes et réduire les pertes.

**Ce que vous allez apprendre :**
- 📊 Analyser des données temporelles (dates, saisonnalité)
- 🧮 Créer des features à partir de dates (jour, mois, jour de la semaine)
- 📉 Modéliser un problème de **Régression** (prédire une quantité)
- 🎯 Évaluer avec MAE et R² Score

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
from datetime import datetime

# Configuration
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

print("✅ Bibliothèques importées !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Étape 1.1 : Charger les Données
Le fichier est `gaspillage_alimentaire.csv`.
"""))

    cells.append(nbf.v4.new_code_cell("""
df = pd.read_csv('gaspillage_alimentaire.csv')

print("📊 Aperçu des données :")
display(df.head())
print(f"\\n✅ Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"\\n📋 Colonnes : {list(df.columns)}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Sanity Check (15 min)

### 📘 Theory: Valeurs Manquantes
Vérifions si nous avons des trous dans nos données.
"""))

    cells.append(nbf.v4.new_code_cell("""
print("🔍 Valeurs manquantes par colonne :")
print(df.isnull().sum())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Exemple : Remplir les Prix Manquants
Pour le prix, nous allons utiliser la **médiane par produit**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Remplir Price par la médiane de chaque produit
df['Price'] = df.groupby('ID_Produit')['Price'].transform(lambda x: x.fillna(x.median()))

print("✅ Prix remplis avec la médiane par produit")
print(f"Valeurs manquantes restantes dans Price : {df['Price'].isnull().sum()}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Remplissez les valeurs manquantes de `Discount` avec la **médiane globale**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Remplir Discount avec la médiane

# mediane_discount = df['Discount'].median()
# df['Discount'].fillna(mediane_discount, inplace=True)
# print(f"✅ Discount rempli avec : {mediane_discount}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (20 min)

### 📊 Visualisation 1 : Distribution des Unités Vendues
Notre variable cible !
"""))

    cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 5))
sns.histplot(df['Unites_Vendues'], bins=50, kde=True, color='green')
plt.title('📈 Distribution des Unités Vendues')
plt.xlabel('Unités Vendues')
plt.ylabel('Fréquence')
plt.show()

print(f"Moyenne : {df['Unites_Vendues'].mean():.2f}")
print(f"Médiane : {df['Unites_Vendues'].median():.2f}")
print(f"Min : {df['Unites_Vendues'].min()}, Max : {df['Unites_Vendues'].max()}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📊 Visualisation 2 : Ventes par Produit
Quels produits se vendent le plus ?
"""))

    cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='ID_Produit', y='Unites_Vendues', palette='Set2')
plt.title('📦 Ventes par Type de Produit')
plt.ylabel('Unités Vendues')
plt.xticks(rotation=45)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ❓ Question
Quel produit a les ventes les plus variables ? Le moins prévisible ?
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Visualisez l'impact du `Discount` sur les ventes avec un **Scatterplot**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Scatterplot Discount vs Unites_Vendues

# plt.figure(figsize=(10, 5))
# sns.scatterplot(data=df, x='Discount', y='Unites_Vendues', alpha=0.5)
# plt.title('💰 Impact des Réductions sur les Ventes')
# plt.xlabel('Taux de Réduction')
# plt.ylabel('Unités Vendues')
# plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : The Art of Feature Engineering (45 min)

## Part 1: The Concept (10 min)

Les ventes alimentaires dépendent du **temps** (jour de la semaine, saison) et des **promotions**. Transformons ces informations en features numériques !

## Part 2: The Lab - Choose Your Recipe (30 min)

### 📅 Recipe 1: Time-Based Features

#### 📘 Theory: Features Temporelles
À partir de la colonne `Date`, nous pouvons extraire :
- Le jour de la semaine (Lundi = 0, Dimanche = 6)
- Le mois de l'année (1-12)
- Est-ce un weekend ? (Samedi/Dimanche)
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Exemple : Extraire des Features de Date
"""))

    cells.append(nbf.v4.new_code_cell("""
# Convertir Date en datetime
df['Date'] = pd.to_datetime(df['Date'])

# Feature 1: Jour de la semaine (0=Lundi, 6=Dimanche)
df['Jour_Semaine'] = df['Date'].dt.dayofweek

# Feature 2: Mois
df['Mois'] = df['Date'].dt.month

# Feature 3: Est weekend (1=oui, 0=non)
df['Est_Weekend'] = (df['Jour_Semaine'] >= 5).astype(int)

print("✅ Features temporelles créées !")
display(df[['Date', 'Jour_Semaine', 'Mois', 'Est_Weekend']].head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez une feature `Jours_Avant_Expiration` = nombre de jours entre `Date` et `Date_Expiration`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer Jours_Avant_Expiration

# df['Date_Expiration'] = pd.to_datetime(df['Date_Expiration'])
# df['Jours_Avant_Expiration'] = (df['Date_Expiration'] - df['Date']).dt.days
# print("✅ Feature Jours_Avant_Expiration créée !")
# display(df[['Date', 'Date_Expiration', 'Jours_Avant_Expiration']].head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🏷️ Recipe 2: Categories (One-Hot Encoding)

Le `ID_Produit` est catégoriel (Bread, Milk, etc.). Encodons-le !
"""))

    cells.append(nbf.v4.new_code_cell("""
# One-Hot Encoding de ID_Produit
df = pd.get_dummies(df, columns=['ID_Produit'], prefix='Produit')

print("✅ Encodage terminé !")
print(f"Nouvelles colonnes créées : {[col for col in df.columns if 'Produit_' in col]}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ➗ Recipe 4: Math Magic

#### 📘 Theory: Interaction Features
Parfois, combiner deux features donne plus d'info. Par exemple :
- **Prix Effectif** = Price × (1 - Discount)
"""))

    cells.append(nbf.v4.new_code_cell("""
# Calculer le prix après réduction
df['Prix_Effectif'] = df['Price'] * (1 - df['Discount'])

print("✅ Feature Prix_Effectif créée !")
display(df[['Price', 'Discount', 'Prix_Effectif']].head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez une feature binaire `Promo_Forte` (1 si Discount > 0.3, sinon 0).
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer Promo_Forte

# df['Promo_Forte'] = (df['Discount'] > 0.3).astype(int)
# print("✅ Feature Promo_Forte créée !")
# print(df['Promo_Forte'].value_counts())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)

Préparons X et y pour le modèle.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Sélectionner features numériques (retirer Date, Date_Expiration)
features_to_drop = ['Date', 'Date_Expiration', 'Unites_Vendues']
features_to_drop = [col for col in features_to_drop if col in df.columns]

X = df.drop(columns=features_to_drop)
y = df['Unites_Vendues']

print(f"✅ Prêt ! X shape: {X.shape}, y shape: {y.shape}")
print(f"Features utilisées : {list(X.columns)}")
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : Building & Trusting Your Model (45 min)

## Part 1: The Split (10 min)

Divisons nos données en Train (80%) et Test (20%).
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Train set : {X_train.shape[0]} lignes")
print(f"✅ Test set  : {X_test.shape[0]} lignes")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training (15 min)

Nous allons utiliser un **RandomForestRegressor** (régression, pas classification).
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestRegressor

# Créer le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraîner
print("🚀 Entraînement en cours...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation (20 min)

### 📘 Theory: Métriques de Régression
Pour la **Régression**, nous utilisons :
- **MAE (Mean Absolute Error)** : Erreur moyenne en unités (facile à comprendre)
- **RMSE (Root Mean Squared Error)** : Pénalise plus les grosses erreurs
- **R² Score** : Pourcentage de variance expliquée (0-1, plus proche de 1 = mieux)

> **💡 Tip:** Pour le gaspillage alimentaire, le MAE est plus parlant : "On se trompe de X unités en moyenne".
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prédictions
y_pred = model.predict(X_test)

# Métriques
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE  : {mae:.2f} unités")
print(f"📊 RMSE : {rmse:.2f} unités")
print(f"📊 R²   : {r2:.3f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📊 Visualisation : Prédictions vs Réel
"""))

    cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Ventes Réelles')
plt.ylabel('Ventes Prédites')
plt.title('🎯 Qualité des Prédictions')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Affichez les **Features Importantes** pour voir ce qui influence le plus les ventes.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Plot Feature Importance

# importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
# plt.figure(figsize=(10, 6))
# importances.plot(kind='barh', color='teal')
# plt.title('🔑 Top 10 Features Importantes')
# plt.xlabel('Importance')
# plt.show()
"""))

    # --- PART 4 BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Analyse Temporelle

**Goal:** Visualiser les ventes au fil du temps pour détecter des tendances saisonnières.

**Approach:**
1. Grouper les ventes par `Date`
2. Créer une courbe temporelle
3. Identifier les pics (fêtes, weekends)
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Analyse temporelle

# ventes_par_jour = df.groupby('Date')['Unites_Vendues'].sum().reset_index()
# plt.figure(figsize=(14, 6))
# plt.plot(ventes_par_jour['Date'], ventes_par_jour['Unites_Vendues'], linewidth=2)
# plt.title('📆 Évolution des Ventes dans le Temps')
# plt.xlabel('Date')
# plt.ylabel('Unités Vendues Totales')
# plt.xticks(rotation=45)
# plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Optimisation du Stock

**Goal:** Calculer le stock optimal pour minimiser le gaspillage.

**Approach:**
1. Pour chaque produit, prédire les ventes moyennes
2. Ajouter une marge de sécurité (ex: +10%)
3. Recommandation de commande
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Recommandations de stock

# # Prédire sur toutes les données (ou un nouveau mois)
# predictions = model.predict(X)
# df['Ventes_Predites'] = predictions

# # Stock recommandé par produit
# stock_recommande = df.groupby('ID_Produit')['Ventes_Predites'].mean() * 1.1  # +10% marge
# print("🛒 Stock Recommandé par Produit :")
# print(stock_recommande)
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Détection d'Anomalies

**Goal:** Identifier les jours avec des ventes inhabituellement hautes ou basses.

**Approach:**
1. Calculer la moyenne et l'écart-type des ventes
2. Marquer les anomalies (ventes > moyenne + 2×écart-type OU < moyenne - 2×écart-type)
3. Investiguer ces jours
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Détection d'anomalies

# moyenne = df['Unites_Vendues'].mean()
# ecart_type = df['Unites_Vendues'].std()

# df['Anomalie'] = ((df['Unites_Vendues'] > moyenne + 2*ecart_type) | 
#                   (df['Unites_Vendues'] < moyenne - 2*ecart_type))

# print(f"🚨 Nombre d'anomalies détectées : {df['Anomalie'].sum()}")
# display(df[df['Anomalie']][['Date', 'ID_Produit', 'Unites_Vendues', 'Discount']])
"""))

    nb['cells'] = cells
    
    with open('Projet_07_Debutant.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Notebook Débutant généré : Projet_07_Debutant.ipynb")

if __name__ == "__main__":
    generer_notebook_debutant()
