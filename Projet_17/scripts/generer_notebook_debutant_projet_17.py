import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    # Titre
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 🛒 PROJET 17 : OPTIMISEUR DE STOCK PÉRISSABLE 🥬

Bienvenue dans ce projet crucial pour les supermarchés et épiceries !

**Le Problème :** Les produits frais (tomates, lait, bananes...) périssent rapidement.
- Commander TROP = ❌ Gaspillage (les produits pourrissent)
- Commander PEU = ❌ Ventes perdues (rupture de stock)

**Votre Mission :** Prédire combien d'unités vous allez vendre DEMAIN pour chaque produit. Ainsi, vous commanderez la quantité parfaite ! 🎯

---

## 📅 VOTRE PROGRAMME

### 📋 SESSION 1 : From Raw Data to Clean Insights (45 min)
- **Part 1: The Setup** - Charger les données de ventes historiques
- **Part 2: The Sanity Check** - Nettoyer les stocks manquants
- **Part 3: Exploratory Data Analysis** - Quels produits se vendent le mieux ?

### 📋 SESSION 2 : The Art of Feature Engineering (45 min)
- **Part 1: The Concept** - Transformer les dates et la météo en variables
- **Part 2: The Lab** - Créer des moyennes mobiles (tendances)
- **Part 3: Final Prep** - Préparer les données pour l'IA

### 📋 SESSION 3 : Building & Trusting Your Model (45 min)
- **Part 1: The Split** - Séparer entraînement et test
- **Part 2: Training** - Entraîner le modèle prédictif
- **Part 3: Evaluation** - Quelle est notre précision ?
- **Part 4: Going Further (BONUS)** - Calculer la quantité optimale à commander

---
"""))

    # SESSION 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : FROM RAW DATA TO CLEAN INSIGHTS
"""))

    # Part 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🏁 Part 1: The Setup (10 min)

Importons nos outils et chargeons les données de ventes.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ Librairies importées !")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 📂 Chargement du fichier stock_perissable.csv
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df = pd.read_csv('stock_perissable.csv')

print("Aperçu des données :")
display(df.head(10))

print("\\nInfos techniques :")
df.info()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
> **💡 Tip:** Le dataset contient :
> - **Date** : Jour de vente
> - **Item** : Produit (Tomato, Banana, Milk, Spinach)
> - **Stock_Initial** : Quantité en rayon au début de la journée
> - **Meteo** : Météo (Hot, Cold, Mild)
> - **Jour_Ferie** : 1 si jour férié, 0 sinon
> - **Sold** : 🎯 NOTRE CIBLE (Quantité vendue)
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🧹 Part 2: The Sanity Check (15 min)

### 1. Valeurs manquantes
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
print("Valeurs manquantes par colonne :")
print(df.isnull().sum())
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
> **⚠️ Warning:** Certaines lignes ont `Stock_Initial` ou `Meteo` manquants. Pour simplifier, on va les supprimer (car difficile d'imputer sans introduire de biais).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

print(f"✅ Nouvelles dimensions : {df.shape}")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 2. Conversion de la date
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df['Date'] = pd.to_datetime(df['Date'])
print("✅ Date convertie en format datetime !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 📊 Part 3: Exploratory Data Analysis (20 min)

### 📈 Ventes par Produit
Quel produit se vend le plus ?
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Item', y='Sold', estimator=np.mean, errorbar=None)
plt.title('Vente Moyenne par Produit')
plt.ylabel('Unités Vendues (Moyenne)')
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
❓ **Question :** Quel produit a le volume de vente le plus élevé en moyenne ?

### 🌦️ Impact de la Météo
La météo influence-t-elle les ventes ?
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Meteo', y='Sold', hue='Item', errorbar=None)
plt.title('Ventes par Météo et Produit')
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
❓ **Question :** Remarquez-vous un pattern ? (Ex: Plus de soupes vendues quand il fait froid ?)
"""))

    # SESSION 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : THE ART OF FEATURE ENGINEERING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🧠 Part 1: The Concept (10 min)

Pour prédire les ventes de demain, le modèle a besoin de "signaux" :
- **Le jour de la semaine** (les ventes sont plus élevées le week-end)
- **La saison / le mois** (les glaces se vendent mieux l'été)
- **Les tendances récentes** (si les ventes augmentent depuis 3 jours...)
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🧪 Part 2: The Lab - Choose Your Recipe (30 min)

### Recipe 1: Dates & Time 🕐
Extrayons des informations de la date.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Extraction de features temporelles
df['Jour'] = df['Date'].dt.day
df['Mois'] = df['Date'].dt.month
df['JourSemaine'] = df['Date'].dt.dayofweek  # 0=Lundi, 6=Dimanche
df['Is_Weekend'] = (df['JourSemaine'] >= 5).astype(int)  # 1 si samedi/dimanche

print("✅ Features temporelles créées !")
display(df[['Date', 'Mois', 'JourSemaine', 'Is_Weekend']].head())
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Recipe 2: Categories 🏷️
Encodons les catégories (`Item` et `Meteo`).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# One-Hot Encoding
df = pd.get_dummies(df, columns=['Item', 'Meteo'], prefix=['Item', 'Meteo'])

print("✅ Encodage terminé !")
display(df.head())
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Recipe 6: Domain-Specific Features 🎯
Créons des features métier.

#### 🔹 Moyenne Mobile (Tendance récente)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Tri par date pour calculer les moyennes mobiles
df = df.sort_values('Date').reset_index(drop=True)

# Moyenne mobile sur 7 jours (tendance de la semaine)
# TODO: Cette feature nécessite un groupby par Item, mais nos Item sont maintenant encodés
# Pour simplifier, on calcule une moyenne globale
df['Sold_MA7'] = df['Sold'].shift(1).rolling(window=7, min_periods=1).mean()

print("✅ Moyenne mobile créée !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🏁 Part 3: Final Prep (5 min)

Supprimons les colonnes inutiles et préparons le dataset final.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Colonnes à supprimer
cols_to_drop = ['Date']  # On garde tout le reste

df_model = df.drop(columns=cols_to_drop)

# Supprimer les lignes avec NaN (créées par rolling)
df_model = df_model.dropna()

print(f"✅ Dataset prêt ! Dimensions : {df_model.shape}")
"""))

    # SESSION 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : BUILDING & TRUSTING YOUR MODEL
"""))

    # Part 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
## ✂️ Part 1: The Split (10 min)

Séparons les features (X) et la cible (y).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X = df_model.drop('Sold', axis=1)
y = df_model['Sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🏋️ Part 2: Training (15 min)

Entraînons un **RandomForestRegressor**.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

print("⏳ Entraînement...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🎯 Part 3: Evaluation (20 min)

Évaluons la précision du modèle.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE (Erreur Moyenne) : {mae:.2f} unités")
print(f"📊 RMSE : {rmse:.2f}")
print(f"📊 R² Score : {r2:.3f}")

# Visualisation
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Ventes Réelles')
plt.ylabel('Ventes Prédites')
plt.title('Vérité vs Prédiction')
plt.show()
"""))

    # Part 4 Bonus
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Calculer la Quantité de Commande Optimale 📦

**Goal:** Recommander combien commander pour demain.

**Why it matters:** Si on prédit 30 ventes mais on en vend parfois 35, on aura une rupture de stock. Il faut une **marge de sécurité**.

**Approach:**
1. Prédire les ventes moyennes
2. Calculer l'écart-type (volatilité)
3. Commande Optimale = Prédiction + (1.5 × Écart-type)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# TODO: Calculer l'écart-type des erreurs de prédiction
errors = y_test - y_pred
std_error = errors.std()

print(f"Écart-type des erreurs : {std_error:.2f}")

# Exemple : Si on prédit 30 unités demain
predicted_sales = 30
safety_margin = 1.5 * std_error
optimal_order = predicted_sales + safety_margin

print(f"\\n📦 Prédiction : {predicted_sales} unités")
print(f"📦 Marge de sécurité : +{safety_margin:.2f}")
print(f"📦 Quantité à commander : {optimal_order:.0f} unités")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Identifier les Articles à Rotation Lente 🐌

**Goal:** Trouver les produits qui se vendent peu (candidats pour les soldes).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Revenir au dataset original pour cette analyse
df_original = pd.read_csv('stock_perissable.csv').dropna()

avg_sales_by_item = df_original.groupby('Item')['Sold'].mean().sort_values()

print("Vente moyenne par produit :")
print(avg_sales_by_item)

# Les produits en dessous de 30 unités/jour sont considérés lents
slow_movers = avg_sales_by_item[avg_sales_by_item < 30]
print(f"\\n🐌 Articles à rotation lente : {list(slow_movers.index)}")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Détecter les Ruptures de Stock 🚨

**Goal:** Identifier les jours où nous avons manqué de stock.

**Approach:** Si `Stock_Initial` < `Sold`, alors nous avons eu une rupture.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Détecter les ruptures (impossible de vendre plus que le stock initial)
# En réalité, si Stock_Initial est proche de Sold, c'est suspect
df_original['Rupture_Probable'] = (df_original['Stock_Initial'] <= df_original['Sold'] * 1.1).astype(int)

ruptures = df_original[df_original['Rupture_Probable'] == 1]
print(f"Nombre de ruptures probables détectées : {len(ruptures)}")

# Top des jours avec ruptures
print("\\nExemples de ruptures :")
display(ruptures[['Date', 'Item', 'Stock_Initial', 'Sold']].head(10))
"""))

    # Sauvegarde
    with open('Projet_17_Stock_Perissable_Debutant.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_debutant()
