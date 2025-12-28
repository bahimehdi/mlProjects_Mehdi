import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "10"
    PROJECT_TITLE = "Recommandation de Voyage Personnalisée"
    DATASET_NAME = "recommandation_voyage.csv"
    TARGET_COL = "Note_Destination"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE}

## 🏁 Objectif : L'Agence de Voyage IA
Les voyageurs sont perdus devant trop de choix. Votre mission est de créer une IA capable de **prédire la note** qu'un utilisateur donnera à un voyage.
Cela nous permettra ensuite de lui recommander les destinations (Style + Climat) qui lui correspondent le mieux !

---

## 📋 Programme des 3 Sessions

### 🕵️‍♀️ SESSION 1 : Enquêteur de Données (45 min)
- **Part 1 :** Chargement et Nettoyage (Le budget est parfois inconnu...)
- **Part 2 :** Analyse Exploratoire (Qu'est-ce qui coûte cher ?)

### 🏗️ SESSION 2 : Architecte de Features (45 min)
- **Part 1 :** Feature Engineering (Transformer les mots en chiffres)
- **Part 2 :** Préparation finale pour l'IA

### 🤖 SESSION 3 : Entraîneur d'IA (45 min)
- **Part 1 :** Entraînement du Modèle (Régression)
- **Part 2 :** Évaluation (À quel point nos prédictions sont-elles proches ?)
- **Part 3 :** Bonus (Le Moteur de Recommandation)

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

# Chargement des données
df = pd.read_csv('{DATASET_NAME}')

print("✅ Données chargées avec succès !")
print(f"📊 Dimensions : {{df.shape[0]}} lignes, {{df.shape[1]}} colonnes")
df.head()
"""))

    # Part 2: Sanity Check
    cells.append(nbf.v4.new_markdown_cell("""
## 🧹 Part 2: The Sanity Check
Regardons si nous avons des trous dans nos données.
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
> **💡 Tip:** Pour le `Budget_Quotidien` manquant, nous allons le remplacer par la **médiane** (valeur du milieu), car elle est moins sensible aux budgets extrêmes (très riches) que la moyenne.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Remplacer les valeurs manquantes dans 'Budget_Quotidien' par la médiane
median_budget = df['Budget_Quotidien'].median()
df['Budget_Quotidien'].fillna(median_budget, inplace=True)

# TODO: Faites la même chose pour 'Age' si nécessaire (ou vérifiez qu'il n'y a pas de manquants)
# median_age = ...
# ...

print("✅ Nettoyage terminé !")
print(df.isnull().sum())
"""))

    # Part 3: EDA
    cells.append(nbf.v4.new_markdown_cell(f"""
## 🔍 Part 3: Exploratory Data Analysis (EDA)
Analysons notre cible : **{TARGET_COL}**.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Distribution des Notes
plt.figure(figsize=(8, 5))
sns.histplot(df['{TARGET_COL}'], kde=True, color='purple')
plt.title("Distribution des Notes de Destination")
plt.xlabel("Note (1 à 5)")
plt.show()

print("❓ Question : Les gens sont-ils généralement satisfaits (notes > 3) ?")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
Regardons le lien entre **Budget** et **Note**. L'argent fait-il le bonheur en voyage ?
"""))

    cells.append(nbf.v4.new_code_cell(f"""
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Budget_Quotidien', y='{TARGET_COL}', hue='Style_Voyage')
plt.title("Note vs Budget")
plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    # Recipe 2: Categories
    cells.append(nbf.v4.new_markdown_cell("""
### 🏷️ Recipe: Categories
Les colonnes `Style_Voyage` (Adventure, Relax...) et `Climat_Prefere` (Tropical, Cold...) sont du texte.
Transformons-les en colonnes numériques avec le **One-Hot Encoding**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Encodage One-Hot
categorical_cols = ['Style_Voyage', 'Climat_Prefere']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("✅ Encodage terminé !")
df_encoded.head()
"""))

    # Recipe 4: Math Magic
    cells.append(nbf.v4.new_markdown_cell("""
### ➗ Recipe: Math Magic (Log Transformation)
Le `Budget_Quotidien` varie énormément. Certains dépensent 50€, d'autres 10000€ !
Cette grande échelle peut perturber le modèle. Utilisons le **Logarithme** pour "tasser" les grandes valeurs.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Création d'une feature Log_Budget
# On ajoute +1 pour éviter log(0) si jamais le budget est 0
df_encoded['Log_Budget'] = np.log1p(df_encoded['Budget_Quotidien'])

# Comparons les distributions
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_encoded['Budget_Quotidien'], ax=ax[0], title='Budget Original')
sns.histplot(df_encoded['Log_Budget'], ax=ax[1], title='Log Budget (Plus normal)')
plt.show()
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🤖 SESSION 3 : Building & Trusting Your Model
"""))

    # Part 1: Split
    cells.append(nbf.v4.new_markdown_cell("""
## ✂️ Part 1: The Split
"""))

    cells.append(nbf.v4.new_code_cell(f"""
from sklearn.model_selection import train_test_split

# On utilise Log_Budget au lieu de Budget_Quotidien
X = df_encoded.drop(['ID_Utilisateur', '{TARGET_COL}', 'Budget_Quotidien'], axis=1)
y = df_encoded['{TARGET_COL}']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shape: {{X_train.shape}}")
print(f"Test shape: {{X_test.shape}}")
"""))

    # Part 2: Training (Regression)
    cells.append(nbf.v4.new_markdown_cell("""
## 🏋️ Part 2: Training (Régression)
Nous voulons prédire une note (nombre continu), donc c'est une **Régression**.
Utilisons `RandomForestRegressor`.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Modèle entraîné !")
"""))

    # Part 3: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""
## 📊 Part 3: Evaluation
Quelle est l'erreur moyenne de notre modèle ? (MAE : Mean Absolute Error)
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE (Erreur Moyenne) : {mae:.2f} points")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.3f}")

# Visualisons Réalité vs Prédiction
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([1, 5], [1, 5], color='red', linestyle='--') # Ligne parfaite
plt.xlabel("Vraie Note")
plt.ylabel("Note Prédite")
plt.title("Réalité vs Prédiction")
plt.show()
"""))

    # Part 4: Bonus
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### 🌟 Bonus Task 1: Le Moteur de Recommandation
Imaginons un nouvel utilisateur. Quel voyage lui recommander ?
Nous allons tester tous les styles/climats possibles pour lui et voir ce que le modèle prédit.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Définissons un profil utilisateur
user_age = 30
user_budget = 500
user_log_budget = np.log1p(user_budget)

# Créons des scénarios de voyage (combinaisons possibles)
# Note: Ceci est une simplification. Dans la réalité, on aurait une liste de destinations.
# Ici, on teste "Style" et "Climat".
# Pour simplifier, on va juste prédire pour un cas manuel.

# TODO: Créez un DataFrame avec une ligne représentant :
# Age=30, Log_Budget=..., Style_Voyage_Adventure=1, Climat_Prefere_Tropical=1...
# Et faites model.predict()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 👥 Bonus Task 2: Tribus de Voyageurs (Clustering)
Pouvons-nous grouper les utilisateurs par budget et âge ?
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.cluster import KMeans

# On prend juste Age et Budget
X_cluster = df[['Age', 'Budget_Quotidien']].dropna()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_cluster)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Budget_Quotidien', hue='Cluster', palette='viridis')
plt.title("Tribus de Voyageurs")
plt.show()
"""))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Recommandation_Debutant.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
