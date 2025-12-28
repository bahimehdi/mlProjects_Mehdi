import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "11"
    PROJECT_TITLE = "Sous-évaluation Immobilière"
    DATASET_NAME = "immobilier.csv"
    TARGET_COL = "Price"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE}

## 🏁 Objectif : Le Chasseur de Bonnes Affaires 🏠
Tout le monde veut acheter une maison moins chère que sa vraie valeur.
Votre mission est de créer une IA capable d'estimer le **Juste Prix** d'une maison.
Si le prix affiché est inférieur à votre estimation... c'est une bonne affaire ! 💰

---

## 📋 Programme des 3 Sessions

### 🕵️‍♀️ SESSION 1 : Enquêteur de Données (45 min)
- **Part 1 :** Chargement et Nettoyage (Attention aux années de construction manquantes !)
- **Part 2 :** Analyse Exploratoire (Le prix dépend-il de la surface ?)

### 🏗️ SESSION 2 : Architecte de Features (45 min)
- **Part 1 :** Feature Engineering (L'âge de la maison compte-t-il ?)
- **Part 2 :** Préparation finale pour l'IA

### 🤖 SESSION 3 : Entraîneur d'IA (45 min)
- **Part 1 :** Entraînement du Modèle (Régression)
- **Part 2 :** Évaluation (Notre estimation est-elle fiable ?)
- **Part 3 :** Bonus (Trouver les maisons sous-évaluées)

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
Les données immobilières sont souvent incomplètes. Vérifions !
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
> **💡 Tip:** Pour l'`Annee_Construction` manquante, nous allons utiliser la **Médiane**. Pour la `Localisation`, nous utiliserons le **Mode** (la valeur la plus fréquente).
"""))

    cells.append(nbf.v4.new_code_cell("""
# Remplacer les valeurs manquantes
# 1. Annee_Construction -> Médiane
median_year = df['Annee_Construction'].median()
df['Annee_Construction'].fillna(median_year, inplace=True)

# 2. Localisation -> Mode
mode_loc = df['Localisation'].mode()[0]
df['Localisation'].fillna(mode_loc, inplace=True)

print("✅ Nettoyage terminé !")
print(df.isnull().sum())
"""))

    # Part 3: EDA
    cells.append(nbf.v4.new_markdown_cell(f"""
## 🔍 Part 3: Exploratory Data Analysis (EDA)
Analysons notre cible : **{TARGET_COL}**.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Distribution des Prix
plt.figure(figsize=(10, 5))
sns.histplot(df['{TARGET_COL}'], kde=True, color='green')
plt.title("Distribution des Prix Immobiliers")
plt.xlabel("Prix")
plt.show()

print("❓ Question : La plupart des maisons sont-elles chères ou abordables ?")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
Regardons le lien évident : **Surface** vs **Prix**. Plus c'est grand, plus c'est cher ?
"""))

    cells.append(nbf.v4.new_code_cell(f"""
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Surface_m2', y='{TARGET_COL}', hue='Localisation', alpha=0.6)
plt.title("Prix vs Surface")
plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    # Recipe 4: Math Magic (Age)
    cells.append(nbf.v4.new_markdown_cell("""
### ➗ Recipe: Math Magic (Calcul de l'Âge)
L'année de construction (ex: 1990) n'est pas très parlante. Ce qui compte, c'est l'**Âge** de la maison.
Créons une feature `Age` = Année Actuelle - Année Construction.
"""))

    cells.append(nbf.v4.new_code_cell("""
CURRENT_YEAR = 2025

# Création de la colonne Age
df['Age'] = CURRENT_YEAR - df['Annee_Construction']

# Vérifions
df[['Annee_Construction', 'Age']].head()
"""))

    # Recipe 2: Categories
    cells.append(nbf.v4.new_markdown_cell("""
### 🏷️ Recipe: Categories
La `Localisation` (Maarif, Anfa...) est cruciale pour le prix.
Transformons-la en chiffres avec le **One-Hot Encoding**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Encodage One-Hot pour Localisation
df_encoded = pd.get_dummies(df, columns=['Localisation'], drop_first=True)

print("✅ Encodage terminé !")
df_encoded.head()
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

# On retire l'ID et la cible
X = df_encoded.drop(['ID_Maison', '{TARGET_COL}'], axis=1)
y = df_encoded['{TARGET_COL}']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shape: {{X_train.shape}}")
print(f"Test shape: {{X_test.shape}}")
"""))

    # Part 2: Training (Regression)
    cells.append(nbf.v4.new_markdown_cell("""
## 🏋️ Part 2: Training (Régression)
Nous voulons prédire un Prix (nombre continu). C'est une **Régression**.
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
À quel point notre estimation est-elle proche du vrai prix ?
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erreur Moyenne (MAE) : {mae:,.0f} DH")
print(f"R² Score : {r2:.3f}")

# Visualisons
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Vrai Prix")
plt.ylabel("Prix Estimé")
plt.title("Vrai Prix vs Estimation")
plt.show()
"""))

    # Part 4: Bonus
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### 💎 Bonus Task 1: Le Détecteur de Bonnes Affaires
Une "bonne affaire", c'est quand le **Prix Affiché** est plus bas que notre **Estimation**.
Cherchons les maisons sous-évaluées de plus de 10% !
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Créons un DataFrame avec les résultats du test
results = X_test.copy()
results['Vrai_Prix'] = y_test
results['Estimation_IA'] = y_pred

# Calculons la différence en pourcentage
# Si (Vrai_Prix < Estimation), c'est une bonne affaire potentielle
results['Difference_Pct'] = (results['Estimation_IA'] - results['Vrai_Prix']) / results['Estimation_IA'] * 100

# Filtrons les "Super Bonnes Affaires" (> 10% moins cher que prévu)
bonnes_affaires = results[results['Difference_Pct'] > 10]

print(f"Nombre de bonnes affaires détectées : {{len(bonnes_affaires)}}")
bonnes_affaires[['Vrai_Prix', 'Estimation_IA', 'Difference_Pct']].head()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🏙️ Bonus Task 2: Les Quartiers les plus Rentables
Quels quartiers ont le plus de bonnes affaires ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# Pour faire ça, il faudrait récupérer le nom du quartier (qui a été encodé).
# C'est un peu complexe car on a perdu la colonne 'Localisation' originale dans X_test.
# Mais on peut regarder les colonnes 'Localisation_...' dans bonnes_affaires.

# Astuce : Regardons simplement dans le dataset original les prix moyens par quartier
mean_price_loc = df.groupby('Localisation')['Price'].mean().sort_values()
mean_price_loc.plot(kind='barh', figsize=(10, 6), color='teal')
plt.title("Prix Moyen par Quartier")
plt.show()
"""))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Immobilier_Debutant.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
