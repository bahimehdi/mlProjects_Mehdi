import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = 14
    PROJECT_TITLE = "Juste Valeur de Voiture d'Occasion"
    DATASET_NAME = "voitures_occasion.csv"
    TARGET_VARIABLE = "Price"
    
    # --- CELLULES ---
    cells = []
    
    # HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE}

Bienvenue dans ce projet de Data Science ! Nous allons construire un modèle pour estimer le **juste prix** d'une voiture d'occasion.

**Objectifs :**
1.  Nettoyer et explorer les données de voitures.
2.  Créer des fonctionnalités (features) intelligentes (âge de la voiture, kilométrage annuel).
3.  Entraîner une Intelligence Artificielle pour prédire le prix.
4.  **BONUS :** Détecter les "Bonnes Affaires" !

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights (45 min)

Dans cette session, nous allons préparer nos données pour l'analyse.
"""))

    # Part 1: Setup
    cells.append(nbf.v4.new_markdown_cell("""
## 1.1 The Setup 🛠️
Importons les outils nécessaires et chargeons les données.
"""))
    
    cells.append(nbf.v4.new_code_cell(f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv("{DATASET_NAME}")

# Premier aperçu
print("📊 Aperçu des données :")
display(df.head())
print(f"\\n📏 Dimensions : {{df.shape}}")
"""))

    # Part 2: Sanity Check
    cells.append(nbf.v4.new_markdown_cell("""
## 1.2 The Sanity Check 🩺
Vérifions la qualité de nos données. Y a-t-il des valeurs manquantes ou bizarres ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# Vérification des valeurs manquantes
print("🔍 Valeurs manquantes :")
print(df.isnull().sum())

# Vérification des doublons
duplicates = df.duplicated().sum()
print(f"\\n👯 Doublons trouvés : {duplicates}")

# Suppression des doublons si nécessaire
if duplicates > 0:
    df = df.drop_duplicates()
    print("✅ Doublons supprimés !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
> **💡 Tip:** Les doublons peuvent fausser nos statistiques. Il est toujours prudent de les retirer.
"""))

    # Part 3: EDA
    cells.append(nbf.v4.new_markdown_cell("""
## 1.3 Exploratory Data Analysis (EDA) 🕵️‍♀️
Comprenons nos données avec des graphiques.

### 📊 Distribution des Prix
Quel est le prix typique d'une voiture ?
"""))

    cells.append(nbf.v4.new_code_cell(f"""
plt.figure(figsize=(10, 6))
sns.histplot(df['{TARGET_VARIABLE}'], kde=True, color='blue')
plt.title('Distribution des Prix')
plt.xlabel('Prix (€)')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Analysez la relation entre le **Kilométrage** et le **Prix**.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# TODO: Créez un scatter plot (nuage de points)
# x = Kilometrage, y = {TARGET_VARIABLE}

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Kilometrage', y='{TARGET_VARIABLE}', alpha=0.6)
plt.title('Prix vs Kilométrage')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
❓ **Question :** Que remarquez-vous ? Plus le kilométrage est élevé, que fait le prix ?
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
---
# 📋 SESSION 2 : The Art of Feature Engineering (45 min)

Nous allons transformer nos données brutes en informations utiles pour l'IA.
"""))

    # Recipe 1: Dates (Age)
    cells.append(nbf.v4.new_markdown_cell("""
## 2.1 Recipe 1: Dates & Time 🕐
La colonne `Year` est utile, mais l'**Âge** de la voiture est plus parlant pour un modèle.
"""))

    cells.append(nbf.v4.new_code_cell("""
import datetime

current_year = datetime.datetime.now().year

# Création de la feature 'Age'
df['Age'] = current_year - df['Year']

print("✅ Colonne 'Age' créée :")
display(df[['Year', 'Age']].head())
"""))

    # Recipe 4: Math Magic (Mileage per Year)
    cells.append(nbf.v4.new_markdown_cell("""
## 2.2 Recipe 4: Math Magic ➗
Une voiture qui a beaucoup roulé en peu de temps est peut-être plus usée. Créons `Km_par_an`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créez la colonne 'Km_par_an'
# Attention à la division par zéro ! Si Age = 0, on peut mettre 1 ou laisser le kilométrage tel quel.

df['Km_par_an'] = df['Kilometrage'] / df['Age'].replace(0, 1)

print("✅ Colonne 'Km_par_an' créée :")
display(df[['Kilometrage', 'Age', 'Km_par_an']].head())
"""))

    # Recipe 2: Categories (Encoding)
    cells.append(nbf.v4.new_markdown_cell("""
## 2.3 Recipe 2: Categories 🏷️
L'ordinateur ne comprend pas "Diesel" ou "BMW". Transformons ces textes en nombres.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Encodage One-Hot pour 'Fuel' et 'Brand'
df_encoded = pd.get_dummies(df, columns=['Fuel', 'Brand'], drop_first=True)

print("✅ Encodage terminé !")
display(df_encoded.head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
> **⚠️ Warning:** `drop_first=True` permet d'éviter la redondance (ex: si ce n'est pas Diesel, c'est Essence).
"""))

    # Final Prep
    cells.append(nbf.v4.new_markdown_cell("""
## 2.4 Final Prep 🏁
Préparons nos variables X (features) et y (target).
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Suppression des colonnes inutiles pour le modèle (ex: ID_Voiture si elle existe)
if 'ID_Voiture' in df_encoded.columns:
    df_encoded = df_encoded.drop('ID_Voiture', axis=1)

X = df_encoded.drop('{TARGET_VARIABLE}', axis=1)
y = df_encoded['{TARGET_VARIABLE}']

print("✅ Données prêtes :")
print(f"X shape: {{X.shape}}")
print(f"y shape: {{y.shape}}")
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
---
# 📋 SESSION 3 : Building & Trusting Your Model (45 min)

C'est le moment d'entraîner notre IA !
"""))

    # Part 1: Split
    cells.append(nbf.v4.new_markdown_cell("""
## 3.1 The Split ✂️
Séparons les données : 80% pour apprendre, 20% pour tester.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Split terminé !")
print(f"Train set: {{X_train.shape}}")
print(f"Test set: {{X_test.shape}}")
"""))

    # Part 2: Training (Regression)
    cells.append(nbf.v4.new_markdown_cell("""
## 3.2 Training 🏋️‍♂️
Nous allons utiliser un **Random Forest Regressor**. C'est un modèle puissant composé de plusieurs arbres de décision.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestRegressor

# Initialisation du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entraînement
print("⏳ Entraînement en cours...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")
"""))

    # Part 3: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""
## 3.3 Evaluation 📝
Est-ce que notre modèle prédit bien les prix ?
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Prédictions
y_pred = model.predict(X_test)

# Métriques
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📉 MAE (Erreur Moyenne) : {mae:.2f} €")
print(f"📉 RMSE : {rmse:.2f} €")
print(f"📈 R² Score (Précision) : {r2:.2%}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
> **💡 Tip:** Le **R²** indique à quel point notre modèle explique les variations de prix. Plus il est proche de 100%, mieux c'est !
"""))

    cells.append(nbf.v4.new_code_cell("""
# Visualisation : Réalité vs Prédiction
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Ligne parfaite
plt.xlabel('Prix Réel')
plt.ylabel('Prix Prédit')
plt.title('Réalité vs Prédiction')
plt.show()
"""))

    # --- PART 4: BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
---
# 🎁 Part 4: Going Further (Bonus)

Notre modèle fonctionne ! Utilisons-le pour des tâches business concrètes.

### Bonus Task 1: Détecteur de "Bonnes Affaires" 💎

**Objectif :** Identifier les voitures vendues en dessous de leur valeur estimée.
**Pourquoi :** Pour acheter malin !

**Approche :**
1. Si `Prix_Réel < Prix_Prédit * 0.9` (10% moins cher), c'est une bonne affaire.
2. Si `Prix_Réel > Prix_Prédit * 1.1` (10% plus cher), c'est trop cher.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Création d'un DataFrame de résultats
results = pd.DataFrame({'Reel': y_test, 'Predit': y_pred})
results['Difference_Pct'] = (results['Reel'] - results['Predit']) / results['Predit']

# Définition des labels
def label_deal(row):
    if row['Difference_Pct'] < -0.10:
        return '💎 Bonne Affaire'
    elif row['Difference_Pct'] > 0.10:
        return '💸 Trop Cher'
    else:
        return '⚖️ Juste Prix'

results['Verdict'] = results.apply(label_deal, axis=1)

print("🔍 Exemples de verdicts :")
display(results.sample(10))

print("\\n📊 Répartition des affaires :")
print(results['Verdict'].value_counts())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Valeur de Revente Future 📉

**Objectif :** Estimer le prix de ces voitures dans 5 ans.
**Approche :**
1. On prend nos données de test.
2. On ajoute 5 ans à l'âge (`Age + 5`).
3. On demande au modèle de prédire le nouveau prix.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Simulation dans 5 ans
X_future = X_test.copy()
X_future['Age'] = X_future['Age'] + 5
# On pourrait aussi augmenter le kilométrage (ex: +15000km/an * 5)
X_future['Kilometrage'] = X_future['Kilometrage'] + (15000 * 5)
# Recalcul de Km_par_an
X_future['Km_par_an'] = X_future['Kilometrage'] / X_future['Age']

# Prédiction
future_price = model.predict(X_future)

# Comparaison
comparison = pd.DataFrame({
    'Prix_Actuel': y_pred,
    'Prix_Dans_5_Ans': future_price
})
comparison['Perte_Valeur'] = comparison['Prix_Actuel'] - comparison['Prix_Dans_5_Ans']

print("📉 Estimation de la perte de valeur sur 5 ans :")
display(comparison.head())
print(f"Perte moyenne : {comparison['Perte_Valeur'].mean():.2f} €")
"""))

    # SAVE
    with open('notebook_debutant_projet_14.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Notebook Débutant généré avec succès !")

if __name__ == "__main__":
    create_notebook()
