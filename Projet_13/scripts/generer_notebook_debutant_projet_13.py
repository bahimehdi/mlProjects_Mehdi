import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "13"
    PROJECT_TITLE = "Prévision des Arrivées de Visiteurs"
    DATASET_NAME = "prevision_visiteurs.csv"
    TARGET_COL = "Visiteurs"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE}

## 🏁 Objectif : Le Prophète du Tourisme 🔮
Les villes sont parfois vides, parfois bondées.
Votre mission est de prédire le nombre de **Visiteurs** pour aider la ville à s'organiser.
Anticiper la foule, c'est éviter le chaos ! 🚌

---

## 📋 Programme des 3 Sessions

### 🕵️‍♀️ SESSION 1 : Enquêteur de Données (45 min)
- **Part 1 :** Chargement et Nettoyage (Attention aux dates !)
- **Part 2 :** Analyse Exploratoire (Y a-t-il des saisons ?)

### 🏗️ SESSION 2 : Architecte de Features (45 min)
- **Part 1 :** Feature Engineering (Extraire le jour, le mois, le weekend)
- **Part 2 :** Préparation finale pour l'IA

### 🤖 SESSION 3 : Entraîneur d'IA (45 min)
- **Part 1 :** Entraînement du Modèle (Régression Temporelle)
- **Part 2 :** Évaluation (Nos prévisions sont-elles justes ?)
- **Part 3 :** Bonus (Alerte Surtourisme)

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
# Important : On dit à pandas que la colonne 'Date' contient des dates
df = pd.read_csv('{DATASET_NAME}', parse_dates=['Date'])

print("✅ Données chargées avec succès !")
print(f"📊 Dimensions : {{df.shape[0]}} lignes, {{df.shape[1]}} colonnes")
df.head()
"""))

    # Part 2: Sanity Check
    cells.append(nbf.v4.new_markdown_cell("""
## 🧹 Part 2: The Sanity Check
Les données réelles ont souvent des trous. Vérifions !
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
> **💡 Tip:** Pour le `Prix_Moyen_Hotel`, nous allons utiliser la **Médiane**. Pour l'`Indicateur_Evenement`, nous mettrons 0 (pas d'événement par défaut).
"""))

    cells.append(nbf.v4.new_code_cell("""
# Remplacer les valeurs manquantes
# 1. Prix_Moyen_Hotel -> Médiane
median_price = df['Prix_Moyen_Hotel'].median()
df['Prix_Moyen_Hotel'].fillna(median_price, inplace=True)

# 2. Indicateur_Evenement -> 0
df['Indicateur_Evenement'].fillna(0, inplace=True)

print("✅ Nettoyage terminé !")
print(df.isnull().sum())
"""))

    # Part 3: EDA
    cells.append(nbf.v4.new_markdown_cell(f"""
## 🔍 Part 3: Exploratory Data Analysis (EDA)
Analysons notre cible : **{TARGET_COL}** au fil du temps.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Évolution du nombre de visiteurs
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='{TARGET_COL}', hue='City')
plt.title("Évolution des Visiteurs par Ville")
plt.xlabel("Date")
plt.ylabel("Nombre de Visiteurs")
plt.show()

print("❓ Question : Voyez-vous des pics réguliers ? (Noël, Été...)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
Regardons l'impact des événements. Y a-t-il plus de monde quand `Indicateur_Evenement` est à 1 ?
"""))

    cells.append(nbf.v4.new_code_cell(f"""
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Indicateur_Evenement', y='{TARGET_COL}', palette='Set2')
plt.title("Impact des Événements sur les Visiteurs")
plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    # Recipe 1: Dates & Time
    cells.append(nbf.v4.new_markdown_cell("""
### 🕐 Recipe: Dates & Time
L'IA ne comprend pas "25 Décembre". Elle préfère "Mois = 12", "Jour = 25".
Découpons la date !
"""))

    cells.append(nbf.v4.new_code_cell("""
# Extraction des informations de date
df['Mois'] = df['Date'].dt.month
df['Jour_Semaine'] = df['Date'].dt.dayofweek # 0=Lundi, 6=Dimanche
df['Jour_Mois'] = df['Date'].dt.day

# Créons une feature "Weekend" (Samedi ou Dimanche)
df['Est_Weekend'] = (df['Jour_Semaine'] >= 5).astype(int)

print("✅ Dates découpées !")
df[['Date', 'Mois', 'Jour_Semaine', 'Est_Weekend']].head()
"""))

    # Recipe 2: Categories
    cells.append(nbf.v4.new_markdown_cell("""
### 🏷️ Recipe: Categories
La `City` (Paris, Kyoto...) est importante. Transformons-la en chiffres.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Encodage One-Hot pour la Ville
df_encoded = pd.get_dummies(df, columns=['City'], drop_first=True)

# Note: On garde la colonne Date pour l'instant pour le tri, mais on l'enlèvera pour l'entraînement
print("✅ Encodage terminé !")
df_encoded.head()
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🤖 SESSION 3 : Building & Trusting Your Model
"""))

    # Part 1: Split
    cells.append(nbf.v4.new_markdown_cell("""
## ✂️ Part 1: The Split (Temporel)
Attention ! Pour le temps, on ne mélange pas (shuffle=False).
On entraîne sur le **Passé**, on teste sur le **Futur**.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# On trie par date pour être sûr
df_encoded = df_encoded.sort_values('Date')

# On définit la coupure (ex: les 20% les plus récents pour le test)
split_index = int(len(df_encoded) * 0.8)

train = df_encoded.iloc[:split_index]
test = df_encoded.iloc[split_index:]

# On sépare X et y
# On retire la Date car l'IA ne sait pas lire "2023-01-01" directement (on a déjà extrait mois/jour)
X_train = train.drop(['Date', '{TARGET_COL}'], axis=1)
y_train = train['{TARGET_COL}']

X_test = test.drop(['Date', '{TARGET_COL}'], axis=1)
y_test = test['{TARGET_COL}']

print(f"Train shape: {{X_train.shape}} (Passé)")
print(f"Test shape: {{X_test.shape}} (Futur)")
"""))

    # Part 2: Training (Regression)
    cells.append(nbf.v4.new_markdown_cell("""
## 🏋️ Part 2: Training (Régression)
Nous voulons prédire un nombre de visiteurs. C'est une **Régression**.
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
Regardons si la courbe prédite suit la courbe réelle.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erreur Moyenne (MAE) : {mae:,.0f} Visiteurs")
print(f"R² Score : {r2:.3f}")

# Visualisons la prédiction vs réalité dans le temps
plt.figure(figsize=(12, 6))
# On utilise l'index du test set pour l'axe X (ou la colonne Date si on l'avait gardée à part)
plt.plot(test['Date'], y_test, label='Réalité', alpha=0.7)
plt.plot(test['Date'], y_pred, label='Prédiction', alpha=0.7, linestyle='--')
plt.title("Prévisions vs Réalité")
plt.legend()
plt.show()
"""))

    # Part 4: Bonus
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### 🚨 Bonus Task 1: Alerte Surtourisme
Définissons un seuil de "Surtourisme" (ex: > 20 000 visiteurs).
Notre modèle peut-il nous prévenir à l'avance ?
"""))

    cells.append(nbf.v4.new_code_cell(f"""
SEUIL_SURTOURISME = 20000

# Créons un DataFrame de résultats
results = pd.DataFrame({{
    'Date': test['Date'],
    'Reel': y_test,
    'Predit': y_pred
}})

# Identifions les jours où l'IA prédit une foule
jours_alerte = results[results['Predit'] > SEUIL_SURTOURISME]

print(f"📅 Nombre de jours d'alerte détectés : {{len(jours_alerte)}}")
print("Prochaines dates critiques :")
print(jours_alerte.head())
"""))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Visiteurs_Debutant.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
