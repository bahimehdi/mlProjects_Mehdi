import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    # Titre
    nb.cells.append(nbf.v4.new_markdown_cell("""
# PROJET 20 : PREDICTION D'EPIDEMIE

Bienvenue dans le dernier projet - le plus important pour la sante publique !

**Le Probleme :** Les hopitaux doivent prevoir les epidemies (Grippe, Dengue) pour preparer lits et medicaments.

**Votre Mission :** Predire le nombre de cas pour la semaine prochaine en analysant la meteo, Google Trends, et les donnees historiques.

---

## VOTRE PROGRAMME

### SESSION 1 : From Raw Data to Clean Insights (45 min)
- **Part 1: The Setup** - Charger les donnees d'epidemie
- **Part 2: The Sanity Check** - Nettoyer les donnees manquantes
- **Part 3: Exploratory Data Analysis** - Analyser les tendances

### SESSION 2 : The Art of Feature Engineering (45 min)
- **Part 1: The Concept** - Comprendre les series temporelles
- **Part 2: The Lab** - Creer des lag features et moyennes mobiles
- **Part 3: Final Prep** - Preparer le dataset

### SESSION 3 : Building & Trusting Your Model (45 min)
- **Part 1: The Split** - Train/Test split
- **Part 2: Training** - RandomForestRegressor
- **Part 3: Evaluation** - MAE, RMSE, R2
- **Part 4: Going Further (BONUS)** - Prediction du pic et systeme d'alerte

---
"""))

    # SESSION 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 1 : FROM RAW DATA TO CLEAN INSIGHTS
"""))

    # Part 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 1: The Setup (10 min)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Librairies importees !")
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df = pd.read_csv('epidemie.csv')

print("Apercu des donnees :")
display(df.head(10))

print("\\nInfos techniques :")
df.info()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
> **Tip:** Le dataset contient :
> - **Week** : Semaine (format date)
> - **Region** : Region geographique
> - **Temp_Moyenne** : Temperature moyenne
> - **Precipitations** : Pluies (favorisent certaines maladies)
> - **Google_Trends** : Recherches Google liees aux symptomes
> - **Cases** : CIBLE (nombre de cas)
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Sanity Check (15 min)

### 1. Valeurs manquantes
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
print("Valeurs manquantes par colonne :")
print(df.isnull().sum())
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Remplir les valeurs manquantes par la mediane
df['Google_Trends'].fillna(df['Google_Trends'].median(), inplace=True)
df['Precipitations'].fillna(df['Precipitations'].median(), inplace=True)

print(f"Nouvelles dimensions : {df.shape}")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 2. Conversion de la date
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df['Week'] = pd.to_datetime(df['Week'])
print("Week convertie en datetime !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (20 min)

### Evolution des cas dans le temps
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(12, 5))
plt.plot(df['Week'], df['Cases'])
plt.title('Evolution des Cas d Epidemie')
plt.xlabel('Semaine')
plt.ylabel('Nombre de Cas')
plt.xticks(rotation=45)
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
Question : Observez-vous des pics epidemiques ?

### Cas par region
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Region', y='Cases', estimator=np.mean, errorbar=None)
plt.title('Cas Moyens par Region')
plt.show()
"""))

    # SESSION 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 2 : THE ART OF FEATURE ENGINEERING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 1: The Concept (10 min)

Pour predire les epidemies :
- **Lag features** : Le nombre de cas de la semaine precedente influence cette semaine
- **Moyennes mobiles** : Tendance sur 4 semaines
- **Saisonnalite** : Certaines maladies ont des pics saisonniers
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Lab (30 min)

### Recipe 1: Dates & Time
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df['Mois'] = df['Week'].dt.month
df['Annee'] = df['Week'].dt.year
df['Week_of_Year'] = df['Week'].dt.isocalendar().week

print("Features temporelles creees !")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Recipe 2: Categories
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df = pd.get_dummies(df, columns=['Region'], prefix='Region')

print("Encodage termine !")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Recipe 6: Domain-Specific Features

#### Feature 1: Lag (Cas de la semaine precedente)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Trier par date
df = df.sort_values('Week').reset_index(drop=True)

# Lag 1 semaine
df['Cases_Lag1'] = df['Cases'].shift(1)

print("Lag feature creee !")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
#### Feature 2: Moyenne mobile (4 semaines)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df['Cases_MA4'] = df['Cases'].shift(1).rolling(window=4, min_periods=1).mean()

print("Moyenne mobile creee !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
cols_to_drop = ['Week']

df_model = df.drop(columns=cols_to_drop)
df_model = df_model.dropna()

print(f"Dataset pret ! Dimensions : {df_model.shape}")
"""))

    # SESSION 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 3 : BUILDING & TRUSTING YOUR MODEL
"""))

    # Part 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 1: The Split (10 min)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X = df_model.drop('Cases', axis=1)
y = df_model['Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training (15 min)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Entrainement...")
model.fit(X_train, y_train)
print("Modele entraine !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation (20 min)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE (Erreur Moyenne) : {mae:.2f} cas")
print(f"RMSE : {rmse:.2f}")
print(f"R2 Score : {r2:.3f}")

# Visualisation
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Cas Reels')
plt.ylabel('Cas Predits')
plt.title('Verite vs Prediction')
plt.show()
"""))

    # Part 4 Bonus
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Classifier le Niveau de Risque

**Goal:** Creer une classification Faible/Moyen/Epidemique basee sur les predictions.

**Why it matters:** Les hopitaux ont besoin d alertes simples, pas juste de chiffres.

**Approach:**
1. Definir seuils : Faible < 200, Moyen 200-500, Epidemique > 500
2. Classifier les predictions
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
def classify_risk(cases):
    if cases < 200:
        return 'Faible'
    elif cases < 500:
        return 'Moyen'
    else:
        return 'Epidemique'

# Appliquer la classification
df_original = pd.read_csv('epidemie.csv')
df_original['Predicted_Risk'] = y_pred[:len(df_original)]
df_original['Risk_Level'] = df_original['Predicted_Risk'].apply(classify_risk)

print("Repartition des niveaux de risque :")
print(df_original['Risk_Level'].value_counts())
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Identifier le Temps de Latence (Pluie -> Epidemie)

**Goal:** Combien de semaines apres la pluie l epidemie eclate-t-elle ?

**Approach:**
1. Calculer correlation entre Precipitations et Cases avec differents lags
2. Trouver le lag avec la correlation maximale
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df_original = pd.read_csv('epidemie.csv').dropna()
df_original = df_original.sort_values('Week').reset_index(drop=True)

# Tester differents lags (0 a 8 semaines)
correlations = []
for lag in range(9):
    df_original[f'Precip_Lag{lag}'] = df_original['Precipitations'].shift(lag)
    corr = df_original['Cases'].corr(df_original[f'Precip_Lag{lag}'])
    correlations.append((lag, corr))
    print(f"Lag {lag} semaines : Correlation = {corr:.3f}")

# Trouver le meilleur lag
best_lag = max(correlations, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
print(f"\\nTemps de latence optimal : {best_lag[0]} semaines")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Corréler Google Trends avec Données Officielles

**Goal:** Google Trends est-il un bon indicateur précoce ?
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df_original = pd.read_csv('epidemie.csv').dropna()

correlation = df_original['Google_Trends'].corr(df_original['Cases'])

print(f"Correlation Google Trends vs Cases : {correlation:.3f}")

if correlation > 0.7:
    print("Google Trends est un EXCELLENT indicateur précoce !")
elif correlation > 0.5:
    print("Google Trends est utile mais pas parfait.")
else:
    print("Google Trends n'est pas fiable pour ce dataset.")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 4: Allouer les Ressources Médicales

**Goal:** Recommander l'allocation de lits d'hôpital par région.

**Approach:**
1. Prédire les cas pour la semaine prochaine par région
2. Allouer 5% des cas prédits comme nombre de lits nécessaires
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Grouper par région et prédire
df_original = pd.read_csv('epidemie.csv').dropna()

avg_cases_by_region = df_original.groupby('Region')['Cases'].mean()

print("Allocation des lits par region (5% des cas moyens) :")
for region, cases in avg_cases_by_region.items():
    beds_needed = int(cases * 0.05)
    print(f"{region}: {beds_needed} lits")
"""))

    # Sauvegarde
    with open('Projet_20_Epidemie_Debutant.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_debutant()
