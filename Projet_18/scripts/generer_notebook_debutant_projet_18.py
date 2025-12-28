import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    # Titre
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 🚗 PROJET 18 : POINTS CHAUDS DE COVOITURAGE 📍

Bienvenue dans ce projet stratégique pour les plateformes de covoiturage !

**Le Problème :** Les chauffeurs roulent dans des rues vides pendant que des passagers attendent ailleurs. C'est du gaspillage de temps et de carburant !

**Votre Mission :** Prédire la demande de courses (Demandes) par zone et par heure. Ainsi, l'app peut envoyer les chauffeurs exactement là où sont les passagers ! 🎯

---

## 📅 VOTRE PROGRAMME

### 📋 SESSION 1 : From Raw Data to Clean Insights (45 min)
- **Part 1: The Setup** - Charger les données historiques de courses
- **Part 2: The Sanity Check** - Nettoyer les données manquantes
- **Part 3: Exploratory Data Analysis** - Où et quand la demande est-elle forte ?

### 📋 SESSION 2 : The Art of Feature Engineering (45 min)
- **Part 1: The Concept** - Extraire des features temporelles
- **Part 2: The Lab** - Créer le ratio Offre/Demande
- **Part 3: Final Prep** - Préparer pour le modèle

### 📋 SESSION 3 : Building & Trusting Your Model (45 min)
- **Part 1: The Split** - Séparer train/test
- **Part 2: Training** - Entraîner le modèle de prédiction
- **Part 3: Evaluation** - Mesurer la précision
- **Part 4: Going Further (BONUS)** - Tarification dynamique et relocalisation

---
"""))

    # SESSION 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : FROM RAW DATA TO CLEAN INSIGHTS
"""))

    # Part 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🏁 Part 1: The Setup (10 min)
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
### 📂 Chargement du fichier covoiturage.csv
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df = pd.read_csv('covoiturage.csv')

print("Aperçu des données :")
display(df.head(10))

print("\\nInfos techniques :")
df.info()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
> **💡 Tip:** Le dataset contient :
> - **ID_Zone** : Identifiant de la zone géographique (1-10)
> - **Horodatage** : Date et heure (format datetime)
> - **Chauffeurs_Actifs** : Nombre de chauffeurs disponibles
> - **Meteo** : Météo (Clear, Rain)
> - **Evenements** : 1 si événement spécial, 0 sinon
> - **Demandes** : 🎯 NOTRE CIBLE (nombre de courses demandées)
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
> **⚠️ Warning:** Nous avons des valeurs manquantes dans `Chauffeurs_Actifs` et `Meteo`. Pour simplifier, supprimons ces lignes.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df = df.dropna()

print(f"✅ Nouvelles dimensions : {df.shape}")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 2. Conversion de l'horodatage
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df['Horodatage'] = pd.to_datetime(df['Horodatage'])
print("✅ Horodatage converti !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 📊 Part 3: Exploratory Data Analysis (20 min)

### 📈 Demande par Zone
Quelles zones sont les plus demandées ?
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='ID_Zone', y='Demandes', estimator=np.mean, errorbar=None)
plt.title('Demande Moyenne par Zone')
plt.ylabel('Nombre de Courses Demandées')
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
❓ **Question :** Quelles zones ont la demande la plus élevée ?

### 🌦️ Impact de la Météo
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Meteo', y='Demandes', errorbar=None)
plt.title('Demande par Météo')
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
❓ **Question :** La pluie augmente-t-elle la demande de courses ?
"""))

    # SESSION 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : THE ART OF FEATURE ENGINEERING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🧠 Part 1: The Concept (10 min)

Pour prédire la demande, le modèle a besoin de :
- **L'heure de la journée** (rush du matin vs nuit calme)
- **Le jour de la semaine** (week-end vs jour de semaine)
- **Le ratio Offre/Demande** (Combien de chauffeurs pour combien de passagers ?)
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🧪 Part 2: The Lab - Choose Your Recipe (30 min)

### Recipe 1: Dates & Time 🕐
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df['Heure'] = df['Horodatage'].dt.hour
df['JourSemaine'] = df['Horodatage'].dt.dayofweek
df['Mois'] = df['Horodatage'].dt.month
df['Is_Weekend'] = (df['JourSemaine'] >= 5).astype(int)

print("✅ Features temporelles créées !")
display(df[['Horodatage', 'Heure', 'JourSemaine', 'Is_Weekend']].head())
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Recipe 2: Categories 🏷️
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df = pd.get_dummies(df, columns=['Meteo'], prefix='Meteo')

print("✅ Encodage terminé !")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Recipe 6: Domain-Specific Features 🎯

#### 🔹 Ratio Offre/Demande
Si on a 100 chauffeurs pour 200 demandes, c'est une pénurie !
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Créer le ratio (attention à division par zéro)
df['Supply_Demand_Ratio'] = df['Chauffeurs_Actifs'] / (df['Demandes'] + 1)

print("✅ Ratio Offre/Demande créé !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🏁 Part 3: Final Prep (5 min)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
cols_to_drop = ['Horodatage']

df_model = df.drop(columns=cols_to_drop)
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
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X = df_model.drop('Demandes', axis=1)
y = df_model['Demandes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🏋️ Part 2: Training (15 min)
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
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE (Erreur Moyenne) : {mae:.2f} courses")
print(f"📊 RMSE : {rmse:.2f}")
print(f"📊 R² Score : {r2:.3f}")

# Visualisation
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Demandes Réelles')
plt.ylabel('Demandes Prédites')
plt.title('Vérité vs Prédiction')
plt.show()
"""))

    # Part 4 Bonus
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Cartographier les Zones de Tarification Dynamique 💰

**Goal:** Identifier les zones où il faut augmenter les prix.

**Why it matters:** Quand la demande est HAUTE et l'offre est BASSE, on applique une "surge pricing" pour attirer plus de chauffeurs.

**Approach:**
1. Définir un seuil : Supply_Demand_Ratio < 0.5 (2 passagers pour 1 chauffeur)
2. Identifier ces moments par zone
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# TODO: Identifier les zones de tarification dynamique
df_original = pd.read_csv('covoiturage.csv').dropna()
df_original['Horodatage'] = pd.to_datetime(df_original['Horodatage'])
df_original['Supply_Demand_Ratio'] = df_original['Chauffeurs_Actifs'] / (df_original['Demandes'] + 1)

# Zones avec ratio < 0.5 (pénurie)
surge_zones = df_original[df_original['Supply_Demand_Ratio'] < 0.5]

print(f"🔥 Nombre d'heures avec tarification dynamique : {len(surge_zones)}")
print("\\nTop 5 zones avec le plus de pénuries :")
print(surge_zones['ID_Zone'].value_counts().head())
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Recommander la Relocalisation des Chauffeurs 🚙

**Goal:** Dire aux chauffeurs de se déplacer des zones à surplus vers les zones en déficit.

**Approach:**
1. Identifier les zones avec trop de chauffeurs (Supply_Demand_Ratio > 2)
2. Identifier les zones en manque (Supply_Demand_Ratio < 0.5)
3. Recommandermouvement : Zone A → Zone B
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Exemple : à l'heure actuelle (dernière heure du dataset)
current_hour = df_original.iloc[-10:]  # Dernières 10 lignes

surplus_zones = current_hour[current_hour['Supply_Demand_Ratio'] > 2][['ID_Zone', 'Chauffeurs_Actifs', 'Demandes']]
deficit_zones = current_hour[current_hour['Supply_Demand_Ratio'] < 0.5][['ID_Zone', 'Chauffeurs_Actifs', 'Demandes']]

print("📈 Zones avec SURPLUS de chauffeurs :")
print(surplus_zones)

print("\\n📉 Zones en DÉFICIT de chauffeurs :")
print(deficit_zones)

print("\\n🚗 Recommandation : Déplacer les chauffeurs des zones en surplus vers les zones en déficit.")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 4: Regrouper les Zones par Type 🏘️

**Goal:** Classifier chaque zone : Résidentielle, Affaires, ou Vie Nocturne.

**Approach:**
- **Résidentielle** : Pic de demande le matin (7h-9h) et soir (18h-20h)
- **Affaires** : Demande élevée en journée (9h-18h)
- **Vie Nocturne** : Pic de demande la nuit (22h-2h)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Calculer la demande moyenne par zone et par heure
df_original['Heure'] = df_original['Horodatage'].dt.hour

demand_by_zone_hour = df_original.groupby(['ID_Zone', 'Heure'])['Demandes'].mean().reset_index()

# Pour chaque zone, identifier l'heure de pic
peak_hours = demand_by_zone_hour.loc[demand_by_zone_hour.groupby('ID_Zone')['Demandes'].idxmax()]

print("Heure de pic par zone :")
print(peak_hours)

# Classification simplifiée
def classify_zone(peak_hour):
    if 7 <= peak_hour <= 9 or 18 <= peak_hour <= 20:
        return 'Résidentielle 🏡'
    elif 9 < peak_hour < 18:
        return 'Affaires 💼'
    else:
        return 'Vie Nocturne 🎉'

peak_hours['Type_Zone'] = peak_hours['Heure'].apply(classify_zone)

print("\\nClassification des zones :")
print(peak_hours[['ID_Zone', 'Heure', 'Type_Zone']])
"""))

    # Sauvegarde
    with open('Projet_18_Covoiturage_Debutant.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_debutant()
