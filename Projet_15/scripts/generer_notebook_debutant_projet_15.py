import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CELLULES DU NOTEBOOK ---
    
    # TITRE ET INTRODUCTION
    cells = [
        nbf.v4.new_markdown_cell("""# 🎓 PROJET 15 : Optimiseur d'Annulation d'Hôtel
## 🏁 Objectif du Projet
Les annulations de dernière minute coûtent cher aux hôtels. Votre mission est de prédire si un client va annuler sa réservation (`Annule = 1`) ou non (`Annule = 0`).
Cela permettra à l'hôtel de mieux gérer ses chambres et d'optimiser son taux d'occupation.

## 📂 Les Données
Le fichier `annulation_hotel.csv` contient 800 réservations avec des informations comme le délai de réservation, le prix, le segment de marché, etc.

---
# 📋 SESSION 1 : From Raw Data to Clean Insights (45 min)
"""),
        
        # PART 1: THE SETUP
        nbf.v4.new_markdown_cell("""## 🛠️ Part 1: The Setup (10 min)
Commençons par importer les outils nécessaires et charger nos données."""),
        
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour un affichage plus joli
sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)

print("✅ Bibliothèques importées avec succès !")"""),
        
        nbf.v4.new_markdown_cell("""### 📥 Chargement des données"""),
        
        nbf.v4.new_code_cell("""# Chargement du dataset
df = pd.read_csv('annulation_hotel.csv')

# Aperçu des 5 premières lignes
print("Aperçu des données :")
display(df.head())

# Informations sur les colonnes
print("\\nInfos du dataset :")
df.info()"""),
        
        nbf.v4.new_markdown_cell("""❓ **Question :** Regardez la colonne `Annule`. Quelles sont les deux valeurs possibles ? Que signifient-elles ?"""),
        
        # PART 2: THE SANITY CHECK
        nbf.v4.new_markdown_cell("""## 🧹 Part 2: The Sanity Check (15 min)
Avant d'analyser, il faut nettoyer ! Vérifions les valeurs manquantes et les doublons."""),
        
        nbf.v4.new_markdown_cell("""### 🔍 Valeurs Manquantes"""),
        
        nbf.v4.new_code_cell("""# Vérification des valeurs manquantes
print("Valeurs manquantes par colonne :")
print(df.isnull().sum())

# Visualisation des manquants
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Carte des valeurs manquantes')
plt.show()"""),
        
        nbf.v4.new_markdown_cell("""📘 **Theory :** Les valeurs manquantes peuvent fausser nos modèles.
- Pour les colonnes numériques (ex: `Prix_Moyen`), on remplace souvent par la **médiane**.
- Pour les colonnes catégorielles (ex: `Segment_Marche`), on remplace par le **mode** (la valeur la plus fréquente) ou "Inconnu"."""),
        
        nbf.v4.new_code_cell("""# 1. Traitement de 'Prix_Moyen' (Numérique) -> Médiane
mediane_prix = df['Prix_Moyen'].median()
df['Prix_Moyen'].fillna(mediane_prix, inplace=True)
print(f"✅ 'Prix_Moyen' rempli avec la médiane : {mediane_prix}")

# TODO: Répétez pour 'Demandes_Speciales' (Numérique) -> Médiane
# Votre code ici

# 2. Traitement de 'Segment_Marche' (Catégoriel) -> Mode
mode_segment = df['Segment_Marche'].mode()[0]
df['Segment_Marche'].fillna(mode_segment, inplace=True)
print(f"✅ 'Segment_Marche' rempli avec le mode : {mode_segment}")

# Vérification finale
assert df.isnull().sum().sum() == 0, "⚠️ Il reste des valeurs manquantes !"
print("✅ Plus aucune valeur manquante !")"""),
        
        nbf.v4.new_markdown_cell("""### 👯 Doublons"""),
        
        nbf.v4.new_code_cell("""# Vérification et suppression des doublons
doublons = df.duplicated().sum()
print(f"Nombre de doublons détectés : {doublons}")

if doublons > 0:
    df.drop_duplicates(inplace=True)
    print(f"✅ {doublons} doublons supprimés.")
else:
    print("✅ Aucun doublon trouvé.")"""),
        
        # PART 3: EDA
        nbf.v4.new_markdown_cell("""## 📊 Part 3: Exploratory Data Analysis (20 min)
Explorons nos données pour comprendre ce qui influence les annulations."""),
        
        nbf.v4.new_markdown_cell("""### 🎯 Analyse de la Cible (`Annule`)"""),
        
        nbf.v4.new_code_cell("""# Distribution des annulations
plt.figure(figsize=(6, 4))
sns.countplot(x='Annule', data=df, palette='pastel')
plt.title('Distribution des Annulations (0=Non, 1=Oui)')
plt.xlabel('Annulé ?')
plt.ylabel('Nombre de réservations')
plt.show()

# Pourcentage
print(df['Annule'].value_counts(normalize=True) * 100)"""),
        
        nbf.v4.new_markdown_cell("""❓ **Question :** Y a-t-il plus d'annulations ou de séjours confirmés ? Est-ce équilibré ?"""),
        
        nbf.v4.new_markdown_cell("""### ⏱️ Délai de Réservation vs Annulation"""),
        
        nbf.v4.new_code_cell("""# Boxplot du Délai de Réservation selon l'Annulation
plt.figure(figsize=(8, 5))
sns.boxplot(x='Annule', y='Delai_Reservation', data=df, palette='Set2')
plt.title('Impact du Délai de Réservation sur l\'Annulation')
plt.show()"""),
        
        nbf.v4.new_markdown_cell("""❓ **Question :** Les gens qui réservent très longtemps à l'avance annulent-ils plus souvent ?"""),
        
        nbf.v4.new_markdown_cell("""---
# 🧪 SESSION 2 : The Art of Feature Engineering (45 min)
Nous allons créer de nouvelles variables pour aider notre modèle."""),
        
        nbf.v4.new_markdown_cell("""## 🍳 Recipe 2: Categories 🏷️
Les ordinateurs ne comprennent pas le texte comme "Online" ou "Corporate". Nous devons encoder ces catégories."""),
        
        nbf.v4.new_code_cell("""# Encodage One-Hot pour 'Segment_Marche'
df = pd.get_dummies(df, columns=['Segment_Marche'], drop_first=True)

print("✅ Colonnes après encodage :")
print(df.columns.tolist())
display(df.head())"""),
        
        nbf.v4.new_markdown_cell("""## 🍳 Recipe 6: Domain-Specific Features 🎯
Créons des variables spécifiques à l'hôtellerie."""),
        
        nbf.v4.new_markdown_cell("""### 1. Catégorie de Délai (Lead Time Category)
Classons les réservations en "Dernière minute", "Normal", "Planifié"."""),
        
        nbf.v4.new_code_cell("""def categorize_lead_time(days):
    if days <= 7:
        return 'Last_Minute'
    elif days <= 30:
        return 'Normal'
    else:
        return 'Planned'

df['Lead_Time_Category'] = df['Delai_Reservation'].apply(categorize_lead_time)

# Vérifions la relation avec l'annulation
plt.figure(figsize=(8, 5))
sns.countplot(x='Lead_Time_Category', hue='Annule', data=df, order=['Last_Minute', 'Normal', 'Planned'])
plt.title('Annulations par Catégorie de Délai')
plt.show()

# Encodage de cette nouvelle variable
df = pd.get_dummies(df, columns=['Lead_Time_Category'], drop_first=True)
print("✅ Feature 'Lead_Time_Category' créée et encodée !")"""),
        
        nbf.v4.new_markdown_cell("""### 2. Demandes Spéciales (Indicateur d'engagement)
Un client qui fait des demandes spéciales est peut-être plus engagé."""),
        
        nbf.v4.new_code_cell("""# Créons une variable binaire : A fait une demande ou non
df['Has_Requests'] = (df['Demandes_Speciales'] > 0).astype(int)

# Visualisons
sns.barplot(x='Has_Requests', y='Annule', data=df)
plt.title('Taux d\'annulation selon s\'il y a des demandes spéciales')
plt.show()"""),
        
        nbf.v4.new_markdown_cell("""---
# 🤖 SESSION 3 : Building & Trusting Your Model (45 min)
C'est le moment de prédire !"""),
        
        nbf.v4.new_markdown_cell("""## ✂️ Part 1: The Split (10 min)
Séparons les données pour l'entraînement et le test."""),
        
        nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split

# Définition des features (X) et de la cible (y)
X = df.drop(['ID_Reservation', 'Annule'], axis=1) # On enlève l'ID qui ne sert à rien
y = df['Annule']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Données divisées : {X_train.shape[0]} entraînement, {X_test.shape[0]} test")"""),
        
        nbf.v4.new_markdown_cell("""## 🏋️ Part 2: Training & Calibration (15 min)
### 📈 CAS 3 : Classification avec Calibration des Probabilités
Pour gérer le surbooking, nous avons besoin de **probabilités fiables**, pas juste d'une prédiction Oui/Non.
Nous allons utiliser un `RandomForestClassifier` et le calibrer."""),
        
        nbf.v4.new_code_cell("""from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

# 1. Entraîner le modèle de base
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
base_model.fit(X_train, y_train)

# 2. Calibrer les probabilités
# Cela ajuste le modèle pour que "70% de probabilité" signifie vraiment "70% de chance d'annuler"
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

print("✅ Modèle entraîné et calibré !")"""),
        
        nbf.v4.new_markdown_cell("""## 🎯 Part 3: Evaluation (20 min)
Vérifions la qualité de nos probabilités."""),
        
        nbf.v4.new_code_cell("""# Prédiction des probabilités sur le test
y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

# Évaluation
auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)

print(f"🎯 ROC-AUC Score : {auc:.3f} (Plus proche de 1 est mieux)")
print(f"🎯 Brier Score : {brier:.3f} (Plus proche de 0 est mieux)")

# Visualisation de la distribution des probabilités
plt.figure(figsize=(8, 5))
sns.histplot(y_pred_proba, bins=20, kde=True)
plt.title('Distribution des Probabilités d\'Annulation Prédites')
plt.xlabel('Probabilité d\'Annulation')
plt.show()"""),
        
        nbf.v4.new_markdown_cell("""## 🎁 Part 4: Going Further (Bonus)
Utilisons nos prédictions pour prendre des décisions business !"""),
        
        nbf.v4.new_markdown_cell("""### 🏨 Bonus Task 1: Calculate Optimal Overbooking Limit
**Goal:** Recommander combien de chambres supplémentaires l'hôtel peut vendre sans risque.
**Why it matters:** Si l'hôtel ne surréserve pas, il perd de l'argent sur les annulations. S'il surréserve trop, il doit reloger des clients (très cher).

**Approche :**
1. Calculer le nombre attendu d'annulations (Somme des probabilités).
2. Appliquer une marge de sécurité (ex: 80% de confiance)."""),
        
        nbf.v4.new_code_cell("""# Imaginons que X_test représente les réservations du mois prochain
expected_cancellations = y_pred_proba.sum()
safe_overbooking = int(expected_cancellations * 0.8) # Marge de sécurité

print(f"📊 Nombre total de réservations : {len(X_test)}")
print(f"🔮 Annulations attendues (Somme des probas) : {expected_cancellations:.1f}")
print(f"✅ Limite de surréservation recommandée (Safe) : {safe_overbooking} chambres")

print(f"\\n💡 Conseil : Vous pouvez vendre {safe_overbooking} chambres de plus que votre capacité totale !")"""),
        
        nbf.v4.new_markdown_cell("""### 👥 Bonus Task 2: Segmenter les clients par Fiabilité
**Goal:** Identifier les clients "À Risque" vs "Fiables".
**Why it matters:** On peut appeler les clients à risque pour confirmer, et laisser tranquilles les fiables."""),
        
        nbf.v4.new_code_cell("""# Création des segments basés sur la probabilité
conditions = [
    (y_pred_proba < 0.3),
    (y_pred_proba >= 0.3) & (y_pred_proba < 0.7),
    (y_pred_proba >= 0.7)
]
choices = ['Fiable', 'Incertain', 'À Risque']

segments = np.select(conditions, choices)

# Visualisation
unique, counts = np.unique(segments, return_counts=True)
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=unique, autopct='%1.1f%%', colors=['#66b3ff', '#ffcc99', '#ff9999'])
plt.title('Segmentation des Réservations Futures')
plt.show()""")
    ]
    
    nb.cells = cells
    
    with open('notebook_debutant_projet_15.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook Débutant généré avec succès !")

if __name__ == "__main__":
    create_notebook()
