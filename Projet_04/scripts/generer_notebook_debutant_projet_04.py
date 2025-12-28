import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --- Titre et Introduction ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🎓 Projet 4 : Système d'Alerte Précoce de Décrochage Scolaire
## Version Débutant - "Je te montre, puis tu répètes"

---

### 🎯 L'Objectif de ce Projet

Le décrochage scolaire a des conséquences graves pour les élèves et les écoles. Votre mission est de **détecter les élèves à risque** (`A_Decroche = 1`) afin de pouvoir intervenir avant qu'il ne soit trop tard.

**Ce que vous allez apprendre :**
- 📊 Analyser des données éducatives (notes, présence, trajet)
- ⚖️ Gérer un problème de **Classification Déséquilibrée** (peu de décrocheurs, mais importants à trouver)
- 🤖 Utiliser `RandomForestClassifier` avec `class_weight='balanced'`
- 🎯 Optimiser le **Rappel (Recall)** pour ne manquer aucun élève en difficulté

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

# Configuration
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

print("✅ Bibliothèques importées !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Étape 1.1 : Charger les Données
Le fichier est `decrochage_scolaire.csv`.
"""))

    cells.append(nbf.v4.new_code_cell("""
df = pd.read_csv('decrochage_scolaire.csv')

print("📊 Aperçu des données :")
display(df.head())
print(f"\\n✅ Dimensions : {df.shape[0]} élèves, {df.shape[1]} colonnes")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Sanity Check (15 min)

### 📘 Theory: Valeurs Manquantes
Vérifions si nous avons des trous dans nos données.
"""))

    cells.append(nbf.v4.new_code_cell("""
print("🔍 Valeurs manquantes :")
print(df.isnull().sum())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Exemple : Remplir 'Temps_Trajet'
Pour le temps de trajet (numérique), nous allons utiliser la **médiane**.
"""))

    cells.append(nbf.v4.new_code_cell("""
mediane_trajet = df['Temps_Trajet'].median()
df['Temps_Trajet'].fillna(mediane_trajet, inplace=True)
print(f"✅ Temps_Trajet rempli avec : {mediane_trajet}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Remplissez les valeurs manquantes de `Education_Parents` (catégorique) avec le **mode** (la valeur la plus fréquente).
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Remplir Education_Parents avec le mode

# mode_edu = df['Education_Parents'].mode()[0]
# df['Education_Parents'].fillna(mode_edu, inplace=True)
# print(f"✅ Education_Parents rempli avec : {mode_edu}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (20 min)

### 📊 Visualisation 1 : Le Déséquilibre de Classe
C'est CRITIQUE. Combien d'élèves ont décroché vs sont restés ?
"""))

    cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='A_Decroche', palette=['green', 'red'])
plt.title('Distribution des Décrochages (0=Non, 1=Oui)')
plt.show()

print("Pourcentages :")
print(df['A_Decroche'].value_counts(normalize=True) * 100)
"""))

    cells.append(nbf.v4.new_markdown_cell("""
> **⚠️ Warning:** Vous voyez ce déséquilibre ? Il y a beaucoup moins de décrocheurs. Si notre modèle dit "Personne ne décroche", il aura raison 90% du temps (Accuracy), mais il sera INUTILE ! Nous devrons utiliser le **Recall** comme métrique.
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Visualisez l'impact de la `Presence` sur le décrochage avec un **Boxplot**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Boxplot Presence vs A_Decroche

# plt.figure(figsize=(8, 5))
# sns.boxplot(data=df, x='A_Decroche', y='Presence', palette=['green', 'red'])
# plt.title('Impact de la Présence sur le Décrochage')
# plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : The Art of Feature Engineering (45 min)

## Part 1: The Concept (10 min)
Transformons nos données brutes en indicateurs de risque.

## Part 2: The Lab - Choose Your Recipe (30 min)

### 🏷️ Recipe 2: Categories
`Education_Parents` est du texte. Transformons-le en nombres avec **One-Hot Encoding**.
"""))

    cells.append(nbf.v4.new_code_cell("""
df = pd.get_dummies(df, columns=['Education_Parents'], prefix='Edu')
print("✅ Encodage terminé !")
display(df.head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ➗ Recipe 4: Math Magic
Créons un **Score de Risque** simple basé sur notre intuition.
Faible Présence + Faibles Notes = Danger.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Exemple : Score inversé (plus c'est haut, plus c'est risqué)
# On normalise grossièrement : (100 - Presence) + (20 - Notes)*5
df['Risk_Score'] = (100 - df['Presence']) + (20 - df['Notes_Precedentes']) * 5

print("✅ Feature Risk_Score créée !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Catégorisez le `Temps_Trajet` en 'Court', 'Moyen', 'Long' (Binning).
- 0-15 min = Court
- 15-45 min = Moyen
- 45+ min = Long
Puis encodez-le (ou gardez-le numérique si vous préférez, mais essayons le binning pour l'exercice).
Pour simplifier ici, créons juste une feature binaire `Trajet_Long` (> 45 min).
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer Trajet_Long

# df['Trajet_Long'] = (df['Temps_Trajet'] > 45).astype(int)
# print("✅ Feature Trajet_Long créée !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)
Préparons X et y.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Supprimer l'ID (inutile)
if 'ID_Etudiant' in df.columns:
    df = df.drop(columns=['ID_Etudiant'])

X = df.drop(columns=['A_Decroche'])
y = df['A_Decroche']

print(f"✅ Prêt ! X shape: {X.shape}, y shape: {y.shape}")
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : Building & Trusting Your Model (45 min)

## Part 1: The Split (10 min)
**IMPORTANT :** Comme nous avons peu de décrocheurs, nous devons utiliser `stratify=y` pour être sûr d'en avoir dans le Train ET dans le Test.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Split stratifié effectué !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training (15 min)

### 🚨 Gestion du Déséquilibre
Nous allons utiliser `class_weight='balanced'`. Cela dit au modèle : "Attention, chaque erreur sur un décrocheur compte DOUBLE (ou plus) !"
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestClassifier

# Modèle avec poids équilibrés
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

print("🚀 Entraînement...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation (20 min)

### 🎯 Focus sur le RECALL (Rappel)
- **Recall** = Capacité à trouver TOUS les décrocheurs.
- Si on rate un élève à risque (Faux Négatif), c'est grave.
- Si on alerte pour rien (Faux Positif), on perd juste un peu de temps de conseiller.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import classification_report, confusion_matrix, recall_score

y_pred = model.predict(X_test)

# Métriques
recall = recall_score(y_test, y_pred)
print(f"🎯 RECALL (Décrocheurs) : {recall:.2%}")
print("\\n📊 Rapport complet :")
print(classification_report(y_test, y_pred))
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Affichez la **Matrice de Confusion** pour voir combien de Faux Négatifs nous avons.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Matrice de Confusion

# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Prédit')
# plt.ylabel('Réel')
# plt.title('Matrice de Confusion')
# plt.show()
"""))

    # --- PART 4 BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Les Facteurs de Risque
**Goal:** Qu'est-ce qui cause le décrochage ? (Feature Importance)
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Plot Feature Importance
# importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# importances.plot(kind='bar', color='teal')
# plt.title('Facteurs de Risque Principaux')
# plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Segmentation des Élèves
**Goal:** Grouper les élèves par profil (ex: "Bons mais loin", "En difficulté partout").
Utilisez KMeans sur `Presence` et `Notes_Precedentes`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: KMeans Clustering
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3, random_state=42)
# df['Cluster'] = kmeans.fit_predict(df[['Presence', 'Notes_Precedentes']])
# sns.scatterplot(data=df, x='Presence', y='Notes_Precedentes', hue='Cluster', palette='viridis')
# plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Recommandation d'Intervention
**Goal:** Créer une règle simple.
Si `Risk_Score` > X, recommander "Entretien Conseiller".
Si `Trajet_Long` == 1, recommander "Aide Transport".
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Logique de recommandation
# def recommend(row):
#     actions = []
#     if row['Risk_Score'] > 80: actions.append("Entretien Pédagogique")
#     if row['Trajet_Long'] == 1: actions.append("Pass Bus")
#     return ", ".join(actions) if actions else "Aucune"

# df['Intervention'] = df.apply(recommend, axis=1)
# display(df[['Risk_Score', 'Trajet_Long', 'Intervention']].head(10))
"""))

    nb['cells'] = cells
    nbf.write(nb, 'donnees_fr/Projet_04/Projet_04_Debutant.ipynb')

if __name__ == "__main__":
    generer_notebook_debutant()
