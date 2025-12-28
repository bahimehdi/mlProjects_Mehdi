import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    # --- Cellules du Notebook ---
    
    cells = []
    
    # Titre et Introduction
    cells.append(nbf.v4.new_markdown_cell("""
# 📰 Projet 6 : Classificateur de Fake News
## Version Débutant - "Je te montre, puis tu répètes"

---

### 🎯 L'Objectif de ce Projet

La désinformation se propage plus vite que la vérité sur les réseaux sociaux. Votre mission est de **détecter les fausses nouvelles** en analysant le titre, le contenu et les patterns de partage des articles.

**Ce que vous allez apprendre :**
- 📝 Analyser des données textuelles (NLP - Natural Language Processing)
- 🔍 Créer des features à partir de texte (longueur, mots-clés, sentiment)
- 🤖 Entraîner un modèle de **Classification Binaire** (Real vs Fake)
- 📊 Évaluer avec F1-Score et Confusion Matrix

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
Nous allons utiliser :
- **pandas** : Pour manipuler le tableau de données
- **numpy** : Pour les calculs mathématiques
- **matplotlib/seaborn** : Pour les graphiques
- **re** : Pour analyser les patterns dans le texte (expressions régulières)
"""))

    cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Configuration pour de beaux graphiques
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

print("✅ Bibliothèques importées !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Étape 1.1 : Charger les Données
Le fichier s'appelle `fake_news.csv`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Charger le dataset
df = pd.read_csv('fake_news.csv')

# Afficher les premières lignes
print("📊 Aperçu des données :")
display(df.head())

print(f"\\n✅ Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"\\n📋 Colonnes : {list(df.columns)}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Sanity Check (15 min)

### 📘 Theory: Distribution de la Cible
Avant de construire un modèle, il faut vérifier si les classes sont équilibrées.
Si on a 99% de vraies news et 1% de fake, le modèle apprendra mal !
"""))

    cells.append(nbf.v4.new_code_cell("""
# Vérifier la distribution de la cible
print("🎯 Distribution des articles :")
print(df['Etiquette'].value_counts())

# Visualiser
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Etiquette', palette='Set2')
plt.title('📊 Répartition Real vs Fake')
plt.ylabel('Nombre d\'articles')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📘 Theory: Valeurs Manquantes
Les données textuelles peuvent avoir des champs vides.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Vérifier les valeurs manquantes
print("🔍 Valeurs manquantes par colonne :")
print(df.isnull().sum())

# Supprimer les lignes avec texte manquant (si nécessaire)
df = df.dropna(subset=['Title', 'Corps_Texte'])
print(f"\\n✅ Dataset nettoyé : {df.shape[0]} lignes restantes")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📘 Theory: Duplicatas
Parfois, le même article est présent plusieurs fois.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Vérifier les duplicatas
print(f"🔍 Nombre de duplicatas : {df.duplicated().sum()}")

# Supprimer les duplicatas
df = df.drop_duplicates()
print(f"✅ Dataset final : {df.shape[0]} lignes")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (20 min)

### 📊 Visualisation 1 : Distribution des Partages
Les fake news ont-elles plus de partages ?
"""))

    cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Etiquette', y='Nb_Partages', palette='coolwarm')
plt.title('📈 Distribution des Partages : Real vs Fake')
plt.ylabel('Nombre de Partages')
plt.yscale('log')  # Échelle log pour mieux voir (partages vont de 0 à 1M)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ❓ Question
Les fake news ont-elles tendance à avoir plus ou moins de partages ? Que remarquez-vous ?
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📊 Visualisation 2 : Longueur du Titre
Les fake news utilisent-elles des titres plus longs ou plus courts ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# Créer une feature temporaire pour l'analyse
df['Title_Length'] = df['Title'].apply(len)

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Title_Length', hue='Etiquette', bins=30, kde=True)
plt.title('📏 Distribution de la Longueur des Titres')
plt.xlabel('Nombre de caractères')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez une visualisation pour comparer la **longueur du texte** (`Corps_Texte`) entre Real et Fake.
Utilisez un histogramme ou un boxplot.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer une feature 'Body_Length' (longueur de Corps_Texte)

# df['Body_Length'] = df['Corps_Texte'].apply(len)

# TODO: Créer un histogramme ou boxplot

# plt.figure(figsize=(10, 5))
# sns.boxplot(data=df, x='Etiquette', y='Body_Length', palette='pastel')
# plt.title('📏 Longueur du Corps de Texte : Real vs Fake')
# plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : The Art of Feature Engineering (45 min)

## Part 1: The Concept (10 min)

Les modèles de Machine Learning ne "lisent" pas le texte comme nous.
Ils ont besoin de **nombres** !

Nous allons transformer le texte en features numériques :
- **Statistiques** : Longueur, nombre de mots
- **Patterns** : Présence de !!!, MAJUSCULES, chiffres
- **Sentiment** : Ton positif/négatif (avancé)

## Part 2: The Lab - Choose Your Recipe (30 min)

### 📝 Recipe 3: Text & NLP

#### 📘 Theory: Features Statistiques
Pour chaque texte, nous pouvons calculer :
- Nombre de mots
- Nombre de caractères
- Longueur moyenne des mots
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ Exemple : Analyser le Titre
Créons des features pour la colonne `Title`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Feature 1: Nombre de mots dans le titre
df['Title_Word_Count'] = df['Title'].apply(lambda x: len(x.split()))

# Feature 2: Nombre de caractères
df['Title_Char_Count'] = df['Title'].apply(len)

# Feature 3: Longueur moyenne des mots
df['Title_Avg_Word_Length'] = df['Title'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0)

print("✅ Features statistiques du titre créées !")
display(df[['Title', 'Title_Word_Count', 'Title_Char_Count', 'Title_Avg_Word_Length']].head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Répétez la même chose pour la colonne `Corps_Texte` (le corps de l'article).
Créez 3 features : `Body_Word_Count`, `Body_Char_Count`, `Body_Avg_Word_Length`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer des features pour Corps_Texte

# df['Body_Word_Count'] = df['Corps_Texte'].apply(lambda x: len(x.split()))
# df['Body_Char_Count'] = df['Corps_Texte'].apply(len)
# df['Body_Avg_Word_Length'] = df['Corps_Texte'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0)

# print("✅ Features du corps créées !")
# display(df[['Corps_Texte', 'Body_Word_Count', 'Body_Char_Count']].head(3))
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📝 Recipe 6: Domain-Specific Features (Détection de Clickbait)

#### 📘 Theory: Clickbait
Les fake news utilisent souvent des titres sensationnalistes :
- "SHOCKING!!!" (points d'exclamation)
- "YOU WON'T BELIEVE" (tout en majuscules)
- Chiffres exagérés

Nous allons créer des features pour détecter ces patterns.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Feature 1: Nombre de points d'exclamation
df['Title_Exclamation_Count'] = df['Title'].apply(lambda x: x.count('!'))

# Feature 2: Pourcentage de majuscules
df['Title_Upper_Ratio'] = df['Title'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)

# Feature 3: Présence de chiffres
df['Title_Has_Numbers'] = df['Title'].apply(lambda x: 1 if bool(re.search(r'\\d', x)) else 0)

print("✅ Features clickbait créées !")
display(df[['Title', 'Title_Exclamation_Count', 'Title_Upper_Ratio', 'Title_Has_Numbers']].head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez une feature binaire `Title_Is_Clickbait` :
- 1 si le titre contient **3 ou plus** points d'exclamation OU **plus de 50%** de majuscules
- 0 sinon
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer la feature Title_Is_Clickbait

# df['Title_Is_Clickbait'] = df.apply(
#     lambda row: 1 if (row['Title_Exclamation_Count'] >= 3 or row['Title_Upper_Ratio'] > 0.5) else 0,
#     axis=1
# )

# print("✅ Feature Clickbait créée !")
# print(df['Title_Is_Clickbait'].value_counts())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ➗ Recipe 4: Math Magic (Transformation des Partages)

#### 📘 Theory: Log Transformation
Les `Nb_Partages` vont de 0 à 1 million. Cette grande échelle peut perturber le modèle.
Une **transformation logarithmique** réduit l'écart et donne plus de poids aux différences entre petits nombres.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Appliquer log(x + 1) pour éviter log(0)
df['Nb_Partages_Log'] = df['Nb_Partages'].apply(lambda x: np.log1p(x))

print("✅ Transformation log des partages créée !")
print(f"Avant : min={df['Nb_Partages'].min()}, max={df['Nb_Partages'].max()}")
print(f"Après : min={df['Nb_Partages_Log'].min():.2f}, max={df['Nb_Partages_Log'].max():.2f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🛠️ À vous de jouer !
Créez une feature `Share_Bucket` qui catégorise les articles :
- 'Viral' si `Nb_Partages` > 10000
- 'Popular' si entre 1000 et 10000
- 'Low' si < 1000
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer la feature Share_Bucket

# def categorize_shares(shares):
#     if shares > 10000:
#         return 'Viral'
#     elif shares >= 1000:
#         return 'Popular'
#     else:
#         return 'Low'

# df['Share_Bucket'] = df['Nb_Partages'].apply(categorize_shares)
# print("✅ Share_Bucket créée !")
# print(df['Share_Bucket'].value_counts())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)

Avant d'entraîner le modèle, nous devons :
1. Encoder la cible (`Etiquette`) en 0/1
2. Sélectionner les features numériques
3. Supprimer les colonnes textuelles originales
"""))

    cells.append(nbf.v4.new_code_cell("""
# Encoder la cible : Real=0, Fake=1
df['Etiquette_Encoded'] = df['Etiquette'].apply(lambda x: 1 if x == 'Fake' else 0)

print("✅ Cible encodée !")
print(df['Etiquette_Encoded'].value_counts())
"""))

    cells.append(nbf.v4.new_code_cell("""
# Sélectionner les features numériques pour le modèle
feature_columns = [
    'Title_Word_Count', 'Title_Char_Count', 'Title_Avg_Word_Length',
    'Title_Exclamation_Count', 'Title_Upper_Ratio', 'Title_Has_Numbers',
    'Nb_Partages_Log'
]

# Vérifier que les colonnes existent
available_features = [col for col in feature_columns if col in df.columns]
print(f"✅ Features disponibles : {available_features}")

# Définir X (features) et y (cible)
X = df[available_features]
y = df['Etiquette_Encoded']

print(f"\\n✅ Prêt pour le modèle ! X shape: {X.shape}, y shape: {y.shape}")
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : Building & Trusting Your Model (45 min)

## Part 1: The Split (10 min)

Nous divisons nos données en deux :
- **Train (80%)** : Pour que le modèle apprenne
- **Test (20%)** : Pour vérifier s'il a bien appris (examen final)
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✅ Train set : {X_train.shape[0]} lignes")
print(f"✅ Test set  : {X_test.shape[0]} lignes")
print(f"\\n📊 Distribution dans Train :")
print(y_train.value_counts(normalize=True))
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training (15 min)

Nous allons utiliser un **RandomForestClassifier**.
C'est un modèle puissant qui combine plusieurs arbres de décision pour voter sur la classe finale.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestClassifier

# Créer le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraîner le modèle
print("🚀 Entraînement en cours...")
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation (20 min)

### 📘 Theory: Métriques de Classification
Pour une **Classification Binaire**, nous utilisons :
- **Accuracy** : Pourcentage de prédictions correctes
- **F1-Score** : Équilibre entre Précision et Rappel (idéal pour classes déséquilibrées)
- **Confusion Matrix** : Tableau montrant Vrai Positifs, Faux Positifs, etc.

> **💡 Tip:** Pour la fake news, on préfère le **F1-Score** car manquer une fake news (Faux Négatif) est aussi grave que bloquer une vraie news (Faux Positif).
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Faire des prédictions
y_pred = model.predict(X_test)

# Calculer les métriques
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"🎯 Accuracy : {accuracy:.2%}")
print(f"🎯 F1-Score : {f1:.3f}")

print("\\n📊 Rapport de Classification :")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📊 Visualisation : Matrice de Confusion
La matrice montre :
- **Top-Left (Vrai Négatif)** : Real prédit comme Real ✅
- **Top-Right (Faux Positif)** : Real prédit comme Fake ❌
- **Bottom-Left (Faux Négatif)** : Fake prédit comme Real ❌
- **Bottom-Right (Vrai Positif)** : Fake prédit comme Fake ✅
"""))

    cells.append(nbf.v4.new_code_cell("""
# Créer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('🔍 Matrice de Confusion')
plt.ylabel('Vraie Étiquette')
plt.xlabel('Prédiction')
plt.show()

print("\\n📊 Interprétation :")
print(f"- Vrai Négatif (Real → Real) : {cm[0, 0]}")
print(f"- Faux Positif (Real → Fake) : {cm[0, 1]} ⚠️")
print(f"- Faux Négatif (Fake → Real) : {cm[1, 0]} ⚠️")
print(f"- Vrai Positif (Fake → Fake) : {cm[1, 1]}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 📊 Features les Plus Importantes
Quelles features aident le plus à détecter les fake news ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# Extraire l'importance des features
feature_importance = pd.DataFrame({
    'Feature': available_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('🔑 Features les Plus Importantes pour Détecter les Fake News')
plt.xlabel('Importance')
plt.show()

print(feature_importance)
"""))

    # --- PART 4 BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 15-30 mins)

Le modèle principal est entraîné ! Maintenant, explorons les tâches secondaires du projet.

### Bonus Task 1: Extraire les Mots-Clés des Fake News

**Goal:** Trouver les mots qui apparaissent le plus souvent dans les titres de fake news.

**Why it matters:** Identifier les patterns linguistiques permet de créer des filtres automatiques.

**Approach:**
1. Filtrer les articles `Etiquette == 'Fake'`
2. Concaténer tous les titres en un seul texte
3. Compter les mots les plus fréquents (après avoir retiré les stop words comme "the", "a")

**Deliverable:** Top 10 des mots-clés dans les fake news
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Extraire les mots-clés des fake news

# from collections import Counter

# # Filtrer les fake news
# fake_titles = df[df['Etiquette'] == 'Fake']['Title']

# # Concaténer et splitter tous les titres
# all_words = ' '.join(fake_titles).lower().split()

# # Compter les mots (optionnel: retirer stop words)
# word_counts = Counter(all_words)

# # Afficher le top 10
# print("🔑 Top 10 mots dans les Fake News :")
# for word, count in word_counts.most_common(10):
#     print(f"  {word}: {count}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Détecter les Patterns "Bot-like"

**Goal:** Identifier les articles avec un ratio partages/longueur anormalement élevé.

**Why it matters:** Les bots partagent massivement sans lire. Un article très court avec énormément de partages est suspect.

**Approach:**
1. Créer une feature `Share_Per_Char = Nb_Partages / Body_Char_Count`
2. Trouver le seuil du 95e percentile
3. Marquer les articles au-dessus comme "Bot-like"

**Deliverable:** Liste des articles suspects
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Détecter les patterns bot-like

# # Créer le ratio partages/longueur
# df['Share_Per_Char'] = df['Nb_Partages'] / (df['Body_Char_Count'] + 1)  # +1 pour éviter division par 0

# # Trouver le seuil (95e percentile)
# threshold = df['Share_Per_Char'].quantile(0.95)

# # Marquer les suspects
# df['Is_Bot_Like'] = df['Share_Per_Char'] > threshold

# print(f"🤖 Seuil Bot-like : {threshold:.2f}")
# print(f"Nombre d'articles suspects : {df['Is_Bot_Like'].sum()}")

# # Afficher quelques exemples
# print("\\nExemples d'articles suspects :")
# display(df[df['Is_Bot_Like']][['Title', 'Nb_Partages', 'Body_Char_Count', 'Share_Per_Char']].head())
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Prédire la Viralité (Régression)

**Goal:** Au lieu de classer Real/Fake, prédire le **nombre de partages** qu'un article aura.

**Why it matters:** Comprendre ce qui rend un contenu viral aide les créateurs de contenu légitime.

**Approach:**
1. Utiliser les mêmes features textuelles (longueur, majuscules, etc.)
2. Changer la cible vers `Nb_Partages_Log` (pour stabiliser)
3. Entraîner un **RandomForestRegressor**
4. Évaluer avec MAE et R²

**Deliverable:** Modèle de prédiction de viralité avec score R²
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Entraîner un modèle de prédiction de viralité

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score

# # Préparer les données (sans Nb_Partages_Log dans les features cette fois)
# X_viral = df[['Title_Word_Count', 'Title_Char_Count', 'Title_Exclamation_Count', 'Title_Upper_Ratio']]
# y_viral = df['Nb_Partages_Log']

# # Split
# X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_viral, y_viral, test_size=0.2, random_state=42)

# # Entraîner
# model_viral = RandomForestRegressor(n_estimators=100, random_state=42)
# model_viral.fit(X_train_v, y_train_v)

# # Prédire
# y_pred_v = model_viral.predict(X_test_v)

# # Évaluer
# mae = mean_absolute_error(y_test_v, y_pred_v)
# r2 = r2_score(y_test_v, y_pred_v)

# print(f"📊 MAE (log scale) : {mae:.2f}")
# print(f"📊 R² Score : {r2:.3f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 4: Topic Clustering (Regrouper par Sujet)

**Goal:** Grouper les articles en catégories automatiques (Politique, Santé, Célébrités).

**Why it matters:** Les fake news se concentrent souvent sur certains sujets sensibles.

**Approach (Avancé):**
1. Utiliser TF-IDF pour vectoriser les titres
2. Appliquer KMeans clustering (k=3 ou 5)
3. Analyser les mots-clés de chaque cluster

**Deliverable:** Distribution des articles par cluster
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Clustering par sujet (Avancé)

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans

# # Vectoriser les titres
# vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
# X_tfidf = vectorizer.fit_transform(df['Title'])

# # Clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# df['Topic_Cluster'] = kmeans.fit_predict(X_tfidf)

# print("📚 Distribution par cluster :")
# print(df['Topic_Cluster'].value_counts().sort_index())

# # Afficher quelques exemples par cluster
# for cluster_id in range(3):
#     print(f"\\n--- Cluster {cluster_id} ---")
#     examples = df[df['Topic_Cluster'] == cluster_id]['Title'].head(3).tolist()
#     for ex in examples:
#         print(f"  - {ex}")
"""))

    # Assign cells to notebook
    nb['cells'] = cells

    # Sauvegarde
    with open('Projet_06_Debutant.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Notebook Débutant généré : Projet_06_Debutant.ipynb")

if __name__ == "__main__":
    generer_notebook_debutant()
