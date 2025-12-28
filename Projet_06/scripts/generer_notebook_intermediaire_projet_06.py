import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Titre et Introduction
    cells.append(nbf.v4.new_markdown_cell("""
# 📰 Projet 6 : Classificateur de Fake News
## Version Intermédiaire - "Voici le chemin, marche seul"

---

### 🎯 L'Objectif de ce Projet

La désinformation se propage plus vite que la vérité. Votre mission est de **construire un système de détection de fake news** en analysant le titre, le contenu textuel, et les patterns de partage.

**Compétences visées :**
- NLP (Natural Language Processing) pour extraire des features textuelles
- Feature engineering créatif pour détecter les patterns de clickbait
- Classification binaire avec métriques adaptées
- Analyse exploratoire des comportements viraux

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights (45 min)

## Part 1: The Setup (10 min)

### Étape 1.1: Imports et Configuration

**Objectif:** Importer les bibliothèques nécessaires pour le NLP et la visualisation.

**Librairies recommandées:**
- `pandas`, `numpy` : Manipulation de données
- `matplotlib`, `seaborn` : Visualisations
- `re` : Expressions régulières pour analyser le texte
- (Optionnel) `nltk` ou `textblob` : Analyse de sentiment

**Conseil:** Configurez `matplotlib` avec une taille de figure par défaut (10, 6) pour de meilleurs graphiques.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.2: Chargement des Données

**Objectif:** Charger `fake_news.csv` et explorer la structure.

**Livrables attendus:**
- Affichage des 5 premières lignes
- Dimensions du dataset (lignes × colonnes)
- Types de données de chaque colonne
- Liste des noms de colonnes

**Conseil:** Utilisez `df.info()` pour avoir un aperçu complet.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Sanity Check (15 min)

### Étape 2.1: Analyse de la Distribution de la Cible

**Objectif:** Vérifier l'équilibre des classes `Etiquette` (Real vs Fake).

**Approches recommandées:**
- `value_counts()` avec normalisation pour voir les pourcentages
- **Visualisation:** Countplot ou barplot pour comparer visuellement

**Livrables attendus:**
- Nombre et pourcentage de Real vs Fake
- Graphique de distribution
- **Décision:** Le dataset est-il équilibré ? (> 30% pour chaque classe = équilibré)

**Conseil:** Si fortement déséquilibré (< 20% d'une classe), noter pour plus tard l'utilisation de SMOTE ou class_weight.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2: Détection des Valeurs Manquantes

**Objectif:** Identifier et traiter les NaN dans les colonnes textuelles.

**Approches recommandées:**
1. **Vérification:** `df.isnull().sum()` pour compter les NaN
2. **Traitement:**
   - Supprimer les lignes avec texte manquant (`dropna`) si < 5% du dataset
   - Remplacer par chaîne vide si nécessaire

**Livrables attendus:**
- Rapport des NaN par colonne
- Dataset nettoyé (nombre de lignes avant/après)

**Conseil:** Pour NLP, il vaut mieux supprimer que remplir avec du texte générique.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.3: Détection des Duplicatas

**Objectif:** Supprimer les articles en double.

**Approche:**
- Utiliser `df.duplicated().sum()` puis `df.drop_duplicates()`
- **Alternative:** Ne considérer que le texte (`subset=['Title', 'Corps_Texte']`)

**Livrable attendu:** Nombre de duplicatas trouvés et supprimés
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (20 min)

### Étape 3.1: Analyse des Partages

**Objectif:** Comparer le comportement de partage entre Real et Fake news.

**Approches recommandées:**
1. **Boxplot:** `sns.boxplot(x='Etiquette', y='Nb_Partages')` avec échelle log (`plt.yscale('log')`)
2. **Statistiques descriptives:** `df.groupby('Etiquette')['Nb_Partages'].describe()`

**Livrables attendus:**
- Graphique comparatif
- **Insight:** Les fake news ont-elles plus ou moins de partages en moyenne ?

**Conseil:** L'échelle log aide à visualiser des données qui varient de 0 à 1 million.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2: Analyse de la Longueur du Texte

**Objectif:** Comparer la longueur des titres et corps de texte entre Real et Fake.

**Approches recommandées:**
1. Créer des features temporaires :
   - `Title_Length = df['Title'].apply(len)`
   - `Body_Length = df['Corps_Texte'].apply(len)`
2. Visualiser avec histogrammes (`hue='Etiquette'`) ou boxplots

**Livrables attendus:**
- 2 graphiques (un pour Title, un pour Body)
- **Insight:** Les fake news ont-elles des titres plus courts/longs ? Texte plus court/long ?

**Conseil:** Les fake news ont souvent des titres sensationnalistes courts et un contenu superficiel.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.3: Exploration des Sources

**Objectif:** Identifier si certaines sources (URLs) sont plus associées aux fake news.

**Approches:**
1. Extraire le domaine principal de `URL_Source` (ex: "cnn.com" depuis "https://cnn.com/article")
2. Compter les articles par source
3. Croiser avec `Etiquette`

**Livrable attendu:** Top 5 sources et leur ratio Fake/Real

**Conseil (Avancé):** Utilisez `urlparse` de la librairie `urllib.parse` ou regex pour extraire le domaine.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : The Art of Feature Engineering (45 min)

## Part 1: The Concept (10 min)

Les modèles de ML ne lisent pas le texte. Vous devez transformer le langage naturel en **vecteurs numériques**.

**Stratégies disponibles:**
1. **Features statistiques** : Longueur, nombre de mots, ponctuation
2. **Features linguistiques** : Sentiment, complexité, clickbait indicators
3. **Vectorisation** : TF-IDF, Count Vectorizer, Word Embeddings (avancé)

## Part 2: The Lab - Choose Your Recipe (30 min)

### Recipe 3: Text & NLP Features

#### Étape 2.1: Features Statistiques du Titre

**Objectif:** Créer des features numériques basées sur le `Title`.

**Features recommandées:**
1. **Word Count** : `len(text.split())`
2. **Character Count** : `len(text)`
3. **Average Word Length** : `sum(len(word) for word in text.split()) / word_count`

**Livrables attendus:**
- Colonnes : `Title_Word_Count`, `Title_Char_Count`, `Title_Avg_Word_Length`
- Vérification : afficher les 5 premières lignes avec ces colonnes

**Conseil:** Utilisez `df['Title'].apply(lambda x: ...)` pour appliquer une fonction à chaque ligne.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.2: Features Statistiques du Corps de Texte

**Objectif:** Répéter la même analyse pour `Corps_Texte`.

**Features à créer:**
- `Body_Word_Count`
- `Body_Char_Count`
- `Body_Avg_Word_Length`

**Livrable attendu:** 3 nouvelles colonnes vérifiées
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Recipe 6: Domain-Specific Features (Clickbait Detection)

#### Étape 2.3: Indicateurs de Clickbait

**Objectif:** Détecter les titres sensationnalistes typiques des fake news.

**Features recommandées:**
1. **Exclamation Count** : `text.count('!')`
2. **Question Mark Count** : `text.count('?')`
3. **Uppercase Ratio** : `sum(1 for c in text if c.isupper()) / len(text)`
4. **Has Numbers** : `1 if re.search(r'\\d', text) else 0`
5. **All Caps Words** : Nombre de mots entièrement en majuscules

**Approches multiples:**
- **Méthode 1 (Simple):** Analyse caractère par caractère
- **Méthode 2 (Regex):** Utiliser `re.findall(r'\\b[A-Z]+\\b', text)` pour détecter mots en majuscules

**Livrables attendus:**
- Minimum 3 features clickbait (exclamation, uppercase ratio, numbers)
- Feature composite : `Title_Is_Clickbait` (1 si >= 3 exclamations OU uppercase_ratio > 0.5)

**Conseil:** Les fake news utilisent souvent "SHOCKING!!!" ou "YOU WON'T BELIEVE!!!"
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.4: Ratio Partages/Longueur (Bot Detection Proxy)

**Objectif:** Créer une feature pour détecter les partages artificiels.

**Hypothèse:** Un article très court avec énormément de partages est suspect (bots).

**Feature à créer:**
```
Share_Per_Word = Nb_Partages / (Body_Word_Count + 1)
```

**Livrable attendu:** Nouvelle colonne `Share_Per_Word`

**Conseil:** Ajoutez +1 au dénominateur pour éviter division par zéro.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Recipe 4: Math Magic (Transformations)

#### Étape 2.5: Log Transformation des Partages

**Objectif:** Normaliser la distribution de `Nb_Partages`.

**Approches:**
1. **Log naturel** : `np.log1p(x)` (log(x+1) pour gérer les 0)
2. **Square Root** : `np.sqrt(x)` (alternative plus douce)

**Livrable attendu:** Colonne `Nb_Partages_Log`

**Conseil:** La transformation log réduit l'impact des valeurs extrêmes (1M partages).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.6: Bucketization (Catégorisation des Partages)

**Objectif:** Créer des buckets de viralité.

**Catégories suggérées:**
- 'Viral' : > 10,000 partages
- 'Popular' : 1,000 - 10,000
- 'Low' : < 1,000

**Approche:** Fonction conditionnelle ou `pd.cut()`

**Livrable attendu:** Colonne `Share_Bucket` (optionnel: encoder en 0/1/2)

**Conseil:** Cette feature peut être utilisée pour une analyse secondaire (pas pour le modèle principal).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)

### Étape 2.7: Encodage de la Cible

**Objectif:** Transformer `Etiquette` (Real/Fake) en valeurs numériques (0/1).

**Approches:**
1. **Lambda:** `df['Etiquette'].apply(lambda x: 1 if x == 'Fake' else 0)`
2. **LabelEncoder:** `from sklearn.preprocessing import LabelEncoder`

**Livrable attendu:** Colonne `Etiquette_Encoded` avec 0=Real, 1=Fake

**Conseil:** Toujours vérifier la distribution après encodage (`value_counts()`).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.8: Sélection des Features

**Objectif:** Créer X (features) et y (target) pour le modèle.

**Features recommandées pour le modèle:**
- Toutes les features statistiques (word count, char count, etc.)
- Toutes les features clickbait (exclamation, uppercase, etc.)
- `Nb_Partages_Log`
- (Optionnel) `Share_Per_Word`

**À EXCLURE:**
- Colonnes textuelles originales (`Title`, `Corps_Texte`, `URL_Source`)
- `ID_Article`
- `Etiquette` (original, non encodée)
- `Nb_Partages` (utiliser la version log)

**Livrables attendus:**
- `X` : DataFrame avec features numériques uniquement
- `y` : Series avec `Etiquette_Encoded`
- Vérification : `X.shape` et `y.shape`

**Conseil:** Créez une liste `feature_columns` puis faites `X = df[feature_columns]`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : Building & Trusting Your Model (45 min)

## Part 1: The Split (10 min)

### Étape 3.1: Train/Test Split

**Objectif:** Diviser les données pour entraînement et évaluation.

**Approches recommandées:**
- **Standard:** 80% train, 20% test
- **Avec Stratification:** `stratify=y` pour garder la même proportion de classes dans train et test

**Paramètres clés:**
- `test_size=0.2`
- `random_state=42` (pour reproductibilité)
- `stratify=y` (IMPORTANT pour classification)

**Livrables attendus:**
- `X_train, X_test, y_train, y_test`
- Affichage des tailles (nombre de lignes)
- Vérification de la distribution des classes dans train et test

**Conseil:** Utilisez `y_train.value_counts(normalize=True)` pour vérifier les proportions.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training (15 min)

### Étape 3.2: Entraînement du Modèle

**Objectif:** Entraîner un classificateur pour détecter les fake news.

**Modèles recommandés:**
1. **RandomForestClassifier** ✅ Recommandé
   - Robuste, gère bien les features multiples
   - Paramètres : `n_estimators=100`, `random_state=42`
   - Avantage : Peut fournir l'importance des features

2. **LogisticRegression** (Alternative)
   - Plus rapide, interprétable
   - Bon pour baseline

3. **GradientBoostingClassifier** (Avancé)
   - Meilleure performance potentielle
   - Plus lent à entraîner

**Livrables attendus:**
- Modèle entraîné et sauvegardé dans une variable `model`
- Message de confirmation d'entraînement

**Conseil:** RandomForest avec 100 arbres est un bon équilibre performance/vitesse.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation (20 min)

### Contexte Métier

**Type de problème:** Classification binaire équilibrée (~60% Real, ~40% Fake)

**Métriques prioritaires:**
1. **F1-Score** ← PRIORITÉ (équilibre précision/rappel)
2. **Accuracy** (acceptable car classes relativement équilibrées)
3. **Confusion Matrix** (pour comprendre les types d'erreurs)

**Pourquoi F1 > Accuracy ?**
- Bloquer une vraie news (Faux Positif) = Censure
- Laisser passer une fake news (Faux Négatif) = Désinformation
- Les deux erreurs sont graves → F1 équilibre les deux

### Étape 3.3: Calcul des Métriques

**Objectif:** Évaluer la performance du modèle.

**Métriques à calculer:**
- `accuracy_score(y_test, y_pred)`
- `f1_score(y_test, y_pred)` ← **PRIORITÉ**
- `classification_report(y_test, y_pred, target_names=['Real', 'Fake'])`

**Livrables attendus:**
- Accuracy (pourcentage)
- F1-Score (0.0 à 1.0)
- Rapport complet (Precision, Recall, F1 par classe)

**Conseil:** Un F1-Score > 0.75 est bon, > 0.85 est excellent pour ce type de problème.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.4: Matrice de Confusion

**Objectif:** Visualiser les types d'erreurs du modèle.

**Interprétation:**
```
                Prédit Real    Prédit Fake
Vrai Real       [TN]           [FP] ← Censure (bad)
Vrai Fake       [FN] ← Désinformation (bad)  [TP]
```

**Approche:**
- Calculer avec `confusion_matrix(y_test, y_pred)`
- Visualiser avec `sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')`

**Livrables attendus:**
- Graphique de la matrice
- Interprétation : Nombre de FP et FN

**Conseil:** Ajoutez les labels `xticklabels` et `yticklabels` pour clarifier.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.5: Feature Importance

**Objectif:** Identifier quelles features aident le plus à détecter les fake news.

**Approche (pour RandomForest):**
```python
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
```

**Livrables attendus:**
- Dataframe trié par importance
- Barplot horizontal (`sns.barplot`)
- **Insight:** Quelle est la feature la plus importante ?

**Conseil:** Si `Nb_Partages_Log` domine, essayez de retirer cette feature et ré-entraîner pour voir l'impact du texte seul.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- PART 4 BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Extraction des Mots-Clés de Fake News

**Goal:** Identifier les mots les plus fréquents dans les titres de fake news.

**Why it matters:** Comprendre le vocabulaire utilisé permet de créer des règles de filtrage automatiques.

**Approche:**
1. Filtrer les articles où `Etiquette == 'Fake'`
2. Concaténer tous les titres en un seul texte
3. Convertir en minuscules et splitter par espaces
4. Utiliser `collections.Counter` pour compter les mots
5. (Optionnel) Retirer les stop words ("the", "a", "is")

**Livrables attendus:**
- Top 10 des mots dans les fake news
- (Bonus) Comparaison avec top 10 des real news

**Conseil:** La librairie `nltk` offre une liste de stop words en anglais : `nltk.corpus.stopwords.words('english')`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Détection de Patterns Bot-like

**Goal:** Identifier les articles avec un ratio partages/longueur anormalement élevé.

**Why it matters:** Les bots partagent massivement sans lire le contenu. Un article court avec énormément de partages est suspect.

**Approche:**
1. Utiliser la feature `Share_Per_Word` créée en Session 2
2. Calculer le 95e percentile : `df['Share_Per_Word'].quantile(0.95)`
3. Marquer les articles au-dessus de ce seuil comme "Bot-like"
4. Analyser la distribution Real/Fake dans ce groupe

**Livrables attendus:**
- Seuil calculé
- Nombre d'articles suspects
- Crosstab : Bot-like × Etiquette
- **Insight:** Les fake news sont-elles plus souvent bot-like ?

**Conseil:** Créez une colonne binaire `Is_Bot_Like` pour faciliter l'analyse.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Prédiction de Viralité (Régression)

**Goal:** Construire un modèle pour prédire le nombre de partages d'un article.

**Why it matters:** Comprendre ce qui rend un contenu viral aide les créateurs de contenu légitime à maximiser leur impact.

**Approche:**
1. Changer la cible : `y_viral = df['Nb_Partages_Log']`
2. Features : Toutes sauf `Nb_Partages` et `Nb_Partages_Log`
3. Modèle : `RandomForestRegressor`
4. Métriques : MAE, RMSE, R²

**Livrables attendus:**
- Modèle de régression entraîné
- MAE et R² Score
- (Bonus) Scatter plot des prédictions vs valeurs réelles

**Conseil:** Un R² > 0.5 serait déjà bon pour ce type de prédiction (comportement viral imprévisible).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 4: Topic Clustering (Regroupement par Sujet)

**Goal:** Grouper automatiquement les articles en catégories thématiques (ex: Politique, Santé, Célébrités).

**Why it matters:** Les fake news se concentrent souvent sur des sujets sensibles (santé, politique). Identifier les topics permet une analyse ciblée.

**Approche (Avancée):**
1. **Vectorisation TF-IDF:**
   - `from sklearn.feature_extraction.text import TfidfVectorizer`
   - `vectorizer = TfidfVectorizer(max_features=50, stop_words='english')`
   - `X_tfidf = vectorizer.fit_transform(df['Title'])`

2. **Clustering KMeans:**
   - `from sklearn.cluster import KMeans`
   - `kmeans = KMeans(n_clusters=3, random_state=42)`
   - `df['Topic_Cluster'] = kmeans.fit_predict(X_tfidf)`

3. **Analyse:**
   - Afficher quelques exemples de titres par cluster
   - Croiser avec `Etiquette` pour voir si certains topics sont plus fake

**Livrables attendus:**
- 3-5 clusters créés
- Exemples de titres par cluster
- Distribution Fake/Real par cluster
- **Interprétation:** Nommer les clusters (ex: "Cluster 0 = Politique")

**Conseil:** Commencez avec 3 clusters, puis augmentez si nécessaire. Analysez les top mots de chaque cluster avec `vectorizer.get_feature_names_out()`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # Assign cells to notebook
    nb['cells'] = cells

    # Sauvegarde
    with open('Projet_06_Intermediaire.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Notebook Intermédiaire généré : Projet_06_Intermediaire.ipynb")

if __name__ == "__main__":
    generer_notebook_intermediaire()
