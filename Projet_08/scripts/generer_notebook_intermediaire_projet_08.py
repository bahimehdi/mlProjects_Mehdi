import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --- Titre ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🎓 Projet 8 : Analyse de Sentiment en Santé Mentale
## Version Intermédiaire - "Voici le chemin, marche seul"

---

### 🎯 L'Objectif
Construire un modèle de classification capable de **détecter l'état mental** à partir de posts sur les réseaux sociaux (`Etiquette` : `Normal`, `Depressed`, `Anxious`).

**Contexte Métier :**
- **Cible** : `Etiquette` (Multi-classe : Normal, Depressed, Anxious)
- **Problème** : Classification multi-classe avec léger déséquilibre (50% Normal, 25% chacune pour les autres).
- **Priorité** : **F1-Score équilibré**. Les classes `Depressed` et `Anxious` sont plus importantes à détecter (impact sur la santé).
- **Spécificité** : Projet **NLP-heavy** - l'essentiel de l'information est dans le texte.

---

### 📋 SESSION 1 : From Raw Data to Clean Insights

#### Étape 1.1 : Chargement et Inspection
**Objectif :** Charger `sante_mentale.csv` et comprendre la structure.

**Livrables attendus :**
- Dimensions et types.
- Identification de la colonne texte (`Texte`), temporelle (`Horodatage`), et catégorielle (`Plateforme`).
- Distribution des étiquettes (check du déséquilibre).
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 1.2 : Nettoyage & Gestion des Manquants
**Objectif :** Remplir les valeurs manquantes dans `Texte`.

**Approches recommandées :**
- `Texte` (Textuel) : Remplacer par un placeholder comme `"No text"` ou `""`.
- Vérifier et supprimer les duplicates si nécessaires.

**Livrables attendus :**
- Dataset propre sans NaN.
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 1.3 : Exploratory Data Analysis (EDA)
**Objectif :** Comprendre les patterns dans les données.

**Analyses à réaliser :**
1. **Countplot** de `Etiquette` : Quantifier le déséquilibre.
2. **Barplot** : Répartition par `Plateforme`.
3. **Distribution de la longueur du texte** : Histogrammes par étiquette (les posts déprimés sont-ils plus courts/longs ?).
4. **WordCloud** (optionnel) : Visualiser les mots les plus fréquents par catégorie.

**Question :** Y a-t-il des différences notables dans le style ou la longueur des textes entre les 3 catégories ?
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
---

### 📋 SESSION 2 : The Art of Feature Engineering

#### Étape 2.1 : Recipe 3 - Text & NLP Features (PRIMARY)
**Objectif :** Extraire des features textuelles riches.

**Features de Base à créer :**
1. **Text_Length** : Nombre de caractères (`str.len()`)
2. **Word_Count** : Nombre de mots (`str.split().str.len()`)
3. **Avg_Word_Length** : Longueur moyenne des mots
4. **Char_Count** : Nombre de caractères (alias de Text_Length)

**Features de Sentiment (TextBlob recommandé) :**
1. **Sentiment_Polarity** : Score de -1 (négatif) à +1 (positif)
2. **Sentiment_Subjectivity** : Score de 0 (objectif) à 1 (subjectif)

**Librairie :** `from textblob import TextBlob`

**Conseil :** Créez des fonctions auxiliaires pour gérer les exceptions (texte vide, etc.).
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.2 : Recipe 6 - Domain-Specific Features (Mental Health)
**Objectif :** Créer des indicateurs spécifiques à la santé mentale.

**Features Métier à créer :**

1. **Negative_Word_Count** : Compter les mots négatifs
   - Liste suggérée : `['sad', 'alone', 'hopeless', 'tired', 'depressed', 'anxious', 'worry', 'fear', 'bad', 'awful']`
   - Approche : Créer une fonction qui compte les occurrences (case-insensitive)

2. **Has_Negative_Words** : Binaire (0 ou 1)

3. **Has_Urgent_Keywords** : Détection d'idées suicidaires ⚠️
   - Liste : `['suicide', 'kill', 'die', 'death', 'end it', 'hurt myself']`
   - Très important pour la sécurité

4. **Exclamation_Count** : Nombre de `!` (peut indiquer l'intensité émotionnelle)

5. **Question_Count** : Nombre de `?` (peut indiquer l'anxiété)

**Conseil :** Vérifiez la corrélation de ces features avec la target.
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.3 : Recipe 1 - Dates & Time Features
**Objectif :** Extraire des patterns temporels.

**Features Temporelles à créer :**
1. Convertir `Horodatage` en datetime
2. **Hour** : Heure du post (0-23)
3. **DayOfWeek** : Jour de la semaine (0=Lundi, 6=Dimanche)
4. **Is_Weekend** : Binaire (1 si samedi/dimanche)
5. **Is_Night** : Binaire (1 si entre 22h et 6h)
6. **Is_Morning** / **Is_Afternoon** / **Is_Evening** (optionnel)

**Hypothèse Métier :** Les posts nocturnes ou du week-end peuvent indiquer une détresse.
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.4 : Recipe 2 - Categories (Plateforme)
**Objectif :** Encoder `Plateforme`.

**Approche :**
- One-Hot Encoding (`pd.get_dummies`) est recommandé.
- Résultat : Colonnes `Platform_Reddit`, `Platform_Twitter`, etc.

**Livrables attendus :**
- Colonnes binaires pour chaque plateforme.
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 2.5 : Final Prep
**Objectif :** Préparer X et y pour la modélisation.

**Étapes :**
1. Supprimer les colonnes non nécessaires : `ID_Post`, `Texte`, `Horodatage`, `DayOfWeek` (si redondant)
2. Créer `X` (features) et `y` (target = `Etiquette`)
3. Vérifier qu'il n'y a pas de NaN dans X

**Livrables attendus :**
- `X` : DataFrame avec toutes les features numériques
- `y` : Series avec les étiquettes
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
---

### 📋 SESSION 3 : Building & Trusting Your Model

#### Étape 3.1 : Split Stratifié
**Objectif :** Diviser Train/Test en gardant la même proportion de chaque classe.

**Contrainte :** Utilisez `stratify=y` dans `train_test_split`.

**Livrables attendus :**
- X_train, X_test, y_train, y_test (80/20 split recommandé)
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 3.2 : Entraînement Multi-Classe
**Objectif :** Entraîner un modèle capable de prédire 3 catégories.

**Modèle recommandé :** `RandomForestClassifier`
- Gère nativement la multi-classe
- Pas besoin de SMOTE (déséquilibre léger)
- Option : `class_weight='balanced'` si vous voulez augmenter le poids des classes minoritaires

**Alternatives (pour comparaison) :**
- Logistic Regression (rapide, interprétable)
- SVC (Support Vector Classifier)

**Livrables attendus :**
- Modèle entraîné
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Étape 3.3 : Évaluation Multi-Classe
**Objectif :** Valider la capacité de détection sur les 3 classes.

**Métriques à calculer :**
1. **Accuracy** : Pourcentage global correct
2. **Classification Report** : Precision, Recall, F1-Score **par classe**
   - Focus sur F1-Score pour `Depressed` et `Anxious`
3. **Confusion Matrix** : Visualiser où le modèle se trompe
   - Axes : [Normal, Depressed, Anxious]

**Questions clés :**
- Quelle classe est la mieux prédite ?
- Le modèle confond-il `Depressed` et `Anxious` entre elles ?
- Y a-t-il beaucoup de faux négatifs pour les classes critiques ?

**Conseil :** Utilisez `sns.heatmap()` pour une matrice de confusion visuelle avec annotations.
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
---

### 🎁 Part 4: Going Further (Bonus Tasks)

#### Bonus Task 1: Analyser les Tendances d'Humeur par Moment de la Journée
**Goal:** Identifier les périodes à risque (ex: plus de dépression la nuit ?).

**Approach:**
1. Recharger les données originales pour avoir `Hour` et `Etiquette`
2. Grouper par `Hour` et `Etiquette`
3. Créer un line plot ou heatmap montrant la distribution horaire

**Livrables attendus :**
- Graphique montrant les tendances par heure
- Insights : Y a-t-il un pic de posts anxieux la nuit ?
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 2: Identifier les Mots Déclencheurs pour Chaque Catégorie
**Goal:** Quels mots apparaissent le plus souvent dans chaque catégorie ?

**Approach:**
1. Séparer le texte par étiquette
2. Tokenizer (split, lower, enlever stopwords basiques)
3. Utiliser `Counter` pour compter les mots
4. Afficher Top 10-15 mots par catégorie

**Librairie suggérée :** `from collections import Counter`

**Optionnel Avancé :** 
- TF-IDF pour identifier les mots discriminants
- WordCloud par catégorie

**Livrables attendus :**
- Liste des mots déclencheurs pour `Normal`, `Depressed`, `Anxious`
- Insights : Quels mots sont uniques à chaque catégorie ?
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 3: Regrouper les Utilisateurs en Groupes de Soutien
**Goal:** Identifier des profils similaires via clustering.

**Approach:**
1. Sélectionner les features pertinentes pour le clustering :
   - `Sentiment_Polarity`, `Negative_Word_Count`, `Is_Night`, `Word_Count`
2. Normaliser les features (StandardScaler recommandé)
3. Appliquer KMeans avec 3-4 clusters
4. Analyser chaque cluster :
   - Composition en termes d'étiquettes
   - Caractéristiques moyennes

**Why it matters:** 
- Créer des groupes de soutien homogènes
- Ex: "Anxieux nocturnes", "Déprimés expressifs", etc.

**Livrables attendus :**
- Clusters assignés
- Visualisation (scatter plot : Polarity vs Negative_Word_Count, coloré par cluster)
- Profil de chaque cluster
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
#### Bonus Task 4: Détecter les Cas Urgents
**Goal:** Système d'alerte pour les posts à risque suicidaire.

**Approach:**
1. Utiliser la feature `Has_Urgent_Keywords` créée précédemment
2. Filtrer les posts avec `Has_Urgent_Keywords == 1`
3. Afficher un tableau récapitulatif :
   - ID_Post
   - Texte
   - Etiquette
   - Horodatage
4. Recommandations automatiques : "Contact ligne d'écoute", "Alerte modérateur"

**Why it matters:** Priorité absolue - intervention immédiate pour sauver des vies.

**Livrables attendus :**
- Nombre de cas urgents détectés
- Tableau détaillé
- Système de recommandation automatique
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))
    
    cells.append(nbf.v4.new_markdown_cell("""
---

## 🎉 Projet Complété !

**Compétences acquises :**
- ✅ Feature Engineering NLP avancé
- ✅ Analyse de sentiment avec TextBlob
- ✅ Classification multi-classe
- ✅ Détection d'urgences en santé mentale

**Next Steps :**
- Testez des modèles de deep learning (LSTM, BERT)
- Intégrez un système de recommandation personnalisé
- Déployez une API de prédiction en temps réel
"""))
    
    nb['cells'] = cells
    nbf.write(nb, 'Projet_08_Intermediaire.ipynb')
    print("✅ Notebook intermédiaire généré : Projet_08_Intermediaire.ipynb")

if __name__ == "__main__":
    generer_notebook_intermediaire()
