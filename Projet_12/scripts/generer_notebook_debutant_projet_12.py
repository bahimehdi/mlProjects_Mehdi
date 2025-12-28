import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = "12"
    PROJECT_TITLE = "Détection de Fraude E-commerce"
    DATASET_NAME = "fraude_ecommerce.csv"
    TARGET_COL = "Est_Frauduleux"
    
    # --- CELLULES ---
    
    cells = []
    
    # 1. HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE}

## 🏁 Objectif : Le Cyber-Détective 🕵️‍♂️
La fraude coûte des milliards. Votre mission est de créer une IA capable de détecter les transactions suspectes.
Nous devons trouver les voleurs sans bloquer les clients honnêtes !

---

## 📋 Programme des 3 Sessions

### 🕵️‍♀️ SESSION 1 : Enquêteur de Données (45 min)
- **Part 1 :** Chargement et Nettoyage (Attention aux pays manquants !)
- **Part 2 :** Analyse Exploratoire (Y a-t-il des pays plus risqués ?)

### 🏗️ SESSION 2 : Architecte de Features (45 min)
- **Part 1 :** Feature Engineering (Créer des profils de risque)
- **Part 2 :** Préparation finale pour l'IA

### 🤖 SESSION 3 : Entraîneur d'IA (45 min)
- **Part 1 :** Entraînement du Modèle (Classification)
- **Part 2 :** Évaluation (Ne pas rater les fraudes !)
- **Part 3 :** Bonus (Calculer l'argent sauvé)

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
Les données de fraude sont souvent incomplètes. Vérifions !
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
> **💡 Tip:** Pour les `Pays` manquants, nous allons mettre "Inconnu". Pour le `Temps`, nous utiliserons la **Médiane**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Remplacer les valeurs manquantes
# 1. Pays_IP et Pays_Carte -> "Inconnu"
df['Pays_IP'].fillna("Inconnu", inplace=True)
df['Pays_Carte'].fillna("Inconnu", inplace=True)

# 2. Temps_Depuis_Derniere -> Médiane
median_time = df['Temps_Depuis_Derniere'].median()
df['Temps_Depuis_Derniere'].fillna(median_time, inplace=True)

print("✅ Nettoyage terminé !")
print(df.isnull().sum())
"""))

    # Part 3: EDA
    cells.append(nbf.v4.new_markdown_cell(f"""
## 🔍 Part 3: Exploratory Data Analysis (EDA)
Analysons notre cible : **{TARGET_COL}**.
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Distribution de la Fraude
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='{TARGET_COL}', palette='Reds')
plt.title("Fraude (1) vs Légitime (0)")
plt.show()

print("❓ Question : Y a-t-il beaucoup de fraudes par rapport aux transactions normales ?")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
Regardons si le **Montant** est un indice. Les fraudeurs volent-ils de gros montants ?
"""))

    cells.append(nbf.v4.new_code_cell(f"""
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='{TARGET_COL}', y='Amount', palette='Set2')
plt.title("Montant des Transactions : Fraude vs Légitime")
plt.show()
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🏗️ SESSION 2 : The Art of Feature Engineering
"""))

    # Recipe 2: Categories
    cells.append(nbf.v4.new_markdown_cell("""
### 🏷️ Recipe: Categories
Les colonnes `Pays` sont du texte. Transformons-les en chiffres avec le **One-Hot Encoding**.
"""))

    cells.append(nbf.v4.new_code_cell("""
# Encodage One-Hot pour les Pays
# Attention : Cela peut créer beaucoup de colonnes si beaucoup de pays !
df_encoded = pd.get_dummies(df, columns=['Pays_IP', 'Pays_Carte'], drop_first=True)

print("✅ Encodage terminé !")
df_encoded.head()
"""))

    # Recipe 6: Domain Specific
    cells.append(nbf.v4.new_markdown_cell("""
### 🎯 Recipe: Domain Specific (Risque Pays)
Si le Pays IP est différent du Pays Carte, est-ce louche ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# Créons une feature "Pays_Different"
# Note: On doit le faire AVANT l'encodage One-Hot, ou utiliser les colonnes originales si on les a gardées.
# Ici, on va le faire simplement :
df['Pays_Different'] = (df['Pays_IP'] != df['Pays_Carte']).astype(int)

# Regardons si c'est lié à la fraude
sns.barplot(data=df, x='Pays_Different', y='Est_Frauduleux')
plt.title("Probabilité de Fraude si Pays Différents")
plt.show()
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

# On doit s'assurer que df_encoded a bien la colonne Pays_Different
# Si on a fait l'encodage avant, il faut refaire l'encodage en incluant la nouvelle feature ou l'ajouter
# Pour simplifier, refaisons l'encodage propre :
df_final = pd.get_dummies(df.drop(['ID_Transaction'], axis=1), columns=['Pays_IP', 'Pays_Carte'], drop_first=True)

X = df_final.drop(['{TARGET_COL}'], axis=1)
y = df_final['{TARGET_COL}']

# Stratify est important car la fraude est rare !
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {{X_train.shape}}")
print(f"Test shape: {{X_test.shape}}")
"""))

    # Part 2: Training (Classification)
    cells.append(nbf.v4.new_markdown_cell("""
## 🏋️ Part 2: Training (Classification Déséquilibrée)
La fraude est rare (Déséquilibre de classe).
Si le modèle dit toujours "Pas Fraude", il aura 95% de réussite... mais il sera inutile !
Nous devons utiliser **SMOTE** pour créer des fausses fraudes d'entraînement.
"""))

    cells.append(nbf.v4.new_code_cell("""
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Avant SMOTE : {{y_train.value_counts()}}")
print(f"Après SMOTE : {{y_train_balanced.value_counts()}}")

# Entraînement
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

print("✅ Modèle entraîné !")
"""))

    # Part 3: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""
## 📊 Part 3: Evaluation
Ici, le **Recall** (Rappel) est roi. On veut attraper TOUTES les fraudes, quitte à vérifier quelques clients honnêtes.
"""))

    cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import classification_report, confusion_matrix, recall_score

y_pred = model.predict(X_test)

print("Rapport de Classification :")
print(classification_report(y_test, y_pred))

recall = recall_score(y_test, y_pred)
print(f"🎯 RECALL (Fraudes détectées) : {{recall:.2%}}")

# Matrice de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.title("Matrice de Confusion")
plt.ylabel("Réalité")
plt.xlabel("Prédiction")
plt.show()
"""))

    # Part 4: Bonus
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### 💰 Bonus Task 1: L'Argent Sauvé
Supposons qu'une fraude détectée est de l'argent sauvé.
Mais une fausse alerte (bloquer un client honnête) coûte 10€ de gestion.
Combien avons-nous gagné ?
"""))

    cells.append(nbf.v4.new_code_cell(f"""
# Récupérons les montants du test set
# Attention : X_test a été mélangé, il faut retrouver les indices
test_indices = X_test.index
amounts = df.loc[test_indices, 'Amount']

# Créons un DataFrame de résultats
results = pd.DataFrame({{
    'Vrai': y_test,
    'Pred': y_pred,
    'Montant': amounts
}})

# Calculons les gains
# Vrai Positif (Fraude stoppée) = + Montant
# Faux Positif (Client bloqué) = - 10€
# Faux Négatif (Fraude ratée) = - Montant

argent_sauve = 0
for index, row in results.iterrows():
    if row['Vrai'] == 1 and row['Pred'] == 1: # Fraude stoppée
        argent_sauve += row['Montant']
    elif row['Vrai'] == 0 and row['Pred'] == 1: # Fausse alerte
        argent_sauve -= 10
    elif row['Vrai'] == 1 and row['Pred'] == 0: # Fraude ratée
        argent_sauve -= row['Montant']

print(f"💰 Bilan financier de l'IA : {{argent_sauve:,.2f}} €")
"""))

    nb['cells'] = cells

    # Save notebook
    with open(f"Projet_{PROJECT_NUMBER}_Fraude_Debutant.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook()
