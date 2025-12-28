import nbformat as nbf

def generer_notebook_debutant():
    nb = nbf.v4.new_notebook()
    
    # Titre
    nb.cells.append(nbf.v4.new_markdown_cell("""
# PROJET 19 : DETECTION DE FRAUDE CARTE DE CREDIT

Bienvenue dans le projet le plus critique pour les banques !

**Le Probleme :** Les fraudeurs volent des milliards chaque annee. Le defi ? Ils representent moins de 3% des transactions (Aiguille dans une Botte de Foin).

**Votre Mission :** Detecter TOUTES les fraudes (Rappel Eleve) meme si ca signifie bloquer quelques transactions legitimes. Manquer une fraude coute 500, bloquer une transaction legitime coute 10. 

---

## VOTRE PROGRAMME

### SESSION 1 : From Raw Data to Clean Insights (45 min)
- **Part 1: The Setup** - Charger les donnees de transactions
- **Part 2: The Sanity Check** - Nettoyer les donnees
- **Part 3: Exploratory Data Analysis** - Analyser le desequilibre

### SESSION 2 : The Art of Feature Engineering (45 min)
- **Part 1: The Concept** - Comprendre les features de fraude
- **Part 2: The Lab** - Creer des features metier
- **Part 3: Final Prep** - Preparer pour le modele

### SESSION 3 : Building & Trusting Your Model (45 min)
- **Part 1: The Split** - Separer train/test
- **Part 2: Training** - SMOTE + RandomForest
- **Part 3: Evaluation** - Rappel prioritaire
- **Part 4: Going Further (BONUS)** - Analyse Cout-Benefice

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
df = pd.read_csv('fraude_carte_credit.csv')

print("Apercu des donnees :")
display(df.head(10))

print("\\nInfos techniques :")
df.info()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
> **Tip:** Le dataset contient :
> - **Class** : CIBLE (0=Legitime, 1=Fraude)
> - **Amount** : Montant de la transaction
> - **Transaction_Type** : Online/POS/ATM
> - **Previous_Fraud_Attempts** : Nombre de fraudes anterieures
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
# Remplir Location_Distance par la mediane
df['Location_Distance'].fillna(df['Location_Distance'].median(), inplace=True)

print(f"Nouvelles dimensions : {df.shape}")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 2. Analyser le desequilibre de classe
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
print("Distribution des classes :")
print(df['Class'].value_counts())
print(f"\\nPourcentage de fraudes : {df['Class'].mean() * 100:.2f}%")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (20 min)

### Visualiser le desequilibre
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Class')
plt.title('Distribution des Classes (0=Legitime, 1=Fraude)')
plt.show()
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
Question : Quelle est la classe minoritaire ?

### Fraude par montant
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Class', y='Amount')
plt.title('Montant par Classe')
plt.show()
"""))

    # SESSION 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
# SESSION 2 : THE ART OF FEATURE ENGINEERING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 1: The Concept (10 min)

Pour detecter la fraude :
- **Transactions nocturnes** (0h-5h) sont suspectes
- **Montants inhabituels** (tres eleves ou en dehors de la norme)
- **Tentatives de fraude anterieures** (red flag)
"""))

    # Part 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Lab (30 min)

### Recipe 2: Categories
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df = pd.get_dummies(df, columns=['Transaction_Type'], prefix='TxType')

print("Encodage termine !")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Recipe 6: Domain-Specific Features

#### Feature 1: Heure Inhabituelle
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df['Is_Night'] = ((df['Time_Hour'] >= 0) & (df['Time_Hour'] <= 5)).astype(int)

print("Feature Is_Night creee !")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
#### Feature 2: Montant Deviation (z-score)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
mean_amount = df['Amount'].mean()
std_amount = df['Amount'].std()

df['Amount_Zscore'] = (df['Amount'] - mean_amount) / std_amount

print("Z-score cree !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
df_model = df.dropna()

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

X = df_model.drop('Class', axis=1)
y = df_model['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
"""))

    # Part 2  
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training avec SMOTE (15 min)

> **Warning:** Les fraudes representent ~3%. SMOTE va creer des exemples synthetiques pour equilibrer.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Appliquer SMOTE
print("Distribution avant SMOTE :")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\\nDistribution apres SMOTE :")
print(y_train_balanced.value_counts())

# Entrainer le modele
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_balanced, y_train_balanced)

print("\\nModele entraine !")
"""))

    # Part 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation - RAPPEL PRIORITAIRE (20 min)

> **Important:** Notre metrique principale est le **RECALL** (classe fraude). Manquer une fraude coute 500.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
from sklearn.metrics import recall_score, classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluer
recall = recall_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"RECALL (Fraude) : {recall:.2%}")
print(f"ROC-AUC : {auc:.3f}")
print("\\nRapport complet :")
print(classification_report(y_test, y_pred, target_names=['Legitime', 'Fraude']))
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion (Focus: Faux Negatifs)')
plt.ylabel('Verite')
plt.xlabel('Prediction')
plt.show()
"""))

    # Part 4 Bonus
    nb.cells.append(nbf.v4.new_markdown_cell("""
## Part 4: Going Further (Bonus - 15-30 mins)

### Bonus Task 1: Analyse Cout-Benefice

**Goal:** Trouver le seuil de probabilite optimal qui minimise les couts.

**Couts:**
- Faux Positif (bloquer transaction legitime) = 10
- Faux Negatif (manquer une fraude) = 500

**Approach:**
1. Tester differents seuils (0.1 to 0.9)
2. Calculer FP et FN pour chaque seuil
3. Calculer cout total = FP * 10 + FN * 500
4. Choisir le seuil avec cout minimum
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Analyse de seuil
thresholds = np.arange(0.1, 0.9, 0.05)
costs = []

for threshold in thresholds:
    y_pred_custom = (y_pred_proba >= threshold).astype(int)
    
    FP = ((y_pred_custom == 1) & (y_test == 0)).sum()
    FN = ((y_pred_custom == 0) & (y_test == 1)).sum()
    
    total_cost = FP * 10 + FN * 500
    costs.append(total_cost)
    
    print(f"Seuil {threshold:.2f}: FP={FP}, FN={FN}, Cout=${total_cost}")

# Seuil optimal
optimal_threshold = thresholds[np.argmin(costs)]
print(f"\\nSeuil optimal : {optimal_threshold:.2f}")
print(f"Cout minimum : ${min(costs)}")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Systeme de Scoring en Temps Reel

**Goal:** Creer un systeme qui score chaque transaction en temps reel.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
def fraud_score(transaction):
    # transaction est un dictionnaire
    # Retourne la probabilite de fraude
    
    # TODO: Preparer les features comme dans le training
    # features = prepare_features(transaction)
    # prob = model.predict_proba([features])[0, 1]
    # return prob
    
    pass

# Exemple
# new_transaction = {'Amount': 500, 'Time_Hour': 3, 'Transaction_Type': 'Online', ...}
# score = fraud_score(new_transaction)
# if score > optimal_threshold:
#     print("ALERTE FRAUDE !")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Analyse des Patterns de Fraude

**Goal:** Identifier les caracteristiques communes des fraudes.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Analyser les fraudes
fraud_df = df_model[df_model['Class'] == 1]
legit_df = df_model[df_model['Class'] == 0]

print("Patterns de fraude :")
print(f"Montant moyen (Fraude) : {fraud_df['Amount'].mean():.2f}")
print(f"Montant moyen (Legitime) : {legit_df['Amount'].mean():.2f}")

print(f"\\nTransactions nocturnes (Fraude) : {fraud_df['Is_Night'].mean() * 100:.1f}%")
print(f"Transactions nocturnes (Legitime) : {legit_df['Is_Night'].mean() * 100:.1f}%")
"""))

    # Sauvegarde
    with open('Projet_19_Fraude_Debutant.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_debutant()
