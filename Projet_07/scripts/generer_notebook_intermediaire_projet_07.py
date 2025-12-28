import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # --- Titre et Introduction ---
    cells.append(nbf.v4.new_markdown_cell("""
# 🥗 Projet 7 : Prévision du Gaspillage Alimentaire
## Version Intermédiaire - "Tu explores, j'oriente"

---

### 🎯 L'Objectif de ce Projet

Réduire le gaspillage alimentaire en prédisant précisément les ventes futures pour optimiser les commandes. Vous devrez explorer les données, identifier les patterns et construire un modèle de prévision robuste.

**Ce que vous allez maîtriser :**
- 📊 Analyse exploratoire approfondie de données temporelles
- 🔧 Feature engineering avancé (features temporelles, interactions)
- 🤖 Optimisation de modèle et sélection de features
- 📈 Validation et interprétation des résultats

---

> **💡 Format de ce notebook :**
> - **Consignes claires** : Chaque section indique ce qu'il faut faire
> - **Code à compléter** : Des TODO pour vous guider
> - **Liberté d'exploration** : Essayez différentes approches !

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights (45 min)

## Part 1: The Setup (5 min)

**Consigne :** Importez les bibliothèques nécessaires et chargez `gaspillage_alimentaire.csv`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Importer pandas, numpy, matplotlib, seaborn, datetime

# TODO: Charger le dataset

# TODO: Afficher les premières lignes et les informations
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: The Sanity Check (15 min)

**Consigne :** Analysez les valeurs manquantes et traitez-les intelligemment.

**Approche recommandée :**
1. Identifiez les colonnes avec valeurs manquantes
2. Pour `Price` et `Discount`, utilisez des méthodes par groupe (par produit)
3. Vérifiez qu'aucune valeur manquante ne reste
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Analyser les valeurs manquantes

# TODO: Remplir Price avec la médiane par ID_Produit

# TODO: Remplir Discount avec la médiane globale

# TODO: Vérifier qu'il n'y a plus de NaN
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Exploratory Data Analysis (25 min)

**Consigne :** Créez 4 visualisations pour comprendre les données :

1. **Distribution de la cible** (`Unites_Vendues`)
2. **Ventes par produit** (boxplot ou violin plot)
3. **Impact du discount** (scatterplot avec couleurs par produit)
4. **Évolution temporelle** (ventes totales par jour)

**Questions clés à répondre :**
- Y a-t-il des valeurs aberrantes (outliers) ?
- Quel produit est le plus stable/variable ?
- Les promotions augmentent-elles vraiment les ventes ?
- Voyez-vous une saisonnalité ou une tendance ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Visualisation 1 - Distribution des ventes

# TODO: Visualisation 2 - Ventes par produit

# TODO: Visualisation 3 - Impact du discount

# TODO: Visualisation 4 - Évolution temporelle
"""))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : The Art of Feature Engineering (45 min)

## Part 1: The Concept (5 min)

Les ventes alimentaires sont influencées par :
- **Le temps** : jour de la semaine, mois, saison
- **La fraîcheur** : jours avant expiration
- **Les promotions** : discount, prix effectif
- **Le produit** : type de produit (encodage)

Votre mission : créer des features pertinentes pour capturer ces patterns.

## Part 2: The Lab - Choose Your Recipe (35 min)

### 📅 Recipe 1: Time-Based Features (15 min)

**Consigne :** À partir des colonnes `Date` et `Date_Expiration`, créez :

1. **Features de base :**
   - `Jour_Semaine` (0-6)
   - `Mois` (1-12)
   - `Jour_Mois` (1-31)
   - `Est_Weekend` (binaire)

2. **Features avancées :**
   - `Jours_Avant_Expiration` (Date_Expiration - Date)
   - `Semaine_Annee` (semaine ISO)
   - `Expire_Bientot` (1 si expire dans 2 jours ou moins, sinon 0)
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Convertir Date et Date_Expiration en datetime

# TODO: Créer les features temporelles de base

# TODO: Créer Jours_Avant_Expiration

# TODO: Créer features avancées (Semaine_Annee, Expire_Bientot)

# TODO: Vérifier les nouvelles colonnes
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### 🏷️ Recipe 2: Categories (10 min)

**Consigne :** Encodez `ID_Produit` avec **One-Hot Encoding**.

**Astuce :** Utilisez `pd.get_dummies()` avec `drop_first=True` pour éviter la multicolinéarité.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: One-Hot Encoding de ID_Produit

# TODO: Afficher les nouvelles colonnes créées
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### ➗ Recipe 4: Math Magic - Interaction Features (10 min)

**Consigne :** Créez des features d'interaction :

1. `Prix_Effectif` = Price × (1 - Discount)
2. `Ratio_Prix_Discount` = Discount / Price (normalisé)
3. `Promo_Forte` = 1 si Discount > 0.3, sinon 0
4. `Urgence_Vente` = Promo_Forte × Expire_Bientot (produit en promo ET proche expiration)
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer Prix_Effectif

# TODO: Créer Ratio_Prix_Discount

# TODO: Créer Promo_Forte

# TODO: Créer Urgence_Vente
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Final Prep (5 min)

**Consigne :** Préparez X et y en supprimant les colonnes non pertinentes (dates brutes, ID textuels).
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Identifier les colonnes à supprimer

# TODO: Créer X (features) et y (Unites_Vendues)

# TODO: Vérifier les dimensions et les colonnes
"""))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : Building & Trusting Your Model (45 min)

## Part 1: The Split (5 min)

**Consigne :** Divisez les données en Train/Test (80/20) avec `random_state=42`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Import train_test_split

# TODO: Créer X_train, X_test, y_train, y_test

# TODO: Afficher les dimensions
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 2: Training (15 min)

### Étape 1 : Modèle de Base

**Consigne :** Entraînez un `RandomForestRegressor` avec les paramètres par défaut + `random_state=42`.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Importer RandomForestRegressor

# TODO: Créer le modèle

# TODO: Entraîner sur X_train, y_train
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2 : Optimisation des Hyperparamètres (Optionnel mais recommandé)

**Consigne :** Testez différents paramètres pour améliorer le modèle :
- `n_estimators` : [50, 100, 200]
- `max_depth` : [10, 20, None]
- `min_samples_split` : [2, 5]

**Astuce :** Utilisez une simple boucle for et comparez les MAE.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO (Optionnel): Tester différents hyperparamètres

# Exemple:
# best_mae = float('inf')
# best_params = {}
# for n_est in [50, 100, 200]:
#     for max_d in [10, 20, None]:
#         model_temp = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, random_state=42)
#         model_temp.fit(X_train, y_train)
#         y_pred_temp = model_temp.predict(X_test)
#         mae_temp = mean_absolute_error(y_test, y_pred_temp)
#         if mae_temp < best_mae:
#             best_mae = mae_temp
#             best_params = {'n_estimators': n_est, 'max_depth': max_d}
# print(f"Meilleurs paramètres: {best_params} avec MAE={best_mae:.2f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""
## Part 3: Evaluation (25 min)

### Étape 1 : Métriques

**Consigne :** Calculez et affichez :
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)  
- R² Score
- MAPE (Mean Absolute Percentage Error) : `mean(|y_true - y_pred| / y_true) × 100`
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Faire les prédictions sur X_test

# TODO: Calculer MAE, RMSE, R²

# TODO: Calculer MAPE

# TODO: Afficher toutes les métriques
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2 : Visualisation des Performances

**Consigne :** Créez 2 graphiques :

1. **Scatter Plot** : Prédictions vs Valeurs Réelles (avec ligne y=x idéale)
2. **Residual Plot** : Erreurs (y_test - y_pred) vs Valeurs Prédites

**Interprétation :** 
- Si les points sont proches de la ligne y=x → bon modèle
- Si les résidus sont aléatoires autour de 0 → pas de biais
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Graphique 1 - Prédictions vs Réel

# TODO: Graphique 2 - Residual Plot
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3 : Feature Importance

**Consigne :** Affichez les 15 features les plus importantes (barplot horizontal).

**Analyse :** Quelles features dominent ? Est-ce cohérent avec votre compréhension du domaine ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Extraire et afficher feature importances

# TODO: Créer un barplot des top 15
"""))

    # --- PART 4 BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus - 30-45 mins)

### Bonus Task 1: Analyse des Erreurs

**Goal:** Identifier sur quels produits/situations le modèle performe mal.

**Approche:**
1. Créer un DataFrame avec `y_test`, `y_pred`, erreur absolue
2. Ajouter les features originales (produit, discount, etc.)
3. Analyser :
   - Sur quel produit l'erreur moyenne est la plus haute ?
   - Les jours avec forte promo sont-ils moins bien prédits ?
   - Les produits proches expiration posent-ils problème ?
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Analyse des erreurs par produit et par contexte
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Prédictions pour le Mois Prochain

**Goal:** Utiliser le modèle pour générer des prévisions réalistes.

**Approche:**
1. Créer un dataset fictif pour les 30 prochains jours
2. Remplir les features (dates, produits, prix moyens, discount moyen)
3. Faire des prédictions
4. Visualiser les prévisions par produit

**Livrable:** Tableau des ventes prévues par produit pour optimiser les commandes.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Créer un dataset pour le mois prochain

# TODO: Générer les prédictions

# TODO: Visualiser et résumer
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 3: Modèle de Détection d'Anomalies

**Goal:** Au lieu de prédire les ventes, détecter les jours "anormaux" où les ventes sont bizarrement hautes ou basses.

**Approche:**
1. Calculer les résidus = y_test - y_pred
2. Marquer comme anomalie si |résidu| > 2 × std(résidus)
3. Investiguer ces jours (date, produit, discount, contexte)

**Application:** Alerter le manager quand quelque chose d'inhabituel se passe.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Détection d'anomalies basée sur les résidus

# TODO: Analyser les anomalies détectées
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 4: Comparaison de Modèles

**Goal:** Comparer `RandomForestRegressor` avec d'autres algorithmes.

**Modèles à tester:**
- Linear Regression (baseline simple)
- Gradient Boosting Regressor
- XGBoost (si installé)

**Livrable:** Tableau comparatif des MAE/R² pour chaque modèle.
"""))

    cells.append(nbf.v4.new_code_cell("""
# TODO: Entraîner et comparer plusieurs modèles

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor

# TODO: Créer un tableau comparatif
"""))

    nb['cells'] = cells
    
    with open('Projet_07_Intermediaire.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Notebook Intermédiaire généré : Projet_07_Intermediaire.ipynb")

if __name__ == "__main__":
    generer_notebook_intermediaire()
