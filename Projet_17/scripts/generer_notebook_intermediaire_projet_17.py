import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    # Titre
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 🛒 PROJET 17 : OPTIMISEUR DE STOCK PÉRISSABLE (Niveau Intermédiaire) 🥬

**Objectif :** Construire un modèle de régression pour prédire les ventes quotidiennes de produits périssables et optimiser les commandes d'inventaire.

---

## 📅 STRUCTURE DU PROJET

### 📋 SESSION 1 : Analyse Exploratoire & Nettoyage
- Chargement et inspection des types
- Gestion des valeurs manquantes
- Analyse des ventes par produit, météo, et jours fériés

### 📋 SESSION 2 : Feature Engineering
- Extraction de features temporelles (jour de la semaine, saisonnalité)
- Encodage des variables catégorielles (Item, Meteo)
- Création de features métier (moyennes mobiles, volatilité)

### 📋 SESSION 3 : Modélisation & Évaluation
- Entraînement d'un modèle de Régression
- Évaluation (MAE, RMSE, R²)
- Analyse de l'importance des features

### 🎁 SESSION 3 - PART 4 : Tâches Bonus
- Calcul de la quantité de commande optimale
- Détection des articles à rotation lente
- Identification des ruptures de stock

---
"""))

    # SESSION 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : DATA EXPLORATION & CLEANING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.1 : Chargement et Inspection
**Objectif :** Charger `stock_perissable.csv` et comprendre la structure.
**Livrables :**
- `df.head()`, `df.info()`
- Identification de la variable cible : `Sold`
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.2 : Nettoyage des Données
**Objectif :** Gérer les valeurs manquantes dans `Stock_Initial` et `Meteo`.
**Approches recommandées :**
- **Suppression** : Si < 10% de lignes manquantes (simple, pas de biais).
- **Imputation** : Médiane pour `Stock_Initial`, mode pour `Meteo` (conserve les données).

**Livrables :**
- Dataset nettoyé sans valeurs manquantes
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.3 : Analyse Exploratoire (EDA)
**Objectif :** Identifier les patterns de vente.
**Visualisations attendues :**
- Ventes moyennes par `Item`
- Ventes par `Meteo` (groupées par Item)
- Évolution temporelle des ventes (Time series plot)

**Conseil :** Utilisez `sns.barplot` et `plt.plot` pour les tendances.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # SESSION 2
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 2 : FEATURE ENGINEERING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.1 : Features Temporelles (Recipe 1)
**Objectif :** Extraire des informations de `Date`.
**Features à créer :**
- `Jour`, `Mois`, `JourSemaine` (0=Lundi)
- `Is_Weekend` (booléen)
- `Trimestre` (optionnel)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Encodage des Catégories (Recipe 2)
**Objectif :** Transformer `Item` et `Meteo` en format numérique.
**Méthode :** One-Hot Encoding (`pd.get_dummies`).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.3 : Features Métier (Recipe 6)
**Objectif :** Créer des variables spécifiques au domaine de l'inventaire.

**Features recommandées :**
1. **Moyenne Mobile (MA7)** : Tendance des 7 derniers jours de ventes.
   - Formule : `df['Sold'].shift(1).rolling(7).mean()`
   - Pourquoi : Capture la tendance récente (si les ventes augmentent).

2. **Volatilité de la demande** : Écart-type des 7 derniers jours.
   - Formule : `df['Sold'].shift(1).rolling(7).std()`
   - Pourquoi : Aide à déterminer la marge de sécurité pour les commandes.

**Conseil :** Utilisez `shift(1)` pour éviter le data leakage (ne pas utiliser la vente du jour même pour prédire).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # SESSION 3
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 3 : MODELING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.1 : Préparation et Split
**Objectif :** Séparer Features (X) et Target (y).
**Target :** `Sold`
**Split :** 80% Train, 20% Test

**Important :** Supprimez les lignes avec NaN créées par `rolling()`.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Entraînement (Régression)
**Modèle recommandé :** RandomForestRegressor
**Pourquoi ?** Robuste aux interactions non-linéaires et ne nécessite pas de normalisation.

**Alternative :** GradientBoostingRegressor (souvent plus précis mais plus lent).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.3 : Évaluation
**Métriques clés :**
- **MAE** (Mean Absolute Error) : Erreur moyenne en unités vendues.
- **R²** : Pourcentage de variance expliquée.

**Feature Importance :** Identifiez les variables les plus influentes.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # PART 4 BONUS
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: Quantité de Commande Optimale
**Goal:** Calculer une recommandation de commande incluant une marge de sécurité.

**Approche :**
1. Prédire les ventes moyennes : `y_pred`
2. Calculer l'écart-type des erreurs de prédiction : `std_error`
3. Formule : `Commande = y_pred + k * std_error` (k=1.5 pour 90% de confiance)

**Deliverables:**
- Fonction `calculate_order_quantity(prediction, std_error, confidence=1.5)`
- Exemple de calcul pour un produit spécifique

### Bonus Task 2: Articles à Rotation Lente
**Goal:** Identifier les produits avec des ventes faibles (candidats pour soldes).

**Approche :**
1. Grouper par `Item` et calculer la moyenne des ventes
2. Définir un seuil (ex: < 30 unités/jour)
3. Lister les produits en dessous du seuil

**Deliverable:** Liste des articles à rotation lente

### Bonus Task 3: Détection des Ruptures de Stock
**Goal:** Trouver les jours où le stock était insuffisant.

**Approche :**
- Si `Stock_Initial` < `Sold`, c'est une rupture avérée
- Si `Stock_Initial` ≈ `Sold` (±10%), c'est une rupture probable

**Deliverable:** DataFrame des ruptures avec Date, Item, Stock_Initial, Sold

### Bonus Task 4: Clustering par Volatilité de Demande (Avancé)
**Goal:** Grouper les produits par stabilité de la demande.

**Approche :**
1. Calculer le coefficient de variation (CV) par produit : `std / mean`
2. Clusters :
   - CV < 0.2 : Demande stable
   - 0.2 < CV < 0.5 : Demande modérée
   - CV > 0.5 : Demande volatile

**Deliverable:** Classification des produits et recommandations de gestion (ex: moins de marge pour demande stable, plus pour volatile)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici pour les bonus
"""))

    # Sauvegarde
    with open('Projet_17_Stock_Perissable_Intermediaire.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_intermediaire()
