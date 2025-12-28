import nbformat as nbf

def generer_notebook_intermediaire():
    nb = nbf.v4.new_notebook()
    
    # Titre
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 🚗 PROJET 18 : POINTS CHAUDS DE COVOITURAGE (Niveau Intermédiaire) 📍

**Objectif :** Construire un modèle de régression pour prédire la demande de courses par zone et par heure.

---

## 📅 STRUCTURE DU PROJET

### 📋 SESSION 1 : Analyse Exploratoire & Nettoyage
- Chargement et conversion de l'horodatage
- Gestion des valeurs manquantes
- Analyse de la demande par zone, météo, et événements

### 📋 SESSION 2 : Feature Engineering
- Extraction de features temporelles (Heure, JourSemaine)
- Encodage des variables catégorielles
- Création de features métier (Supply_Demand_Ratio, lag features)

### 📋 SESSION 3 : Modélisation & Évaluation
- Entraînement d'un modèle de Régression
- Évaluation (MAE, RMSE, R²)
- Feature importance

### 🎁 SESSION 3 - PART 4 : Tâches Bonus
- Zones de tarification dynamique
- Recommandation de relocalisation
- Clustering par type de zone

---
"""))

    # SESSION 1
    nb.cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : DATA EXPLORATION & CLEANING
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.1 : Chargement et Inspection
**Objectif :** Charger `covoiturage.csv` et convertir l'horodatage.
**Livrables :**
- `df.head()`, `df.info()`
- Variable cible identifiée : `Demandes`
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.2 : Nettoyage
**Objectif :** Gérer les valeurs manquantes dans `Chauffeurs_Actifs` et `Meteo`.
**Approches recommandées :**
- Suppression (si < 10%)
- Imputation (médiane pour Chauffeurs_Actifs, mode pour Meteo)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.3 : Analyse Exploratoire
**Objectif :** Comprendre les patterns de demande.
**Visualisations attendues :**
- Demande moyenne par `ID_Zone`
- Demande par `Meteo`
- Demande par `Heure` (time series)

**Conseil :** Créez un graphique de demande par heure pour identifier les heures de pointe.
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
**Objectif :** Extraire des informations de `Horodatage`.
**Features à créer :**
- `Heure`, `JourSemaine`, `Mois`
- `Is_Weekend`
- `Is_Rush_Hour` (7-9h ou 17-19h)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Encodage des Catégories (Recipe 2)
**Objectif :** Transformer `Meteo` en format numérique.
**Méthode :** One-Hot Encoding (`pd.get_dummies`).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.3 : Features Métier (Recipe 6)
**Objectif :** Créer des variables spécifiques au covoiturage.

**Features recommandées :**
1. **Supply_Demand_Ratio** : `Chauffeurs_Actifs / (Demandes + 1)`
   - Interprétation : < 0.5 = Pénurie, > 2 = Surplus

2. **Lag Features** : Demande à l'heure précédente (par zone).
   - Formule : `df.groupby('ID_Zone')['Demandes'].shift(1)`
   - Pourquoi : La demande actuelle dépend souvent de la tendance récente.

**Conseil :** Triez par `Horodatage` avant de créer les lag features.
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
**Target :** `Demandes`
**Split :** 80% Train, 20% Test

**Important :** Supprimez `Horodatage` et les lignes avec NaN (créées par lag).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Entraînement
**Modèle recommandé :** RandomForestRegressor
**Alternative :** GradientBoostingRegressor (plus précis, plus lent).
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.3 : Évaluation
**Métriques clés :**
- **MAE** : Erreur moyenne en nombre de courses.
- **R²** : Pourcentage de variance expliquée.

**Feature Importance :** Identifiez les variables les plus influentes.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici
"""))

    # PART 4 BONUS
    nb.cells.append(nbf.v4.new_markdown_cell("""
## 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: Zones de Tarification Dynamique
**Goal:** Identifier les moments où appliquer une surcharge (surge pricing).

**Approche :**
1. Calculer `Supply_Demand_Ratio` pour chaque ligne
2. Définir seuil : Ratio < 0.5 = Pénurie → Surge Pricing
3. Grouper par `ID_Zone` et compter les heures de pénurie

**Deliverable :** Top 5 des zones avec le plus d'heures de pénurie.

### Bonus Task 2: Recommandation de Relocalisation
**Goal:** Suggérer aux chauffeurs de se déplacer des zones en surplus vers celles en déficit.

**Approche :**
1. Pour l'heure actuelle (ou dernière heure du dataset) :
   - Surplus : Ratio > 2
   - Déficit : Ratio < 0.5
2. Créer une table de recommandations : "Déplacer de Zone X vers Zone Y"

**Deliverable :** Table de relocalisation.

### Bonus Task 4: Clustering de Zones
**Goal:** Classifier les zones : Résidentielle, Affaires, ou Vie Nocturne.

**Approche :**
1. Calculer la demande moyenne par zone et par heure
2. Identifier l'heure de pic pour chaque zone
3. Classification :
   - Pic 7-9h ou 18-20h : Résidentielle
   - Pic 9h-18h : Affaires
   - Pic 22h-2h : Vie Nocturne

**Deliverable :** Classification des 10 zones.

### Bonus Task 3: Prévision pour le Réveillon (Optionnel)
**Goal:** Prédire la demande pour le 31 décembre à minuit.

**Approche :**
1. Créer une ligne fictive : Date=31/12, Heure=0, Evenements=1
2. Remplir les autres features avec des moyennes
3. Utiliser `model.predict()`

**Deliverable :** Prédiction de la demande par zone pour le réveillon.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Votre code ici pour les bonus
"""))

    # Sauvegarde
    with open('Projet_18_Covoiturage_Intermediaire.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    generer_notebook_intermediaire()
