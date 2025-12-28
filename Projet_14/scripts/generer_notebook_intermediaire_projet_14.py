import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CONFIGURATION ---
    PROJECT_NUMBER = 14
    PROJECT_TITLE = "Juste Valeur de Voiture d'Occasion"
    DATASET_NAME = "voitures_occasion.csv"
    
    # --- CELLULES ---
    cells = []
    
    # HEADER
    cells.append(nbf.v4.new_markdown_cell(f"""
# 🎓 PROJET {PROJECT_NUMBER} : {PROJECT_TITLE} (Version Intermédiaire)

**Objectif :** Développer un modèle de régression pour estimer le prix des voitures d'occasion et identifier les opportunités d'achat.

---
"""))

    # --- SESSION 1 ---
    cells.append(nbf.v4.new_markdown_cell("""
# 📋 SESSION 1 : From Raw Data to Clean Insights

### Étape 1.1 : Chargement et Inspection
**Objectif :** Charger `voitures_occasion.csv` et identifier les types de données.
**Livrables :**
- DataFrame chargé
- Résumé des infos (`info()`, `describe()`)
- Vérification des doublons
"""))
    
    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 1.2 : Analyse Exploratoire (EDA)
**Objectif :** Comprendre les facteurs influençant le prix.
**Approches recommandées :**
- Histogramme de la variable cible (`Price`)
- Boxplots : Prix par Marque (`Brand`), Prix par Carburant (`Fuel`)
- Scatterplot : Prix vs Kilométrage
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 2 ---
    cells.append(nbf.v4.new_markdown_cell("""
---
# 📋 SESSION 2 : Feature Engineering

### Étape 2.1 : Création de Features Temporelles
**Objectif :** Transformer l'année en âge.
**Conseil :** `Age = Année_Actuelle - Year`
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.2 : Features Mathématiques
**Objectif :** Créer un indicateur d'intensité d'usage.
**Idée :** `Km_par_an = Kilometrage / Age`
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 2.3 : Encodage des Catégories
**Objectif :** Convertir `Brand` et `Fuel` en format numérique.
**Méthodes :**
- `pd.get_dummies()` (One-Hot Encoding)
- `LabelEncoder` (si ordre hiérarchique, moins recommandé ici)
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- SESSION 3 ---
    cells.append(nbf.v4.new_markdown_cell("""
---
# 📋 SESSION 3 : Modélisation & Décision

### Étape 3.1 : Préparation et Split
**Objectif :** Séparer Features (X) et Target (y), puis Train/Test sets (80/20).
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.2 : Entraînement (Régression)
**Modèle recommandé :** `RandomForestRegressor`
**Pourquoi :** Gère bien les relations non-linéaires et les interactions entre variables.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Étape 3.3 : Évaluation
**Métriques attendues :**
- **MAE** (Erreur absolue moyenne en €)
- **RMSE** (Pénalise les grosses erreurs)
- **R²** (Coefficient de détermination)

**Visualisation :** Tracez un graphique "Prix Réel vs Prix Prédit".
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # --- PART 4: BONUS ---
    cells.append(nbf.v4.new_markdown_cell("""
---
# 🎁 Part 4: Going Further (Bonus)

### Bonus Task 1: Algorithme de "Bonne Affaire" 💎

**Goal:** Créer une colonne `Verdict` pour chaque voiture du test set.

**Approach:**
1. Calculez l'écart en % : `(Prix_Réel - Prix_Prédit) / Prix_Prédit`
2. Définissez des seuils :
   - < -10% : "Bonne Affaire" (Sous-cotée)
   - > +10% : "Trop Cher" (Sur-cotée)
   - Entre les deux : "Juste Prix"

**Deliverable:** Un DataFrame avec les colonnes `Prix_Reel`, `Prix_Predit`, `Ecart_Pct`, `Verdict`.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    cells.append(nbf.v4.new_markdown_cell("""
### Bonus Task 2: Analyse de la Dépréciation 📉

**Goal:** Quelle marque perd le plus de valeur ?

**Approach:**
1. Simulez le prix de toutes les voitures dans 5 ans (Age + 5, Km + 75000).
2. Calculez la perte de valeur (`Prix_Actuel - Prix_Futur`).
3. Groupez par `Brand` et calculez la perte moyenne.

**Deliverable:** Un bar chart montrant la perte de valeur moyenne par marque.
"""))

    cells.append(nbf.v4.new_code_cell("# Votre code ici"))

    # SAVE
    with open('notebook_intermediaire_projet_14.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("✅ Notebook Intermédiaire généré avec succès !")

if __name__ == "__main__":
    create_notebook()
