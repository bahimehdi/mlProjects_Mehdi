# 🎯 Guide de Notation et Réponses Attendues

Ce document fournit les réponses attendues pour chaque question technique posée lors des soutenances. Chaque réponse est notée sur une échelle de 0 à 5.

## 📊 Barème de Notation (0-5)

| Note | Description | Critères |
| :--- | :--- | :--- |
| **0** | **Non Répondu / Hors Sujet** | L'étudiant ne sait pas répondre ou donne une réponse incohérente. |
| **1** | **Très Insuffisant** | Réponse vague, manque de vocabulaire technique, incompréhension du concept. |
| **2** | **Insuffisant** | Comprend vaguement le concept mais incapable d'expliquer l'implémentation ou le "pourquoi". |
| **3** | **Acceptable (Moyen)** | Réponse correcte mais basique. Sait "comment" mais pas forcément "pourquoi" (ex: "J'ai utilisé ça parce que c'était dans le cours"). |
| **4** | **Bien** | Bonne compréhension technique et théorique. Explique clairement la démarche. |
| **5** | **Excellent** | Maîtrise totale. Justifie le choix par rapport aux données, critique ses propres résultats, propose des améliorations. |

---

## 📋 Workflow Data Science Attendu (Tous Projets)

**Ordre correct des opérations :**
1. **Explorer** (EDA Initial)
2. **Nettoyer** (Gérer NaNs, valeurs aberrantes, types)
3. **Split train/test** (Toujours avant le preprocessing avancé)
4. **Préprocesser le train set** (Fit & Transform)
5. **Appliquer au test set** (Transform uniquement, utiliser les stats du train)
6. **SMOTE** (Seulement sur le Train Set transformé)
7. **Modéliser & Évaluer**

**Réponse Idéale pour les questions Workflow (5/5) :** "J'ai tout nettoyé d'abord, puis splitté. J'ai ensuite calculé mes scalers/imputers sur le train set uniquement pour éviter le Data Leakage, et j'ai appliqué SMOTE uniquement sur le train pour ne pas créer de fausses données de test."

---

## 1. Projet 16 : Prédiction Box-Office

### Q0: EDA Initial
- **Réponse Idéale (5/5) :** "J'ai commencé par `df.info()` pour voir les types. J'ai trouvé que Budget avait des 0 suspects, que Genre était catégoriel, et que la distribution des recettes était très skewed. J'ai aussi vérifié les NaN avec `df.isnull().sum()`."
- **Réponse Médiocre (1/5) :** "J'ai juste regardé les premières lignes avec `head()`."

### Q1: Gestion des Budgets/Recettes nuls ou négatifs
- **Réponse Idéale (5/5) :** "J'ai analysé ces lignes. Comme un budget de 0 est impossible pour un film commercial, j'ai considéré cela comme une valeur manquante. J'ai soit supprimé ces lignes (si peu nombreuses), soit remplacé par la médiane du même Genre."
- **Réponse Médiocre (1/5) :** "Je n'ai rien fait" ou "J'ai mis 0".

### Q2: Feature Engineering (Dates)
- **Réponse Idéale (5/5) :** "Oui, j'ai extrait le mois. J'ai remarqué (via un barplot) que les films sortis en Été (Juin/Juillet) et en Décembre ont des revenus moyens nettement supérieurs."

### Q3: Encodage du Genre
- **Réponse Idéale (5/5) :** "J'ai utilisé le One-Hot Encoding (`get_dummies`) car il n'y a pas d'ordre de grandeur entre 'Action' et 'Comédie'. Un Label Encoding (1, 2, 3...) aurait faussé le modèle en introduisant une hiérarchie inexistante."

### Q4: Transformation Log du Budget
- **Réponse Idéale (5/5) :** "La distribution du budget est très étalée (skewed) avec quelques blockbusters énormes. Le Log permet de 'tasser' ces valeurs extrêmes et de rendre la distribution plus normale (gaussienne), ce qui aide l'algorithme à mieux apprendre."

### Q5: Workflow (Split vs Imputation)
- **Réponse Idéale (5/5) :** "J'ai splitté d'abord. Si je calcule la médiane sur tout le dataset pour remplacer les 0, j'utilise des infos du futur (test set). Il faut calculer la médiane sur le train et l'appliquer au test."

### Q6: Score R² et Feature Importance
- **Réponse Idéale (5/5) :** "Mon R² est de 0.X (ex: 0.65). Le Feature Importance montre que le 'Budget' (ou Log_Budget) est de loin la variable la plus prédictive, suivie par le nombre de votes ou le casting."

---

## 2. Projet 10 : Recommandation de Voyage

### Q0: EDA Initial
- **Réponse Idéale (5/5) :** "On a analysé les types avec `df.info()`, trouvé X% de NaN dans Budget_Quotidien, et observé que les notes suivent une distribution normale centrée sur 7."
- **Réponse Médiocre (1/5) :** "On n'a pas vraiment exploré, on a directement modélisé."

### Q1: Imputation du Budget_Quotidien
- **Réponse Idéale (5/5) :** "J'ai utilisé la médiane car la moyenne est trop sensible aux valeurs extrêmes (voyages de luxe). La médiane représente mieux le touriste typique."

### Q2: Relation Budget vs Note
- **Réponse Idéale (5/5) :** "Le scatterplot montre un nuage de points assez dispersé. La corrélation est positive mais faible. Payer cher ne garantit pas une note de 10/10, d'autres facteurs comme le Climat jouent beaucoup."

### Q3: Encodage Style_Voyage
- **Réponse Idéale (5/5) :** "Oui, One-Hot Encoding. Comme un utilisateur peut aimer plusieurs styles, on a des colonnes binaires `Style_Aventure`, `Style_Luxe`, etc."

### Q4: Workflow (Split avant imputation)
- **Réponse Idéale (5/5) :** "Oui, on a fait le split d'abord. Sinon, la médiane calculée inclurait des infos du test set, ce qui créerait du data leakage et surestimerait nos performances."
- **Réponse Médiocre (1/5) :** "On a tout imputé avant le split" ou "Je ne sais pas pourquoi c'est important."

### Q5: Retrait ID Utilisateur
- **Réponse Idéale (5/5) :** "L'ID est un identifiant unique aléatoire. Il n'a aucune valeur prédictive. Si on le laisse, le modèle risque d'apprendre par coeur les IDs du train set et ne saura pas généraliser aux nouveaux utilisateurs (Overfitting)."

### Q6: MAE et Interprétation
- **Réponse Idéale (5/5) :** "Notre MAE est de 0.8. Cela signifie qu'en moyenne, notre prédiction de note se trompe de +/- 0.8 point sur une échelle de 10. C'est acceptable pour une recommandation."

---

## 3. Projet 06 : Fake News (NLP)

### Q0: EDA Initial
- **Réponse Idéale (5/5) :** "On a vérifié `df.info()`, trouvé que toutes les colonnes sont du texte, pas de NaN. On a analysé la distribution des longueurs de texte et le ratio Fake/Real."
- **Réponse Médiocre (1/5) :** "On a juste chargé les données et commencé."

### Q1: Équilibre du Dataset
- **Réponse Idéale (5/5) :** "Il est relativement équilibré (ex: 60/40 ou 50/50). Je n'ai donc pas eu besoin d'utiliser de techniques complexes de rééquilibrage comme SMOTE, l'accuracy reste une métrique valide."

### Q2: Word_Count Feature
- **Réponse Idéale (5/5) :** "Oui. J'ai observé que les Fake News ont tendance à être soit très courtes (juste une accroche), soit très longues (théories du complot), alors que les vrais articles ont une longueur plus standard."

### Q3: Clickbait Detection
- **Réponse Idéale (5/5) :** "Les titres de Fake News utilisent beaucoup de MAJUSCULES et de '!!!'. J'ai chiffré ça avec une feature `Uppercase_Ratio`. C'est l'une des variables les plus discriminantes dans mon modèle."

### Q4: Workflow NLP (Data Leakage)
- **Réponse Idéale (5/5) :** "J'ai fait `vectorizer.fit_transform(X_train)` et `vectorizer.transform(X_test)`. Si je 'fit' sur tout avant le split, le modèle connaît tous les mots du test set (vocabulaire), ce qui est de la triche."

### Q5: F1-Score vs Accuracy
- **Réponse Idéale (5/5) :** "L'Accuracy peut être trompeuse. Le F1-Score est meilleur car il fait la moyenne harmonique entre Précision et Rappel. Ici, il est crucial de bien détecter les Fake (Rappel) sans censurer les Vrais (Précision)."

### Q5: Matrice de Confusion et Gravité
- **Réponse Idéale (5/5) :** "Voici la matrice. On voit que j'ai 50 Faux Négatifs (Fake prédits comme Vrais), ce qui est le plus dangereux car les fausses infos se propagent. Les Faux Positifs (Vrais censurés) sont aussi problématiques mais moins critiques."

---

## 4. Projet 19 : Fraude Carte Crédit

### Q0: EDA Initial & Nettoyage
- **Réponse Idéale (5/5) :** "J'ai checké `df.info()` et trouvé des incohérences (ex: ' Class' avec un espace, ou des NaNs). J'ai nettoyé tout ça en premier. Modéliser sur des données sales = Échec garanti."
- **Réponse Médiocre (1/5) :** "J'ai lancé SMOTE direct sans regarder les données, et j'ai eu des erreurs."

### Q1: Gestion du Déséquilibre (Imbalanced)
- **Réponse Idéale (5/5) :** "C'est le point critique (3% de fraude). J'ai utilisé SMOTE sur le train set uniquement pour générer des fraudes synthétiques et permettre au modèle de voir assez d'exemples positifs."

### Q2: Workflow SMOTE (Critique)
- **Réponse Idéale (5/5) :** "SMOTE doit être fait **APRÈS** le split train/test et **UNIQUEMENT** sur le train set. Si on le fait avant, on crée des copies de données qui se retrouvent dans le test set (Data Leakage), rendant le score final faux (trop optimiste)."
- **Réponse Médiocre (1/5) :** "J'ai fait SMOTE sur tout le dataset avant le split."

### Q3: Features Métier (Night/Zscore)
- **Réponse Idéale (5/5) :** "J'ai créé `Is_Night` car les fraudes arrivent souvent la nuit. Le Z-Score aide à détecter les montants aberrants pour un client donné (ex: dépenser 5000€ alors qu'on dépense d'habitude 50€)."

### Q4: Recall (Rappel) et Justification
- **Réponse Idéale (5/5) :** "Je vise un Recall > 0.85 pour la classe Fraude. C'est la priorité : il vaut mieux bloquer une transaction par erreur (Faux Positif = client mécontent) que de laisser passer une fraude de 10 000€ (Faux Négatif = perte sèche)."

### Q5: Ajustement du Seuil
- **Réponse Idéale (5/5) :** "Par défaut le seuil est 0.5. Je l'ai baissé à 0.3 pour être plus agressif sur la détection de fraude, ce qui a augmenté mon Recall de 10% (mais aussi les Faux Positifs)."

### Q6: Cost-Benefit Analysis (Bonus)
- **Réponse Idéale (5/5) :** "J'ai calculé le coût total : `Coût = (FP * 10) + (FN * 500)`. J'ai tracé ce coût pour différents seuils et choisi celui qui minimise la perte financière totale (souvent autour de 0.1 ou 0.2)."
- **Pas de Bonus (0/5) :** L'étudiant n'a pas abordé cette analyse.

---

## 5. Projet 07 : Gaspillage Alimentaire

### Q0: EDA Initial
- **Réponse Idéale (5/5) :** "On a exploré avec `df.info()`, trouvé que Price avait 15% de NaN, et que les dates de péremption étaient au bon format. On a aussi visualisé les ventes par jour de la semaine."
- **Réponse Médiocre (1/5) :** "On n'a pas vraiment exploré les données."

### Q1: Imputation Prix manquants
- **Réponse Idéale (5/5) :** "J'ai remplacé les NaN par la médiane des prix *de ce produit spécifique*. Remplacer par la moyenne globale aurait été faux car une pomme ne coûte pas le même prix qu'un steak."

### Q2: Jours Avant Expiration
- **Réponse Idéale (5/5) :** "C'est une feature clé. Plus la date d'expiration approche, plus les ventes augmentent (souvent aidées par des promos 'date courte'). La corrélation est négative (moins de jours = plus de ventes)."

### Q3: Feature Urgence_Vente
- **Réponse Idéale (5/5) :** "C'est une interaction : `Expire_Bientot * Promo_Forte`. C'est là que les volumes de ventes explosent. Le modèle capture très bien cet effet 'bon plan de dernière minute'."

### Q4: Workflow Imputation
- **Réponse Idéale (5/5) :** "On a fait le split d'abord, puis calculé la médiane par produit sur le train set, et appliqué ces mêmes valeurs au test set."
- **Réponse Médiocre (1/5) :** "On a imputé sur tout le dataset."

### Q5: Prédictions vs Réel
- **Réponse Idéale (5/5) :** "Le modèle suit bien la tendance globale et les saisonnalités hebdo (pics du samedi). Il a un peu plus de mal sur les pics extrêmes de fin d'année."

---

## 6. Projet 08 : Santé Mentale

### Q0: EDA Initial
- **Réponse Idéale (5/5) :** "On a analysé la distribution des 3 classes, trouvé un léger déséquilibre (40% Normal, 35% Anxious, 25% Depressed). On a aussi vérifié les textes vides et dupliqués."
- **Réponse Médiocre (1/5) :** "On a directement fait le modèle."

### Q1: Polarité (TextBlob)
- **Réponse Idéale (5/5) :** "La polarité va de -1 à 1. Les tweets 'Depressed' ont une polarité très négative (proche de -0.8), alors que 'Anxious' est parfois plus neutre mais avec beaucoup de subjectivité."

### Q2: Mots Clés Spécifiques
- **Réponse Idéale (5/5) :** "J'ai trouvé que 'tired', 'alone', 'sleep' sont typiques de la dépression. Pour l'anxiété, c'est plutôt 'worry', 'scared', 'future', 'what if'."

### Q3: Équilibre des classes
- **Réponse Idéale (5/5) :** "Les classes n'étaient pas parfaitement équilibrées. On a utilisé `class_weight='balanced'` dans le modèle pour compenser."

### Q4: Workflow NLP (Vocabulaire)
- **Réponse Idéale (5/5) :** "Même principe que pour le scaling : le vocabulaire doit être construit uniquement sur les tweets du train set. Les mots inconnus du test set seront ignorés ou marqués comme 'unknown'."

### Q5: Confusion Anxious/Depressed
- **Réponse Idéale (5/5) :** "Oui, il y a de la confusion car les symptômes se chevauchent. Le modèle distingue très bien 'Normal' des deux autres, mais a plus de mal à séparer Anxiété et Dépression."

### Q5: Système d'Alerte (Bonus)
- **Réponse Idéale (5/5) :** "J'ai fait un filtre simple : si le texte contient 'suicide', 'kill myself' ou 'die', le système lève un drapeau rouge immédiat, quelle que soit la prédiction du modèle ML."

---

## 7. Projet X : Performance Développeurs AI

### Q0: EDA Initial
- **Réponse Idéale (5/5) :** "On a fait une heatmap de corrélation. On a vu que AI_Usage est corrélé positivement avec Productivity jusqu'à un certain point, et que Stress est négativement corrélé avec Success_Rate."
- **Réponse Médiocre (1/5) :** "On n'a pas exploré, on a juste entraîné le modèle."

### Q1: Relation AI Usage vs Productivité
- **Réponse Idéale (5/5) :** "Ce n'est pas linéaire. L'utilisation de l'IA augmente la productivité jusqu'à un certain point (effet d'aide), mais trop d'usage (copier-coller sans comprendre) peut faire baisser la qualité ou le taux de succès (courbe en cloche ou plateau)."

### Q2: Features Créatives (Team de 4)
- **Réponse Idéale (5/5) :** "On a créé le ratio `Code_Efficiency = Lines_of_Code / Hours_Worked`. On a aussi combiné `Stress_Level` et `AI_Usage` pour voir si l'IA réduit le stress."

### Q3: Workflow
- **Réponse Idéale (5/5) :** "Le preprocessing (scaling, encoding) a été fait après le split, en fittant sur le train set et en transformant le test set avec les mêmes paramètres."
- **Réponse Médiocre (1/5) :** "On a tout normalisé avant le split."

### Q4: RMSE Modèle Régression
- **Réponse Idéale (5/5) :** "Notre RMSE est de X. Cela veut dire qu'on prédit le taux de succès à +/- X% près. Le RandomForest a mieux marché que la Régression Linéaire car les relations ne sont pas linéaires."

### Q5: Classification Low/High
- **Réponse Idéale (5/5) :** "On a coupé à la médiane du `Task_Success_Rate` pour avoir deux classes équilibrées. Ce qui distingue les 'High Performers', c'est souvent l'expérience couplée à un usage modéré et intelligent de l'IA."

### Q6: Risque Burnout (Bonus)
- **Réponse Idéale (5/5) :** "Les profils à risque sont ceux qui combinent `High Hours` + `High Stress`. Paradoxalement, ceux qui n'utilisent PAS du tout l'IA semblent plus stressés car ils font tout manuellement."

### Q7: Synthèse IA
- **Réponse Idéale (5/5) :** "L'IA est un multiplicateur de force pour les seniors, mais peut être une béquille risquée pour les juniors s'ils ne vérifient pas le code. Il faut encourager l'usage supervisé."
