# 📝 Feuille de Notation - Soutenances Data Science
**Date :** 23 Décembre 2025
**Examinateur :** Mehdi

---

## 1. Projet 16 : Prédiction Box-Office
**Membre(s) :** Lamya lajmi (1 personne)
**Niveau Attendu :** Intermédiaire (Feature Engineering & Random Forest)

**Questions Techniques :**
- [ ] **EDA Initial :** "Avant de modéliser, qu'as-tu découvert lors de l'exploration ? Types de données, valeurs manquantes, distributions ?"
- [ ] **Data Cleaning :** "Comment as-tu géré les films avec un `Budget` ou `Recette` de 0 ou négatif ?"
- [ ] **Feature Engineering (Dates) :** "As-tu extrait le `Mois` de sortie ? Y a-t-il un mois plus rentable ?"
- [ ] **Feature Engineering (Encodage) :** "Comment as-tu encodé la colonne `Genre` ? (Attendu : One-Hot Encoding)"
- [ ] **Math :** "Pourquoi appliquer une transformation Log (`np.log1p`) sur le Budget ? Montre la distribution avant/après."
- [ ] **Workflow :** "Avez-vous fait le split train/test **AVANT** de remplacer les valeurs manquantes/0 ? Pourquoi ?"
- [ ] **Modèle :** "Quel est ton score R² sur le test set ? Quelles variables influencent le plus le succès ?"

**Note :** ______ / 20
**Commentaires :**

---

## 2. Projet 10 : Recommandation de Voyage Personnalisée
**Membre(s) :** ELKHALLADI SAFOUANE, FTIH HAJAR (2 personnes)
**Objectif :** Prédire la `Note_Destination` et recommander.

**Questions Techniques :**
- [ ] **EDA Initial :** "Montrez-moi votre exploration initiale. Quelles anomalies avez-vous détectées dans les données ?"
- [ ] **Data Cleaning :** "Le `Budget_Quotidien` avait des valeurs manquantes. Par quoi avez-vous remplacé les NaN ? (Attendu : Médiane)"
- [ ] **EDA Approfondi :** "Montrez le graphe `Budget` vs `Note`. Est-ce que payer plus cher garantit une meilleure note ?"
- [ ] **Feature Engineering :** "Avez-vous utilisé `pd.get_dummies` pour `Style_Voyage` ?"
- [ ] **Workflow :** "Avez-vous fait le split train/test AVANT l'imputation ? Pourquoi est-ce important ?"
- [ ] **Modèle :** "Avez-vous retiré l'ID utilisateur avant l'entraînement ? Pourquoi ?"
- [ ] **Évaluation :** "Quelle est votre MAE (Erreur Absolue Moyenne) ? En moyenne, de combien vous trompez-vous sur la note ?"

**Note :** ______ / 20
**Commentaires :**

---

## 3. Projet 06 : Classificateur de Fake News
**Membre(s) :** Ahmed Saifeddine Nakhli, Mouna Belhask (2 personnes)
**Objectif :** NLP - Détection Fake vs Real.

**Questions Techniques :**
- [ ] **EDA Initial :** "Montrez-moi votre exploration. Avez-vous analysé `df.info()`, les types de données, les valeurs nulles ?"
- [ ] **Équilibre :** "Le dataset est-il équilibré ? (Ratio Fake/Real)"
- [ ] **NLP Features :** "Avez-vous créé des features simples comme `Word_Count` ? Les Fake News sont-elles plus courtes ?"
- [ ] **Clickbait :** "Avez-vous détecté les majuscules ou points d'exclamation abusifs ? Est-ce discriminant ?"
- [ ] **Workflow NLP :** "Avez-vous 'fit' le Vectorizer (TF-IDF/CountVec) **UNIQUEMENT** sur le train set ? Pourquoi ?"
- [ ] **Modèle & Métrique :** "Quel est votre F1-Score ? Pourquoi l'Accuracy seule ne suffit pas ici ?"
- [ ] **Matrice de Confusion :** "Montrez la matrice. Faites-vous plus de Faux Positifs ou Faux Négatifs ? Lequel est plus grave ici ?"

**Note :** ______ / 20
**Commentaires :**

---

## 4. Projet 19 : Détection de fraude carte de crédit
**Membre(s) :** Aya haddaoui, Jihane Benradi (2 personnes)
**Objectif :** Détection d'anomalies (Imbalanced Dataset).

**Questions Techniques :**
- [ ] **EDA Initial & Nettoyage :** "Quelles anomalies avez-vous trouvées (NaNs, valeurs ' Class' dans la cible) ? Avez-vous nettoyé AVANT de tenter le split ou SMOTE ?"
- [ ] **Déséquilibre :** "Il y a très peu de fraudes (~3%). Comment avez-vous géré ça ? (Attendu : SMOTE ou Class Weights)"
- [ ] **Workflow SMOTE (Critique) :** "Avez-vous appliqué SMOTE **APRÈS** le split train/test et **UNIQUEMENT** sur le train set ? Pourquoi est-ce une faute grave de le faire avant ?"
- [ ] **Features Métier :** "Avez-vous créé une feature `Is_Night` (fraudes nocturnes) ou `Amount_Zscore` (montants aberrants) ?"
- [ ] **Métrique Critique :** "Quel est votre **Recall** sur la classe Fraude ? Pourquoi le Recall est plus important que la Précision ici ?"
- [ ] **Compromis :** "Avez-vous ajusté le seuil (threshold) de probabilité ? Si oui, à combien ? Quel effet sur Recall/Précision ?"
- [ ] **Bonus (Cost-Benefit) :** "Avez-vous calculé le coût total (FP*10 + FN*500) pour trouver le seuil optimal ?"

**Note :** ______ / 20
**Commentaires :**

---

## 5. Projet 07 : Réduction du Gaspillage Alimentaire
**Membre(s) :** Sanae Amenouad, Rim Bassou (2 personnes)
**Objectif :** Régression Temporelle (Prédiction des ventes).

**Questions Techniques :**
- [ ] **EDA Initial :** "Montrez-moi votre exploration. Types des colonnes, valeurs manquantes par colonne, distributions ?"
- [ ] **Nettoyage :** "Comment avez-vous remplacé les `Price` manquants ? (Attendu : par Produit, pas globalement)"
- [ ] **Features Temporelles :** "Avez-vous calculé `Jours_Avant_Expiration` ? Est-ce que ça influence les ventes ?"
- [ ] **Interactions :** "Avez-vous créé la feature `Urgence_Vente` (Promo + Péremption proche) ?"
- [ ] **Workflow :** "L'imputation a été faite avant ou après le split ? Pourquoi ça compte ?"
- [ ] **Modèle :** "Montrez le graphe Prédictions vs Réel. Les pics de ventes sont-ils bien prédits ?"
- [ ] **Feature Importance :** "Qu'est-ce qui fait vendre le plus : Le Prix (Discount) ou la Date ?"

**Note :** ______ / 20
**Commentaires :**

---

## 6. Projet 08 : Santé Mentale
**Membre(s) :** Hala Rahal, Hind Sadok, ILIAS HAIFA (3 personnes)
**Objectif :** NLP Multi-classe (Normal, Anxious, Depressed).

**Questions Techniques :**
- [ ] **EDA Initial :** "Montrez l'exploration des données. Distribution des classes ? Textes vides ou dupliqués ?"
- [ ] **NLP & Sentiment :** "Avez-vous utilisé TextBlob pour la polarité ? Comment se comporte la polarité des tweets 'Anxious' ?"
- [ ] **Mots Clés :** "Avez-vous cherché des mots spécifiques (suicide, kill, hopeless) ? (Bonus Task)"
- [ ] **Équilibre :** "Les 3 classes sont-elles équilibrées ? Si non, qu'avez-vous fait ?"
- [ ] **Workflow NLP :** "Le vocabulaire du Vectorizer a-t-il été appris uniquement sur le train set (pas de fuite) ?"
- [ ] **Modèle Multi-classe :** "Le modèle confond-il souvent 'Anxious' et 'Depressed' ? Montrez la matrice de confusion."
- [ ] **Team Size (3) - Bonus :** "Avez-vous implémenté le système d'alerte pour les cas urgents ? Comment fonctionne-t-il ?"
- [ ] **Analyse Temporelle :** "Y a-t-il une heure de la journée où les posts dépressifs sont plus fréquents ?"

**Note :** ______ / 20
**Commentaires :**

---

## 7. Projet X (21) : Performance des Développeurs AI
**Membre(s) :** Rania Srir, Wiam Chmicha, Aya Belghazi, Sophia Yassfouli (4 personnes)
**Objectif :** Prédire `Task_Success_Rate` & Classifier la performance.

**Questions Techniques :**
- [ ] **EDA Initial :** "Montrez votre exploration. Corrélations entre variables ? Outliers détectés ?"
- [ ] **EDA & Data :** "Quelle est la relation entre `AI_Usage_Hours` et la `Productivité` ? Est-elle linéaire ?"
- [ ] **Feature Engineering :** "Vu que vous êtes 4, quelles nouvelles features créatives avez-vous inventées ? (ex: Ratio Code/Heure)"
- [ ] **Workflow :** "Le preprocessing a été fait sur tout le dataset ou seulement sur le train set ?"
- [ ] **Modèle 1 (Régression) :** "Vous deviez prédire le Taux de Succès. Quel est votre RMSE ?"
- [ ] **Modèle 2 (Classification) :** "Vous avez classifié les devs (Low vs High). Comment avez-vous défini le seuil ?"
- [ ] **Stress (Bonus) :** "Avez-vous identifié les profils à risque de Burnout ? Quels sont les signes ?"
- [ ] **Synthèse :** "Selon votre modèle, faut-il encourager ou limiter l'usage de l'IA pour les débutants ?"

**Note :** ______ / 20
**Commentaires :**
