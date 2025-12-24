# 🤖 Prompts Gemini pour Google Sheets (Notation Automatique)

Ce fichier contient les prompts à utiliser dans **Google Sheets** (avec l'extension *Gemini for Sheets*) ou directement dans le chat Gemini pour noter les réponses des étudiants.

---

## 1. Prompt "Correcteur Unitaire" (Pour une cellule Google Sheets)

Utilisez ce prompt si vous avez une colonne `Réponse Etudiant` et une colonne `Réponse Idéale`.

**Formule Google Sheets :**
```excel
=GEMINI("Tu es un expert Data Science strict. Note la réponse de l'étudiant sur 5 selon le barème suivant :
0 = Non répondu / Hors sujet
1 = Très insuffisant (Vague, manque vocabulaire)
2 = Insuffisant (Comprend pas le 'pourquoi')
3 = Acceptable (Correct mais basique)
4 = Bien (Bonne compréhension)
5 = Excellent (Maîtrise totale, justification critique)

Question : " & A2 & "
Réponse Attendue : " & B2 & "
Réponse Etudiant : " & C2 & "

Tâche :
1. Compare la réponse de l'étudiant à la réponse attendue.
2. Vérifie si les mots-clés techniques (ex: SMOTE, Data Leakage, Split) sont présents.
3. Donne UNIQUEMENT le format : [NOTE]/5 - [Court Commentaire]")
```
*(Remplacez A2, B2, C2 par vos cellules Question, Réponse Idéale, Réponse Etudiant)*

---

## 2. Prompt "Analyse de Transcript" (Pour le Chat)

Si vous copiez-collez une transcription brute de la soutenance, utilisez ce prompt pour extraire les notes.

**Prompt :**
```text
Tu es un juré d'examen Data Science. Voici la transcription d'une soutenance pour le [Projet X].
Ton objectif est de remplir la grille de notation.

Voici les critères et réponses attendues (issue du Scoring Guide) :
[COPIER LE CONTENU DU SCORING_GUIDE.MD POUR CE PROJET ICI]

Voici la transcription de l'étudiant :
"""
[COLLER LA TRANSCRIPTION OU LES NOTES ICI]
"""

Pour chaque question technique listée dans le guide :
1. Identifie si l'étudiant a abordé le sujet.
2. Évalue la justesse de sa réponse (attention au Data Leakage et Nettoyage).
3. Attribue une note de 0 à 5.
4. Justifie la note en une phrase.

Format de sortie :
- **[Nom Question]** : X/5. Justification : ...
```

---

## 3. Prompt "Générateur de Feedback" (Post-Soutenance)

Pour générer le paragraphe de commentaire final à envoyer à l'étudiant.

**Prompt :**
```text
Basé sur les notes suivantes, rédige un feedback constructif mais direct pour l'étudiant.
Utilise le ton : "Professionnel, encourageant mais ferme sur la rigueur méthodologique".

Notes :
- EDA/Nettoyage : [Note]/5
- Feature Engineering : [Note]/5
- Workflow/Split : [Note]/5
- Modélisation : [Note]/5
- Bonus/Business : [Note]/5

Points Clés à mentionner :
- Si la note Workflow est < 3 : Explique pourquoi le Data Leakage est grave.
- Si la note Nettoyage est < 3 : Rappelle que "Garbage In, Garbage Out".
- Félicite les points forts (notes 4 ou 5).

Sortie attendue : Un paragraphe de 5-6 lignes.
```

---

## 4. Prompt Spécial "Vérification Workflow" (Binaires)

Pour vérifier rapidement si les erreurs critiques ont été commises (utile pour Project 19/16/06).

**Formule Google Sheets :**
```excel
=GEMINI("Analyse cette réponse et dis-moi si l'étudiant a commis une erreur de Data Leakage (Fuite de données).
Réponse : " & C2 & "

Règles :
- Si l'étudiant dit 'j'ai imputé avant le split' -> LEAKAGE
- Si l'étudiant dit 'j'ai vectorisé avant le split' -> LEAKAGE
- Si l'étudiant dit 'j'ai fait SMOTE avant le split' -> LEAKAGE
- Sinon -> OK

Réponds uniquement par 'LEAKAGE DETECTED' ou 'CLEAN WORKFLOW'.")
```
