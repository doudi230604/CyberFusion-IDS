# Mini-projet : Supervised, Unsupervised and Deep Learning for IDS

Student(s): [NOM(S) / Prénom(s)]

Institution: [Your Institution]

Date: [DD/MM/YYYY]

---

## Sommaire

1. Introduction
2. Objectif
3. Datasets (tableau)
4. Prétraitement (par dataset)
5. Sélection des caractéristiques (Top 20)
6. Méthodologie
   - Supervised Learning (Random Forest, Decision Tree)
   - Unsupervised Learning (Isolation Forest)
   - Deep Learning (LSTM)
7. Résultats
   - Tableaux de résultats par dataset
   - Figures (matrices de confusion, courbes ROC)
   - Comparaison: Toutes les caractéristiques vs Top 20
8. Conclusion
9. Annexes
   - A: Description du pipeline LSTM
   - B: Isolation Forest — seuils testés et sélection
   - C: Code (référence aux scripts du dépôt)

---

## 1. Introduction

Paragraphe 1 — IDS

Les Intrusion Detection Systems (IDS) constituent une composante essentielle des défenses informatiques. Ils surveillent le trafic réseau et les événements sur hôte afin d'identifier des comportements anormaux ou malveillants. L'objectif est d'alerter les opérateurs ou de déclencher des actions automatiques pour limiter les attaques.

Paragraphe 2 — Machine Learning (supervised & unsupervised)

L'apprentissage supervisé (SL) permet d'entraîner des modèles sur des exemples étiquetés (normal vs attaque). L'apprentissage non-supervisé (USL) détecte des anomalies sans étiquettes en recherchant des points qui se démarquent du comportement normal; l'Isolation Forest est un exemple courant.

Paragraphe 3 — Deep Learning

Les approches de Deep Learning (DL), comme les LSTM, capturent des dépendances temporelles et peuvent apprendre des représentations complexes de séquences réseau; elles sont utiles lorsque la structure temporelle est importante.

Paragraphe 4 — Objectif du projet

Ce projet compare des approches SL, USL et DL sur plusieurs jeux de données (UNSW‑NB15, CICIDS2017, TonIoT). Le but est d'identifier le meilleur modèle en termes de détection d'attaques (privilégiant le recall) et d'évaluer l'effet de la sélection de caractéristiques (Top 20 vs toutes).

---

## 2. Objectif

- Comparer les performances des modèles RF, DT, Isolation Forest et LSTM sur chaque dataset.
- Mesurer l'impact de la sélection des Top 20 caractéristiques.
- Fournir des recommandations (meilleur modèle SL, meilleures pratiques de prétraitement et seuils pour IForest).

---

## 3. Datasets (tableau)

| Dataset | Fichiers utilisés | Taille (approx.) | Nombre de caractéristiques | Notes (attaques) |
|---------|-------------------|------------------:|---------------------------:|------------------|
| UNSW-NB15 | [fichiers] | 2.5M | 49 | mix de classes, multi-catégories (attack_cat) |
| CICIDS2017 | [fichiers] | 2.8M | 84 | trafic réel, divers types d'attaques |
| TonIoT | [fichiers] | ~650K | variable | données IoT, attaques spécifiques IoT |

> Remplir les colonnes "Fichiers utilisés" et "Taille" avec vos valeurs réelles.

---

## 4. Prétraitement (par dataset)

Pour chaque dataset, fournir (texte, pas de code) :
- Fichiers et partitions utilisés (ex : `UNSW_NB15_training-set.csv`)
- Colonnes supprimées (ex : Flow ID, IP source/destination, timestamp)
- Nettoyage appliqué (valeurs manquantes traitées par médiane / suppression, conversion types, suppression de valeurs infinies)
- Encodage catégoriel (ex : protocole, service) — méthode (LabelEncoder ou encodage fréquence)
- Normalisation (StandardScaler)
- Taille finale des caractéristiques (ex : de 49 -> 30 -> Top 20)

Exemple (UNSW-NB15):
- Fichiers: training + testing
- Colonnes retirées: `id`, `src_ip`, `dst_ip`, `timestamp`
- Opérations: suppression de colonnes inutiles, remplissage des NaN par la médiane, encodage des colonnes catégorielles, standardisation
- Final: 40 caractéristiques initiales → 20 (Top 20 sélectionnées)

> Note : Ajoutez ici vos décisions et le nombre final de caractéristiques pour chaque dataset.

---

## 5. Sélection des caractéristiques (Top 20)

Pour chaque dataset ajouter une table listant le Top 20 (ordre décroissant d'importance) :

### UNSW-NB15 — Top 20 (exemple)
1. feature_1
2. feature_2
...
20. feature_20

### CICIDS2017 — Top 20 (exemple)
1. feature_a
...

### TonIoT — Top 20 (exemple)
1. ...

> Remplacer les listes par les vraies caractéristiques extraites via RandomForest ou méthode choisie.

---

## 6. Méthodologie

### 6.1 Supervised Learning (RF, DT)
- Split : entraînement/test — **Exemple**: 80% train, 20% test, stratifié
- Validation: cross-validation 5‑fold sur l'ensemble d'entraînement
- Paramètres (exemples) :
  - RandomForest: n_estimators=100, max_depth=15, min_samples_split=10, random_state=42
  - DecisionTree: max_depth=10, min_samples_leaf=10, random_state=42
- Mesures rapportées: Accuracy, Recall (sensibilité), Precision, F1-score, AUC

### 6.2 Unsupervised Learning (Isolation Forest)
- Entraînement uniquement sur données normales (X_train_normal)
- Paramètres: contamination (testé), n_estimators=100, random_state=42
- Sélection du seuil: méthode ROC / Youden's J ou maximisation (tpr - fpr) sur ensemble de validation
- Rapports: AUC (ROC), seuil choisi, matrice de confusion, précision/recall à ce seuil

### 6.3 Deep Learning (LSTM)
- Pipeline (voir Annexe A): séquençage des données en fenêtres temporelles (timesteps), normalisation, architecture LSTM
- Hyperparamètres typiques: timesteps=10/20, batch_size=32, epochs=50, learning_rate=0.001
- Validation: split validation (20%) ou cross-validation par groupe temporel si nécessaire

---

## 7. Résultats

> **Instructions**: Remplissez ces tableaux avec les métriques réelles issues de vos expérimentations. Les tables ci‑dessous sont des modèles à compléter.

### 7.1 UNSW-NB15 — Résultats (All features vs Top 20)

| Modèle | Features | Accuracy | Precision | Recall | F1-Score | AUC | Commentaires |
|--------|---------:|---------:|----------:|-------:|---------:|----:|-------------|
| RandomForest | All | TBD | TBD | TBD | TBD | TBD |  |
| RandomForest | Top 20 | TBD | TBD | TBD | TBD | TBD |  |
| DecisionTree | All | TBD | TBD | TBD | TBD | TBD |  |
| DecisionTree | Top 20 | TBD | TBD | TBD | TBD | TBD |  |
| IsolationForest | All | TBD | TBD | TBD | TBD | TBD | threshold: TBD |
| LSTM | All (sequence) | TBD | TBD | TBD | TBD | TBD |  |

### 7.2 CICIDS2017 — Résultats

(Same table layout — fill for each model and feature set)

### 7.3 TonIoT — Résultats

(Same table layout)

### 7.4 Comparaisons transversales

- Table récapitulative par dataset indiquant le meilleur modèle SL (par recall), la performance du meilleur IForest (AUC), et le LSTM si applicable.

---

## 8. Isolation Forest — seuils testés & sélection (Annexe B)

- Méthode de sélection de seuil recommandée: **Maximiser Youden's J statistic (J = TPR - FPR)** sur un ensemble de validation.
- Exemple de valeurs testées (ancre) : [-1.0, -0.5, -0.2, -0.1, 0.0]
- Table de test — ex :

| Seuil | TPR | FPR | Youden J = TPR-FPR | AUC |
|-------:|----:|----:|-------------------:|----:|
| -0.5 | 0.xx | 0.yy | 0.zz | 0.### |
| -0.3 | ... | ... | ... | ... |

- Indiquer seuil retenu et justification (ex : meilleur J)

---

## 9. LSTM pipeline (Annexe A)

Texte décrivant la chaîne de traitement **(sans code)** :

1. Choix des features et fenêtrage (définir `timesteps`)
2. Transformation en séquences (reshape: [samples, timesteps, features])
3. Normalisation (fit scaler sur split train uniquement)
4. Architecture (ex: LSTM(64)->Dropout(0.2)->LSTM(32)->Dense)
5. Entraînement (batch_size, epochs, validation_split)
6. Sélection du modèle (checkpoint sur validation loss/recall)
7. Évaluation sur ensemble test : matrice de confusion, courbe ROC, métriques

---

## 10. Conclusion

- Résumer brièvement lequel des modèles SL offre la meilleure détection (baseline : choisir selon le Recall).
- Indiquer l'impact de la sélection Top 20 (perte/gain en performance, coût de calcul réduit).
- Recommandations pour la mise en production: utiliser modèle SL optimisé pour recall, seuil dynamique pour IForest, pipeline de monitoring des drifts.

---

## Annexes

### Annexe A — LSTM pipeline (description, hyperparamètres recommandés)

(voir Section 9)

### Annexe B — Isolation Forest (seuils testés / tableau détaillé)

(voir Section 8)

### Annexe C — Code

Les scripts et notebooks se trouvent dans le dépôt :
- `models/decision_tree/` — decision tree scripts
- `models/random_forest/` — random forest scripts
- `models/isolation_forest/` — isolation forest scripts
- `models/decision_tree/decision_treeT.py` — TON-IoT specific
- `models/decision_tree/decision_treeC.py` — CICIDS2017 specific
- `models/decision_tree/decision_tree.py` — UNSW-NB15 specific

> Conformément à la consigne IA.docx, le code source n'est pas inclus dans le corps du rapport mais référencé en annexe: vous pouvez copier/joindre les scripts pertinents comme appendice séparé.

---

## Next steps

- Option A: I can **populate the tables** with real metrics **if you provide** the results CSVs (predictions, metrics) or allow me to run the scripts in this environment and collect results.
- Option B: I can stop here and you will fill the placeholders manually.

Please reply with one choice: `Fill with results` (attach results or allow run), `Fill placeholders manually` (I leave as-is), or `Make modifications` (specify which parts to change).
