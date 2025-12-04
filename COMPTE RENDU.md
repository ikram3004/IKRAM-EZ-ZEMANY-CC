# Compte Rendu : Dataset "Give Me Some Credit"
## Projet de Credit Scoring - Data Science & Machine Learning
# Nom complet : IKRAM EZ-ZEMANY
---
![WhatsApp Image 2025-12-04 at 22 07 15_0b7997ff](https://github.com/user-attachments/assets/e1eadcde-ae29-4285-8d89-8f5cfbb0518c)

## 1. PRÉSENTATION DU DATASET

### 1.1 Source et Contexte
- **Nom du Dataset** : Give Me Some Credit - 2011 Competition Data
- **Source** : Kaggle Competition (2011)
- **Lien** : https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset
- **Licence** : Données publiques (usage éducatif et recherche autorisé)

### 1.2 Contexte Métier
Les banques jouent un rôle crucial dans l'économie en décidant qui peut obtenir du crédit et à quelles conditions. Les algorithmes de credit scoring permettent d'estimer la probabilité de défaut de paiement d'un client, aidant ainsi les institutions financières à prendre des décisions éclairées sur l'octroi de prêts.

**Problématique** : Prédire la probabilité qu'un individu connaisse des difficultés financières dans les deux prochaines années.

---

## 2. CARACTÉRISTIQUES TECHNIQUES DU DATASET

### 2.1 Structure des Données

| Caractéristique | Valeur |
|----------------|--------|
| **Nombre d'observations** | 150,000 lignes |
| **Nombre de variables** | 11 colonnes |
| **Taille du fichier** | ~5.2 MB (compressé) |
| **Format** | CSV (Comma-Separated Values) |
| **Fichiers disponibles** | cs-training.csv, cs-test.csv, sampleEntry.csv |

### 2.2 Type de Problème
- **Catégorie** : Classification Binaire Supervisée
- **Variable Cible (Target)** : `SeriousDlqin2yrs`
  - **0** = Client sans défaut de paiement
  - **1** = Client en défaut de paiement (difficultés financières)

---

## 3. DICTIONNAIRE DES VARIABLES

### 3.1 Variable Cible

| Variable | Type | Description |
|----------|------|-------------|
| **SeriousDlqin2yrs** | Integer (0/1) | Indicateur de défaut : le client a connu des difficultés financières sérieuses (retard de paiement de 90 jours ou plus) dans les 2 années suivantes |

### 3.2 Variables Explicatives (Features)

| Variable | Type | Description | Signification Business |
|----------|------|-------------|------------------------|
| **RevolvingUtilizationOfUnsecuredLines** | Float | Taux d'utilisation des lignes de crédit renouvelables non garanties | Ratio entre le solde total des cartes de crédit et la limite totale (ex: 0.5 = 50% utilisé) |
| **age** | Integer | Âge du client en années | Indicateur de maturité financière et stabilité |
| **NumberOfTime30-59DaysPastDueNotWorse** | Integer | Nombre de fois où le client a eu un retard de paiement de 30-59 jours (pas pire) dans les 2 dernières années | Indicateur de difficultés financières modérées |
| **DebtRatio** | Float | Ratio dette/revenu mensuel | Total des dettes divisé par le revenu mensuel (ex: 0.3 = 30% du revenu consacré aux dettes) |
| **MonthlyIncome** | Float | Revenu mensuel du client | Capacité financière globale |
| **NumberOfOpenCreditLinesAndLoans** | Integer | Nombre de crédits et prêts ouverts | Indicateur du niveau d'endettement actif |
| **NumberOfTimes90DaysLate** | Integer | Nombre de fois où le client a eu un retard de paiement de 90 jours ou plus | Indicateur de difficultés financières graves |
| **NumberRealEstateLoansOrLines** | Integer | Nombre de prêts immobiliers ou lignes de crédit immobilier | Engagement financier à long terme |
| **NumberOfTime60-89DaysPastDueNotWorse** | Integer | Nombre de fois où le client a eu un retard de paiement de 60-89 jours | Indicateur de difficultés financières sérieuses |
| **NumberOfDependents** | Float | Nombre de personnes à charge financièrement (hors le client) | Impact sur les charges financières du foyer |

---

## 4. ANALYSE PRÉLIMINAIRE DU DATASET

### 4.1 Distribution de la Variable Cible

Le dataset présente un **déséquilibre important** entre les classes :
- **Classe 0 (Pas de défaut)** : ~93% des observations
- **Classe 1 (Défaut)** : ~7% des observations

**Implications** :
- Nécessité d'utiliser des techniques de gestion du déséquilibre (SMOTE, class_weight)
- Métriques d'évaluation à privilégier : F1-Score, ROC-AUC, Recall (plutôt qu'Accuracy)

### 4.2 Qualité des Données

#### Valeurs Manquantes Identifiées

| Variable | Valeurs Manquantes | Pourcentage |
|----------|-------------------|-------------|
| **MonthlyIncome** | ~29,731 | ~19.8% |
| **NumberOfDependents** | ~3,924 | ~2.6% |
| **Autres variables** | 0 | 0% |

**Stratégies d'imputation à envisager** :
- MonthlyIncome : Imputation par la médiane par groupe d'âge ou régression
- NumberOfDependents : Imputation par la médiane (probablement 0 ou 1)

#### Valeurs Aberrantes (Outliers)

Plusieurs variables présentent des valeurs extrêmes :
- **age** : Quelques valeurs à 0 (aberration logique)
- **RevolvingUtilizationOfUnsecuredLines** : Valeurs supérieures à 1 (>100%)
- **DebtRatio** : Valeurs très élevées (ratios impossibles)
- **Variables de retard** : Valeurs extrêmes (96, 98)

**Décisions à prendre** :
- Suppression ou correction des âges = 0
- Winsorization ou cap des valeurs extrêmes
- Analyse approfondie pour distinguer erreurs de saisie vs vraies valeurs

---

## 5. INSIGHTS CLÉS (ANALYSE EXPLORATOIRE)

### 5.1 Corrélations avec la Variable Cible

**Variables les plus corrélées avec le défaut de paiement** :
1. **NumberOfTimes90DaysLate** : Forte corrélation positive (++)
2. **NumberOfTime60-89DaysPastDueNotWorse** : Forte corrélation positive (++)
3. **NumberOfTime30-59DaysPastDueNotWorse** : Corrélation positive (+)
4. **age** : Corrélation négative (-) → Les clients jeunes sont plus à risque
5. **RevolvingUtilizationOfUnsecuredLines** : Corrélation positive (+)

**Interprétation** :
- L'historique de retards de paiement est le meilleur prédicteur de défaut futur
- Les clients plus jeunes présentent un risque plus élevé (moins d'expérience financière)
- Un taux d'utilisation élevé du crédit revolving indique un risque accru

### 5.2 Profil des Clients à Risque

**Caractéristiques typiques d'un client à haut risque** :
- Historique de retards de paiement (30, 60 ou 90+ jours)
- Jeune (< 30 ans)
- Taux d'utilisation du crédit élevé (> 80%)
- Ratio dette/revenu élevé (> 0.5)
- Revenu mensuel faible

---

## 6. PRÉPARATION DES DONNÉES (PREPROCESSING)

### 6.1 Étapes de Nettoyage Nécessaires

1. **Traitement des valeurs manquantes**
   - Imputation de MonthlyIncome (méthode : médiane par groupe ou régression)
   - Imputation de NumberOfDependents (méthode : médiane = 0)

2. **Gestion des outliers**
   - Correction des âges aberrants (0 → médiane ou suppression)
   - Cap des valeurs extrêmes (winsorization au 1er et 99e percentile)
   - Vérification et correction des ratios impossibles

3. **Feature Engineering**
   - Création de variables agrégées :
     - `TotalPastDue` = somme de tous les retards
     - `HasPastDue` = indicateur binaire (au moins 1 retard)
     - `IncomePerDependent` = MonthlyIncome / (NumberOfDependents + 1)
     - `CreditUtilizationCategory` = catégorisation du taux d'utilisation

4. **Normalisation/Standardisation**
   - StandardScaler pour les variables numériques continues
   - Conservation des variables de comptage (déjà sur des échelles comparables)

5. **Gestion du déséquilibre**
   - Application de SMOTE (Synthetic Minority Over-sampling Technique)
   - Ou ajustement des poids de classe (`class_weight='balanced'`)

---

## 7. MODÉLISATION ENVISAGÉE

### 7.1 Algorithmes à Tester

| Algorithme | Justification | Avantages |
|------------|---------------|-----------|
| **Logistic Regression** | Baseline interprétable | Simple, rapide, coefficients interprétables |
| **Random Forest** | Gestion des non-linéarités | Robuste aux outliers, importance des features |
| **XGBoost** | Performance optimale | État de l'art pour la classification, gestion du déséquilibre |
| **LightGBM** (optionnel) | Alternative performante | Plus rapide que XGBoost |

### 7.2 Stratégie de Validation

- **Split initial** : 80% Train / 20% Test
- **Cross-Validation** : 5-Fold Stratified CV (conserve la proportion des classes)
- **Optimisation** : GridSearchCV ou RandomizedSearchCV sur le jeu d'entraînement

### 7.3 Métriques d'Évaluation

| Métrique | Importance | Justification |
|----------|------------|---------------|
| **ROC-AUC** | ⭐⭐⭐ | Mesure la capacité à discriminer les classes (insensible au déséquilibre) |
| **F1-Score** | ⭐⭐⭐ | Équilibre entre Precision et Recall |
| **Recall** | ⭐⭐ | Important de détecter les vrais défauts (coût élevé des faux négatifs) |
| **Precision** | ⭐⭐ | Éviter de rejeter trop de bons clients |
| **Accuracy** | ⭐ | Moins pertinent (biaisé par le déséquilibre) |

---

## 8. VALEUR BUSINESS DU PROJET

### 8.1 Objectifs Métier

1. **Réduction du risque de crédit**
   - Identifier les clients à haut risque avant l'octroi du prêt
   - Réduire le taux de défaut de paiement

2. **Optimisation des décisions**
   - Automatisation partielle du processus de décision
   - Réduction du temps de traitement des demandes

3. **Amélioration de la rentabilité**
   - Diminution des pertes liées aux impayés
   - Meilleure allocation du capital

### 8.2 Impacts Attendus

- **Réduction des pertes** : Identification proactive des clients à risque
- **Gain de temps** : Automatisation du scoring (décision en temps réel)
- **Équité** : Décisions basées sur des données objectives (moins de biais humains)

---

## 9. LIMITES ET PRÉCAUTIONS

### 9.1 Limites du Dataset

1. **Temporalité** : Données de 2011 (potentiellement obsolètes, contexte économique différent)
2. **Généralisation** : Modèle entraîné sur un échantillon spécifique (population américaine probable)
3. **Variables manquantes** : Certaines informations pertinentes pourraient être absentes (score FICO, historique bancaire complet)

### 9.2 Considérations Éthiques

- **Biais potentiels** : L'âge est corrélé au risque (discrimination possible)
- **Transparence** : Nécessité d'expliquer les décisions (RGPD, réglementation bancaire)
- **Équité** : Éviter la discrimination envers certains groupes démographiques

---

## 10. PLAN DE TRAVAIL

### Phase 1 : Exploration (1 semaine)
- [x] Chargement et première analyse du dataset
- [x] Analyse exploratoire complète (EDA)
- [x] Identification des problèmes de qualité

### Phase 2 : Preprocessing (1 semaine)
- [ ] Imputation des valeurs manquantes
- [ ] Gestion des outliers
- [ ] Feature Engineering
- [ ] Normalisation et préparation finale

### Phase 3 : Modélisation (1 semaine)
- [ ] Entraînement de 3+ algorithmes
- [ ] Cross-validation et optimisation des hyperparamètres
- [ ] Sélection du meilleur modèle

### Phase 4 : Évaluation et Livrable (1 semaine)
- [ ] Analyse des performances (métriques, courbes ROC, matrice de confusion)
- [ ] Rédaction du rapport scientifique
- [ ] Création de la vidéo de présentation
- [ ] Mise en ligne sur GitHub

---

## 11. CONCLUSION

Le dataset "Give Me Some Credit" constitue une excellente base pour développer un modèle de credit scoring robuste et performant. La problématique est claire, les données sont réalistes (avec leurs imperfections), et le cas d'usage a une forte valeur business.

**Points forts du dataset** :
✅ Taille suffisante (150,000 observations)  
✅ Variables métier pertinentes et interprétables  
✅ Défi technique intéressant (déséquilibre des classes)  
✅ Application réelle dans le secteur financier

**Prochaines étapes** :
1. Télécharger le fichier `cs-training.csv` depuis Kaggle
2. Exécuter le notebook d'exploration
3. Passer au preprocessing et à la modélisation

---

**Date** : Décembre 2025  
**Projet** : Data Science & Machine Learning - Credit Scoring  
**Dataset** : Give Me Some Credit (Kaggle 2011)
