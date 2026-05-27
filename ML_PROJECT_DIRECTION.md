# Direction ML du projet Traine_aide_decission

## Objectif actuel

Le projet vise à entraîner un modèle d’aide à la décision poker capable de prédire une action parmi :

- CHECK
- FOLD
- CALL
- RAISE

L’objectif n’est pas encore de brancher automatiquement le modèle au bot live, mais de construire une base ML propre, testable et alignée avec des labels théoriques.

---

## Décision principale

Le pipeline principal devient :

```text
PokerBench oracle
→ features normalisées
→ modèle 4 classes
→ étude graphique
→ prédiction offline
````

PokerBench est désormais la source principale pour l’entraînement, car les labels sont issus d’un oracle/solver et représentent mieux l’idée de “bonne décision théorique”.

---

## Ce qui est conservé

### `pokerbench_oracle_baseline_v1`

Baseline principale actuelle.

Points forts :

* 73 200 lignes utilisables
* 4 classes natives : CHECK / FOLD / CALL / RAISE
* labels séparés comme `pokerbench_solver_oracle`
* étude graphique générée
* métriques par street disponibles
* prédiction offline OK
* pas de branchement live

Résultat actuel :

```text
accuracy ≈ 0.695
macro F1 ≈ 0.712
```

Ce score est considéré comme plus crédible que les anciens scores très élevés, car le dataset est plus difficile et moins artificiel.

---

## Ce qui est déprécié

### `live_bb_baseline_v1`

Ce pipeline est conservé uniquement comme historique.

Il ne doit plus être utilisé comme référence principale, car son score élevé venait principalement de données augmentées/resamplées.

Limites :

* score autour de 96 % non comparable à PokerBench
* dépendance à du resampling / jitter
* faible valeur comme vérité théorique
* labels legacy, pas oracle solver

Statut :

```text
deprecated / archive only
```

Il peut rester dans le projet pour comparaison historique, mais il ne doit pas être affiché comme meilleur modèle.

---

## Position sur les données live

Les données live ne sont pas une vérité d’entraînement.

Elles sont utiles pour :

* tester le parsing
* vérifier les features réelles
* valider la compatibilité avec le bot
* faire du shadow mode
* repérer les formats de spots réellement rencontrés

Mais elles ne répondent pas à la question :

```text
Quelle était la meilleure décision théorique ?
```

Pour cette raison, les logs live ne doivent pas devenir la source principale de labels.

---

## Position sur le solver local

Le solver local reste intéressant, mais il n’est pas encore utilisable comme générateur principal.

Constat actuel :

```text
solver_connected = true
solver_generated_rows très faible
resampling majoritaire
```

Il ne doit donc pas être présenté comme une source massive de labels tant que la génération réelle de spots solver n’est pas fiable.

---

## Prochaine expérimentation prioritaire

Créer une version 3 intentions :

```text
NO_INVEST
CALL
RAISE
```

Mapping :

```text
CHECK -> NO_INVEST
FOLD  -> NO_INVEST
CALL  -> CALL
RAISE -> RAISE
```

Puis au moment de l’action finale :

```text
NO_INVEST + check possible     -> CHECK
NO_INVEST + check impossible   -> FOLD
```

But :

* simplifier le problème
* réduire la confusion CHECK/FOLD
* vérifier si le modèle devient plus robuste
* conserver le modèle 4 classes comme baseline

Nom proposé :

```text
pokerbench_oracle_3intent_v1
```

---

## Prochaine amélioration modèle

Tester une séparation preflop / postflop :

```text
PREFLOP -> modèle preflop
FLOP/TURN/RIVER -> modèle postflop
```

Justification :

Les performances actuelles montrent que le modèle global apprend mieux le preflop que le postflop.

Résultats observés :

```text
PREFLOP : meilleur score
FLOP   : plus faible
TURN   : plus faible
RIVER  : plus faible
```

Nom proposé :

```text
pokerbench_oracle_router_v1
```

---

## Règles de projet

1. Ne pas brancher automatiquement le modèle au bot live.
2. Garder le mode shadow uniquement tant que la robustesse n’est pas validée.
3. Ne pas utiliser les colonnes label/debug/audit/raw text dans X_train.
4. Ne pas comparer directement un score issu de resampling avec un score oracle.
5. Ne pas gonfler artificiellement les datasets pour améliorer les métriques.
6. Privilégier les vraies lignes oracle à la génération artificielle.
7. Garder les anciens pipelines en archive, pas en référence principale.


