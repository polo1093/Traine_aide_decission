# Pipeline dataset solver

Ce dossier peut maintenant generer un dataset `ml_decision_snapshot` labelise
par solver sans modifier l'export live `training_dataset.jsonl`.

Le solver sert uniquement a produire le label. Les `features` restent limitees
aux informations observables ou reconstructibles par le projet live : cartes
hero, board, street, position, joueurs actifs, pot, montant a payer, boutons,
profils adverses et features derivees. Les cartes adverses cachees, resultats
de main, EV internes solver, historique complet non structure et reads humains
ne sont jamais ecrits dans `features`.

## Format

La sortie suit strictement `example_training_dataset.jsonl` :

- `metadata.label_source` vaut `solver` ;
- `metadata.decision_mode` vaut `solver` ;
- le label solver normalise est ecrit dans `labels.legacy_action` et
  `labels.final_action`, car ce sont les champs cibles du contrat existant ;
- les actions autorisees sont seulement `FOLD`, `CHECK`, `CALL`, `RAISE` ;
- `BET` est normalise en `RAISE` ;
- `WAIT`, les cartes invalides, les streets incoherentes et les actions
  impossibles selon `buttons_active` sont rejetees.

Le contrat strict de l'exemple ne contient que `debug.decision_reason` et
`debug.scan_status`. Les sorties brutes du solver ne sont donc pas ajoutees au
JSONL strict ; elles restent disponibles au niveau adapter pour audit externe.

## Equites et features derivees

Le writer reutilise `feature_rebuilder.rebuild_derived_features(...)` pour
recalculer :

- `to_call_pot_ratio`
- `equity_required`
- `ev`
- `call_max`
- `buttons_active`
- `has_check`, `has_call`, `has_raise`
- `opponent_profiles`

Si les dependances du projet principal sont disponibles,
`feature_rebuilder.estimate_project_equity(...)` peut recalculer l'equite table.
Sinon, le mode synthetique utilise un fallback simple derive uniquement des
cartes observables, afin de permettre les tests et le branchement du solver reel
plus tard.

## Generer un dataset synthetique

Depuis ce dossier :

```bash
python -m solver_dataset_cli generate-solver-dataset --mode synthetic --n-spots 1000 --seed 42 --solver mock --output solver_training_dataset.jsonl
```

Depuis le dossier parent, si `ml_dataset_export` est importable :

```bash
python -m ml_dataset_export generate-solver-dataset --mode synthetic --n-spots 1000 --seed 42 --solver mock --output ml_dataset_export/solver_training_dataset.jsonl
```

## Relabeler des snapshots existants

```bash
python -m solver_dataset_cli generate-solver-dataset --mode relabel-existing --input-existing training_dataset.jsonl --n-spots 1000 --solver mock --output solver_training_dataset.jsonl
```

## Valider le JSONL

```bash
python -m solver_dataset_cli validate-solver-dataset --input solver_training_dataset.jsonl --example example_training_dataset.jsonl
```

La validation detecte les champs manquants, champs inconnus, mauvais types,
actions invalides, `WAIT`, features interdites, cartes invalides, doublons et
boards incoherents avec la street.

## Brancher un vrai solver

Deux options existent :

- implementer une classe compatible avec `SolverAdapter.solve_spot(...)` ;
- exposer un module Python avec une fonction `solve_spot(spot)` puis lancer :

```bash
python -m solver_dataset_cli generate-solver-dataset --solver external --external-solver-module mon_solver --mode synthetic --n-spots 1000 --output solver_training_dataset.jsonl
```

Le module externe doit retourner un `SolverDecision` ou un dict avec `action`,
`raise_amount`, `confidence` et `raw`. Le solver peut utiliser autant
d'informations internes que necessaire pour resoudre le spot, mais seules les
donnees observables du `SolverSpot` sont recopiees dans `features`.
