# Export dataset ML poker

Ce dossier n'est pas un export du projet complet. Il sert uniquement de contrat
simple pour partager ou reconstruire un dataset d'entrainement a partir des logs
du projet.

## Fichiers

- `example_training_dataset.jsonl` : petit exemple de lignes au format
  `ml_decision_snapshot`.
- `training_dataset.jsonl` : fichier genere automatiquement pendant les parties,
  avec seulement les lignes directement utilisables pour l'entrainement.
- `feature_rebuilder.py` : fonctions copiables dans un autre projet pour
  reconstruire les features derivees et les features "maison".

Le vrai dataset doit etre regenere depuis ce projet avec la telemetry active.
Depuis maintenant, `ml_decision_snapshot` est active par defaut au lancement. Les
futures mains seront donc ecrites dans :

```text
logs/telemetry/<game>/sessions/<session_id>/hands/hand_<hand_id>.jsonl
```

En plus des logs complets, le projet ajoute automatiquement les lignes
entrainables ici :

```text
dist/ml_dataset_export/training_dataset.jsonl
```

Ce fichier est volontairement plus simple que les logs complets : il ne contient
pas les `cycle_snapshot`, pas les `player_snapshot`, et pas les labels non
entrainables comme `WAIT`.

## Ce que le projet sait capter

Ces champs sont realistes parce qu'ils existent dans le scanner, le moteur de
decision ou la telemetry actuelle :

- cartes hero detectees : `hero_cards`
- board detecte : `board_cards`
- street : `PREFLOP`, `FLOP`, `TURN`, `RIVER`
- position hero si detectee/configuree : `hero_position`
- joueurs actifs/de depart : `player_active`, `player_start`
- pot, montant a payer, ratio `to_call/pot`
- montants normalises en big blinds : `pot_bb`, `to_call_bb`,
  `legacy_raise_amount_bb`, `buttons[].value_bb`, `players[].stack_bb`
- boutons visibles : `buttons`, `buttons_active`, `has_check`, `has_call`,
  `has_raise`
- profils adverses calcules : `opponent_profiles`
- equity calculee : `equity_table`, `equity_1v1`, `equity_required`
- EV et seuil de call : `ev`, `call_max`
- decision legacy : `legacy_action`, `legacy_reason`, `legacy_raise_amount`
  et `legacy_raise_amount_bb`
- flags qualite : cartes incertaines, board transitoire, boutons incoherents,
  compteur adversaires incertain, etc.

## Features reconstructibles

Le dataset garde les features observees et plusieurs features deja calculees.
Si tu veux repartir de logs plus bruts dans un autre projet, `feature_rebuilder.py`
permet de reconstruire :

- `to_call_pot_ratio` : `to_call / pot`
- `amount_unit`, `amount_unit_value`, `amount_unit_source`
- les montants suffixes `_bb` en divisant par la big blind courante
- `equity_required` : `to_call / (pot + to_call)`
- `ev` : `equity_table * (pot + to_call) - to_call`
- `call_max` : `(equity_table * pot) / (1 - equity_table)`
- `buttons_active`, `has_check`, `has_call`, `has_raise`
- `opponent_profiles` a partir des joueurs actifs, actions et stacks
- `starting_hand_strength` et `preflop_notation` a partir des cartes hero
- filtre `is_trainable_row(...)` identique a l'export live minimal

Exemple :

```python
import json
from feature_rebuilder import rebuild_derived_features, is_trainable_row

with open("training_dataset.jsonl", "r", encoding="utf-8") as handle:
    row = json.loads(next(handle))

if is_trainable_row(row):
    rebuilt = rebuild_derived_features(row)
    print(rebuilt["equity_required"], rebuilt["ev"], rebuilt["call_max"])
```

Par defaut, le fichier ne relance pas la simulation d'equite. Si tu executes ce
module depuis le depot complet avec les dependances poker disponibles, tu peux
demander un recalcul exact de l'equite table :

```python
rebuilt = rebuild_derived_features(row, recompute_equity=True, simulations=600)
```

Dans un repo ML separe, le plus simple est souvent de garder `equity_table`
stockee dans le JSONL, puis de recalculer les ratios, EV, call max et profils si
tu regeneres des exemples depuis des logs plus bruts.

## Ce qu'il ne faut pas inventer

Ne pas mettre dans les features :

- cartes adverses cachees ;
- resultat final de la main si le projet ne l'a pas observe ;
- action history complete si elle n'est pas encore structuree ;
- solver label ou EV theorique externe ;
- reads humains non captes par la telemetry.

Ces informations pourront etre ajoutees plus tard, mais pas dans ce dataset de
base.

## Filtre minimal pour entrainer

Pour entrainer un modele supervise sur les labels du moteur legacy, garder
seulement les lignes qui respectent :

```text
type == "ml_decision_snapshot"
metadata.decision_engine_version == "decision_engine_v2"
metadata.legacy_rules_version == "legacy_rules_v2"
quality_flags.usable_for_training == true
labels.label_valid == true
labels.known_bug_risk == false
quality_flags.amount_unit_missing == false
```

Les actions `WAIT` ne doivent pas servir de labels d'action.

## Format d'une ligne

Chaque ligne JSONL est autonome :

- `metadata` identifie la main, la street, le mode et la version des regles ;
- `features` contient les inputs observables ;
- `labels` contient la decision prise par le moteur legacy ;
- `confidence` contient les confiances OCR/detection quand disponibles ;
- `quality_flags` permet de filtrer les lignes sales ;
- `debug` aide a auditer une ligne sans servir directement au modele.

## Usage attendu dans un autre projet

Copier seulement ce dossier ou seulement le fichier JSONL d'exemple. Dans le
projet principal, regenerer ensuite le vrai dataset depuis les logs live. Le
dataset reel doit rester separe du code applicatif et peut etre versionne dans
un repo dedie si besoin.
