# Export dataset ML poker

Ce dossier n'est pas un export du projet complet. Il sert uniquement de contrat
simple pour partager ou reconstruire un dataset d'entrainement a partir des logs
du projet.

## Fichiers

- `example_training_dataset.jsonl` : petit exemple de lignes au format
  `ml_decision_snapshot`.
- `training_dataset.jsonl` : fichier genere automatiquement pendant les parties,
  avec seulement les lignes directement utilisables pour l'entrainement.

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
- boutons visibles : `buttons`, `buttons_active`, `has_check`, `has_call`,
  `has_raise`
- profils adverses calcules : `opponent_profiles`
- equity calculee : `equity_table`, `equity_1v1`, `equity_required`
- EV et seuil de call : `ev`, `call_max`
- decision legacy : `legacy_action`, `legacy_reason`, `legacy_raise_amount`
- flags qualite : cartes incertaines, board transitoire, boutons incoherents,
  compteur adversaires incertain, etc.

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
