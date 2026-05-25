# PokerTH Action Replay

Le replayer reconstruit des contextes de decision hero a partir d'un `hand_summary`
deja parse par `pokerth.history_parser`.

Il sert uniquement a valider la chaine technique :

```text
PokerTH hand history
-> parse hand_summary
-> replay actions
-> detecter les decisions hero
-> produire des decision_snapshots ml_dataset_v1
-> mapper vers solver_job_v1
```

Il ne produit pas de label ML et ne doit pas etre utilise comme source de
strategie fiable.

## Pot reconstruit

Le pot est reconstruit en rejouant les actions dans l'ordre :

- small blind et big blind ajoutent leur montant au pot ;
- call ajoute exactement le montant a payer ;
- bet ajoute le montant mise par le joueur sur la street ;
- raise est interprete comme un montant total de contribution sur la street ;
- check et fold n'ajoutent rien.
- all-in est accepte seulement si le montant reste coherent avec call/bet/raise
  simple, sans side pot ni call partiel.

Cette reconstruction reste estimee. Les snapshots produits gardent donc :

```json
{
  "pot_is_estimated": true,
  "pot_reconstruction_method": "sum_posted_bet_call_raise_amounts"
}
```

## Calcul de `to_call`

Pour chaque decision hero, le replayer capture l'etat avant l'action hero.

```text
to_call = contribution_max_sur_la_street - contribution_hero_sur_la_street
```

Si hero peut checker gratuitement, `to_call` vaut `0.0` parce que le replay l'a
calcule explicitement. Il n'y a pas de fallback silencieux a `0.0`.

Si le replay ne peut pas reconstruire la decision, le resultat est `failed`.

## Exemple de contexte

```json
{
  "hero_name": "polo",
  "street": "FLOP",
  "hero_action": "CALL",
  "pot_before_action": 1120.0,
  "to_call": 480.0,
  "can_check": false,
  "can_call": true,
  "can_raise": true,
  "active_opponents": 1,
  "board_cards": ["9c", "5c", "8s"],
  "hero_cards": ["8h", "9s"],
  "villain_hand": ["8c", "Kc"],
  "decision_context_known": true
}
```

## Rejets stricts

Le replayer refuse les mains qui sortent du cadre simple :

- `multiway_context_not_supported`
- `side_pot_not_supported`
- `all_in_complex_not_supported`
- `amount_parse_failed`
- `pot_reconstruction_failed`
- `unknown_action`
- `villain_hand_missing`
- `hero_hand_missing`

Ces rejets sont volontaires : on prefere ne rien produire plutot que construire
un faux spot propre.

## Limites actuelles

- heads-up uniquement ;
- side pots refuses ;
- all-in simple accepte seulement quand le montant est explicite et coherent ;
- all-in partiel ou complexe refuse ;
- pas de rake ;
- pas de split pot detaille ;
- `raise` interprete comme montant total atteint sur la street ;
- `can_raise` reste une approximation no-limit simple ;
- les snapshots restent `usable_for_training: false` ;
- aucun `training_label`, `gto_label` ou `label_action` n'est genere.

## Tests

```bash
python -m pytest tests/test_pokerth_action_replayer.py
```
