````md
# Traine_aide_decission

Projet expérimental offline pour générer, filtrer et résoudre des spots de poker avec PokerSolver, afin de préparer plus tard un dataset ML exploitable par un modèle de décision.

Le projet est séparé du projet live `aide_decision`.

## Objectif

Ce projet ne sert pas à jouer en live.

Il sert à construire une chaîne offline :

```text
synthetic spots / PokerTH histories
→ solver_job_v1
→ eligibility filter
→ subprocess solver runner
→ solver_run_result JSONL
→ analyse runtime
→ plus tard : labels candidats
→ plus tard : dataset ML
→ plus tard : modèle bootstrap
````

Pour l’instant, les sorties solver sont des traces techniques, pas des labels ML.

## État actuel

Le projet contient déjà :

* un adapter propre vers PokerSolver ;
* un backend Rust PokerSolver fonctionnel ;
* une couche `solver_jobs` avec validation ;
* un mapper snapshot ML vers solver job ;
* un générateur de spots synthétiques ;
* un runner de fichiers JSONL ;
* un filtre d’éligibilité avant solver ;
* un runner subprocess avec hard timeout ;
* un parser PokerTH conservateur ;
* un pipeline PokerTH vers solver batch ;
* des scripts expérimentaux bornés ;
* des tests unitaires couvrant les modules principaux.

## Ce que le projet ne fait pas encore

Le projet ne fait pas encore :

* entraînement ML ;
* génération de labels ML fiables ;
* création de `training_label` ;
* modèle RandomForest / LightGBM ;
* prédiction live ;
* intégration dans `aide_decision` ;
* solve massif non borné ;
* solve FLOP complexe fiable ;
* décision GTO exploitable directement.

## Structure du projet

```text
Traine_aide_decission/
│
├── projet importer/
│   └── poker_solver-main/
│       └── copie locale de PokerSolver
│
├── solvers/
│   ├── poker_solver_adapter.py
│   └── ...
│
├── solver_jobs/
│   ├── job_schema.py
│   ├── job_builder.py
│   ├── job_runner.py
│   ├── job_file_runner.py
│   ├── eligibility.py
│   ├── subprocess_runner.py
│   ├── solver_worker.py
│   └── snapshot_mapper.py
│
├── synthetic/
│   ├── deck.py
│   └── spot_generator.py
│
├── pokerth/
│   ├── history_parser.py
│   ├── snapshot_builder.py
│   └── pipeline.py
│
├── experiments/
│   ├── generate_synthetic_solver_jobs.py
│   ├── run_synthetic_solver_jobs.py
│   ├── calibrate_solver_runtime.py
│   ├── run_pokerth_solver_pipeline.py
│   └── ...
│
├── docs/
│   ├── solver_adapter.md
│   ├── solver_jobs.md
│   ├── synthetic_spot_generation.md
│   ├── solver_job_file_runner.md
│   ├── solver_subprocess_runner.md
│   ├── solver_runtime_calibration.md
│   ├── solver_eligibility.md
│   └── pokerth_import.md
│
├── tests/
│   └── tests unitaires
│
├── outputs/
│   └── fichiers générés localement, ignorés par Git
│
├── requirements.txt
├── pytest.ini
└── README.md
```

## Installation

### 1. Installer les dépendances Python

```powershell
python -m pip install -r requirements.txt
```

Dépendance minimale actuelle :

```text
psutil
```

### 2. Installer Rust

PokerSolver utilise un backend Rust via `maturin`.

Installer Rust stable :

```powershell
winget install --id Rustlang.Rustup -e
```

Puis rouvrir le terminal et vérifier :

```powershell
cargo --version
rustc --version
```

### 3. Installer maturin

```powershell
python -m pip install "maturin>=1.7,<2.0"
```

### 4. Installer les Visual Studio Build Tools

Sur Windows, la compilation Rust native nécessite `link.exe`.

Installer :

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

Dans l’installateur, sélectionner :

```text
Desktop development with C++
```

Composants nécessaires :

* MSVC ;
* Windows SDK ;
* C++ Build Tools.

Si `link.exe` n’est pas visible dans PowerShell standard, utiliser :

* Developer PowerShell for VS 2022 ;
* x64 Native Tools Command Prompt for VS 2022 ;
* ou charger `vcvars64.bat`.

### 5. Installer PokerSolver localement

Depuis le dossier PokerSolver :

```powershell
cd "projet importer\poker_solver-main"
python -m pip install -e .
```

Vérifier l’import Rust :

```powershell
python -c "import poker_solver._rust; print('ok')"
```

## Vérifier l’adapter PokerSolver

```powershell
python -m pytest tests/test_poker_solver_adapter.py
```

Résultat attendu :

```text
passed
```

Le backend doit indiquer :

```json
{
  "available": true,
  "rust_backend_available": true,
  "version": "1.7.0"
}
```

## Générer des jobs synthétiques

Le générateur crée des `solver_job_v1`.

Il ne lance pas le solver.

Exemple : générer 10 jobs

```powershell
python experiments/generate_synthetic_solver_jobs.py --count 10 --seed 42 --profile random_turn_spot --iterations 1 --timeout-s 5 --output outputs/synthetic_turn_10.jsonl
```

Exemple : générer 100 jobs

```powershell
python experiments/generate_synthetic_solver_jobs.py --count 100 --seed 42 --profile random_river_spot --iterations 1 --timeout-s 5 --output outputs/synthetic_river_100.jsonl
```

Profils existants :

```text
random_flop_spot
random_turn_spot
random_river_spot
drawy_board_spot
paired_board_spot
made_hand_vs_draw_spot
top_pair_spot
two_pair_plus_spot
```

Attention : tous les profils ne sont pas solver-safe actuellement.

## Filtre d’éligibilité solver

La calibration runtime a montré que seuls certains profils passent de manière stable.

Actuellement, le filtre autorise strictement :

```text
random_turn_spot
random_river_spot
```

avec :

```text
iterations <= 5
timeout_s <= 5
street in TURN/RIVER
```

Les profils suivants sont refusés pour l’instant :

```text
random_flop_spot
drawy_board_spot
paired_board_spot
top_pair_spot
two_pair_plus_spot
made_hand_vs_draw_spot
```

Un job refusé n’est pas “mauvais poker”.
Il est seulement considéré non sûr côté runtime.

## Lancer un batch solver borné

Exemple : solver seulement 5 jobs

```powershell
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_turn_10.jsonl --max-jobs 5 --output outputs/solver_runs/turn_5_results.jsonl
```

Validation sans solve réel :

```powershell
python experiments/run_synthetic_solver_jobs.py --input outputs/synthetic_turn_10.jsonl --max-jobs 5 --output outputs/solver_runs/turn_5_dry_run.jsonl --dry-run
```

Par défaut :

* `max_jobs = 5` ;
* refus de `max_jobs > 50` sans flag explicite ;
* subprocess tuable par job ;
* timeout dur ;
* aucun label ML généré.

## Subprocess runner

Chaque solve réel est lancé dans un processus séparé.

Avantages :

* le solve Rust peut être tué si timeout ;
* le batch ne reste pas bloqué ;
* les erreurs sont encapsulées proprement ;
* chaque résultat est écrit en JSONL.

Exemple de timeout :

```json
{
  "status": "failed",
  "solver_status": "timeout",
  "error": "solver_subprocess_timeout:5s",
  "quality": {
    "is_label_candidate": false,
    "exclusion_reason": "timeout"
  }
}
```

## Calibration runtime

Script :

```powershell
python experiments/calibrate_solver_runtime.py --jobs-per-profile 1 --iterations 1 5 --timeout-s 5 --output outputs/solver_calibration/calibration_short.jsonl
```

Résultat observé sur calibration courte :

```text
total_run: 16
successes: 4
timeouts: 12
errors: 0
```

Profils stables pour smoke test :

```text
random_turn_spot
random_river_spot
```

Profils trop lourds actuellement :

```text
random_flop_spot
drawy_board_spot
paired_board_spot
top_pair_spot
two_pair_plus_spot
made_hand_vs_draw_spot
```

## Pipeline PokerTH

Le projet sait aussi parser des historiques PokerTH.

Chaîne :

```text
PokerTH history
→ hand_summary
→ snapshot ml_dataset_v1
→ solver_job_v1
→ solver batch
→ solver_run_result JSONL
```

Le parser est conservateur.

Il rejette notamment :

```text
showdown_missing
villain_hand_missing
hero_hand_missing
multiway_context_not_supported
side_pot_not_supported
invalid_board
to_call_unknown
pot_reconstruction_failed
```

Les snapshots PokerTH gardent :

```json
{
  "usable_for_training": false,
  "usable_for_solver": true
}
```

Le résultat d’une main PokerTH n’est jamais converti automatiquement en label stratégique.

## Format solver_job_v1

Exemple :

```json
{
  "solver_job_id": "synthetic_solver_job_random_turn_spot_seed_42_000000",
  "source_snapshot_id": "synthetic_snapshot_random_turn_spot_seed_42_000000",
  "schema_version": "solver_job_v1",
  "source_type": "synthetic",
  "units": "chips",
  "street": "TURN",
  "hero_hand": ["Ah", "Kh"],
  "villain_hand": ["Qd", "Qc"],
  "villain_range": null,
  "board": ["2h", "7h", "9d", "Ts"],
  "pot": 100.0,
  "to_call": 20.0,
  "stack": 1000.0,
  "bet_sizes": [0.33],
  "iterations": 1,
  "timeout_s": 5.0,
  "backend": "rust",
  "label_intent": "solver_smoke",
  "generation_seed": 42,
  "generation_profile": "random_turn_spot"
}
```

## Format solver_run_result

Chaque ligne JSONL est autonome :

```json
{
  "record_type": "solver_run_result",
  "solver_job_id": "synthetic_solver_job_random_turn_spot_seed_42_000000",
  "source_snapshot_id": "synthetic_snapshot_random_turn_spot_seed_42_000000",
  "source_type": "synthetic",
  "solver_status": "ok",
  "solver_job": {},
  "solver_result": {},
  "quality": {
    "is_label_candidate": false,
    "exclusion_reason": "iterations_too_low"
  },
  "error": null,
  "warnings": [],
  "recorded_at": "2026-05-25T15:17:33+00:00"
}
```

Champs volontairement absents :

```text
training_label
gto_label
label_action
```

## Tests

Lancer toute la suite :

```powershell
python -m pytest
```

Tests ciblés utiles :

```powershell
python -m pytest tests/test_poker_solver_adapter.py
python -m pytest tests/test_solver_jobs.py
python -m pytest tests/test_synthetic_spot_generator.py
python -m pytest tests/test_solver_job_file_runner.py
python -m pytest tests/test_solver_subprocess_runner.py
python -m pytest tests/test_solver_eligibility.py
python -m pytest tests/test_pokerth_pipeline.py
```

## Discipline de développement

Règles du projet :

* écrire les tests avant ou pendant l’implémentation ;
* chaque bug doit ajouter un test de non-régression ;
* pas de refactor massif non validé ;
* aucun solve massif par défaut ;
* pas de label ML sans validation ;
* pas de modification du solver importé ;
* pas de modification de `aide_decision`.

## Roadmap

### Étape 1 — Stabilisé

* Adapter PokerSolver.
* Backend Rust activé.
* Solver jobs.
* Génération synthétique.
* Batch runner.
* Subprocess hard-timeout.
* Filtre d’éligibilité.
* Pipeline PokerTH conservateur.

### Étape 2 — En cours / prochaine étape

* Lancer de petits batches réels sur profils éligibles.
* Analyser les résultats solver.
* Identifier les profils utilisables.
* Construire un rapport de stabilité runtime.

### Étape 3 — Plus tard

* Définir des critères `is_label_candidate = true`.
* Transformer certains solver_run_result en labels candidats.
* Construire un dataset ML filtré.
* Entraîner un modèle bootstrap simple.
* Évaluer les erreurs dangereuses :

  * modèle prédit RAISE alors que le label est FOLD ;
  * modèle prédit CALL alors que le label est FOLD ;
  * modèle propose une action agressive sur données incertaines.

### Étape 4 — Beaucoup plus tard

* Export vers `aide_decision`.
* Mode `ml_shadow`.
* Comparaison legacy vs ML.
* Jamais de remplacement direct sans fallback.

## Limites actuelles

* Les solve FLOP timeoutent souvent.
* Les profils structurés complexes ne sont pas encore solver-safe.
* Le solver heavy est coûteux.
* Les jobs synthétiques ne représentent pas forcément une vraie distribution PokerTH.
* Les historiques PokerTH sont utiles pour validation, mais pas pour générer automatiquement des labels fiables.
* Les sorties solver actuelles sont des traces, pas des labels.
* Aucun modèle ML n’est encore entraîné.

## Résumé

Ce projet est une base offline propre pour préparer un futur dataset ML poker.

Le bon flux actuel :

```text
generate synthetic jobs
→ filter eligibility
→ run solver in subprocess
→ store solver_run_result
→ analyze runtime
```

Le flux à éviter :

```text
generate jobs
→ solver massif
→ labels ML automatiques
→ entraînement direct
```

Le projet privilégie la robustesse, la traçabilité et les garde-fous avant la performance.

```
```
