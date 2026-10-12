# Synthetic longitudinal student generator — public demo

This repository is a fully synthetic demonstration of a longitudinal student-data generator. It does not represent a real university, a real academic program, or observed students. Course names, identifiers, cohorts, admission routes, scores, reference values, and model parameters are fictitious and exist only to demonstrate the software workflow.

The public configuration is scientifically separated from any applied research configuration. It preserves the generic simulation mechanisms and output contract, but it does not reproduce an institutional curriculum, population, calibration, or result.

## Quick start

Install the pinned dependencies and run the demonstration with seed 42:

```powershell
python -m pip install -r requirements.txt
python -B scripts/run_demo_generator.py --seed 42 --output outputs/demo_seed_42
python -B scripts/verify_demo_run.py --seed 42 --run-directory outputs/demo_seed_42
```

The output directory is never overwritten. Choose a new path for another run.

## Demo inputs

The generator reads exactly two workbooks from `inputs/demo/`:

- `Demo_Courses.xlsx`: 50 fictitious courses, ten levels, a newly designed acyclic prerequisite graph, synthetic course-period references, and generic mechanism settings.
- `Demo_Trajectories.xlsx`: fictitious cohorts, synthetic admission profiles, demonstration routes, and synthetic longitudinal references.

Both files were created from scratch for this public package. They are not renamed copies of applied data.

## Output contract

Each run produces six canonical CSV tables:

- `Ramos.csv`
- `Estudiantes.csv`
- `Matriculas_estudiante_periodo.csv`
- `Inscripciones_estudiantes.csv`
- `Agregados_curso_periodo.csv`
- `Auditoria.csv`

It also produces `metadata.json`, `table_hashes.json`, `run_summary.md`, and `synthetic_student_data.xlsx`. The analysis-ready workbook contains the five non-audit tables plus `Data_dictionary`; internal audit fields remain in the separate audit CSV.

## Validation and privacy

Run the public test suite and the deidentification gate:

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/public_deidentification_gate.py --root .
```

The tests cover input loading, preflight checks, DAG acyclicity, complete generation, structural audit, output completeness, analysis-ready workbook structure, deterministic reproducibility, and workbook metadata hygiene.

## Scope

This package demonstrates data generation and reproducibility. It is not an empirical dataset, a validated forecasting model, a causal model, or evidence about any educational organization. See `docs/PUBLIC_DEMO_CONFIGURATION.md`, `docs/DATA_CONTRACT.md`, and `docs/LIMITATIONS.md` for details.
