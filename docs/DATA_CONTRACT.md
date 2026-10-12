# Data contract

## Run identity

Every generated table carries `scenario_id`, `seed`, and `run_id`. A student is uniquely identified across runs by `(scenario_id, seed, student_id)`.

## Logical tables

- `Ramos`: curriculum references and prerequisite structure.
- `Estudiantes`: pre-entry features plus final trajectory state.
- `Matriculas_estudiante_periodo`: one longitudinal row per active student-period.
- `Inscripciones_estudiantes`: one course registration per student, course, and period.
- `Agregados_curso_periodo`: course-period summaries derived from registrations.
- `Auditoria`: internal stochastic state and technical diagnostics.

## Temporal roles

- `FEATURE_PRE_T`: information available before period `t` is generated.
- `OUTCOME_POST_T`: result known only after period `t` or the trajectory closes.
- `IDENTIFIER`: key used for identity or joins.
- `REFERENCE`: structural or configured context.
- `AUDIT_ONLY`: internal state excluded from the five non-audit datasets.

The analysis-ready workbook contains the five non-audit tables and a `Data_dictionary` sheet. It never includes the audit table.

## Reproducibility

The RNG manager derives deterministic substreams from the run seed and stable entity keys. Repeating a run with the same public inputs, code, environment, scenario, and seed produces byte-identical canonical CSV tables. `table_hashes.json` records their SHA-256 values.
