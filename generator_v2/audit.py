"""Zero-tolerance structural checks over a completed simulation."""

from __future__ import annotations

from math import isinf
from typing import Any

from .config import InputBundle
from .curriculum import Curriculum
from .schema import (
    FEATURE_PRE_T,
    LOGICAL_TABLE_NAMES,
    PUBLIC_COLUMN_ROLES,
    RUN_KEYS,
    public_contract_violations,
)


class StructuralAuditError(AssertionError):
    """Raised when generated data violate a structural invariant."""


def _contains_infinite(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_infinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_infinite(item) for item in value)
    try:
        return bool(isinf(float(value)))
    except (TypeError, ValueError):
        return False


def run_structural_audit(result: Any, bundle: InputBundle, curriculum: Curriculum) -> dict[str, Any]:
    """Validate temporal, schema, identity, curricular and event integrity."""

    checks: dict[str, bool] = {}
    details: dict[str, str] = {}

    def add(name: str, condition: bool, detail: str = "") -> None:
        checks[name] = bool(condition)
        if detail:
            details[name] = detail

    tables = result.tables()
    contract_violations = public_contract_violations(tables)
    add("six_logical_tables", set(tables) == LOGICAL_TABLE_NAMES)
    add("public_availability_contract", not contract_violations, repr(contract_violations))
    add(
        "pre_t_contract_excludes_outcomes",
        all(
            role == FEATURE_PRE_T
            for table in PUBLIC_COLUMN_ROLES.values()
            for column, role in table.items()
            if column.endswith("_pre") or column.startswith("z_") or column == "nota_previa"
        ),
    )

    expected_identity = {
        "scenario_id": str(result.metadata["scenario_id"]),
        "seed": int(result.metadata["seed"]),
        "run_id": str(result.metadata["run_id"]),
    }
    identity_ok = True
    identity_detail = ""
    for table_name, frame in tables.items():
        if not RUN_KEYS <= set(frame.columns):
            identity_ok = False
            identity_detail = f"{table_name}: faltan {sorted(RUN_KEYS-set(frame.columns))}"
            break
        if any(not frame[key].eq(value).all() for key, value in expected_identity.items()):
            identity_ok = False
            identity_detail = f"{table_name}: identidad de corrida inconsistente"
            break
    add("consistent_run_identity", identity_ok, identity_detail)
    add("metadata_has_no_infinity", not _contains_infinite(result.metadata))

    ramos = result.ramos
    add(
        "fifty_courses",
        len(ramos) == 50 and ramos["course_id"].is_unique and ramos["dag_id"].is_unique,
    )
    expected_crosswalk = dict(
        zip(bundle.courses["course_id"].astype(str), bundle.courses["dag_id"].astype(str), strict=True)
    )
    actual_crosswalk = dict(zip(ramos["course_id"].astype(str), ramos["dag_id"].astype(str), strict=True))
    add("course_id_dag_id_crosswalk", actual_crosswalk == expected_crosswalk)

    expected_students = int(bundle.active_cohorts["n_estudiantes_simulados"].sum())
    students = result.estudiantes
    add(
        "student_count",
        len(students) == expected_students
        and not students.duplicated(["scenario_id", "seed", "student_id"]).any(),
        f"expected={expected_students}, found={len(students)}",
    )
    cohort_counts = students.groupby("cohorte").size().to_dict()
    expected_counts = {
        int(row.cohorte): int(row.n_estudiantes_simulados)
        for row in bundle.active_cohorts.itertuples(index=False)
    }
    add("cohort_counts", cohort_counts == expected_counts, f"found={cohort_counts}")
    student_audit = result.auditoria.loc[result.auditoria["record_type"].eq("student_effect")]
    physical_pool_sizes = bundle.profiles.groupby("cohorte").size().astype(int).to_dict()
    sampling_by_cohort_ok = True
    sampling_details: dict[int, dict[str, int]] = {}
    for cohort, simulated_size in expected_counts.items():
        physical_pool_size = int(physical_pool_sizes.get(cohort, 0))
        cohort_audit = student_audit.loc[student_audit["cohorte"].eq(cohort)]
        actual = cohort_audit["tipo_muestreo_perfil"].value_counts().to_dict()
        expected = {
            "ORIGINAL": physical_pool_size if simulated_size >= physical_pool_size else 0,
            "SUBMUESTREO_SIN_REEMPLAZO": simulated_size if simulated_size < physical_pool_size else 0,
            "BOOTSTRAP_ADICIONAL": max(0, simulated_size - physical_pool_size),
        }
        sampling_details[cohort] = {key: int(actual.get(key, 0)) for key in expected}
        sampling_by_cohort_ok = sampling_by_cohort_ok and (
            len(cohort_audit) == simulated_size
            and all(int(actual.get(key, 0)) == value for key, value in expected.items())
            and set(actual) <= set(expected)
        )
    add(
        "profile_sampling_by_cohort",
        sampling_by_cohort_ok,
        f"found={sampling_details}",
    )

    registrations = result.inscripciones
    add(
        "unique_student_course_period",
        not registrations.duplicated(
            ["scenario_id", "seed", "student_id", "course_id", "periodo"]
        ).any(),
    )
    add(
        "registration_id_crosswalk",
        registrations["course_id"].map(expected_crosswalk).eq(registrations["dag_id"]).all(),
    )
    add("grade_range", registrations["nota_oficial"].between(1.0, 7.0, inclusive="both").all())
    grade_audit = result.auditoria.loc[result.auditoria["record_type"].eq("grade_internal")]
    add(
        "raw_grades_audit_only",
        len(grade_audit) == len(registrations)
        and grade_audit["nota_cruda"].between(1.0, 7.0, inclusive="both").all(),
    )
    expected_approved = registrations["nota_oficial"].ge(
        float(bundle.course_parameters["nota_aprobacion"])
    ).astype(int)
    add("approval_from_grade", expected_approved.equals(registrations["aprobado"].astype(int)))
    add("attempts_between_1_and_3", registrations["intento"].between(1, 3).all())

    attempt_ok = True
    for _, rows in registrations.sort_values("periodo").groupby(
        ["student_id", "dag_id"], sort=False
    ):
        attempts = rows["intento"].astype(int).tolist()
        if attempts != list(range(1, len(attempts) + 1)):
            attempt_ok = False
            break
    add("attempts_sequential", attempt_ok)

    s1 = registrations.loc[registrations["semestres_desde_ingreso_pre"].eq(1)]
    add(
        "s1_pre_t_zero_history",
        not s1.empty
        and s1["promedio_acumulado_pre"].isna().all()
        and s1["promedio_ultimo_semestre_pre"].isna().all()
        and s1["nota_previa"].isna().all()
        and s1[
            [
                "aprobados_acumulados_pre",
                "reprobados_acumulados_pre",
                "avance_real_pre",
                "avance_esperado_pre",
                "rezago_pre",
                "z_promedio_acumulado_pre",
                "z_promedio_ultimo_semestre_pre",
                "z_nota_previa_mismo_ramo",
            ]
        ].fillna(0).eq(0).all().all(),
    )

    panel = result.matriculas
    add("load_maximum", panel["carga"].le(int(bundle.general_parameters["carga_maxima"])).all())
    add(
        "s1_load",
        panel.loc[panel["semestres_desde_ingreso"].eq(1), "carga"].le(
            int(bundle.general_parameters["carga_S1"])
        ).all(),
    )

    prerequisites_ok = True
    passed_by_student: dict[str, set[str]] = {}
    for period in bundle.active_periods:
        current = registrations.loc[registrations["periodo"].eq(period)]
        for row in current.itertuples(index=False):
            passed = passed_by_student.setdefault(str(row.student_id), set())
            if not curriculum.prerequisites[str(row.dag_id)].issubset(passed):
                prerequisites_ok = False
                break
        if not prerequisites_ok:
            break
        approved = current.loc[current["aprobado"].eq(1)]
        for student_id, rows in approved.groupby("student_id", sort=False):
            passed_by_student.setdefault(str(student_id), set()).update(rows["dag_id"].astype(str))
    add("prerequisites_strictly_pre_t", prerequisites_ok)

    period_order = {period: index for index, period in enumerate(bundle.active_periods)}
    post_event_ok = True
    terminal = panel.loc[panel["evento_cierre"].isin(["GRADUADO", "ELIMINADO", "NO_CONTINUIDAD"])]
    for row in terminal.itertuples(index=False):
        later = registrations.loc[
            registrations["student_id"].eq(row.student_id)
            & registrations["periodo"].map(period_order).gt(period_order[str(row.periodo)])
        ]
        if not later.empty:
            post_event_ok = False
            break
    add("no_post_terminal_records", post_event_ok)
    add("all_final_states_resolved", not students["estado_final"].eq("ACTIVO").any())
    add(
        "final_states_valid",
        students["estado_final"].isin(
            ["CENSURADO", "GRADUADO", "ELIMINADO", "NO_CONTINUIDAD"]
        ).all(),
    )
    last_period = bundle.active_periods[-1]
    censored = students.loc[students["estado_final"].eq("CENSURADO")]
    add(
        "censoring_exactly_at_horizon",
        not censored.empty
        and censored["periodo_terminal"].eq(last_period).all()
        and panel.loc[panel["censurado"].eq(1), "periodo"].eq(last_period).all(),
    )
    terminal_panel = panel.loc[panel["evento_cierre"].notna()]
    add(
        "one_terminal_or_censor_row_per_student",
        len(terminal_panel) == len(students)
        and terminal_panel.groupby("student_id").size().eq(1).all(),
    )

    course_audit = result.auditoria.loc[result.auditoria["record_type"].eq("course_state")]
    add("course_state_rows", len(course_audit) == 50 * len(bundle.active_periods))
    add(
        "course_audit_id_crosswalk",
        course_audit["course_id"].map(expected_crosswalk).eq(course_audit["dag_id"]).all(),
    )
    add(
        "initial_delta_zero",
        course_audit.loc[
            course_audit["periodo"].eq(bundle.active_periods[0]), ["delta_mu", "delta_sd"]
        ].astype(float).eq(0.0).all().all(),
    )
    add(
        "beta_kappa_positive",
        course_audit["kappa_raw"].astype(float).gt(0.0).all()
        and course_audit["kappa"].astype(float).gt(0.0).all(),
    )
    add(
        "beta_adjustment_audited",
        course_audit["ajuste_beta"].isin([0, 1]).all()
        and course_audit["motivo_ajuste"].notna().all(),
    )

    aggregates = result.agregados
    add(
        "aggregate_rows",
        len(aggregates) == 50 * len(bundle.active_periods)
        and not aggregates.duplicated(["scenario_id", "seed", "course_id", "periodo"]).any(),
    )
    add(
        "aggregate_id_crosswalk",
        aggregates["course_id"].map(expected_crosswalk).eq(aggregates["dag_id"]).all(),
    )
    add(
        "aggregate_enrollment_reconciliation",
        int(aggregates["N"].sum()) == len(registrations),
        f"aggregate={int(aggregates['N'].sum())}, registrations={len(registrations)}",
    )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        explanation = "; ".join(f"{name}: {details.get(name, '')}" for name in failed)
        raise StructuralAuditError(f"Fallas estructurales: {explanation}")
    return {"passed": True, "n_checks": len(checks), "checks": checks}

