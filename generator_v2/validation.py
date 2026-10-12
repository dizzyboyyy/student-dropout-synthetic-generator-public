"""Structural preflight validation for the effective input bundle."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import isclose
from typing import Iterable

import numpy as np
import pandas as pd

from .config import InputBundle


class InputValidationError(ValueError):
    """Raised when an input violates a zero-tolerance structural rule."""


@dataclass(frozen=True)
class CheckResult:
    """One named preflight assertion."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    """Collection of preflight assertions."""

    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [item for item in self.checks if not item.passed]

    def require_success(self) -> None:
        if not self.passed:
            details = "; ".join(f"{item.name}: {item.detail}" for item in self.failed)
            raise InputValidationError(f"Preflight BLOCKING: {details}")


def _expected_periods(first_year: int, last_year: int) -> list[str]:
    return [f"{year}-{semester:02d}" for year in range(first_year, last_year + 1) for semester in (1, 2)]


def _is_acyclic(nodes: Iterable[str], edges: list[tuple[str, str]]) -> bool:
    node_set = set(nodes)
    if any(source not in node_set or target not in node_set for source, target in edges):
        return False
    indegree = {node: 0 for node in node_set}
    successors: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        successors[source].append(target)
        indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in successors[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(indegree)


def validate_inputs(bundle: InputBundle) -> ValidationReport:
    """Run all structural assertions required before the motor may start."""

    checks: list[CheckResult] = []

    def add(name: str, condition: bool, detail: str = "") -> None:
        checks.append(CheckResult(name=name, passed=bool(condition), detail=detail))

    courses = bundle.courses
    course_ids = courses["course_id"].tolist()
    dag_ids = courses["dag_id"].tolist()
    expected_course_ids = [f"COURSE_{index:03d}" for index in range(1, 51)]
    expected_dag_ids = [f"COURSE_{index:03d}" for index in range(1, 51)]
    add("exactamente 50 cursos", len(courses) == 50, f"n={len(courses)}")
    add("COURSE_001-050 como course_id únicos", course_ids == expected_course_ids and len(set(course_ids)) == 50)
    add("COURSE_001-050 como dag_id únicos", dag_ids == expected_dag_ids and len(set(dag_ids)) == 50)
    add(
        "crosswalk 1:1",
        all(course_id == dag_id for course_id, dag_id in zip(course_ids, dag_ids, strict=True)),
    )
    levels = dict(zip(dag_ids, courses["nivel"].astype(int), strict=True))
    add("niveles demo 1-10", all(1 <= level <= 10 for level in levels.values()))

    prerequisite_map = dict(
        zip(
            bundle.prerequisites["codigo_asignatura"],
            bundle.prerequisites["prerequisite_list"],
            strict=True,
        )
    )
    edges = [(prerequisite, course) for course, prerequisites in prerequisite_map.items() for prerequisite in prerequisites]
    add("50 filas de prerrequisitos", len(prerequisite_map) == 50)
    add("prerrequisitos cubren catálogo", set(prerequisite_map) == set(dag_ids))
    add("prerrequisitos existentes", all(source in levels for source, _ in edges))
    add("sin self-loops", all(source != target for source, target in edges))
    add(
        "prerrequisitos en nivel anterior",
        all(
            source in levels and target in levels and levels[source] < levels[target]
            for source, target in edges
        ),
    )
    add("DAG acíclico", _is_acyclic(dag_ids, edges))
    expected_final = {
        "COURSE_031",
        "COURSE_034",
        "COURSE_038",
        "COURSE_040",
        "COURSE_045",
    }
    add(
        "curso terminal usa el cierre demo",
        set(prerequisite_map.get("COURSE_050", [])) == expected_final,
        repr(prerequisite_map.get("COURSE_050", [])),
    )

    periods = bundle.periods
    period_ids = periods["periodo_id"].tolist()
    expected_periods = _expected_periods(2031, 2044)
    expected_active = _expected_periods(2040, 2044)
    add("periodos de referencia completos", period_ids == expected_periods)
    add("horizonte baseline exacto", bundle.active_periods == expected_active)
    add(
        "períodos de referencia demo no simulados",
        not periods.loc[periods["anio"].astype(int) < 2040, "usar_en_simulacion"].eq(True).any(),
    )

    for name, matrix, lower, upper in (
        ("Media_curso_periodo", bundle.media_course_period, 1.0, 7.0),
        ("DE_curso_periodo", bundle.sd_course_period, 0.0, 3.0),
    ):
        values = matrix.to_numpy(dtype=float)
        add(f"{name} dimensión", matrix.shape == (28, 50), f"shape={matrix.shape}")
        add(f"{name} cursos", matrix.columns.tolist() == expected_course_ids)
        add(f"{name} periodos", matrix.index.tolist() == expected_periods)
        add(f"{name} sin faltantes", bool(np.isfinite(values).all()))
        add(
            f"{name} rango",
            bool(((values > lower) & (values < upper)).all()),
            f"min={np.nanmin(values)}, max={np.nanmax(values)}",
        )

    cohort_map = bundle.cohorts.set_index("cohorte")
    active_years = set(bundle.active_cohorts["cohorte"].astype(int))
    add("cohortes activas demo 2040-2044", active_years == set(range(2040, 2045)))

    simulated_sizes = pd.to_numeric(
        bundle.cohorts["n_estudiantes_simulados"], errors="coerce"
    )
    integral_nonnegative_sizes = (
        simulated_sizes.notna()
        & np.isfinite(simulated_sizes)
        & simulated_sizes.eq(simulated_sizes.round())
        & simulated_sizes.ge(0)
    )
    add(
        "tamaños de cohorte enteros no negativos",
        bool(integral_nonnegative_sizes.all()),
    )
    active_mask = bundle.cohorts["usar_en_simulacion"].eq(True)
    add(
        "tamaños de cohortes activas positivos",
        bool(
            integral_nonnegative_sizes.loc[active_mask].all()
            and simulated_sizes.loc[active_mask].gt(0).all()
        ),
    )
    add(
        "cohortes históricas desactivadas",
        all(
            cohort_map.loc[year, "usar_en_simulacion"] == False
            and pd.to_numeric(
                pd.Series([cohort_map.loc[year, "n_estudiantes_simulados"]]),
                errors="coerce",
            ).eq(0).all()
            for year in range(2031, 2040)
        ),
    )

    profiles = bundle.profiles
    profile_counts = Counter(profiles["cohorte"].astype(int))
    add("perfil IDs únicos", profiles["id_perfil"].is_unique and profiles["id_perfil"].notna().all())
    add(
        "pools físicos disponibles para cohortes activas",
        all(profile_counts[year] > 0 for year in active_years),
    )
    add(
        "perfiles pertenecen a cohortes configuradas",
        set(profile_counts) <= set(bundle.cohorts["cohorte"].astype(int)),
    )
    demo_routes = {"REGULAR", "ADMISSION_A", "ADMISSION_B", "ADMISSION_C"}
    add("vías demo válidas", set(profiles["via_ingreso"]) <= demo_routes)
    add(
        "preferencias válidas",
        bool((pd.to_numeric(profiles["preferencia"], errors="coerce") >= 1).all()),
    )
    expected_first = profiles["preferencia"].astype(int).eq(1)
    add("primera_opcion correcta", expected_first.equals(profiles["primera_opcion"]))

    signal_errors: list[float] = []
    population_reference_ok = True
    for year, cohort_row in cohort_map.loc[sorted(active_years)].iterrows():
        cohort_profiles = profiles.loc[profiles["cohorte"].astype(int) == int(year)]
        regular_scores = cohort_profiles.loc[
            cohort_profiles["via_ingreso"].eq("REGULAR"), "puntaje_admision"
        ].astype(float)
        population_reference_ok &= (
            len(regular_scores) == int(cohort_row["n_regular_referencia"])
            and isclose(float(regular_scores.mean()), float(cohort_row["media_puntaje_regular"]), abs_tol=1e-12)
            and isclose(float(regular_scores.std(ddof=0)), float(cohort_row["de_puntaje_regular"]), abs_tol=1e-12)
        )
        for row in cohort_profiles.itertuples(index=False):
            expected = (
                0.0
                if row.via_ingreso == "ADMISSION_B"
                else (float(row.puntaje_admision) - float(cohort_row["media_puntaje_regular"]))
                / float(cohort_row["de_puntaje_regular"])
            )
            signal_errors.append(abs(expected - float(row.senal_admision)))
    add("referencias REGULAR ddof=0", population_reference_ok)
    add("señal de admisión", max(signal_errors, default=float("inf")) <= 1e-12)
    neutral_route = profiles.loc[profiles["via_ingreso"].eq("ADMISSION_B")]
    add(
        "vía demo neutral con señal cero",
        len(neutral_route) > 0
        and neutral_route["senal_admision"].astype(float).eq(0.0).all(),
    )
    routes_per_cohort = profiles.groupby("cohorte")["via_ingreso"].agg(set)
    add("cuatro vías demo por cohorte", all(routes == demo_routes for routes in routes_per_cohort))

    required_course = {
        "distribucion_notas", "nota_minima", "nota_maxima", "nota_aprobacion",
        "decimales_nota_oficial", "factor_kappa_inicial", "rho_media",
        "variacion_media_95_abs", "rho_de", "variacion_de_95_rel",
        "beta_binomial_habilitada", "usar_tasa_aprobacion_para_generar",
        "usar_n_inscritos_como_capacidad", "cursos_disponibles_todos_semestres",
    }
    required_general = {
        "periodo_inicio_simulacion", "periodo_fin_simulacion", "ingreso_regular_semestre",
        "carga_S1", "carga_nominal_desde_S2", "carga_maxima",
        "activar_sexto_si_hay_extra", "criterio_sexto_ramo", "retiros_asignatura",
        "stopout_reingreso", "eliminacion_tercer_intento",
        "graduacion_por_trabajo_titulacion", "seed_desarrollo", "replicas_baseline",
        "semillas_baseline",
    }
    required_person = {
        "distribucion_b_i", "sigma_b", "beta_admision", "peso_residual_admision",
        "vida_media_admision_semestres", "beta_H_promedio_acumulado",
        "beta_H_ultimo_semestre", "beta_H_nota_previa_mismo_ramo",
        "distribucion_phi_i", "media_phi_i", "sigma_phi_i", "p1_no_continuidad",
        "p_inf_no_continuidad", "lambda_D_no_continuidad", "gamma_R_senal_reciente",
        "gamma_L_rezago", "distribucion_u_i", "sigma_u_i",
        "prob_school_public", "prob_school_private_aided",
        "prob_school_private", "prob_school_other",
        "referencia_senal_admision", "senal_admision_neutral",
    }
    add("parámetros curso completos", required_course <= set(bundle.course_parameters))
    add("parámetros generales completos", required_general <= set(bundle.general_parameters))
    add("parámetros persona completos", required_person <= set(bundle.person_parameters))

    probabilities = [
        float(bundle.person_parameters[key])
        for key in (
            "prob_school_public",
            "prob_school_private_aided",
            "prob_school_private",
            "prob_school_other",
        )
    ]
    add(
        "probabilidades dependencia",
        all(0.0 <= probability <= 1.0 for probability in probabilities)
        and isclose(sum(probabilities), 1.0, abs_tol=1e-12),
        f"sum={sum(probabilities)}",
    )
    p1 = float(bundle.person_parameters["p1_no_continuidad"])
    p_inf = float(bundle.person_parameters["p_inf_no_continuidad"])
    add("probabilidades no continuidad", 0.0 < p_inf <= p1 < 1.0)
    add(
        "escalas y concentración admisibles",
        all(
            float(value) > 0.0
            for value in (
                bundle.course_parameters["factor_kappa_inicial"],
                bundle.course_parameters["variacion_media_95_abs"],
                bundle.course_parameters["variacion_de_95_rel"],
                bundle.person_parameters["sigma_b"],
                bundle.person_parameters["sigma_u_i"],
                bundle.person_parameters["vida_media_admision_semestres"],
            )
        )
        and float(bundle.person_parameters["sigma_phi_i"]) >= 0.0,
    )
    add(
        "persistencias AR(1)",
        all(abs(float(bundle.course_parameters[key])) < 1.0 for key in ("rho_media", "rho_de")),
    )
    add(
        "reglas demo",
        bundle.course_parameters["distribucion_notas"] == "BETA_REESCALADA_1_7"
        and bundle.course_parameters["beta_binomial_habilitada"] is False
        and bundle.course_parameters["usar_tasa_aprobacion_para_generar"] is False
        and bundle.course_parameters["usar_n_inscritos_como_capacidad"] is False
        and bundle.course_parameters["cursos_disponibles_todos_semestres"] is True
        and bundle.general_parameters["retiros_asignatura"] is False
        and bundle.general_parameters["stopout_reingreso"] is False
        and bundle.general_parameters["eliminacion_tercer_intento"] is True
        and bundle.general_parameters["graduacion_por_trabajo_titulacion"] is True,
    )
    add(
        "reglas de carga demo",
        int(bundle.general_parameters["carga_S1"]) == 4
        and int(bundle.general_parameters["carga_nominal_desde_S2"]) == 5
        and int(bundle.general_parameters["carga_maxima"]) == 6
        and bundle.general_parameters["activar_sexto_si_hay_extra"] is True
        and bundle.general_parameters["criterio_sexto_ramo"]
        == "MENOR_NIVEL_REMANENTE_EMPATE_ALEATORIO",
    )
    add(
        "configuración ejecución",
        int(bundle.general_parameters["seed_desarrollo"]) == 42
        and int(bundle.general_parameters["replicas_baseline"]) == 10
        and bundle.general_parameters["semillas_baseline"] == "7001-7010",
    )

    return ValidationReport(checks)

