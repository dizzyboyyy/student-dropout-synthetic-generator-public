"""Student sampling, persistent effects and longitudinal state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import InputBundle
from .rng import RNGManager


DEPENDENCY_PARAMETERS = (
    ("PUBLIC", "prob_school_public"),
    ("PRIVATE_AIDED", "prob_school_private_aided"),
    ("PRIVATE", "prob_school_private"),
    ("OTHER", "prob_school_other"),
)


@dataclass
class StudentState:
    """Mutable internal state for one simulated student."""

    student_id: str
    id_perfil_fuente: str
    cohorte: int
    puntaje_admision: float
    senal_admision: float
    via_ingreso: str
    preferencia: int
    primera_opcion: bool
    dependencia_escolar: str
    b_i: float
    phi_i: float
    u_i: float
    profile_draw_type: str
    status: str = "ACTIVO"
    terminal_period: str | None = None
    attempts: dict[str, int] = field(default_factory=dict)
    previous_grade: dict[str, float] = field(default_factory=dict)
    passed: set[str] = field(default_factory=set)
    all_official_grades: list[float] = field(default_factory=list)
    last_semester_grades: list[float] = field(default_factory=list)
    approved_attempts: int = 0
    failed_attempts: int = 0

    @property
    def cumulative_average(self) -> float | None:
        return float(np.mean(self.all_official_grades)) if self.all_official_grades else None

    @property
    def last_semester_average(self) -> float | None:
        return float(np.mean(self.last_semester_grades)) if self.last_semester_grades else None

    @property
    def historical_pass_ratio(self) -> float | None:
        total = self.approved_attempts + self.failed_attempts
        return self.approved_attempts / total if total else None

    def snapshot(
        self, semester_number: int, expected_progress_pre: int, blocked_count: int
    ) -> dict[str, Any]:
        """Freeze information available strictly before the current period."""

        actual_progress = len(self.passed)
        lag = max(0, expected_progress_pre - actual_progress)
        return {
            "semestres_desde_ingreso_pre": int(semester_number),
            "promedio_acumulado_pre": self.cumulative_average,
            "promedio_ultimo_semestre_pre": self.last_semester_average,
            "aprobados_acumulados_pre": int(self.approved_attempts),
            "reprobados_acumulados_pre": int(self.failed_attempts),
            "proporcion_aprobacion_pre": self.historical_pass_ratio,
            "avance_real_pre": int(actual_progress),
            "avance_esperado_pre": int(expected_progress_pre),
            "rezago_pre": int(lag),
            "bloqueos_pre": int(blocked_count),
        }

    def apply_period_results(self, results: Iterable[dict[str, Any]]) -> None:
        """Close the semester atomically after every current-period grade exists."""

        rows = list(results)
        self.last_semester_grades = [float(row["nota_oficial"]) for row in rows]
        for row in rows:
            course = str(row["dag_id"])
            official = float(row["nota_oficial"])
            approved = bool(row["aprobado"])
            self.attempts[course] = int(row["intento"])
            self.previous_grade[course] = official
            self.all_official_grades.append(official)
            if approved:
                self.passed.add(course)
                self.approved_attempts += 1
            else:
                self.failed_attempts += 1


def _sample_profiles(
    profiles: pd.DataFrame,
    n_sim: int,
    rng: np.random.Generator,
) -> list[tuple[pd.Series, str]]:
    """Use all, sample without replacement, or append reproducible bootstrap rows."""

    ordered = profiles.reset_index(drop=True)
    n_pool = len(ordered)
    if n_sim == n_pool:
        return [(ordered.iloc[index], "ORIGINAL") for index in range(n_pool)]
    if n_sim < n_pool:
        indices = rng.choice(n_pool, size=n_sim, replace=False)
        return [(ordered.iloc[int(index)], "SUBMUESTREO_SIN_REEMPLAZO") for index in indices]
    extra_indices = rng.choice(n_pool, size=n_sim - n_pool, replace=True)
    base = [(ordered.iloc[index], "ORIGINAL") for index in range(n_pool)]
    extras = [(ordered.iloc[int(index)], "BOOTSTRAP_ADICIONAL") for index in extra_indices]
    return base + extras


def create_students(bundle: InputBundle, rng_manager: RNGManager) -> list[StudentState]:
    """Create all enabled cohorts with immutable profile observables and effects."""

    person = bundle.person_parameters
    labels = [label for label, _ in DEPENDENCY_PARAMETERS]
    probabilities = np.array([float(person[key]) for _, key in DEPENDENCY_PARAMETERS], dtype=float)
    students: list[StudentState] = []
    for cohort_row in bundle.active_cohorts.sort_values("cohorte").itertuples(index=False):
        cohort = int(cohort_row.cohorte)
        pool = bundle.profiles.loc[bundle.profiles["cohorte"].astype(int).eq(cohort)].copy()
        selections = _sample_profiles(
            pool,
            int(cohort_row.n_estudiantes_simulados),
            rng_manager.generator("profile_sampling", cohort),
        )
        for sequence, (profile, draw_type) in enumerate(selections, start=1):
            student_id = f"S{cohort}_{sequence:04d}"
            admission_score = float(profile["puntaje_admision"])
            admission_route = str(profile["via_ingreso"])
            admission_signal = (
                0.0
                if admission_route == "ADMISSION_B"
                else (admission_score - float(cohort_row.media_puntaje_regular))
                / float(cohort_row.de_puntaje_regular)
            )
            dependency = str(
                rng_manager.generator("school_dependency", student_id).choice(labels, p=probabilities)
            )
            students.append(
                StudentState(
                    student_id=student_id,
                    id_perfil_fuente=str(profile["id_perfil"]),
                    cohorte=cohort,
                    puntaje_admision=admission_score,
                    senal_admision=admission_signal,
                    via_ingreso=admission_route,
                    preferencia=int(profile["preferencia"]),
                    primera_opcion=int(profile["preferencia"]) == 1,
                    dependencia_escolar=dependency,
                    b_i=float(
                        rng_manager.generator("student_b", student_id).normal(
                            0.0, float(person["sigma_b"])
                        )
                    ),
                    phi_i=float(
                        rng_manager.generator("student_phi", student_id).normal(
                            float(person["media_phi_i"]), float(person["sigma_phi_i"])
                        )
                    ),
                    u_i=float(
                        rng_manager.generator("student_u", student_id).normal(
                            0.0, float(person["sigma_u_i"])
                        )
                    ),
                    profile_draw_type=draw_type,
                )
            )
    return students


def students_observable_frame(
    students: list[StudentState], *, scenario_id: str, seed: int, run_id: str
) -> pd.DataFrame:
    """Create the public student table under the explicit column-role contract."""

    return pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "seed": int(seed),
                "run_id": run_id,
                "student_id": student.student_id,
                "id_perfil_fuente": student.id_perfil_fuente,
                "cohorte": student.cohorte,
                "puntaje_admision": student.puntaje_admision,
                "senal_admision": student.senal_admision,
                "via_ingreso": student.via_ingreso,
                "preferencia": student.preferencia,
                "primera_opcion": student.primera_opcion,
                "dependencia_escolar": student.dependencia_escolar,
                "estado_final": student.status,
                "periodo_terminal": student.terminal_period,
            }
            for student in students
        ]
    )

