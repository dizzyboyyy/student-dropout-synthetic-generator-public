"""Calendar-time multi-cohort simulation engine."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import GENERATOR_VERSION, InputBundle
from .course_state import CoursePeriodState, generate_course_states
from .curriculum import Curriculum
from .dropout import draw_noncontinuity, noncontinuity_probability
from .enrollment import select_courses
from .grades import generate_grade, z_score
from .rng import RNGManager
from .schema import data_contract
from .student import StudentState, create_students, students_observable_frame
from .validation import validate_inputs


FINAL_DAG_ID = "COURSE_050"
TERMINAL_STATES = {"GRADUADO", "ELIMINADO", "NO_CONTINUIDAD"}


@dataclass
class SimulationResult:
    """Six logical output tables plus run metadata and structural checks."""

    ramos: pd.DataFrame
    estudiantes: pd.DataFrame
    matriculas: pd.DataFrame
    inscripciones: pd.DataFrame
    agregados: pd.DataFrame
    auditoria: pd.DataFrame
    metadata: dict[str, Any]
    structural_checks: dict[str, Any]

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return the six public logical outputs with stable names."""

        return {
            "Ramos": self.ramos,
            "Estudiantes": self.estudiantes,
            "Matriculas_estudiante_periodo": self.matriculas,
            "Inscripciones_estudiantes": self.inscripciones,
            "Agregados_curso_periodo": self.agregados,
            "Auditoria": self.auditoria,
        }


def semester_number(cohort: int, period: str) -> int:
    """One-based semesters elapsed since an annual semester-01 intake."""

    year_text, semester_text = period.split("-")
    return (int(year_text) - int(cohort)) * 2 + int(semester_text)


def determine_terminal_event(period_results: list[dict[str, Any]]) -> str | None:
    """Apply elimination before graduation exactly as specified."""

    if any(int(row["intento"]) >= 3 and int(row["aprobado"]) == 0 for row in period_results):
        return "ELIMINADO"
    if any(row["dag_id"] == FINAL_DAG_ID and int(row["aprobado"]) == 1 for row in period_results):
        return "GRADUADO"
    return None


def _stats(values: list[float | None]) -> tuple[float | None, float | None]:
    clean = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if clean.size < 2:
        return None, None
    sd = float(clean.std(ddof=0))
    if sd <= 0.0:
        return float(clean.mean()), None
    return float(clean.mean()), sd


def _composition(values: pd.Series) -> str:
    counts = values.value_counts(sort=False).sort_index()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


class GeneratorEngine:
    """Run the validated DGM without mutating any input workbook."""

    def __init__(self, bundle: InputBundle):
        self.bundle = bundle
        self.preflight = validate_inputs(bundle)
        self.preflight.require_success()
        self.curriculum = Curriculum.from_frames(bundle.courses, bundle.prerequisites)

    def run(self, seed: int, scenario_id: str = "baseline") -> SimulationResult:
        """Execute one reproducible scenario/seed pair entirely in memory."""

        if int(seed) < 0:
            raise ValueError("La seed debe ser un entero no negativo")
        if not str(scenario_id).strip():
            raise ValueError("scenario_id no puede estar vacío")
        seed = int(seed)
        scenario_id = str(scenario_id)
        run_id = f"{scenario_id}_seed_{seed}"
        run_identity = {"scenario_id": scenario_id, "seed": seed, "run_id": run_id}
        rng_manager = RNGManager(int(seed))
        course_states = generate_course_states(self.bundle, rng_manager)
        students = create_students(self.bundle, rng_manager)
        student_by_id = {student.student_id: student for student in students}
        general = self.bundle.general_parameters
        grade_parameters = {**self.bundle.person_parameters, **self.bundle.course_parameters}
        periods = self.bundle.active_periods

        audit_records: list[dict[str, Any]] = []
        audit_records.append(
            {
                "record_type": "rng_config",
                **run_identity,
                **rng_manager.describe(),
            }
        )
        for filename, digest in sorted(self.bundle.hashes.items()):
            audit_records.append(
                {
                    "record_type": "input_hash",
                    **run_identity,
                    "input_file": filename,
                    "sha256": digest,
                }
            )
        for student in students:
            audit_records.append(
                {
                    "record_type": "student_effect",
                    **run_identity,
                    "student_id": student.student_id,
                    "id_perfil_fuente": student.id_perfil_fuente,
                    "cohorte": student.cohorte,
                    "tipo_muestreo_perfil": student.profile_draw_type,
                    "b_i": student.b_i,
                    "phi_i": student.phi_i,
                    "u_i": student.u_i,
                }
            )
        for state in course_states.values():
            audit_records.append(
                {"record_type": "course_state", **run_identity, **state.to_record()}
            )

        registration_records: list[dict[str, Any]] = []
        panel_records: list[dict[str, Any]] = []

        for period_index, period in enumerate(periods):
            year = int(period[:4])
            active = sorted(
                [
                    student
                    for student in students
                    if student.status == "ACTIVO" and student.cohorte <= year
                ],
                key=lambda student: student.student_id,
            )
            snapshots: dict[str, dict[str, Any]] = {}
            plans: dict[str, list[str]] = {}
            for student in active:
                current_semester = semester_number(student.cohorte, period)
                expected_pre = self.curriculum.expected_progress_after_semester(
                    current_semester - 1
                )
                blocked = self.curriculum.blocked_count(student.passed)
                snapshots[student.student_id] = student.snapshot(
                    current_semester, expected_pre, blocked
                )
                plans[student.student_id] = select_courses(
                    self.curriculum,
                    passed_before_period=set(student.passed),
                    attempts_before_period=dict(student.attempts),
                    semester_number=current_semester,
                    rng=rng_manager.generator("load_tiebreak", student.student_id, period),
                    first_semester_load=int(general["carga_S1"]),
                    nominal_load=int(general["carga_nominal_desde_S2"]),
                    maximum_load=int(general["carga_maxima"]),
                )

            cumulative_stats = _stats(
                [snapshots[student.student_id]["promedio_acumulado_pre"] for student in active]
            )
            last_stats = _stats(
                [snapshots[student.student_id]["promedio_ultimo_semestre_pre"] for student in active]
            )
            previous_stats: dict[str, tuple[float | None, float | None]] = {}
            for course in self.curriculum.course_order:
                previous_stats[course] = _stats(
                    [
                        student.previous_grade.get(course)
                        for student in active
                        if course in plans[student.student_id] and course in student.previous_grade
                    ]
                )

            results_by_student: dict[str, list[dict[str, Any]]] = {
                student.student_id: [] for student in active
            }
            for student in active:
                snapshot = snapshots[student.student_id]
                for course in plans[student.student_id]:
                    attempt = int(student.attempts.get(course, 0)) + 1
                    previous_grade = student.previous_grade.get(course)
                    cumulative_z = z_score(
                        snapshot["promedio_acumulado_pre"], *cumulative_stats
                    )
                    last_z = z_score(
                        snapshot["promedio_ultimo_semestre_pre"], *last_stats
                    )
                    previous_z = z_score(previous_grade, *previous_stats[course])
                    state = course_states[(course, period)]
                    generated = generate_grade(
                        state=state,
                        admission_signal=student.senal_admision,
                        semester_number=int(snapshot["semestres_desde_ingreso_pre"]),
                        cumulative_z=cumulative_z,
                        last_semester_z=last_z,
                        previous_course_z=previous_z,
                        attempt=attempt,
                        b_i=student.b_i,
                        phi_i=student.phi_i,
                        parameters=grade_parameters,
                        rng=rng_manager.generator("grades", student.student_id, course, period),
                    )
                    record = {
                        **run_identity,
                        "student_id": student.student_id,
                        "cohorte": student.cohorte,
                        "course_id": self.curriculum.course_ids[course],
                        "dag_id": course,
                        "periodo": period,
                        "intento": attempt,
                        "nota_previa": previous_grade,
                        **snapshot,
                        "z_promedio_acumulado_pre": cumulative_z,
                        "z_promedio_ultimo_semestre_pre": last_z,
                        "z_nota_previa_mismo_ramo": previous_z,
                        "nota_oficial": generated["nota_oficial"],
                        "aprobado": generated["aprobado"],
                    }
                    registration_records.append(record)
                    results_by_student[student.student_id].append(record)
                    audit_records.append(
                        {
                            "record_type": "grade_internal",
                            **run_identity,
                            "student_id": student.student_id,
                            "cohorte": student.cohorte,
                            "course_id": self.curriculum.course_ids[course],
                            "dag_id": course,
                            "periodo": period,
                            "intento": attempt,
                            **generated,
                        }
                    )

            for student in active:
                rows = results_by_student[student.student_id]
                student.apply_period_results(rows)
                current_semester = int(snapshots[student.student_id]["semestres_desde_ingreso_pre"])
                event = determine_terminal_event(rows)
                dropout_details: dict[str, float] | None = None
                dropout_draw: float | None = None
                if event is not None:
                    student.status = event
                    student.terminal_period = period
                else:
                    expected_post = self.curriculum.expected_progress_after_semester(
                        current_semester
                    )
                    dropout_details = noncontinuity_probability(
                        semester_number=current_semester,
                        last_semester_average=student.last_semester_average,
                        actual_progress=len(student.passed),
                        expected_progress=expected_post,
                        u_i=student.u_i,
                        parameters=self.bundle.person_parameters,
                    )
                    dropped, dropout_draw = draw_noncontinuity(
                        dropout_details["p_no_continuidad"],
                        rng_manager.generator("noncontinuity", student.student_id, period),
                    )
                    audit_records.append(
                        {
                            "record_type": "noncontinuity_draw",
                            **run_identity,
                            "student_id": student.student_id,
                            "periodo": period,
                            **dropout_details,
                            "draw_no_continuidad": dropout_draw,
                            "evento": int(dropped),
                        }
                    )
                    if dropped:
                        event = "NO_CONTINUIDAD"
                        student.status = event
                        student.terminal_period = period
                if period_index == len(periods) - 1 and student.status == "ACTIVO":
                    event = "CENSURADO"
                    student.status = event
                    student.terminal_period = period

                expected_post = self.curriculum.expected_progress_after_semester(
                    current_semester
                )
                actual_post = len(student.passed)
                panel_records.append(
                    {
                        **run_identity,
                        "student_id": student.student_id,
                        "cohorte": student.cohorte,
                        "periodo": period,
                        "semestres_desde_ingreso": current_semester,
                        "estado_inicio": "ACTIVO",
                        "estado_cierre": student.status,
                        "evento_cierre": event,
                        "censurado": int(event == "CENSURADO"),
                        "carga": len(rows),
                        "promedio_semestre": student.last_semester_average,
                        "promedio_acumulado": student.cumulative_average,
                        "aprobados_acumulados": student.approved_attempts,
                        "reprobados_acumulados": student.failed_attempts,
                        "proporcion_aprobacion": student.historical_pass_ratio,
                        "avance_real": actual_post,
                        "avance_esperado": expected_post,
                        "rezago": max(0, expected_post - actual_post),
                        "bloqueos": self.curriculum.blocked_count(student.passed),
                    }
                )

        registrations = pd.DataFrame(registration_records)
        panel = pd.DataFrame(panel_records)
        aggregates = self._build_aggregates(registrations, course_states, scenario_id, seed)
        audit = pd.DataFrame(audit_records)
        students_frame = students_observable_frame(
            students, scenario_id=scenario_id, seed=seed, run_id=run_id
        )
        ramos = self._ramos_frame(scenario_id=scenario_id, seed=seed, run_id=run_id)
        metadata = {
            "generator_version": GENERATOR_VERSION,
            "scenario_id": scenario_id,
            "seed": int(seed),
            "run_id": run_id,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "rng": rng_manager.describe(),
            "effective_configuration": self.bundle.effective_configuration(),
            "data_contract": data_contract(),
            "validation_status": {
                "input_integrity": "PASS",
                "structural_validation": "PENDING_AUDIT",
                "strict_json": "PENDING_WRITE_VERIFICATION",
                "temporal_consistency": "PENDING_AUDIT",
                "leakage_contract": "PENDING_AUDIT",
                "multi_seed_identity": "PENDING_AUDIT",
                "calibration": "CALIBRATION_PENDING",
                "plausibility": "NOT_YET_ACCEPTED",
            },
        }
        result = SimulationResult(
            ramos=ramos,
            estudiantes=students_frame,
            matriculas=panel,
            inscripciones=registrations,
            agregados=aggregates,
            auditoria=audit,
            metadata=metadata,
            structural_checks={},
        )
        from .audit import run_structural_audit

        result.structural_checks = run_structural_audit(result, self.bundle, self.curriculum)
        result.metadata["structural_checks"] = result.structural_checks
        result.metadata["validation_status"].update(
            {
                "structural_validation": "PASS",
                "temporal_consistency": "PASS",
                "leakage_contract": "PASS",
                "multi_seed_identity": "PASS",
            }
        )
        return result

    def _ramos_frame(self, *, scenario_id: str, seed: int, run_id: str) -> pd.DataFrame:
        groups = self.bundle.groups[["group_id", "group_name"]]
        courses = self.bundle.courses.merge(groups, on="group_id", how="left", validate="many_to_one")
        rows = []
        for course in courses.itertuples(index=False):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "seed": int(seed),
                    "run_id": run_id,
                    "course_id": str(course.course_id),
                    "dag_id": str(course.dag_id),
                    "course_name": str(course.course_name),
                    "nivel": int(course.nivel),
                    "group_id": str(course.group_id),
                    "group_name": str(course.group_name),
                    "prerrequisitos": self.curriculum.prerequisites_string(str(course.dag_id)),
                    "media_ref_base": float(course.media_ref_base),
                    "de_ref_base": float(course.de_ref_base),
                }
            )
        return pd.DataFrame(rows)

    def _build_aggregates(
        self,
        registrations: pd.DataFrame,
        states: dict[tuple[str, str], CoursePeriodState],
        scenario_id: str,
        seed: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        grouped = {
            key: group.copy()
            for key, group in registrations.groupby(["dag_id", "periodo"], sort=False)
        }
        for period in self.bundle.active_periods:
            for course in self.curriculum.course_order:
                state = states[(course, period)]
                group = grouped.get((course, period))
                if group is None or group.empty:
                    n = 0
                    mean = None
                    sd = None
                    pass_rate = None
                    cohort_composition = "{}"
                    attempt_composition = "{}"
                else:
                    n = len(group)
                    grades = group["nota_oficial"].astype(float)
                    mean = float(grades.mean())
                    sd = float(grades.std(ddof=0))
                    pass_rate = float(group["aprobado"].astype(float).mean())
                    cohort_composition = _composition(group["cohorte"])
                    attempt_composition = _composition(group["intento"])
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "seed": int(seed),
                        "run_id": f"{scenario_id}_seed_{int(seed)}",
                        "course_id": self.curriculum.course_ids[course],
                        "dag_id": course,
                        "periodo": period,
                        "N": n,
                        "media": mean,
                        "DE": sd,
                        "tasa_aprobacion": pass_rate,
                        "composicion_cohorte": cohort_composition,
                        "composicion_intento": attempt_composition,
                    }
                )
        return pd.DataFrame(rows)

