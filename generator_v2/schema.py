"""Explicit availability contract for generated columns."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


IDENTIFIER = "IDENTIFIER"
FEATURE_PRE_T = "FEATURE_PRE_T"
OUTCOME_POST_T = "OUTCOME_POST_T"
REFERENCE = "REFERENCE"
AUDIT_ONLY = "AUDIT_ONLY"

PUBLIC_TABLE_NAMES = {
    "Ramos",
    "Estudiantes",
    "Matriculas_estudiante_periodo",
    "Inscripciones_estudiantes",
    "Agregados_curso_periodo",
}
LOGICAL_TABLE_NAMES = PUBLIC_TABLE_NAMES | {"Auditoria"}

RUN_KEYS = {"scenario_id", "seed", "run_id"}

AUDIT_ONLY_COLUMN_NAMES = {
    "tipo_muestreo_perfil",
    "nota_cruda",
    "peso_admision",
    "eta_nota",
    "media_individual_beta",
    "alpha_beta",
    "beta_beta",
    "b_i",
    "phi_i",
    "u_i",
    "mu_ref",
    "sd_ref",
    "delta_mu",
    "delta_sd",
    "theta_mu_raw",
    "theta_mu_beta",
    "sd_raw",
    "m0_raw",
    "m_min",
    "m_max",
    "m0_beta",
    "s",
    "kappa_raw",
    "kappa",
    "ajuste_beta",
    "motivo_ajuste",
    "draw_no_continuidad",
    "eta_no_continuidad",
    "p_no_continuidad",
}


def _roles(columns: set[str], role: str) -> dict[str, str]:
    return {column: role for column in columns}


PUBLIC_COLUMN_ROLES: dict[str, dict[str, str]] = {
    "Ramos": {
        **_roles(RUN_KEYS | {"course_id", "dag_id"}, IDENTIFIER),
        **_roles(
            {
                "course_name",
                "nivel",
                "group_id",
                "group_name",
                "prerrequisitos",
                "media_ref_base",
                "de_ref_base",
            },
            REFERENCE,
        ),
    },
    "Estudiantes": {
        **_roles(RUN_KEYS | {"student_id", "id_perfil_fuente"}, IDENTIFIER),
        **_roles(
            {
                "cohorte",
                "puntaje_admision",
                "senal_admision",
                "via_ingreso",
                "preferencia",
                "primera_opcion",
                "dependencia_escolar",
            },
            FEATURE_PRE_T,
        ),
        **_roles({"estado_final", "periodo_terminal"}, OUTCOME_POST_T),
    },
    "Matriculas_estudiante_periodo": {
        **_roles(RUN_KEYS | {"student_id", "cohorte", "periodo"}, IDENTIFIER),
        **_roles({"semestres_desde_ingreso", "estado_inicio"}, FEATURE_PRE_T),
        **_roles(
            {
                "estado_cierre",
                "evento_cierre",
                "censurado",
                "carga",
                "promedio_semestre",
                "promedio_acumulado",
                "aprobados_acumulados",
                "reprobados_acumulados",
                "proporcion_aprobacion",
                "avance_real",
                "avance_esperado",
                "rezago",
                "bloqueos",
            },
            OUTCOME_POST_T,
        ),
    },
    "Inscripciones_estudiantes": {
        **_roles(
            RUN_KEYS
            | {"student_id", "cohorte", "course_id", "dag_id", "periodo"},
            IDENTIFIER,
        ),
        **_roles(
            {
                "intento",
                "nota_previa",
                "semestres_desde_ingreso_pre",
                "promedio_acumulado_pre",
                "promedio_ultimo_semestre_pre",
                "aprobados_acumulados_pre",
                "reprobados_acumulados_pre",
                "proporcion_aprobacion_pre",
                "avance_real_pre",
                "avance_esperado_pre",
                "rezago_pre",
                "bloqueos_pre",
                "z_promedio_acumulado_pre",
                "z_promedio_ultimo_semestre_pre",
                "z_nota_previa_mismo_ramo",
            },
            FEATURE_PRE_T,
        ),
        **_roles({"nota_oficial", "aprobado"}, OUTCOME_POST_T),
    },
    "Agregados_curso_periodo": {
        **_roles(RUN_KEYS | {"course_id", "dag_id", "periodo"}, IDENTIFIER),
        **_roles(
            {
                "N",
                "media",
                "DE",
                "tasa_aprobacion",
                "composicion_cohorte",
                "composicion_intento",
            },
            OUTCOME_POST_T,
        ),
    },
}


def data_contract() -> dict[str, object]:
    """Return the machine-readable predictor/outcome/audit contract."""

    return {
        "roles": {
            "FEATURE_PRE_T": "observable disponible antes de generar el período t",
            "OUTCOME_POST_T": "resultado conocido sólo después de cerrar el período t o la trayectoria",
            "AUDIT_ONLY": "estado interno o artefacto procedimental, excluido de datasets predictivos",
            "IDENTIFIER": "clave de identificación y unión",
            "REFERENCE": "atributo estructural o referencia configurada",
        },
        "public_column_roles": PUBLIC_COLUMN_ROLES,
        "audit_table": "Auditoria",
        "audit_table_default_role": AUDIT_ONLY,
        "multi_seed_student_key": ["scenario_id", "seed", "student_id"],
        "run_key": ["scenario_id", "seed", "run_id"],
    }


def public_contract_violations(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    """Return schema/leakage violations for the five non-audit tables."""

    violations: list[str] = []
    for table_name in sorted(PUBLIC_TABLE_NAMES):
        if table_name not in tables:
            violations.append(f"falta tabla pública {table_name}")
            continue
        columns = set(tables[table_name].columns)
        leaked = sorted(columns & AUDIT_ONLY_COLUMN_NAMES)
        if leaked:
            violations.append(f"{table_name}: columnas AUDIT_ONLY expuestas {leaked}")
        declared = set(PUBLIC_COLUMN_ROLES[table_name])
        if columns != declared:
            violations.append(
                f"{table_name}: contrato distinto; faltan={sorted(declared-columns)}, "
                f"sobran={sorted(columns-declared)}"
            )
    return violations

