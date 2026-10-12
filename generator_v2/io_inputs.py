"""Read-only parsers for the two operational Excel workbooks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np
import pandas as pd

from .config import InputBundle, InputPaths


EXPECTED_COURSE_SHEETS = [
    "Guia_Usuario",
    "Parametros_Curso",
    "Catalogo_Grupos",
    "Catalogo_Cursos",
    "Prerrequisitos",
    "Periodos",
    "Media_curso_periodo",
    "DE_curso_periodo",
    "N_inscritos_obs",
    "N_aprobados_obs",
    "Tasa_aprobacion_obs",
    "Fuente_curso_periodo",
]
EXPECTED_TRAJECTORY_SHEETS = [
    "Guia_Usuario",
    "Parametros_Generales",
    "Parametros_Persona",
    "Cohortes",
    "Perfiles_Admision",
    "Referencias_Cohorte_Periodo",
]

TRUE_STRINGS = {"TRUE", "VERDADERO", "SI", "YES", "1"}
FALSE_STRINGS = {"FALSE", "FALSO", "NO", "0"}
COURSE_BOOLEAN_PARAMETERS = {
    "beta_binomial_habilitada",
    "usar_tasa_aprobacion_para_generar",
    "usar_n_inscritos_como_capacidad",
    "cursos_disponibles_todos_semestres",
}
GENERAL_BOOLEAN_PARAMETERS = {
    "activar_sexto_si_hay_extra",
    "retiros_asignatura",
    "stopout_reingreso",
    "eliminacion_tercer_intento",
    "graduacion_por_trabajo_titulacion",
}


def parse_bool(value: Any, *, field: str = "valor booleano") -> bool:
    """Parse supported spreadsheet booleans and reject ambiguous values."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value) in (0.0, 1.0):
            return bool(int(value))
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKD", value.strip().upper())
        token = "".join(character for character in normalized if not unicodedata.combining(character))
        if token in TRUE_STRINGS:
            return True
        if token in FALSE_STRINGS:
            return False
    raise ValueError(f"{field}: booleano ambiguo o desconocido {value!r}")


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute a file SHA-256 without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_table(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a workbook table whose machine header is on Excel row 4."""

    frame = pd.read_excel(path, sheet_name=sheet_name, header=3, engine="openpyxl")
    frame = frame.dropna(how="all").copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.reset_index(drop=True)


def _parameters(
    frame: pd.DataFrame,
    sheet_name: str,
    boolean_parameters: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    required = {"parametro", "valor"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{sheet_name}: faltan columnas {sorted(missing)}")
    keys = frame["parametro"].astype(str).str.strip()
    if keys.duplicated().any():
        duplicates = keys[keys.duplicated(keep=False)].tolist()
        raise ValueError(f"{sheet_name}: parámetros duplicados {duplicates}")
    values = dict(zip(keys, frame["valor"], strict=True))
    for parameter in boolean_parameters:
        if parameter in values:
            values[parameter] = parse_bool(values[parameter], field=f"{sheet_name}.{parameter}")
    return values


def _boolean_column(frame: pd.DataFrame, column: str, context: str) -> None:
    if column not in frame.columns:
        raise ValueError(f"{context}: falta columna booleana {column}")
    frame[column] = frame[column].map(
        lambda value: parse_bool(value, field=f"{context}.{column}")
    )


def _read_matrix(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a course-period matrix, excluding its human-readable name row."""

    frame = _read_table(path, sheet_name)
    if "periodo_id" not in frame.columns:
        raise ValueError(f"{sheet_name}: falta periodo_id")
    frame = frame.loc[frame["periodo_id"].astype(str).str.strip() != "Nombre"].copy()
    frame["periodo_id"] = frame["periodo_id"].astype(str).str.strip()
    value_columns = [column for column in frame.columns if column != "periodo_id"]
    frame[value_columns] = frame[value_columns].apply(pd.to_numeric, errors="coerce")
    return frame.set_index("periodo_id")


def _read_text_matrix(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a course-period matrix whose cells are provenance labels."""

    frame = _read_table(path, sheet_name)
    if "periodo_id" not in frame.columns:
        raise ValueError(f"{sheet_name}: falta periodo_id")
    frame = frame.loc[frame["periodo_id"].astype(str).str.strip() != "Nombre"].copy()
    frame["periodo_id"] = frame["periodo_id"].astype(str).str.strip()
    return frame.set_index("periodo_id")


def _clean_codes(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].astype("string").str.strip()
    return out


def load_inputs(root: str | Path) -> InputBundle:
    """Load and normalize the current operational workbooks from ``root``."""

    paths = InputPaths.from_root(root)
    paths.require_operational_files()

    with pd.ExcelFile(paths.courses, engine="openpyxl") as course_book:
        course_sheets = course_book.sheet_names
    with pd.ExcelFile(paths.trajectories, engine="openpyxl") as trajectory_book:
        trajectory_sheets = trajectory_book.sheet_names
    if course_sheets != EXPECTED_COURSE_SHEETS:
        raise ValueError(
            f"{paths.courses.name}: hojas distintas de la especificación; "
            f"esperadas={EXPECTED_COURSE_SHEETS}, encontradas={course_sheets}"
        )
    if trajectory_sheets != EXPECTED_TRAJECTORY_SHEETS:
        raise ValueError(
            f"{paths.trajectories.name}: hojas distintas de la especificación; "
            f"esperadas={EXPECTED_TRAJECTORY_SHEETS}, encontradas={trajectory_sheets}"
        )

    course_parameters_frame = _read_table(paths.courses, "Parametros_Curso")
    general_parameters_frame = _read_table(paths.trajectories, "Parametros_Generales")
    person_parameters_frame = _read_table(paths.trajectories, "Parametros_Persona")

    courses = _clean_codes(
        _read_table(paths.courses, "Catalogo_Cursos"), ["course_id", "dag_id", "group_id"]
    )
    groups = _clean_codes(_read_table(paths.courses, "Catalogo_Grupos"), ["group_id"])
    prerequisites = _clean_codes(
        _read_table(paths.courses, "Prerrequisitos"), ["codigo_asignatura"]
    )
    prerequisites["prerequisite_list"] = prerequisites["codigos_prerrequisitos"].map(
        lambda value: []
        if pd.isna(value) or not str(value).strip() or str(value).strip() == "-"
        else [part.strip() for part in str(value).split(";") if part.strip()]
    )

    periods = _read_table(paths.courses, "Periodos")
    periods["periodo_id"] = periods["periodo_id"].astype(str).str.strip()
    _boolean_column(periods, "usar_en_simulacion", "Periodos")

    cohorts = _read_table(paths.trajectories, "Cohortes")
    _boolean_column(cohorts, "usar_en_simulacion", "Cohortes")
    profiles = _read_table(paths.trajectories, "Perfiles_Admision")
    profiles["id_perfil"] = profiles["id_perfil"].astype(str).str.strip()
    profiles["via_ingreso"] = profiles["via_ingreso"].astype(str).str.strip()
    _boolean_column(profiles, "primera_opcion", "Perfiles_Admision")
    macro_references = _read_table(paths.trajectories, "Referencias_Cohorte_Periodo")
    macro_references["periodo"] = macro_references["periodo"].astype(str).str.strip()
    _boolean_column(
        macro_references, "usar_como_referencia", "Referencias_Cohorte_Periodo"
    )

    observed = {
        sheet: _read_matrix(paths.courses, sheet)
        for sheet in ("N_inscritos_obs", "N_aprobados_obs", "Tasa_aprobacion_obs")
    }

    bundle = InputBundle(
        paths=paths,
        hashes={
            paths.courses.name: file_sha256(paths.courses),
            paths.trajectories.name: file_sha256(paths.trajectories),
        },
        course_parameters=_parameters(
            course_parameters_frame, "Parametros_Curso", COURSE_BOOLEAN_PARAMETERS
        ),
        general_parameters=_parameters(
            general_parameters_frame, "Parametros_Generales", GENERAL_BOOLEAN_PARAMETERS
        ),
        person_parameters=_parameters(person_parameters_frame, "Parametros_Persona"),
        courses=courses,
        groups=groups,
        prerequisites=prerequisites,
        periods=periods,
        media_course_period=_read_matrix(paths.courses, "Media_curso_periodo"),
        sd_course_period=_read_matrix(paths.courses, "DE_curso_periodo"),
        observed_course_period=observed,
        source_course_period=_read_text_matrix(paths.courses, "Fuente_curso_periodo"),
        cohorts=cohorts,
        profiles=profiles,
        macro_references=macro_references,
    )
    return bundle

