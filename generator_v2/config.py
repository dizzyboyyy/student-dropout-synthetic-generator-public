"""Typed configuration objects and project-local path controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


GENERATOR_VERSION = "2.1.0-public-demo"
COURSES_FILENAME = "Demo_Courses.xlsx"
TRAJECTORIES_FILENAME = "Demo_Trajectories.xlsx"


@dataclass(frozen=True)
class InputPaths:
    """Resolved operational input paths constrained to one project root."""

    root: Path
    courses: Path
    trajectories: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "InputPaths":
        """Build paths without searching parents or external directories."""

        resolved_root = Path(root).resolve()
        paths = cls(
            root=resolved_root,
            courses=resolved_root / COURSES_FILENAME,
            trajectories=resolved_root / TRAJECTORIES_FILENAME,
        )
        for path in (paths.courses, paths.trajectories):
            try:
                path.resolve().relative_to(resolved_root)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Ruta fuera de la raíz exclusiva: {path}") from exc
        return paths

    def require_operational_files(self) -> None:
        """Fail if an operational input is absent from the project root."""

        missing = [str(path) for path in (self.courses, self.trajectories) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Faltan inputs operacionales dentro de la raíz exclusiva: " + ", ".join(missing)
            )


@dataclass
class InputBundle:
    """Normalized, validated input tables used by one simulation scenario."""

    paths: InputPaths
    hashes: dict[str, str]
    course_parameters: dict[str, Any]
    general_parameters: dict[str, Any]
    person_parameters: dict[str, Any]
    courses: pd.DataFrame
    groups: pd.DataFrame
    prerequisites: pd.DataFrame
    periods: pd.DataFrame
    media_course_period: pd.DataFrame
    sd_course_period: pd.DataFrame
    observed_course_period: dict[str, pd.DataFrame]
    source_course_period: pd.DataFrame
    cohorts: pd.DataFrame
    profiles: pd.DataFrame
    macro_references: pd.DataFrame

    @property
    def active_periods(self) -> list[str]:
        """Simulation periods in calendar order."""

        mask = self.periods["usar_en_simulacion"].eq(True)
        return self.periods.loc[mask, "periodo_id"].astype(str).tolist()

    @property
    def active_cohorts(self) -> pd.DataFrame:
        """Cohorts enabled for the baseline/scenario."""

        return self.cohorts.loc[self.cohorts["usar_en_simulacion"].eq(True)].copy()

    def effective_configuration(self) -> dict[str, Any]:
        """Return the complete effective configuration for run metadata."""

        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            return frame.to_dict(orient="records")

        return {
            "generator_version": GENERATOR_VERSION,
            "input_hashes": dict(sorted(self.hashes.items())),
            "course_parameters": dict(sorted(self.course_parameters.items())),
            "general_parameters": dict(sorted(self.general_parameters.items())),
            "person_parameters": dict(sorted(self.person_parameters.items())),
            "courses": records(self.courses),
            "prerequisites": records(
                self.prerequisites.drop(columns=["prerequisite_list"], errors="ignore")
            ),
            "periods": records(self.periods),
            "cohorts": records(self.cohorts),
            "media_course_period": self.media_course_period.to_dict(orient="index"),
            "sd_course_period": self.sd_course_period.to_dict(orient="index"),
        }

