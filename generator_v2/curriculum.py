"""Curriculum DAG, strict pre-period eligibility and load selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Curriculum:
    """Validated curriculum graph keyed by public DAG course IDs."""

    course_order: tuple[str, ...]
    levels: Mapping[str, int]
    prerequisites: Mapping[str, frozenset[str]]
    course_ids: Mapping[str, str]
    names: Mapping[str, str]

    @classmethod
    def from_frames(cls, courses: pd.DataFrame, prerequisites: pd.DataFrame) -> "Curriculum":
        """Construct a curriculum from normalized input tables."""

        course_order = tuple(courses["dag_id"].astype(str))
        return cls(
            course_order=course_order,
            levels=dict(zip(course_order, courses["nivel"].astype(int), strict=True)),
            prerequisites={
                str(row.codigo_asignatura): frozenset(row.prerequisite_list)
                for row in prerequisites.itertuples(index=False)
            },
            course_ids=dict(zip(course_order, courses["course_id"].astype(str), strict=True)),
            names=dict(zip(course_order, courses["course_name"].astype(str), strict=True)),
        )

    def eligible(self, passed_before_period: set[str]) -> list[str]:
        """Return pending courses whose prerequisites were passed before ``t``."""

        return [
            course
            for course in self.course_order
            if course not in passed_before_period
            and self.prerequisites[course].issubset(passed_before_period)
        ]

    def blocked_count(self, passed: set[str]) -> int:
        """Count pending courses not yet enabled by the prerequisite DAG."""

        return sum(
            1
            for course in self.course_order
            if course not in passed and not self.prerequisites[course].issubset(passed)
        )

    def expected_progress_after_semester(self, semester_number: int) -> int:
        """Expected completions after ``semester_number``; zero before semester one."""

        if int(semester_number) <= 0:
            return 0
        expected_level = min(int(semester_number), max(self.levels.values()))
        return sum(level <= expected_level for level in self.levels.values())

    def expected_progress(self, semester_number: int) -> int:
        """Backward-compatible alias for post-semester expected progress."""

        return self.expected_progress_after_semester(semester_number)

    def select_load(
        self,
        passed_before_period: set[str],
        attempts_before_period: Mapping[str, int],
        semester_number: int,
        rng: np.random.Generator,
        first_semester_load: int = 4,
        nominal_load: int = 5,
        maximum_load: int = 6,
    ) -> list[str]:
        """Select courses using retry priority and the explicit sixth-course rule."""

        candidates = self.eligible(passed_before_period)
        if not candidates:
            return []
        if semester_number <= 1:
            ranked = sorted(candidates, key=lambda course: (self.levels[course], course))
            return ranked[: min(first_semester_load, len(ranked))]

        retry_tiebreak = {course: float(rng.random()) for course in candidates}

        def priority(course: str) -> tuple[float | int | str, ...]:
            previous_attempts = int(attempts_before_period.get(course, 0))
            level = self.levels[course]
            if previous_attempts > 0:
                return (0, -previous_attempts, level, retry_tiebreak[course])
            if level < semester_number:
                category = 1
            elif level == semester_number:
                category = 2
            else:
                category = 3
            return (category, level, course)

        ranked = sorted(candidates, key=priority)
        selected = ranked[: min(nominal_load, len(ranked))]
        remaining = [course for course in ranked if course not in selected]
        if remaining and len(selected) < maximum_load:
            minimum_level = min(self.levels[course] for course in remaining)
            tied = [course for course in remaining if self.levels[course] == minimum_level]
            sixth = tied[int(rng.integers(0, len(tied)))]
            selected.append(sixth)
        return selected

    def prerequisites_string(self, course: str) -> str:
        """Return a stable compact representation for outputs."""

        return "; ".join(sorted(self.prerequisites[course]))


def prior_approvals_by_student(
    registrations: pd.DataFrame, periods: Sequence[str]
) -> dict[tuple[str, str], set[str]]:
    """Reconstruct approved courses strictly before every student-period."""

    result: dict[tuple[str, str], set[str]] = {}
    if registrations.empty:
        return result
    by_period = {period: index for index, period in enumerate(periods)}
    for student_id, rows in registrations.groupby("student_id", sort=False):
        approved: set[str] = set()
        for period in periods:
            result[(str(student_id), period)] = set(approved)
            current = rows.loc[
                rows["periodo"].eq(period) & rows["aprobado"].eq(1), "dag_id"
            ]
            approved.update(current.astype(str))
        unknown = set(rows["periodo"]) - set(by_period)
        if unknown:
            raise ValueError(f"Períodos desconocidos en inscripciones: {sorted(unknown)}")
    return result

