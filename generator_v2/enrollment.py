"""Enrollment helpers kept separate from the simulation orchestrator."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .curriculum import Curriculum


def select_courses(
    curriculum: Curriculum,
    *,
    passed_before_period: set[str],
    attempts_before_period: Mapping[str, int],
    semester_number: int,
    rng: np.random.Generator,
    first_semester_load: int,
    nominal_load: int,
    maximum_load: int,
) -> list[str]:
    """Select a valid load using only information available before ``t``."""

    return curriculum.select_load(
        passed_before_period=passed_before_period,
        attempts_before_period=attempts_before_period,
        semester_number=semester_number,
        rng=rng,
        first_semester_load=first_semester_load,
        nominal_load=nominal_load,
        maximum_load=maximum_load,
    )

