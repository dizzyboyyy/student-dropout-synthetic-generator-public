"""Discrete non-continuity hazard after academic events."""

from __future__ import annotations

from math import exp, log
from typing import Any

import numpy as np


def logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Probabilidad fuera de (0,1): {probability}")
    return log(probability / (1.0 - probability))


def noncontinuity_probability(
    *,
    semester_number: int,
    last_semester_average: float | None,
    actual_progress: int,
    expected_progress: int,
    u_i: float,
    parameters: dict[str, Any],
) -> dict[str, float]:
    """Compute the specified baseline hazard after closing the semester."""

    p1 = float(parameters["p1_no_continuidad"])
    p_inf = float(parameters["p_inf_no_continuidad"])
    lambda_dropout = float(parameters["lambda_D_no_continuidad"])
    p0 = p_inf + (p1 - p_inf) * exp(-lambda_dropout * (semester_number - 1))
    recent = 0.0 if last_semester_average is None else float(
        np.clip((4.0 - float(last_semester_average)) / 3.0, -1.0, 1.0)
    )
    lag = min(
        1.0,
        max(0.0, float(expected_progress - actual_progress)) / max(1.0, float(expected_progress)),
    )
    eta = (
        logit(p0)
        + float(parameters["gamma_R_senal_reciente"]) * recent
        + float(parameters["gamma_L_rezago"]) * lag
        + float(u_i)
    )
    return {
        "p0": p0,
        "R_recent": recent,
        "L_rezago": lag,
        "eta_no_continuidad": eta,
        "p_no_continuidad": logistic(eta),
    }


def draw_noncontinuity(probability: float, rng: np.random.Generator) -> tuple[bool, float]:
    """Draw one permanent non-continuity event."""

    draw = float(rng.random())
    return draw < float(probability), draw

