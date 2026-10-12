"""Leakage-free grade predictor and re-scaled Beta draw."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from math import exp, log
from typing import Any

import numpy as np

from .course_state import CoursePeriodState


def half_up_one_decimal(value: float) -> float:
    """Round an official grade to one decimal with decimal half-up semantics."""

    return float(Decimal(str(float(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def logit(probability: float) -> float:
    """Numerically safe logit for a probability strictly inside (0,1)."""

    if not 0.0 < probability < 1.0:
        raise ValueError(f"Probabilidad fuera de (0,1): {probability}")
    return log(probability / (1.0 - probability))


def logistic(value: float) -> float:
    """Stable logistic transform."""

    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1.0 + exp_value)


def z_score(value: float | None, mean: float | None, sd: float | None) -> float:
    """Return zero for unavailable/degenerate pre-period standardization."""

    if value is None or mean is None or sd is None or sd <= 0.0:
        return 0.0
    return (float(value) - float(mean)) / float(sd)


def admission_weight(semester_number: int, residual_weight: float, half_life: float) -> float:
    """Compute the specified decaying admission-signal weight."""

    if half_life <= 0.0:
        raise ValueError("vida_media_admision_semestres debe ser positiva")
    lambda_admission = log(2.0) / half_life
    return residual_weight + (1.0 - residual_weight) * exp(
        -lambda_admission * (semester_number - 1)
    )


def generate_grade(
    *,
    state: CoursePeriodState,
    admission_signal: float,
    semester_number: int,
    cumulative_z: float,
    last_semester_z: float,
    previous_course_z: float,
    attempt: int,
    b_i: float,
    phi_i: float,
    parameters: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Draw one grade; approval is derived only from the official grade."""

    weight = admission_weight(
        semester_number,
        float(parameters["peso_residual_admision"]),
        float(parameters["vida_media_admision_semestres"]),
    )
    eta = (
        logit(state.m0_beta)
        + float(parameters["beta_admision"]) * weight * float(admission_signal)
        + float(parameters["beta_H_promedio_acumulado"]) * float(cumulative_z)
        + float(parameters["beta_H_ultimo_semestre"]) * float(last_semester_z)
        + float(parameters["beta_H_nota_previa_mismo_ramo"])
        * float(previous_course_z)
        * int(attempt > 1)
        + float(b_i)
        + float(phi_i) * (int(attempt) - 1)
    )
    individual_mean = min(max(logistic(eta), 1e-12), 1.0 - 1e-12)
    alpha = individual_mean * state.kappa
    beta = (1.0 - individual_mean) * state.kappa
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError(f"Parámetros Beta no positivos: alpha={alpha}, beta={beta}")
    x_value = float(rng.beta(alpha, beta))
    raw_grade = 1.0 + 6.0 * x_value
    official = min(max(half_up_one_decimal(raw_grade), 1.0), 7.0)
    approved = int(official >= float(parameters.get("nota_aprobacion", 4.0)))
    return {
        "peso_admision": weight,
        "eta_nota": eta,
        "media_individual_beta": individual_mean,
        "alpha_beta": alpha,
        "beta_beta": beta,
        "nota_cruda": raw_grade,
        "nota_oficial": official,
        "aprobado": approved,
    }

