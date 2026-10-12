"""Latent course-period AR(1) state and Beta feasibility projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log, sqrt

from .config import InputBundle
from .rng import RNGManager


EPSILON_BETA = 1e-6


class BetaFeasibilityError(ValueError):
    """Raised when no Beta distribution can match the requested dispersion."""


@dataclass(frozen=True)
class CoursePeriodState:
    """Auditable raw and feasible course-period context."""

    course_id: str
    dag_id: str
    periodo: str
    mu_ref: float
    sd_ref: float
    delta_mu: float
    delta_sd: float
    theta_mu_raw: float
    sd_raw: float
    m0_raw: float
    s: float
    m_min: float
    m_max: float
    m0_beta: float
    theta_mu_beta: float
    kappa_raw: float
    kappa: float
    ajuste_beta: int
    motivo_ajuste: str

    def to_record(self) -> dict[str, float | int | str]:
        return asdict(self)


def beta_feasible_state(
    *,
    course_id: str,
    dag_id: str,
    periodo: str,
    mu_ref: float,
    sd_ref: float,
    delta_mu: float,
    delta_sd: float,
    factor_kappa: float,
    epsilon: float = EPSILON_BETA,
) -> CoursePeriodState:
    """Project only the Beta mean while preserving the raw AR(1) state."""

    theta_mu_raw = float(mu_ref) + float(delta_mu)
    sd_raw = float(sd_ref) * exp(float(delta_sd))
    if sd_raw <= 0.0:
        raise BetaFeasibilityError(f"{course_id}/{dag_id} {periodo}: sd_raw={sd_raw} debe ser >0")
    s = sd_raw / 6.0
    if s >= 0.5:
        raise BetaFeasibilityError(
            f"{course_id}/{dag_id} {periodo}: sd_raw={sd_raw} (s={s}) no admite ninguna Beta"
        )
    m0_raw = (theta_mu_raw - 1.0) / 6.0
    root = sqrt(1.0 - 4.0 * s * s)
    m_min = (1.0 - root) / 2.0
    m_max = (1.0 + root) / 2.0
    lower = m_min + epsilon
    upper = m_max - epsilon
    if lower >= upper:
        raise BetaFeasibilityError(
            f"{course_id}/{dag_id} {periodo}: margen numérico deja intervalo Beta vacío"
        )
    m0_beta = min(max(m0_raw, lower), upper)
    if m0_beta != m0_raw:
        reason = "MEDIA_BAJO_LIMITE" if m0_raw < lower else "MEDIA_SOBRE_LIMITE"
        adjusted = 1
    else:
        reason = "SIN_AJUSTE"
        adjusted = 0
    theta_mu_beta = 1.0 + 6.0 * m0_beta
    kappa_raw = m0_beta * (1.0 - m0_beta) / (s * s) - 1.0
    kappa = float(factor_kappa) * kappa_raw
    if kappa_raw <= 0.0 or kappa <= 0.0:
        raise BetaFeasibilityError(
            f"{course_id}/{dag_id} {periodo}: kappa_raw={kappa_raw}, kappa={kappa}"
        )
    return CoursePeriodState(
        course_id=course_id,
        dag_id=dag_id,
        periodo=periodo,
        mu_ref=float(mu_ref),
        sd_ref=float(sd_ref),
        delta_mu=float(delta_mu),
        delta_sd=float(delta_sd),
        theta_mu_raw=theta_mu_raw,
        sd_raw=sd_raw,
        m0_raw=m0_raw,
        s=s,
        m_min=m_min,
        m_max=m_max,
        m0_beta=m0_beta,
        theta_mu_beta=theta_mu_beta,
        kappa_raw=kappa_raw,
        kappa=kappa,
        ajuste_beta=adjusted,
        motivo_ajuste=reason,
    )


def generate_course_states(
    bundle: InputBundle, rng_manager: RNGManager
) -> dict[tuple[str, str], CoursePeriodState]:
    """Generate the shared course-period state for every active period."""

    parameters = bundle.course_parameters
    periods = bundle.active_periods
    start = str(bundle.general_parameters["periodo_inicio_simulacion"])
    rho_mu = float(parameters["rho_media"])
    rho_sd = float(parameters["rho_de"])
    sigma_delta_mu = float(parameters["variacion_media_95_abs"]) / 1.96
    sigma_eta_mu = sigma_delta_mu * sqrt(1.0 - rho_mu * rho_mu)
    sigma_delta_sd = log(1.0 + float(parameters["variacion_de_95_rel"])) / 1.96
    sigma_eta_sd = sigma_delta_sd * sqrt(1.0 - rho_sd * rho_sd)
    factor_kappa = float(parameters["factor_kappa_inicial"])

    states: dict[tuple[str, str], CoursePeriodState] = {}
    previous_mu: dict[str, float] = {}
    previous_sd: dict[str, float] = {}
    for period in periods:
        for course in bundle.courses.itertuples(index=False):
            dag_id = str(course.dag_id)
            course_id = str(course.course_id)
            if period == start:
                delta_mu = 0.0
                delta_sd = 0.0
            else:
                eta_mu = float(
                    rng_manager.generator("course_mean_state", dag_id, period).normal(
                        0.0, sigma_eta_mu
                    )
                )
                eta_sd = float(
                    rng_manager.generator("course_sd_state", dag_id, period).normal(
                        0.0, sigma_eta_sd
                    )
                )
                delta_mu = rho_mu * previous_mu[dag_id] + eta_mu
                delta_sd = rho_sd * previous_sd[dag_id] + eta_sd
            state = beta_feasible_state(
                course_id=course_id,
                dag_id=dag_id,
                periodo=period,
                mu_ref=float(bundle.media_course_period.loc[period, course_id]),
                sd_ref=float(bundle.sd_course_period.loc[period, course_id]),
                delta_mu=delta_mu,
                delta_sd=delta_sd,
                factor_kappa=factor_kappa,
            )
            states[(dag_id, period)] = state
            previous_mu[dag_id] = delta_mu
            previous_sd[dag_id] = delta_sd
    return states

