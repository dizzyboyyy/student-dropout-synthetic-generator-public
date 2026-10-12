"""Generador Sintético Longitudinal V2."""

from .config import GENERATOR_VERSION, InputBundle, InputPaths
from .engine import GeneratorEngine, SimulationResult
from .io_inputs import load_inputs

__all__ = [
    "GENERATOR_VERSION",
    "GeneratorEngine",
    "InputBundle",
    "InputPaths",
    "SimulationResult",
    "load_inputs",
]

