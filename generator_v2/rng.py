"""Stable named random-number substreams."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RNGManager:
    """Derive independent PCG64 streams from stable SHA-256 keys."""

    master_seed: int

    algorithm: str = "numpy.random.PCG64"
    derivation: str = "SHA-256(master_seed, namespace, identifiers)"

    def derived_seed(self, namespace: str, *identifiers: Any) -> int:
        """Return a stable unsigned 128-bit seed for a named component."""

        payload = json.dumps(
            [int(self.master_seed), str(namespace), *identifiers],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:16], "big", signed=False)

    def generator(self, namespace: str, *identifiers: Any) -> np.random.Generator:
        """Create a deterministic generator for one unique random operation."""

        return np.random.Generator(np.random.PCG64(self.derived_seed(namespace, *identifiers)))

    def describe(self) -> dict[str, Any]:
        """Return auditable RNG metadata."""

        return {
            "master_seed": int(self.master_seed),
            "algorithm": self.algorithm,
            "derivation": self.derivation,
            "namespaces": [
                "profile_sampling",
                "school_dependency",
                "student_b",
                "student_phi",
                "student_u",
                "course_mean_state",
                "course_sd_state",
                "load_tiebreak",
                "grades",
                "noncontinuity",
            ],
        }

