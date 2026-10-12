"""Verify one public synthetic demonstration run without resimulating it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = RELEASE_ROOT / "inputs" / "demo"
SCENARIO_ID = "public_demo"
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from generator_v2.io_inputs import load_inputs  # noqa: E402
from generator_v2.outputs import verify_run_directory  # noqa: E402
from generator_v2.validation import validate_inputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-directory", required=True, type=Path)
    args = parser.parse_args()

    preflight = validate_inputs(load_inputs(INPUT_ROOT))
    preflight.require_success()
    result = verify_run_directory(
        args.run_directory,
        expected_scenario=SCENARIO_ID,
        expected_seed=args.seed,
    )
    audit_checks = int(result["metadata"]["structural_checks"]["n_checks"])
    if len(preflight.checks) != 49 or audit_checks != 34:
        raise RuntimeError(
            f"Unexpected control counts: preflight={len(preflight.checks)}, audit={audit_checks}"
        )
    print("RUN VERIFICATION PASS")
    print(f"PREFLIGHT: {len(preflight.checks)}/49")
    print(f"STRUCTURAL AUDIT: {audit_checks}/34")
    print(f"RUN_ID: {result['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
