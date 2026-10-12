"""Run the public synthetic demonstration with one explicit seed."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = RELEASE_ROOT / "inputs" / "demo"
SCENARIO_ID = "public_demo"

import sys

if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from generator_v2.engine import GeneratorEngine  # noqa: E402
from generator_v2.io_inputs import load_inputs  # noqa: E402
from generator_v2.outputs import (  # noqa: E402
    ANALYSIS_READY_WORKBOOK,
    verify_run_directory,
    write_run,
)
from generator_v2.validation import validate_inputs  # noqa: E402


def non_negative_seed(value: str) -> int:
    """Parse one non-negative integer seed."""

    try:
        seed = int(value.strip())
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--seed must be a non-negative integer") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("--seed must be a non-negative integer")
    return seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public synthetic demonstration once."
    )
    parser.add_argument("--seed", required=True, type=non_negative_seed)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_inputs(INPUT_ROOT)
    preflight = validate_inputs(bundle)
    preflight.require_success()
    if len(preflight.checks) != 49:
        raise RuntimeError(f"Expected 49 preflight checks, observed {len(preflight.checks)}")
    print(f"PREFLIGHT PASS: {len(preflight.checks)}/49")

    result = GeneratorEngine(bundle).run(seed=args.seed, scenario_id=SCENARIO_ID)
    audit_checks = int(result.structural_checks["n_checks"])
    if audit_checks != 34:
        raise RuntimeError(f"Expected 34 structural audit checks, observed {audit_checks}")
    print(f"STRUCTURAL AUDIT PASS: {audit_checks}/34")

    with tempfile.TemporaryDirectory(
        prefix=".generator_v2_stage_", dir=target.parent
    ) as staging_text:
        staging = Path(staging_text)
        produced = write_run(
            result,
            bundle,
            output_root=staging,
            central_reports_dir=None,
        )
        verify_run_directory(
            produced,
            expected_scenario=SCENARIO_ID,
            expected_seed=args.seed,
        )
        produced.replace(target)

    verify_run_directory(
        target,
        expected_scenario=SCENARIO_ID,
        expected_seed=args.seed,
    )
    print(f"RUN VERIFICATION PASS: scenario={SCENARIO_ID} seed={args.seed}")
    print(f"ANALYSIS-READY WORKBOOK: {target / ANALYSIS_READY_WORKBOOK}")
    print(f"OUTPUT: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
