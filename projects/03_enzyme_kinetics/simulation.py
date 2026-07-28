"""Starter simulation scaffold for: Enzyme kinetics for specific molecules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domain_catalog import DOMAINS
from template_simulation import run_mass_spring_demo


DOMAIN_KEY = "03_enzyme_kinetics"


def build_project_context() -> dict[str, object]:
    """Return metadata plus a shared mechanics sanity check."""
    domain_info = DOMAINS[DOMAIN_KEY]
    baseline = run_mass_spring_demo(domain_key=DOMAIN_KEY)

    return {
        "domain_key": DOMAIN_KEY,
        "domain": domain_info,
        "baseline_demo": baseline,
        "next_steps": [
            "Load real domain data into a consistent tensor format.",
            "Define domain-specific Lagrangian or Hamiltonian terms.",
            "Calibrate parameters with optimization or Bayesian inference.",
            "Validate conservation laws or dissipation metrics.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(build_project_context(), indent=2))
