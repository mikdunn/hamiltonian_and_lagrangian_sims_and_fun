"""Shared starter for domain-specific simulations."""

from __future__ import annotations

import numpy as np

from physics_core import (
    canonical_hamiltonian,
    check_energy_conservation,
    kinetic_energy_point_masses,
    lagrangian,
    principle_registry,
)


def run_mass_spring_demo() -> dict[str, object]:
    """Run a small shared demo that each domain can reuse as a smoke test."""
    masses = np.array([1.0])
    positions = np.array([[0.1]])
    velocities = np.array([[0.0]])
    spring_constant = 2.0

    kinetic = kinetic_energy_point_masses(masses, velocities)
    potential = 0.5 * spring_constant * float(np.sum(positions * positions))
    lagrangian_value = lagrangian(kinetic, potential)

    momenta = masses[:, None] * velocities
    hamiltonian_value = canonical_hamiltonian(momenta, masses, potential)

    energies = np.array([hamiltonian_value, hamiltonian_value * 1.0001, hamiltonian_value * 0.9999])
    conservation = check_energy_conservation(energies)

    return {
        "kinetic": kinetic,
        "potential": potential,
        "lagrangian": lagrangian_value,
        "hamiltonian": hamiltonian_value,
        "energy_check": conservation,
        "principles": principle_registry(),
    }
