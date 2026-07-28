"""Reusable physics helpers for Hamiltonian/Lagrangian and field-based modeling."""

from .hamiltonian_lagrangian import (
    action_from_trajectory,
    canonical_hamiltonian,
    field_lagrangian_density,
    gravitational_potential,
    hamilton_equations,
    kinetic_energy_point_masses,
    lagrangian,
    spring_potential,
)
from .scientific_principles import (
    check_energy_conservation,
    diffusion_equation_residual,
    discrete_least_action,
    euler_lagrange_residual,
    noether_linear_momentum,
    poisson_bracket,
    principle_registry,
    wave_equation_residual,
)

__all__ = [
    "action_from_trajectory",
    "canonical_hamiltonian",
    "field_lagrangian_density",
    "gravitational_potential",
    "hamilton_equations",
    "kinetic_energy_point_masses",
    "lagrangian",
    "spring_potential",
    "check_energy_conservation",
    "diffusion_equation_residual",
    "discrete_least_action",
    "euler_lagrange_residual",
    "noether_linear_momentum",
    "poisson_bracket",
    "principle_registry",
    "wave_equation_residual",
]
