"""Core mechanics and field-theory utility functions."""

from __future__ import annotations

from typing import Callable

import numpy as np

ArrayLike = np.ndarray


def kinetic_energy_point_masses(masses: ArrayLike, velocities: ArrayLike) -> float:
    """Return total kinetic energy for point masses."""
    masses = np.asarray(masses, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    speed_sq = np.sum(velocities * velocities, axis=1)
    return float(0.5 * np.sum(masses * speed_sq))


def gravitational_potential(
    positions: ArrayLike,
    masses: ArrayLike,
    gravitational_constant: float = 6.67430e-11,
) -> float:
    """Return Newtonian pairwise gravitational potential energy."""
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    n_bodies = len(masses)
    potential = 0.0

    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= 0.0:
                continue
            potential -= gravitational_constant * masses[i] * masses[j] / distance

    return float(potential)


def spring_potential(displacements: ArrayLike, spring_constants: ArrayLike) -> float:
    """Return potential for independent Hookean springs."""
    displacements = np.asarray(displacements, dtype=float)
    spring_constants = np.asarray(spring_constants, dtype=float)
    return float(0.5 * np.sum(spring_constants * displacements * displacements))


def lagrangian(kinetic_energy: float, potential_energy: float) -> float:
    """Return L = T - V."""
    return float(kinetic_energy - potential_energy)


def canonical_hamiltonian(
    momenta: ArrayLike,
    masses: ArrayLike,
    potential_energy: float,
) -> float:
    """Return H = sum(p^2 / (2m)) + V for separable systems."""
    momenta = np.asarray(momenta, dtype=float)
    masses = np.asarray(masses, dtype=float)

    if momenta.ndim == 1:
        momenta = momenta[:, None]

    momentum_sq = np.sum(momenta * momenta, axis=1)
    kinetic = 0.5 * np.sum(momentum_sq / masses)
    return float(kinetic + potential_energy)


def _finite_difference_gradient(
    fn: Callable[[ArrayLike, ArrayLike, float], float],
    q: ArrayLike,
    p: ArrayLike,
    t: float,
    with_respect_to: str,
    eps: float,
) -> ArrayLike:
    """Numerical gradient used by hamilton_equations for generic Hamiltonians."""
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)

    grad = np.zeros_like(q if with_respect_to == "q" else p)
    for idx in range(grad.size):
        q_plus = q.copy()
        q_minus = q.copy()
        p_plus = p.copy()
        p_minus = p.copy()

        if with_respect_to == "q":
            q_plus[idx] += eps
            q_minus[idx] -= eps
        else:
            p_plus[idx] += eps
            p_minus[idx] -= eps

        f_plus = fn(q_plus, p_plus, t)
        f_minus = fn(q_minus, p_minus, t)
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)

    return grad


def hamilton_equations(
    hamiltonian_fn: Callable[[ArrayLike, ArrayLike, float], float],
    q: ArrayLike,
    p: ArrayLike,
    t: float,
    eps: float = 1e-6,
) -> tuple[ArrayLike, ArrayLike]:
    """Return (q_dot, p_dot) from Hamilton's equations using finite differences."""
    q_dot = _finite_difference_gradient(hamiltonian_fn, q, p, t, with_respect_to="p", eps=eps)
    p_dot = -_finite_difference_gradient(hamiltonian_fn, q, p, t, with_respect_to="q", eps=eps)
    return q_dot, p_dot


def action_from_trajectory(
    lagrangian_fn: Callable[[ArrayLike, ArrayLike, float], float],
    q_path: ArrayLike,
    qdot_path: ArrayLike,
    t_path: ArrayLike,
) -> float:
    """Approximate action integral S = integral L(q, q_dot, t) dt with trapezoids."""
    q_path = np.asarray(q_path, dtype=float)
    qdot_path = np.asarray(qdot_path, dtype=float)
    t_path = np.asarray(t_path, dtype=float)

    lagrangian_values = np.array(
        [lagrangian_fn(q, qdot, t) for q, qdot, t in zip(q_path, qdot_path, t_path, strict=True)],
        dtype=float,
    )
    return float(np.trapezoid(lagrangian_values, t_path))


def field_lagrangian_density(
    time_derivative: ArrayLike,
    spatial_gradient: ArrayLike,
    mass_density: float = 1.0,
    stiffness: float = 1.0,
    potential_density: float | ArrayLike = 0.0,
) -> ArrayLike:
    """Return generic scalar-field Lagrangian density.

    Useful for string/wave/acoustic approximations:
        L = 1/2 rho * u_t^2 - 1/2 k * |grad u|^2 - V(u)
    """
    time_derivative = np.asarray(time_derivative, dtype=float)
    spatial_gradient = np.asarray(spatial_gradient, dtype=float)
    potential_density = np.asarray(potential_density, dtype=float)

    grad_sq = np.sum(spatial_gradient * spatial_gradient, axis=-1)
    return 0.5 * mass_density * time_derivative * time_derivative - 0.5 * stiffness * grad_sq - potential_density
