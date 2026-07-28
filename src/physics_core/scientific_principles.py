"""Reusable scientific principles for mechanics, waves, and diffusion modeling."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ArrayLike = np.ndarray


def euler_lagrange_residual(d_dt_dL_dqdot: ArrayLike, dL_dq: ArrayLike) -> ArrayLike:
    """Compute Euler-Lagrange residual: d/dt(dL/dq_dot) - dL/dq."""
    return np.asarray(d_dt_dL_dqdot, dtype=float) - np.asarray(dL_dq, dtype=float)


def discrete_least_action(
    lagrangian_fn: Callable[[ArrayLike, ArrayLike, float], float],
    q_path: ArrayLike,
    t_path: ArrayLike,
) -> float:
    """Compute a discrete action estimate from a trajectory."""
    q_path = np.asarray(q_path, dtype=float)
    t_path = np.asarray(t_path, dtype=float)

    dt = np.diff(t_path)
    qdot = np.diff(q_path, axis=0) / dt[:, None]
    q_mid = 0.5 * (q_path[1:] + q_path[:-1])
    t_mid = 0.5 * (t_path[1:] + t_path[:-1])

    lagrangian_values = np.array(
        [lagrangian_fn(q, qd, t) for q, qd, t in zip(q_mid, qdot, t_mid, strict=True)],
        dtype=float,
    )
    return float(np.sum(lagrangian_values * dt))


def check_energy_conservation(energies: ArrayLike, tolerance: float = 1e-3) -> dict[str, float | bool]:
    """Return diagnostic metrics for energy drift in a simulation."""
    energies = np.asarray(energies, dtype=float)
    initial = energies[0]
    drift = float(np.max(np.abs(energies - initial)))
    relative = drift / max(abs(initial), 1e-12)
    return {
        "initial_energy": float(initial),
        "max_abs_drift": drift,
        "max_relative_drift": float(relative),
        "conserved_within_tolerance": bool(relative <= tolerance),
    }


def poisson_bracket(
    grad_f_q: ArrayLike,
    grad_f_p: ArrayLike,
    grad_g_q: ArrayLike,
    grad_g_p: ArrayLike,
) -> float:
    """Compute canonical Poisson bracket {f, g}."""
    grad_f_q = np.asarray(grad_f_q, dtype=float)
    grad_f_p = np.asarray(grad_f_p, dtype=float)
    grad_g_q = np.asarray(grad_g_q, dtype=float)
    grad_g_p = np.asarray(grad_g_p, dtype=float)
    return float(np.dot(grad_f_q, grad_g_p) - np.dot(grad_f_p, grad_g_q))


def noether_linear_momentum(masses: ArrayLike, velocities: ArrayLike) -> ArrayLike:
    """Total momentum from translation symmetry in mechanics models."""
    masses = np.asarray(masses, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    return np.sum(masses[:, None] * velocities, axis=0)


def wave_equation_residual(d2u_dt2: ArrayLike, laplacian_u: ArrayLike, wave_speed: float) -> ArrayLike:
    """Residual for u_tt - c^2 nabla^2 u = 0."""
    return np.asarray(d2u_dt2, dtype=float) - (wave_speed**2) * np.asarray(laplacian_u, dtype=float)


def diffusion_equation_residual(dT_dt: ArrayLike, laplacian_T: ArrayLike, diffusivity: float) -> ArrayLike:
    """Residual for T_t - alpha nabla^2 T = 0."""
    return np.asarray(dT_dt, dtype=float) - diffusivity * np.asarray(laplacian_T, dtype=float)


def principle_registry() -> dict[str, str]:
    """Human-readable registry of shared scientific principles."""
    return {
        "least_action": "Stationary action principle for trajectories and fields.",
        "euler_lagrange": "Equation of motion from variational derivatives of a Lagrangian.",
        "hamiltonian_flow": "Canonical phase-space dynamics preserving symplectic structure.",
        "noether_symmetry": "Conservation laws induced by continuous symmetries.",
        "wave_dynamics": "Hyperbolic field dynamics for strings, acoustics, and EM analogs.",
        "diffusion_gradient_flow": "Parabolic transport and Onsager-style dissipative dynamics.",
        "poisson_structure": "Bracket algebra for observables in Hamiltonian systems.",
    }
