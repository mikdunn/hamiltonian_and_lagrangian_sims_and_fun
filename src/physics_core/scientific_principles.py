"""Reusable scientific principles for mechanics, waves, and diffusion modeling."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ArrayLike: TypeAlias = NDArray[np.float64]


def _central_gradient_nd(
    field: ArrayLike, spacing: float | tuple[float, ...]
) -> list[ArrayLike]:
    """Return central-difference gradients for an N-D scalar field."""
    field = np.asarray(field, dtype=float)
    spacings = (
        spacing if isinstance(spacing, tuple) else (spacing,) * field.ndim
    )
    return list(np.gradient(field, *spacings, edge_order=2))


def euler_lagrange_residual(
    d_dt_dL_dqdot: ArrayLike, dL_dq: ArrayLike
) -> ArrayLike:
    """Compute Euler-Lagrange residual: d/dt(dL/dq_dot) - dL/dq."""
    return np.asarray(d_dt_dL_dqdot, dtype=float) - np.asarray(
        dL_dq, dtype=float
    )


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
        [
            lagrangian_fn(q, qd, t)
            for q, qd, t in zip(q_mid, qdot, t_mid, strict=True)
        ],
        dtype=float,
    )
    return float(np.sum(lagrangian_values * dt))


def check_energy_conservation(
    energies: ArrayLike, tolerance: float = 1e-3
) -> dict[str, float | bool]:
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


def noether_linear_momentum(
    masses: ArrayLike, velocities: ArrayLike
) -> ArrayLike:
    """Total momentum from translation symmetry in mechanics models."""
    masses = np.asarray(masses, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    return np.sum(masses[:, None] * velocities, axis=0)


def wave_equation_residual(
    d2u_dt2: ArrayLike, laplacian_u: ArrayLike, wave_speed: float
) -> ArrayLike:
    """Residual for u_tt - c^2 nabla^2 u = 0."""
    return np.asarray(d2u_dt2, dtype=float) - (wave_speed**2) * np.asarray(
        laplacian_u, dtype=float
    )


def diffusion_equation_residual(
    dT_dt: ArrayLike, laplacian_T: ArrayLike, diffusivity: float
) -> ArrayLike:
    """Residual for T_t - alpha nabla^2 T = 0."""
    return np.asarray(dT_dt, dtype=float) - diffusivity * np.asarray(
        laplacian_T, dtype=float
    )


def divergence_vector_field(
    vector_field: ArrayLike, spacing: float | tuple[float, ...] = 1.0
) -> ArrayLike:
    """Compute divergence of a vector field with components on axis 0."""
    vector_field = np.asarray(vector_field, dtype=float)
    dims = vector_field.shape[0]
    spacings = spacing if isinstance(spacing, tuple) else (spacing,) * dims

    div = np.zeros_like(vector_field[0])
    for axis in range(dims):
        div += np.gradient(
            vector_field[axis], spacings[axis], axis=axis, edge_order=2
        )
    return div


def curl_vector_field(
    vector_field: ArrayLike, spacing: float | tuple[float, float, float] = 1.0
) -> ArrayLike:
    """Compute curl for a 3D vector field with components on axis 0."""
    vector_field = np.asarray(vector_field, dtype=float)
    if vector_field.shape[0] != 3:
        raise ValueError(
            "curl_vector_field expects 3 components (Fx, Fy, Fz)."
        )

    sx, sy, sz = (
        spacing if isinstance(spacing, tuple) else (spacing, spacing, spacing)
    )
    fx, fy, fz = vector_field

    d_fz_dy = np.gradient(fz, sy, axis=1, edge_order=2)
    d_fy_dz = np.gradient(fy, sz, axis=2, edge_order=2)
    d_fx_dz = np.gradient(fx, sz, axis=2, edge_order=2)
    d_fz_dx = np.gradient(fz, sx, axis=0, edge_order=2)
    d_fy_dx = np.gradient(fy, sx, axis=0, edge_order=2)
    d_fx_dy = np.gradient(fx, sy, axis=1, edge_order=2)

    return np.stack(
        [
            d_fz_dy - d_fy_dz,
            d_fx_dz - d_fz_dx,
            d_fy_dx - d_fx_dy,
        ],
        axis=0,
    )


def laplacian_scalar_field(
    field: ArrayLike, spacing: float | tuple[float, ...] = 1.0
) -> ArrayLike:
    """Compute scalar Laplacian using repeated central differences."""
    field = np.asarray(field, dtype=float)
    spacings = (
        spacing if isinstance(spacing, tuple) else (spacing,) * field.ndim
    )

    result = np.zeros_like(field)
    for axis in range(field.ndim):
        first = np.gradient(field, spacings[axis], axis=axis, edge_order=2)
        second = np.gradient(first, spacings[axis], axis=axis, edge_order=2)
        result += second
    return result


def spherical_to_cartesian(
    radius: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> ArrayLike:
    """Convert spherical coordinates (r, theta, phi) to Cartesian (x, y, z)."""
    radius = np.asarray(radius, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.stack([x, y, z], axis=0)


def cartesian_to_spherical(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Convert Cartesian coordinates (x, y, z) to (r, theta, phi)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    radius = np.sqrt(x * x + y * y + z * z)
    radius_safe = np.maximum(radius, 1e-12)
    theta = np.arccos(np.clip(z / radius_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return radius, theta, phi


def radial_laplacian_spherical(
    radius: ArrayLike, scalar_field_r: ArrayLike
) -> ArrayLike:
    """Compute radial-only spherical Laplacian: (1/r^2) d/dr (r^2 dF/dr)."""
    radius = np.asarray(radius, dtype=float)
    scalar_field_r = np.asarray(scalar_field_r, dtype=float)

    dF_dr = np.gradient(scalar_field_r, radius, edge_order=2)
    flux = radius * radius * dF_dr
    d_flux_dr = np.gradient(flux, radius, edge_order=2)
    radius_safe = np.maximum(radius * radius, 1e-12)
    return d_flux_dr / radius_safe


def greens_function_poisson_3d(radius: ArrayLike) -> ArrayLike:
    """Return Green's function for 3D Poisson equation: G = -1/(4*pi*r)."""
    radius = np.asarray(radius, dtype=float)
    radius_safe = np.maximum(radius, 1e-12)
    return -1.0 / (4.0 * np.pi * radius_safe)


def greens_function_helmholtz_3d(
    radius: ArrayLike, wavenumber: float
) -> ArrayLike:
    """Return outgoing 3D Helmholtz Green's function magnitude |exp(ikr)/(4*pi*r)|."""
    radius = np.asarray(radius, dtype=float)
    radius_safe = np.maximum(radius, 1e-12)
    complex_value = np.exp(1j * wavenumber * radius_safe) / (
        4.0 * np.pi * radius_safe
    )
    return np.abs(complex_value)


def greens_function_heat_kernel_1d(
    x: ArrayLike, time: float, diffusivity: float
) -> ArrayLike:
    """Return 1D heat kernel G(x,t) = (4*pi*alpha*t)^(-1/2) exp(-x^2/(4*alpha*t))."""
    x = np.asarray(x, dtype=float)
    t_safe = max(time, 1e-12)
    alpha_safe = max(diffusivity, 1e-12)
    norm = 1.0 / np.sqrt(4.0 * np.pi * alpha_safe * t_safe)
    return norm * np.exp(-(x * x) / (4.0 * alpha_safe * t_safe))


def incompressible_navier_stokes_residual_2d(
    u: ArrayLike,
    v: ArrayLike,
    pressure: ArrayLike,
    viscosity: float,
    density: float,
    spacing: float,
    dt: float,
    u_prev: ArrayLike,
    v_prev: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return momentum and continuity residuals for 2D incompressible flow."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    u_prev = np.asarray(u_prev, dtype=float)
    v_prev = np.asarray(v_prev, dtype=float)

    du_dt = (u - u_prev) / dt
    dv_dt = (v - v_prev) / dt

    du_dx = np.gradient(u, spacing, axis=0, edge_order=2)
    du_dy = np.gradient(u, spacing, axis=1, edge_order=2)
    dv_dx = np.gradient(v, spacing, axis=0, edge_order=2)
    dv_dy = np.gradient(v, spacing, axis=1, edge_order=2)

    dp_dx = np.gradient(pressure, spacing, axis=0, edge_order=2)
    dp_dy = np.gradient(pressure, spacing, axis=1, edge_order=2)

    lap_u = laplacian_scalar_field(u, spacing=spacing)
    lap_v = laplacian_scalar_field(v, spacing=spacing)

    residual_u = (
        du_dt + (u * du_dx + v * du_dy) + dp_dx / density - viscosity * lap_u
    )
    residual_v = (
        dv_dt + (u * dv_dx + v * dv_dy) + dp_dy / density - viscosity * lap_v
    )
    continuity = du_dx + dv_dy

    return residual_u, residual_v, continuity


def discrete_fourier_transform(signal: ArrayLike) -> ArrayLike:
    """Return DFT coefficients using numpy FFT backend."""
    signal = np.asarray(signal, dtype=float)
    return np.fft.fft(signal)


def inverse_discrete_fourier_transform(coefficients: ArrayLike) -> ArrayLike:
    """Return real part of inverse DFT reconstruction."""
    coefficients = np.asarray(coefficients)
    reconstructed = np.fft.ifft(coefficients)
    return np.real(reconstructed)


def discrete_wavelet_transform_haar(
    signal: ArrayLike,
) -> tuple[ArrayLike, ArrayLike]:
    """Single-level Haar DWT returning approximation and detail coefficients."""
    signal = np.asarray(signal, dtype=float)
    if signal.size % 2 != 0:
        signal = np.pad(signal, (0, 1), mode="edge")

    even = signal[0::2]
    odd = signal[1::2]
    approx = (even + odd) / np.sqrt(2.0)
    detail = (even - odd) / np.sqrt(2.0)
    return approx, detail


def inverse_discrete_wavelet_transform_haar(
    approx: ArrayLike, detail: ArrayLike
) -> ArrayLike:
    """Inverse single-level Haar DWT reconstruction."""
    approx = np.asarray(approx, dtype=float)
    detail = np.asarray(detail, dtype=float)
    even = (approx + detail) / np.sqrt(2.0)
    odd = (approx - detail) / np.sqrt(2.0)

    out = np.empty(approx.size + detail.size, dtype=float)
    out[0::2] = even
    out[1::2] = odd
    return out


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
        "vector_calculus": "Differential operators: divergence, curl, and Laplacian on discretized fields.",
        "spherical_modeling": "Coordinate transforms and radial operators for central-force/field problems.",
        "greens_functions": "Fundamental solutions for Poisson, Helmholtz, and heat equations.",
        "navier_stokes": "Incompressible fluid momentum and continuity residual diagnostics.",
        "spectral_methods": "DFT and wavelet decompositions for frequency- and scale-domain analysis.",
    }
