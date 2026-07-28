"""Shared starter for domain-specific simulations."""

from __future__ import annotations

import numpy as np

from physics_core import (
    canonical_hamiltonian,
    check_energy_conservation,
    curl_vector_field,
    divergence_vector_field,
    discrete_fourier_transform,
    discrete_wavelet_transform_haar,
    greens_function_heat_kernel_1d,
    greens_function_helmholtz_3d,
    greens_function_poisson_3d,
    incompressible_navier_stokes_residual_2d,
    inverse_discrete_fourier_transform,
    inverse_discrete_wavelet_transform_haar,
    kinetic_energy_point_masses,
    laplacian_scalar_field,
    lagrangian,
    principle_registry,
    radial_laplacian_spherical,
    spherical_to_cartesian,
    cartesian_to_spherical,
)


VECTOR_FIELD_DOMAINS = {
    "06_circuit_analysis",
    "10_weather_dynamics",
    "11_fluid_dynamics",
    "14_em_resonators",
    "17_water_waves",
    "18_acoustical_dynamics",
    "19_seismic_elastic_waves",
    "20_plasma_waves",
    "22_traffic_crowd_info_waves",
}

SPHERICAL_DOMAINS = {
    "01_celestial_mechanics",
    "10_weather_dynamics",
    "17_water_waves",
    "19_seismic_elastic_waves",
    "20_plasma_waves",
    "21_quantum_wave_mechanics",
}

GREENS_DOMAINS = {
    "13_strings_beams_membranes",
    "14_em_resonators",
    "15_heat_diffusion_waves",
    "16_molecular_waves",
    "17_water_waves",
    "18_acoustical_dynamics",
    "19_seismic_elastic_waves",
    "20_plasma_waves",
    "21_quantum_wave_mechanics",
}

NAVIER_STOKES_DOMAINS = {
    "10_weather_dynamics",
    "11_fluid_dynamics",
    "17_water_waves",
    "18_acoustical_dynamics",
    "22_traffic_crowd_info_waves",
}

SPECTRAL_DOMAINS = {
    "02_molecular_dynamics",
    "03_enzyme_kinetics",
    "04_protein_binding_energy",
    "08_music_acoustics",
    "09_bond_energy_pathways",
    "13_strings_beams_membranes",
    "15_heat_diffusion_waves",
    "16_molecular_waves",
    "18_acoustical_dynamics",
    "21_quantum_wave_mechanics",
}


def run_domain_diagnostics(domain_key: str) -> dict[str, object]:
    """Return domain-aware advanced diagnostics from shared math/physics tools."""
    diagnostics: dict[str, object] = {}

    if domain_key in VECTOR_FIELD_DOMAINS:
        axis = np.linspace(-1.0, 1.0, 12)
        xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
        vector_field = np.stack([yy, -xx, zz], axis=0)
        div = divergence_vector_field(vector_field, spacing=axis[1] - axis[0])
        curl = curl_vector_field(vector_field, spacing=axis[1] - axis[0])
        scalar = np.exp(-(xx * xx + yy * yy + zz * zz))
        lap = laplacian_scalar_field(scalar, spacing=axis[1] - axis[0])
        diagnostics["vector_calculus"] = {
            "divergence_rms": float(np.sqrt(np.mean(div * div))),
            "curl_rms": float(np.sqrt(np.mean(curl * curl))),
            "laplacian_rms": float(np.sqrt(np.mean(lap * lap))),
        }

    if domain_key in SPHERICAL_DOMAINS:
        radius = np.linspace(0.2, 2.0, 24)
        theta = np.linspace(0.3, 2.4, 24)
        phi = np.linspace(-2.8, 2.8, 24)
        xyz = spherical_to_cartesian(radius, theta, phi)
        radius_back, theta_back, phi_back = cartesian_to_spherical(xyz[0], xyz[1], xyz[2])
        radial_profile = np.exp(-radius) / radius
        radial_lap = radial_laplacian_spherical(radius, radial_profile)
        diagnostics["spherical_modeling"] = {
            "radius_roundtrip_max_error": float(np.max(np.abs(radius_back - radius))),
            "theta_roundtrip_max_error": float(np.max(np.abs(theta_back - theta))),
            "radial_laplacian_rms": float(np.sqrt(np.mean(radial_lap * radial_lap))),
        }

    if domain_key in GREENS_DOMAINS:
        radius = np.linspace(0.1, 2.0, 64)
        x = np.linspace(-1.0, 1.0, 128)
        poisson_vals = greens_function_poisson_3d(radius)
        helmholtz_vals = greens_function_helmholtz_3d(radius, wavenumber=2.5)
        heat_vals = greens_function_heat_kernel_1d(x, time=0.2, diffusivity=0.05)
        diagnostics["greens_functions"] = {
            "poisson_at_r1": float(poisson_vals[np.argmin(np.abs(radius - 1.0))]),
            "helmholtz_mean_magnitude": float(np.mean(helmholtz_vals)),
            "heat_kernel_integral": float(np.trapezoid(heat_vals, x)),
        }

    if domain_key in NAVIER_STOKES_DOMAINS:
        axis = np.linspace(-1.0, 1.0, 16)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        u_prev = np.sin(np.pi * xx) * np.cos(np.pi * yy)
        v_prev = -np.cos(np.pi * xx) * np.sin(np.pi * yy)
        u = 0.99 * u_prev
        v = 0.99 * v_prev
        pressure = 0.5 * (xx * xx + yy * yy)
        ru, rv, rc = incompressible_navier_stokes_residual_2d(
            u=u,
            v=v,
            pressure=pressure,
            viscosity=1e-2,
            density=1.0,
            spacing=axis[1] - axis[0],
            dt=1e-2,
            u_prev=u_prev,
            v_prev=v_prev,
        )
        diagnostics["navier_stokes"] = {
            "u_residual_rms": float(np.sqrt(np.mean(ru * ru))),
            "v_residual_rms": float(np.sqrt(np.mean(rv * rv))),
            "continuity_rms": float(np.sqrt(np.mean(rc * rc))),
        }

    if domain_key in SPECTRAL_DOMAINS:
        n = 128
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        signal = np.sin(2.0 * np.pi * 5.0 * t) + 0.5 * np.sin(2.0 * np.pi * 12.0 * t)
        dft = discrete_fourier_transform(signal)
        reconstructed = inverse_discrete_fourier_transform(dft)
        approx, detail = discrete_wavelet_transform_haar(signal)
        wavelet_reconstruction = inverse_discrete_wavelet_transform_haar(approx, detail)[: signal.size]

        diagnostics["spectral_analysis"] = {
            "dft_peak_index": int(np.argmax(np.abs(dft[: n // 2]))),
            "dft_reconstruction_max_error": float(np.max(np.abs(reconstructed - signal))),
            "dwt_reconstruction_max_error": float(np.max(np.abs(wavelet_reconstruction - signal))),
        }

    return diagnostics


def run_mass_spring_demo(domain_key: str | None = None) -> dict[str, object]:
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

    result = {
        "kinetic": kinetic,
        "potential": potential,
        "lagrangian": lagrangian_value,
        "hamiltonian": hamiltonian_value,
        "energy_check": conservation,
        "principles": principle_registry(),
    }

    if domain_key is not None:
        result["applicable_diagnostics"] = run_domain_diagnostics(domain_key)

    return result
