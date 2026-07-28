# Hamiltonian and Lagrangian Sims and Fun

This repository is organized as a reusable Python framework for building physics-informed projects using:

- Hamiltonian mechanics
- Lagrangian and variational methods
- Field-theoretic wave and diffusion models

It includes separate folders for 22 real-data project domains and a shared core library you can reuse across future projects.

## Project Layout

```text
.
├── projects/
│   ├── 01_celestial_mechanics/
│   ├── 02_molecular_dynamics/
│   ├── ...
│   └── 22_traffic_crowd_info_waves/
├── src/
│   ├── physics_core/
│   │   ├── __init__.py
│   │   ├── hamiltonian_lagrangian.py
│   │   └── scientific_principles.py
│   ├── domain_catalog.py
│   └── template_simulation.py
└── requirements.txt
```

## Reusable Core Functions

The shared mechanics API lives in `src/physics_core/hamiltonian_lagrangian.py`:

- `kinetic_energy_point_masses`
- `gravitational_potential`
- `spring_potential`
- `lagrangian`
- `canonical_hamiltonian`
- `hamilton_equations`
- `action_from_trajectory`
- `field_lagrangian_density`

The shared scientific-principles API lives in `src/physics_core/scientific_principles.py`:

- `euler_lagrange_residual`
- `discrete_least_action`
- `check_energy_conservation`
- `poisson_bracket`
- `noether_linear_momentum`
- `wave_equation_residual`
- `diffusion_equation_residual`
- `principle_registry`

Extended operators and methods:

- `divergence_vector_field`
- `curl_vector_field`
- `laplacian_scalar_field`
- `spherical_to_cartesian`
- `cartesian_to_spherical`
- `radial_laplacian_spherical`
- `greens_function_poisson_3d`
- `greens_function_helmholtz_3d`
- `greens_function_heat_kernel_1d`
- `incompressible_navier_stokes_residual_2d`
- `discrete_fourier_transform`
- `inverse_discrete_fourier_transform`
- `discrete_wavelet_transform_haar`
- `inverse_discrete_wavelet_transform_haar`

These are intended to be imported and reused in any project folder or external repository.

## Domain Folders

Each folder under `projects/` contains:

- A domain-specific `README.md` (system, model angle, real data, CS relevance)
- A `simulation.py` starter wired to shared core functions

Domain list:

1. Celestial mechanics and star-chart predictors
2. Molecular dynamics simulations
3. Enzyme kinetics for specific molecules
4. Protein binding energy predictions
5. DNA repair simulations
6. Circuit analysis and EM fields
7. Computer engineering information-flow dynamics
8. Music and acoustical signal dynamics
9. Bond energies and reaction pathways
10. Weather dynamics simulations
11. Fluid dynamics
12. Coupled oscillators and mechanical field analogs
13. Vibrating strings, beams, and membranes
14. Electromagnetic experiments and coupled resonators
15. Heat conduction and diffusion waves
16. Molecular waves and spectra
17. Water waves and free-surface dynamics
18. Acoustical dynamics in rooms and media
19. Seismic and elastic waves
20. Plasma waves and collective modes
21. Quantum wave mechanics
22. Traffic, crowd, and information-flow waves

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run a Starter Simulation

Example:

```bash
python projects/01_celestial_mechanics/simulation.py
```

The output is JSON containing domain metadata, a baseline mass-spring mechanics sanity check, and suggested next steps for fitting real data.

Each simulation now also includes domain-aware diagnostics where applicable, such as:

- Vector calculus (`div`, `curl`, `laplacian`) on sample fields
- Spherical coordinate modeling checks
- Green's-function evaluations for Poisson/Helmholtz/heat kernels
- Navier-Stokes residual diagnostics for fluid-like domains
- DFT/DWT spectral diagnostics for signal/spectroscopy/wave domains

## Suggested Next Build Step

Pick one domain folder and add:

1. A data loader (`data_loader.py`) for your real dataset.
2. A model file (`model.py`) with your domain-specific Hamiltonian/Lagrangian terms.
3. A trainer/inference script (`fit.py`) for parameter estimation.
4. Validation checks using `check_energy_conservation`, wave/diffusion residuals, or custom invariants.
