# Quantum wave mechanics

## System and field

This project focuses on: Quantum wave mechanics.

## Modeling angle

Schrodinger/Dirac equations from action principles

## Independent and dependent variables

- Independent variables: position (`x` or `r`) and time `t`.
- Dependent variable: complex wavefunction `psi(x,t)`.
- Why: observable probabilities are derived from `|psi|^2`, so model interpretation must be anchored to wavefunction dynamics.

## Trivial-case verification

The notebook starts from the infinite-square-well ground state:

- `phi_1(x) = sqrt(2/L) sin(pi x / L)`
- `E_1 = pi^2 hbar^2 / (2 m L^2)`
- `psi_1(x,t) = phi_1(x) exp(-i E_1 t / hbar)`

The derivation and plots verify stationary probability density before richer quantum-wave modeling.

## Real data sources

Spectra, scattering cross sections, cold-atom data

## Why it matters for abstract CS

Quantum simulation and state-learning methods

## Starter usage

Run the local starter simulation:

```bash
python projects/21_quantum_wave_mechanics/simulation.py
```
