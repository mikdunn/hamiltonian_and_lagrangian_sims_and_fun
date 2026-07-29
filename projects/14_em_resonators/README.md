# Electromagnetic experiments and coupled resonators

## System and field

This project focuses on: Electromagnetic experiments and coupled resonators.

## Modeling angle

EM action plus discrete resonator-network Hamiltonians

## Independent and dependent variables

- Independent variables: space `(x, y, z)` and time `t`.
- Dependent variables: fields `E, H, D, B` and reduced modal amplitudes `a_n(t)`.
- Why: resonator data is low-dimensional, but the governing model is electromagnetic field dynamics in spacetime.

## Electrodynamics model basis

The notebook starts from Maxwell equations and material closure, then projects to cavity/resonator modes. In source-free homogeneous media this yields wave equations such as:

- `nabla^2 E - mu eps d^2E/dt^2 = 0`

## Trivial-case verification

The first benchmark is the fundamental 1D cavity standing mode:

- `E(x,t) = E0 sin(kx) cos(omega t)`
- `H(x,t) = (E0/Z) cos(kx) sin(omega t)`
- `k = pi/L`, `omega = v k`

The notebook shows this case both analytically and graphically before moving to richer coupling.

## Real data sources

S-parameters, field maps, time-domain responses

## Why it matters for abstract CS

Inverse design and structured EM model reduction

## Starter usage

Run the local starter simulation:

```bash
python projects/14_em_resonators/simulation.py
```
