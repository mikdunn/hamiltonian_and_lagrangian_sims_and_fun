# Circuit analysis and EM fields

## System and field

This project focuses on: Circuit analysis and EM fields.

## Modeling angle

Flux-charge Lagrangians and EM field action principles

## Independent and dependent variables

- Independent variables: time `t` and spatial coordinates `(x, y, z)` at the field level.
- Dependent variables: electromagnetic fields `E, H, D, B` and reduced circuit states `q(t), i(t), phi(t)`.
- Why: full Maxwell equations govern field transport, while circuit states are low-dimensional observables used for fitting and control.

## Maxwell-equation coverage

This project treats all four Maxwell equations with constitutive relations:

- `div D = rho_f`
- `div B = 0`
- `curl E = -dB/dt`
- `curl H = J_f + dD/dt`
- Closure: `D = eps E`, `B = mu H`, `J_f = sigma E`

## Trivial-case verification

Before complex circuit interactions, we solve the lossless LC case:

- `L q_ddot + (1/C) q = 0`
- `q(t) = Q0 cos(omega0 t)`, `i(t) = -Q0 omega0 sin(omega0 t)`
- Constant total energy check.

The notebook includes both mathematical derivation and plots for this sanity check.

## Real data sources

Voltage/current traces, S-parameters, impedance spectra

## Why it matters for abstract CS

Model reduction, control, network synthesis

## Starter usage

Run the local starter simulation:

```bash
python projects/06_circuit_analysis/simulation.py
```
