# Heat conduction and diffusion waves

## System and field

This project focuses on: Heat conduction and diffusion waves.

## Modeling angle

Diffusion/extended-thermodynamics models in variational form

## Independent and dependent variables

- Independent variables: thermodynamic coordinates (for example `T, V`) and field coordinates `(x, t)`.
- Dependent variables: `S(T,V)`, `P(T,V)`, temperature field `T(x,t)`, and heat flux terms.
- Why: Maxwell relations constrain thermodynamic couplings, while PDE variables capture spatial-temporal evolution.

## Maxwell relations for complex thermal events

The project uses thermodynamic Maxwell identities to model cross-coupled responses:

- `(dS/dV)_T = (dP/dT)_V`
- `(dS/dP)_T = -(dV/dT)_P`
- `(dT/dV)_S = -(dP/dS)_V`
- `(dT/dP)_S = (dV/dS)_P`

These relations are used as consistency constraints when extending beyond simple diffusion.

## Trivial-case verification

The notebook first checks an ideal-gas benchmark:

- `P = nRT/V`
- `(dS/dV)_T = (dP/dT)_V = nR/V`

Both sides are computed and plotted to validate the thermodynamic layer before more complex modeling.

## Real data sources

IR thermography, thermistor arrays

## Why it matters for abstract CS

Inverse heat conduction and thermal control

## Starter usage

Run the local starter simulation:

```bash
python projects/15_heat_diffusion_waves/simulation.py
```
