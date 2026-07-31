# Molecular dynamics simulations

## System and field

This project focuses on: Molecular dynamics simulations.

## Modeling angle

Atomistic/coarse-grained Hamiltonians and variational coarse-graining

## Real data sources

PDB, NMR, SAXS, cryo-EM trajectories

## Why it matters for abstract CS

Sampling, differentiable simulation, ML force fields

## Starter usage

Run the local starter simulation:

```bash
python projects/02_molecular_dynamics/simulation.py
```

Run a headless thermodynamic combustion check (shows CH4 oxidation products):

```bash
python projects/02_molecular_dynamics/simulation.py --no-ui --phase gas --mixture CH4:6,O2:12,N2:4 --temperature 980
```
