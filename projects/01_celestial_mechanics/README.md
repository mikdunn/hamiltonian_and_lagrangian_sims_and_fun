# Celestial mechanics and star-chart predictors

## System and field

This project now includes an interactive night-sky simulator that:

- maps visible stars and planets,
- overlays major constellation line segments,
- labels known stars (including Polaris / North Star),
- labels visible planets with descriptive notes,
- labels visible constellations with common names,
- simulates planetary motion through the night with a time slider.

## Hamiltonian and Lagrangian modeling

Planet positions are propagated with a symplectic leapfrog integrator under a
central solar gravitational potential. This uses Hamiltonian dynamics for
orbital updates and reports Hamiltonian/Lagrangian drift diagnostics over the
night window.

## Default location

The default observer location is the University of Michigan (Ann Arbor):

- latitude: `42.2780`
- longitude: `-83.7382`
- timezone: `America/Detroit`

## Run the simulator

```bash
python projects/01_celestial_mechanics/simulation.py
```

This opens the interactive map and time slider.

## Useful options

Custom location:

```bash
python projects/01_celestial_mechanics/simulation.py --lat 34.0522 --lon -118.2437
```

Custom date and time window:

```bash
python projects/01_celestial_mechanics/simulation.py --date 2026-10-15 --start-hour 19 --duration-hours 12
```

Headless visibility report (no UI):

```bash
python projects/01_celestial_mechanics/simulation.py --no-ui
```

Save an initial frame image:

```bash
python projects/01_celestial_mechanics/simulation.py --save-frame ann_arbor_sky.png
```
