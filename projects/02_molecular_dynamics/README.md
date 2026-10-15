# Molecular dynamics simulations

## System and field

This project focuses on: Molecular dynamics simulations.

## Modeling angle

Reduced coarse-grained center dynamics, plus a separate optional atomistic
DFT geometry/electron-density calculation.

## Real data sources

PDB, NMR, SAXS, and cryo-EM trajectories are potential sources for future
validation; the current starter simulation does not load or fit those data.

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

## Scope and optional electron-density calculation

The interactive trajectory is a reduced coarse-grained model: molecule centers
interact through a softened, truncated Lennard–Jones-like potential, while
rigid intramolecular templates are used for rendering. It does not calculate
bond stretching, angle/torsion energies, atom-level orbital overlap, or
electron-density-dependent forces. Thermostat/noise, damping, box rescaling,
and stochastic reaction rules mean the full trajectory does not conserve the
conservative reference Hamiltonian described in the notebook.

The notebook provides a separate optional PySCF DFT workflow
for user-entered molecular geometries. It optimizes the specified geometry and
plots the resulting total molecular electron density using atom-centered basis
functions (H–Kr, up to 24 atoms), and shows the signed HOMO isosurface. These
electronic-structure results are not coupled to the coarse-grained trajectory.
Install the optional dependencies in the
notebook kernel and restart it:

```ipython
%pip install -r projects/16_molecular_waves/requirements-quantum-chemistry.txt
```

The DFT results depend on the supplied structure, charge, spin, functional,
and basis set; the default is B3LYP/def2-SVP. The density is a molecular total,
not a unique per-atom partition. Displayed contact-derived angles are geometry
descriptors, not angle potentials in the molecular-dynamics engine.

The last notebook section is a bounded organic product calculator and
reaction-network simulator. Enable two to four reactants; select atom counts
(C/H/N/O/S/F/Cl/Br), ring presence, and a matching explicit isomer. A small
catalog includes ketones, aldehydes, carboxylic acids, alcohols, and some
nonreactive comparison structures. This is not exhaustive isomer enumeration.
Custom neutral, closed-shell SMILES are supported with matching formula/ring
settings, up to 24 heavy atoms per structure. Formula and ring presence alone
cannot determine molecular connectivity or reactivity. Install the optional
graph dependency in the notebook kernel and restart it:

```ipython
%pip install -r projects/02_molecular_dynamics/requirements-organic-reactions.txt
```

### Product rules, dynamics and yield optimization

`organic_product_simulator.py` implements two declared reaction families:
alpha-H ketone + aldehyde aldol addition, and carboxylic acid + aliphatic alcohol
esterification. Atom-mapped rules enumerate constitutional product candidates,
deduplicate equivalent sites and check elemental balance including esterification
water. Unsupported pairings produce an explicit explanation, not fabricated
products. Selecting a reaction family does not establish that experimental
conditions permit the reaction; catalysts, solvents, reverse reactions,
stereoselectivity and transition barriers are not predicted.

All candidate channels compete in a fixed-volume, 1 L concentration model.
For channel j, `v_j = k_f(T) c_a c_b / m_j`, with equal branching across `m_j`
regio-products of a pair. Product forms at `v_j` and is lost at `k_d(T) p_j`.
A tracked product-equivalent loss pool retains the material balance.
The demonstration constants, shared by both families, are:

- `k_f(T) = 0.035 exp[(35000/8.314)(1/310 - 1/T)]` L/mol/min.
- `k_d(T) = 0.0015 exp[(70000/8.314)(1/310 - 1/T)]` per minute.

These rates and activation energies are illustrative, not measured values.
Yield is the selected product divided by the initial target substrate,
reported as a percentage; it is not experimentally isolated yield.
The optimizer searches 6 temperatures (280-380 K), 5 multipliers (0.5-3)
of the target partner feed and 5 durations (10-120 minutes), plus the baseline.
It updates controls to the best of 151 model evaluations, never performing worse
than the baseline in that model. This is not a continuous/global optimum,
experimental process optimization, or a synthesis recommendation. Reagent cost
and work-up recovery are not modeled. Multiple reactants can produce competing
products; loss is a bookkeeping pool, not an identified degradation molecule.

Playback uses RDKit ETKDG/UFF conformers and schematic approach/fading based on
concentration trajectories. It is not atomistic molecular dynamics or a
simulation of a bond-breaking path. The separate DFT and coarse-grained MD
calculations above remain unchanged.

For R reactants and P product channels, each ODE evaluation is O(R+P).
Output memory is O(F(R+2P)) for F frames. The finite optimizer integrates
151 condition sets and reintegrates the best for playback, retaining only the
best case. Product-rule matching can grow
combinatorially with graph size/matching sites; inputs are bounded and more
than 32 channels are rejected. Rendering scales with displayed atom/bond counts.

The earlier acetone pKa/kinetics/lifted-DMD teaching implementation remains in
`organic_reactions.py` with its tests, but is no longer the last notebook UI.

### Notebook output and input checks

The phase simulator displays molecular motion and full-trajectory diagnostics
as soon as its cell runs. Diagnostics summarize the whole precomputed run, not
just the displayed frame, and refresh when simulation controls change.
The organic simulator displays its default trajectory on startup. Button
callback tracebacks are captured in the labeled organic and DFT output panels.

The DFT multiplicity selector only offers electron-count-compatible spin states
for the entered geometry and charge; this does not establish the ground state.
Neutral water has 10 electrons and uses multiplicity 1 (singlet), not 59.
Invalid geometry or charge disables the calculation and displays an explanation.
Use `%pip` or VS Code's notebook package installation in the selected kernel:
installing RDKit in a different Python environment will not fix a notebook
`ModuleNotFoundError`. Rerun the simulator cells after a kernel restart.
