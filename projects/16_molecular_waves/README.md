# Molecular waves and spectra

## System and field

This project focuses on: Molecular waves and spectra.

## Modeling angle

Quantum Hamiltonians for vibrational/rotational spectra

## Real data sources

IR/Raman/microwave spectra, pump-probe data

## Why it matters for abstract CS

Spectral inversion, quantum control, ML spectra

## Starter usage

Run the local starter simulation:

```bash
python projects/16_molecular_waves/simulation.py
```

## Interactive 3D orbital explorer

Run the **3D orbital shapes** section of [the molecular-waves notebook](16_molecular_waves_equations_and_derivation.ipynb).
It loads [orbital_explorer.py](orbital_explorer.py); keep that file alongside the notebook.
Install the repository requirements in the selected notebook kernel (especially Plotly and scikit-image).

- Start with the **d orbital gallery**: four separate clovers and the dz² lobes/torus.
- Select one atomic orbital rather than superposing a whole shell.
- In **ethylene**, select the pz–pz pi overlap and compare its atomic contributors,
  bonding MO, antibonding MO and reinforcing/cancelling overlap.
- In **benzene**, inspect six-electron delocalized pi density or individual occupied/empty MOs.
  Pi bonding is side-on overlap; resonance is delocalization, not switching sigma bonds.
- Water/HF hydrogen-bond presets and up to four interacting molecules remain available.

The model uses spherical harmonics, Slater radial functions and generalized extended-Hückel
eigenproblems. It is a qualitative teaching visualization, not a quantum-chemistry structure solver.
The spd/spdf demonstrations are not evidence for d/f participation in real hypervalent bonds.
The surface slider is a fraction of each field's peak magnitude, not a probability threshold.

Run focused regression tests with:

```bash
python -m unittest discover -s projects/16_molecular_waves -p 'test_orbital_explorer.py'
```

## Equation-driven orbital explorer

The notebook's new final section loads [equation_explorer.py](equation_explorer.py).
Enter a formula or equation, such as `2 H2 + O2 -> 2 H2O`, then click **Parse equation**.
It reports stoichiometric atom counts and checks atom/charge conservation without
automatically balancing the equation. Parentheses/brackets, hydrates, phase labels,
Unicode subscripts and explicit charges (`Fe^2+`, `Cl^-`) are supported.

**Atomic orbital** mode lets you choose a species, element, occupied subshell and
individual s/p/d/f orbital. All 118 element symbols are recognized; configurations
are Aufbau estimates with common exceptions, not spectroscopic ground-state data.
Monatomic ion populations are adjusted; polyatomic ions show neutral-atom references.
Hydrogenic radial functions include radial nodes. The adjustable effective nuclear
charge is a manual size parameter, not a many-electron screening calculation.

**Molecular** modes reuse H2, N2, C2H4, C2H2, H2O, HF, C6H6, PF5 and SF6 templates.
Other formulas request explicit coordinates/orbital sets instead of guessing isomers
or geometry. Each structure row is:

```text
element orbital-set x y z [aim-x aim-y aim-z [twist-degrees]]
```

Coordinates are in angstroms; aim defaults to +x. For example, two H2 rows are
`H s 0 0 0 1 0 0` and `H s 0.74 0 0 -1 0 0`. Structure composition must match
the selected formula. Molecular parameters are limited to H/C/N/O/F/P/S/Cl and
neutral species; custom structures support 24 atoms and a 30 angstrom span.
Coefficients affect counts only, not the number of displayed molecule copies.
These are qualitative illustrations, not reaction trajectories or spin-resolved
quantum-chemistry calculations.

Validate both explorers with:

```bash
python -m unittest discover -s projects/16_molecular_waves -p 'test_*explorer.py'
```

### Optional self-consistent DFT density and geometry

The equation-driven explorer also has a separate **PySCF DFT** mode. Install
the optional solver dependencies in the same Python environment as the
notebook:

```python
%pip install -r projects/16_molecular_waves/requirements-quantum-chemistry.txt
```

Restart the kernel, choose a species, and enter one starting geometry row per
atom as `element x y z` (angstroms), or use a built-in geometry template.
The formula sets composition and charge; it does not determine molecular
isomer or starting coordinates. Set spin multiplicity explicitly, then run
the optimizer. The initial def2-SVP workflow accepts elements H through Kr
(Z ≤ 36), uses B3LYP by default, and limits an interactive calculation to 24
atoms. Results include the DFT total energy, optimized coordinates,
total electron-density isosurface, and signed molecular-orbital isosurfaces.
Geometry follows the DFT energy surface, including nuclear repulsion; bond
angles are not imposed by a hand-tuned geometric penalty.
Contact distances and angles are a covalent-radius visualization heuristic,
not a bond-order analysis. A local geometry optimization can find a
nearby stationary structure rather than the global minimum; the result is
not experimental electron density or a reaction/bond energy.
