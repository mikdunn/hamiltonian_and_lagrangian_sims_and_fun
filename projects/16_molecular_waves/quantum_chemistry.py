"""Optional PySCF DFT calculations for explicitly specified molecules."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np

MAX_GRID_POINTS = 350_000
BOHR_ANGSTROM = 0.529177210903
SYMBOLS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn '
    'Fe Co Ni Cu Zn Ga Ge As Se Br Kr'
).split()
ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(SYMBOLS)}

# Single-bond covalent radii in angstroms, used only to suggest geometry
# contacts for the report. They are not used by the electronic calculation.
COVALENT_RADII = dict(zip(
    SYMBOLS,
    (
        .31, .28, 1.28, .96, .84, .76, .71, .66, .57, .58,
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
        2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32,
        1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20,
        1.20, 1.16,
    ),
))


def validate_multiplicity(
    symbols: list[str], charge: int, multiplicity: int
) -> int:
    """Validate the spin state and return PySCF's N_alpha - N_beta spin."""
    if not isinstance(multiplicity, int) or multiplicity < 1:
        raise ValueError('Spin multiplicity must be a positive integer.')
    unsupported = sorted(
        {symbol for symbol in symbols if symbol not in ATOMIC_NUMBERS}
    )
    if unsupported:
        raise ValueError(
            'The initial def2-SVP solver supports elements H through Kr; '
            f'unsupported: {", ".join(unsupported)}.'
        )
    electron_count = sum(ATOMIC_NUMBERS[symbol] for symbol in symbols) - charge
    spin = multiplicity - 1
    if electron_count <= 0:
        raise ValueError('The selected charge leaves no electrons.')
    if spin > electron_count or (electron_count - spin) % 2:
        raise ValueError(
            f'{electron_count} electrons are incompatible with '
            f'multiplicity {multiplicity}; choose a spin-compatible state.'
        )
    return spin


def parse_geometry(text: str, expected_counts: dict[str, int]):
    """Parse one atom and Cartesian x/y/z coordinate per non-empty row."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(
            'Enter one solver-geometry row per atom: element x y z.'
        )
    if len(rows) > 24:
        raise ValueError(
            'The initial interactive solver supports up to 24 atoms.'
        )
    atoms = []
    counts: dict[str, int] = {}
    for row in rows:
        if len(row) != 4:
            raise ValueError(
                'Each solver-geometry row must be: element x y z '
                '(coordinates in angstroms).'
            )
        symbol = row[0]
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(
                f'{symbol} is outside the initial H-through-Kr solver range.'
            )
        try:
            position = tuple(float(value) for value in row[1:])
        except ValueError as exc:
            raise ValueError('Solver coordinates must be numbers.') from exc
        if not np.isfinite(position).all():
            raise ValueError('Solver coordinates must be finite.')
        atoms.append((symbol, position))
        counts[symbol] = counts.get(symbol, 0) + 1
    if counts != expected_counts:
        raise ValueError(
            f'Solver geometry composition {counts} does not match '
            f'the selected formula: {expected_counts}.'
        )
    coordinates = np.asarray([position for _, position in atoms])
    distances = np.linalg.norm(
        coordinates[:, None] - coordinates[None, :], axis=-1
    )
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 0.1:
        raise ValueError('Nuclei cannot coincide or be less than 0.1 Å apart.')
    if np.ptp(coordinates, axis=0).max() > 30:
        raise ValueError('Solver geometries must span at most 30 Å.')
    return atoms


def infer_geometry_contacts(
    symbols: list[str], coordinates: np.ndarray
) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, int, float]]]:
    """Return approximate close-pair distances and contact-based angles."""
    positions = np.asarray(coordinates, dtype=float)
    if positions.shape != (len(symbols), 3) or not np.isfinite(positions).all():
        raise ValueError('Coordinates must be a finite N-by-3 array.')
    unsupported = sorted(
        {symbol for symbol in symbols if symbol not in COVALENT_RADII}
    )
    if unsupported:
        raise ValueError(
            f'Missing contact-radius data for {", ".join(unsupported)}.'
        )
    contacts = []
    for i, j in combinations(range(len(symbols)), 2):
        distance = float(np.linalg.norm(positions[i] - positions[j]))
        cutoff = 1.25 * (
            COVALENT_RADII[symbols[i]] + COVALENT_RADII[symbols[j]]
        )
        if distance <= cutoff:
            contacts.append((i, j, distance))

    neighbours: dict[int, list[int]] = {i: [] for i in range(len(symbols))}
    for i, j, _ in contacts:
        neighbours[i].append(j)
        neighbours[j].append(i)
    angles = []
    for center, connected in neighbours.items():
        for first, second in combinations(connected, 2):
            a = positions[first] - positions[center]
            b = positions[second] - positions[center]
            cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            angle = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
            angles.append((first, center, second, angle))
    return contacts, angles


def _grid_axes(
    coordinates: np.ndarray, spacing: float, padding: float = 4.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('Density grid spacing must be a positive number.')
    coordinates = np.asarray(coordinates, dtype=float)
    low = coordinates.min(axis=0) - padding
    high = coordinates.max(axis=0) + padding
    axes = tuple(
        np.arange(low[i], high[i] + spacing / 2, spacing)
        for i in range(3)
    )
    count = math.prod(len(axis) for axis in axes)
    if count > MAX_GRID_POINTS:
        raise ValueError(
            f'Density grid would contain {count:,} points; the limit is '
            f'{MAX_GRID_POINTS:,}. Increase grid spacing or use a smaller '
            'molecule.'
        )
    return axes


def _density_matrix(mf):
    density = np.asarray(mf.make_rdm1())
    return density.sum(axis=0) if density.ndim == 3 else density


def is_unrestricted(mf) -> bool:
    return np.asarray(mf.mo_coeff).ndim == 3


def _evaluate_field(mol, axes, density_matrix, coefficients=None):
    x, y, z = np.meshgrid(*axes, indexing='ij')
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    density = np.empty(len(points), dtype=float)
    orbital = np.empty(len(points), dtype=float) if coefficients is not None else None
    for start in range(0, len(points), 10_000):
        stop = min(start + 10_000, len(points))
        ao = mol.eval_gto('GTOval', points[start:stop] / BOHR_ANGSTROM)
        density[start:stop] = np.einsum(
            'pi,ij,pj->p', ao, density_matrix, ao, optimize=True
        )
        if orbital is not None:
            orbital[start:stop] = ao @ coefficients
    shape = tuple(len(axis) for axis in axes)
    return (
        density.reshape(shape),
        orbital.reshape(shape) if orbital is not None else None,
    )


def calculate_dft(
    atoms: list[tuple[str, tuple[float, float, float]]],
    charge: int = 0,
    multiplicity: int = 1,
    basis: str = 'def2-SVP',
    functional: str = 'B3LYP',
    max_steps: int = 50,
    grid_spacing: float = 0.4,
) -> dict:
    """Optimize a supplied geometry and calculate its DFT density and MOs."""
    if not atoms:
        raise ValueError('Provide at least one atom and its Cartesian position.')
    if not 1 <= max_steps <= 200:
        raise ValueError('Geometry optimization steps must be from 1 to 200.')
    symbols = [atom[0] for atom in atoms]
    coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
    if coordinates.shape != (len(atoms), 3) or not np.isfinite(coordinates).all():
        raise ValueError('Every atom needs three finite Cartesian coordinates.')
    spin = validate_multiplicity(symbols, charge, multiplicity)
    axes = _grid_axes(coordinates, grid_spacing)

    try:
        from pyscf import dft, gto
        from pyscf.geomopt.geometric_solver import (
            GeomOptNotConvergedError,
            optimize,
        )
    except ImportError as exc:
        raise ImportError(
            'The optional DFT solver requires PySCF and geomeTRIC. '
            'Install them in this notebook kernel with '
            '`%pip install -r projects/16_molecular_waves/'
            'requirements-quantum-chemistry.txt`, then restart the kernel.'
        ) from exc

    mol = gto.M(
        atom=atoms,
        basis=basis,
        charge=charge,
        spin=spin,
        unit='Angstrom',
        symmetry=False,
        verbose=0,
    )
    mean_field_type = dft.RKS if spin == 0 else dft.UKS
    initial = mean_field_type(mol)
    initial.xc = functional
    initial.max_cycle = 100
    initial.conv_tol = 1e-8
    try:
        optimized_mol = optimize(initial, maxsteps=max_steps)
    except GeomOptNotConvergedError as exc:
        raise RuntimeError(
            f'Geometry optimization did not converge in {max_steps} steps. '
            'Try a better starting geometry or allow more optimization steps.'
        ) from exc
    mean_field = mean_field_type(optimized_mol)
    mean_field.xc = functional
    mean_field.max_cycle = 100
    mean_field.conv_tol = 1e-8
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(
            'The DFT self-consistent-field calculation did not converge. '
            'Try a different initial geometry, functional, or spin state.'
        )

    optimized_coordinates = np.asarray(optimized_mol.atom_coords(unit='Angstrom'))
    axes = _grid_axes(optimized_coordinates, grid_spacing)
    density_matrix = _density_matrix(mean_field)
    density, _ = _evaluate_field(optimized_mol, axes, density_matrix)
    contacts, angles = infer_geometry_contacts(symbols, optimized_coordinates)
    return {
        'symbols': symbols,
        'coordinates': optimized_coordinates,
        'charge': charge,
        'multiplicity': multiplicity,
        'basis': basis,
        'functional': functional,
        'energy_hartree': float(mean_field.e_tot),
        'density_matrix': density_matrix,
        'density': density,
        'axes': axes,
        'mol': optimized_mol,
        'mean_field': mean_field,
        'contacts': contacts,
        'angles': angles,
        'grid_spacing': grid_spacing,
    }


def molecular_orbital(result: dict, index: int, spin_channel: int = 0):
    """Evaluate one real-space MO and return its field and occupation."""
    mf = result['mean_field']
    if is_unrestricted(mf):
        if spin_channel not in (0, 1):
            raise ValueError('Open-shell spin channel must be 0 (alpha) or 1 (beta).')
        coefficients = np.asarray(mf.mo_coeff[spin_channel])
        occupations = np.asarray(mf.mo_occ[spin_channel])
        label = 'alpha' if spin_channel == 0 else 'beta'
    else:
        if spin_channel != 0:
            raise ValueError('Closed-shell calculations have one MO channel.')
        coefficients = np.asarray(mf.mo_coeff)
        occupations = np.asarray(mf.mo_occ)
        label = 'restricted'
    if not 0 <= index < coefficients.shape[1]:
        raise ValueError('Selected molecular orbital is out of range.')
    _, field = _evaluate_field(
        result['mol'],
        result['axes'],
        result['density_matrix'],
        coefficients[:, index],
    )
    return field, float(occupations[index]), label


def density_figure(result: dict, threshold_fraction: float = 0.02):
    """Create an isosurface of total electron density in electrons/bohr^3."""
    import plotly.graph_objects as go

    x, y, z = np.meshgrid(*result['axes'], indexing='ij')
    rho = result['density']
    lower = max(float(rho.max()) * threshold_fraction, 1e-8)
    fig = go.Figure(
        go.Isosurface(
            x=x.ravel(),
            y=y.ravel(),
            z=z.ravel(),
            value=rho.ravel(),
            isomin=lower,
            isomax=float(rho.max()) * 0.45,
            surface_count=3,
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title='e⁻/bohr³'),
            name='Total electron density',
        )
    )
    _add_nuclei(fig, result)
    fig.update_layout(
        title='Self-consistent total electron density',
        scene=dict(aspectmode='data', xaxis_title='x (Å)',
                   yaxis_title='y (Å)', zaxis_title='z (Å)'),
    )
    return fig


def orbital_figure(
    result: dict,
    index: int,
    spin_channel: int = 0,
    threshold_fraction: float = 0.12,
):
    """Create signed MO isosurfaces; color indicates wavefunction phase."""
    import plotly.graph_objects as go

    field, occupancy, label = molecular_orbital(result, index, spin_channel)
    x, y, z = np.meshgrid(*result['axes'], indexing='ij')
    cutoff = max(float(np.abs(field).max()) * threshold_fraction, 1e-8)
    fig = go.Figure(
        go.Isosurface(
            x=x.ravel(),
            y=y.ravel(),
            z=z.ravel(),
            value=field.ravel(),
            isomin=-cutoff,
            isomax=cutoff,
            surface_count=2,
            colorscale=[[0, '#d62728'], [0.5, '#ffffff'], [1, '#1f77b4']],
            cmid=0,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
            name=f'{label} MO {index + 1}',
        )
    )
    _add_nuclei(fig, result)
    fig.update_layout(
        title=f'{label.capitalize()} MO {index + 1} ({occupancy:g} e⁻)',
        scene=dict(aspectmode='data', xaxis_title='x (Å)',
                   yaxis_title='y (Å)', zaxis_title='z (Å)'),
    )
    return fig, occupancy, label


def _add_nuclei(fig, result):
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter3d(
            x=result['coordinates'][:, 0],
            y=result['coordinates'][:, 1],
            z=result['coordinates'][:, 2],
            mode='markers+text',
            text=result['symbols'],
            textposition='top center',
            marker=dict(size=5, color='#333333', line=dict(color='white', width=1)),
            name='Nuclei',
        )
    )
