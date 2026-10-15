"""Equation composition, atomic orbitals and explicitly specified molecules."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import html
import importlib.util
import math
from pathlib import Path
import re

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display
from scipy.special import eval_genlaguerre

_spec = importlib.util.spec_from_file_location(
    'equation_orbital_model', Path(__file__).with_name('orbital_explorer.py')
)
if _spec is None or _spec.loader is None:
    raise ImportError('Cannot load the adjacent orbital_explorer.py.')
model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model)

_qc_spec = importlib.util.spec_from_file_location(
    'molecular_quantum_chemistry',
    Path(__file__).with_name('quantum_chemistry.py'),
)
if _qc_spec is None or _qc_spec.loader is None:
    raise ImportError('Cannot load the adjacent quantum_chemistry.py.')
quantum_chemistry = importlib.util.module_from_spec(_qc_spec)
_qc_spec.loader.exec_module(quantum_chemistry)

SYMBOLS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe '
    'Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In '
    'Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf '
    'Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am '
    'Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'
).split()
ATOMIC_NUMBERS = {symbol: i + 1 for i, symbol in enumerate(SYMBOLS)}
SUBSHELLS = sorted(
    [(n, ell) for n in range(1, 8) for ell in range(min(n, 4))],
    key=lambda shell: (sum(shell), shell[0]),
)
LETTERS = 'spdf'
SUBSCRIPTS = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
SUPERSCRIPTS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+-')

# Common neutral ground-state promotions. Other elements remain explicitly
# labelled Aufbau estimates, rather than asserted spectroscopic configurations.
PROMOTIONS = {
    'Cr': ((4, 0), (3, 2), 1),
    'Cu': ((4, 0), (3, 2), 1),
    'Nb': ((5, 0), (4, 2), 1),
    'Mo': ((5, 0), (4, 2), 1),
    'Ru': ((5, 0), (4, 2), 1),
    'Rh': ((5, 0), (4, 2), 1),
    'Pd': ((5, 0), (4, 2), 2),
    'Ag': ((5, 0), (4, 2), 1),
    'La': ((4, 3), (5, 2), 1),
    'Ce': ((4, 3), (5, 2), 1),
    'Gd': ((4, 3), (5, 2), 1),
    'Pt': ((6, 0), (5, 2), 1),
    'Au': ((6, 0), (5, 2), 1),
    'Ac': ((5, 3), (6, 2), 1),
    'Th': ((5, 3), (6, 2), 2),
    'Pa': ((5, 3), (6, 2), 1),
    'U': ((5, 3), (6, 2), 1),
    'Np': ((5, 3), (6, 2), 1),
    'Cm': ((5, 3), (6, 2), 1),
    'Lr': ((6, 2), (7, 1), 1),
}


@dataclass(frozen=True)
class Species:
    formula: str
    coefficient: int
    counts: dict[str, int]
    charge: int = 0
    phase: str = ''


def _positive_integer(text: str) -> int:
    number = int(text) if text else 1
    if not 1 <= number <= 1_000_000_000:
        raise ValueError(
            'Coefficients and subscripts must be positive integers '
            'no larger than 1,000,000,000.'
        )
    return number


def _formula_counts(formula: str) -> dict[str, int]:
    tokens = re.findall(r'[A-Z][a-z]?|\d+|[()[\]]', formula)
    if ''.join(tokens) != formula:
        raise ValueError(f'Unsupported formula syntax: {formula!r}.')
    stack = [(None, Counter())]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ('(', '['):
            if len(stack) >= 16:
                raise ValueError('Formula groups are nested too deeply.')
            stack.append((token, Counter()))
            i += 1
            continue
        if token in (')', ']'):
            if len(stack) == 1 or stack[-1][0] != {')': '(', ']': '['}[token]:
                raise ValueError('Unmatched parentheses/brackets in formula.')
            _, counts = stack.pop()
            if not counts:
                raise ValueError('Empty formula group.')
        elif token in ATOMIC_NUMBERS:
            counts = Counter({token: 1})
        else:
            raise ValueError(
                f'Unknown element or misplaced subscript: {token}.'
            )
        i += 1
        multiplier = 1
        if i < len(tokens) and tokens[i].isdigit():
            multiplier = _positive_integer(tokens[i])
            i += 1
        for element, count in counts.items():
            stack[-1][1][element] += count * multiplier
    if len(stack) != 1 or not stack[0][1]:
        raise ValueError('Incomplete or empty formula.')
    return dict(stack[0][1])


def parse_species(text: str) -> Species:
    text = text.strip().translate(SUBSCRIPTS)
    text = re.sub(
        r'([⁰¹²³⁴⁵⁶⁷⁸⁹]*[⁺⁻])',
        lambda m: '^' + m[0].translate(SUPERSCRIPTS),
        text,
    )
    phase_match = re.search(r'\((aq|s|l|g)\)$', text)
    phase = phase_match[1] if phase_match else ''
    if phase_match:
        text = text[: phase_match.start()].strip()
    coefficient_match = re.match(r'(\d+)\s*', text)
    coefficient = 1
    if coefficient_match:
        coefficient = _positive_integer(coefficient_match[1])
        text = text[coefficient_match.end():]
    text = re.sub(r'\s+', '', text)
    charge_match = re.search(r'\^(\d*)([+-])$', text)
    charge = 0
    if charge_match:
        charge = _positive_integer(charge_match[1])
        charge *= 1 if charge_match[2] == '+' else -1
        text = text[: charge_match.start()]
    if text == 'e' and charge == -1:
        return Species('e^-', coefficient, {}, charge, phase)
    total = Counter()
    for i, part in enumerate(re.split(r'[.·]', text)):
        multiple = 1
        if i:
            match = re.match(r'(\d+)', part)
            if match:
                multiple = _positive_integer(match[1])
                part = part[match.end():]
        for element, count in _formula_counts(part).items():
            total[element] += multiple * count
    return Species(text, coefficient, dict(total), charge, phase)


def parse_equation(text: str) -> tuple[list[Species], list[Species]]:
    if not text.strip() or len(text) > 8192:
        raise ValueError(
            'Enter a formula or equation (up to 8192 characters).'
        )
    sides = re.split(r'<=>|<->|->|=>|→|⇌|↔|=', text.strip())
    if len(sides) > 2 or any(not side.strip() for side in sides):
        raise ValueError('Use one reaction arrow with nonempty sides.')
    # Ionic positive signs are consumed with their caret charge, not as
    # separators. Require caret notation: Fe^3+, NH4^+, Cl^-.

    def side_terms(side: str) -> list[Species]:
        protected = re.sub(
            r'\^\d*\+|[⁰¹²³⁴⁵⁶⁷⁸⁹]*⁺',
            lambda m: m[0].replace('+', '\x00'),
            side,
        )
        return [
            parse_species(term.replace('\x00', '+'))
            for term in protected.split('+')
        ]

    left = side_terms(sides[0])
    right = side_terms(sides[1]) if len(sides) == 2 else []
    return left, right


def totals(species: list[Species]) -> tuple[dict[str, int], int]:
    counts = Counter()
    for entry in species:
        for element, count in entry.counts.items():
            counts[element] += entry.coefficient * count
    return dict(counts), sum(s.coefficient * s.charge for s in species)


def electron_configuration(
    symbol: str, charge: int = 0
) -> dict[tuple[int, int], int]:
    if symbol not in ATOMIC_NUMBERS:
        raise ValueError(f'Unknown element {symbol!r}.')
    z = ATOMIC_NUMBERS[symbol]
    if not -2 <= charge <= z or z - charge > 118:
        raise ValueError(
            'Atomic ion views support charges from -2 up to Z; '
            'more negative ions need an explicit electronic model.'
        )
    remaining = z
    counts = {}
    for shell in SUBSHELLS:
        _, ell = shell
        counts[shell] = min(remaining, 2 * (2 * ell + 1))
        remaining -= counts[shell]
    if symbol in PROMOTIONS:
        source, target, amount = PROMOTIONS[symbol]
        counts[source] -= amount
        counts[target] += amount
    if charge > 0:
        remaining = charge
        for shell in sorted(counts, reverse=True):
            removed = min(remaining, counts[shell])
            counts[shell] -= removed
            remaining -= removed
    elif charge < 0:
        remaining = -charge
        for shell in SUBSHELLS:
            added = min(remaining, 2 * (2 * shell[1] + 1) - counts[shell])
            counts[shell] += added
            remaining -= added
    return {shell: count for shell, count in counts.items() if count}


def orbital_occupancies(count: int, ell: int) -> list[int]:
    size = 2 * ell + 1
    if not 0 <= count <= 2 * size:
        raise ValueError('Invalid subshell occupancy.')
    return [int(count > k) + int(count > size + k) for k in range(size)]


def atomic_field(
    n: int, ell: int, orbital: str, zeff: float = 1.0
) -> tuple[np.ndarray, dict]:
    if not (1 <= n <= 7 and 0 <= ell < min(n, 4)):
        raise ValueError('Select a valid n/l shell (n=1..7, s/p/d/f).')
    if orbital not in model.PURE[LETTERS[ell]] or not zeff > 0:
        raise ValueError('Invalid atomic orbital or effective nuclear charge.')
    # Odd-sized symmetric grid contains the nucleus and nodal planes exactly.
    extent = 4.0 * n * n * model.BOHR / zeff
    axes = np.linspace(-extent, extent, 81)
    h = float(axes[1] - axes[0])
    points = np.stack(np.meshgrid(axes, axes, axes, indexing='ij'), -1)
    r = np.linalg.norm(points, axis=-1)
    unit = np.divide(
        points, r[..., None], out=np.zeros_like(points), where=r[..., None] > 0
    )
    rho = 2 * zeff * r / (n * model.BOHR)
    norm = math.sqrt(
        (2 * zeff / (n * model.BOHR)) ** 3
        * math.factorial(n - ell - 1)
        / (2 * n * math.factorial(n + ell))
    )
    radial = (
        norm
        * np.exp(-rho / 2)
        * rho**ell
        * eval_genlaguerre(n - ell - 1, 2 * ell + 1, rho)
    )
    angular = model.YLM[orbital](*(unit[..., k] for k in range(3)))
    field = radial * angular
    grid = dict(
        shape=field.shape,
        h=h,
        lo=np.repeat(-extent, 3),
        pts=points.reshape(-1, 3),
    )
    return field, grid


def atomic_figure(
    symbol: str,
    charge: int,
    n: int,
    ell: int,
    k: int,
    level: float,
    opacity: float,
    zeff: float,
) -> go.Figure:
    orbital = model.PURE[LETTERS[ell]][k]
    field, grid = atomic_field(n, ell, orbital, zeff)
    fig = go.Figure(
        model.iso_meshes(
            field,
            grid,
            level,
            model.PHASE['ao'],
            f"{symbol} {n}{orbital}",
            opacity,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(size=5, color='black'),
            name=f'{symbol} nucleus (Z={ATOMIC_NUMBERS[symbol]})',
        )
    )
    fig.update_layout(
        title=f'{symbol} charge {
            charge:+d}: {n}{
            model.PRETTY.get(
                orbital,
                orbital)}',
        height=600,
        scene=dict(
            aspectmode='data',
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
        ),
        uirevision='equation-atomic-camera',
    )
    return fig


def _preset(prefix: str) -> list[dict]:
    return next(
        specs
        for name, specs in model.PRESETS.items()
        if name.startswith(prefix)
    )


MOLECULES = {
    'H2': _preset('H₂'),
    'N2': _preset('N₂'),
    'C2H4': _preset('Ethylene'),
    'C2H2': _preset('Acetylene'),
    'H2O': _preset('Water dimer')[:3],
    'HF': _preset('HF dimer')[:2],
    'C6H6': _preset('Benzene'),
    'PF5': _preset('PF₅'),
    'SF6': _preset('SF₆'),
}


def explicit_structure(text: str, species: Species) -> list[dict]:
    """Rows are element, orbital set, x/y/z, and optional aim-x/y/z/twist."""
    if species.charge:
        raise ValueError(
            'Molecular ions need explicit electron populations; '
            'this neutral bonding model does not support them.'
        )
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError('Enter a structure row for every atom.')
    if len(rows) > 24:
        raise ValueError(
            'Custom molecular views support up to 24 atoms per '
            'structure to keep the orbital grid interactive. '
            'Equation atom counts have no such visualization limit.'
        )
    counts = Counter()
    specs = []
    for row in rows:
        if len(row) not in (5, 8, 9):
            raise ValueError(
                'Each row: element orbital-set x y z '
                '[aim-x aim-y aim-z [twist-degrees]].'
            )
        symbol, orbital = row[:2]
        if symbol not in model.ELEMENTS:
            raise ValueError(
                f'Molecular model lacks {symbol} parameters. '
                'Atomic visualization is still available.'
            )
        if orbital not in model.ORBITAL_SETS:
            raise ValueError(f'Choose an orbital set from {
                model.ORBITAL_SETS}.')
        try:
            values = np.array([float(v) for v in row[2:]])
        except ValueError as exc:
            raise ValueError(
                'Coordinates, aim and twist must be numbers.'
            ) from exc
        if not np.isfinite(values).all():
            raise ValueError('Structure values must be finite.')
        aim = values[3:6] if len(values) >= 6 else np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(aim) < 1e-9:
            raise ValueError('An orbital aim vector cannot be zero.')
        twist = float(values[6]) if len(values) == 7 else 0.0
        specs.append(
            model._A(
                symbol, orbital, pos=values[:3], aim=tuple(aim), twist=twist
            )
        )
        counts[symbol] += 1
    if dict(counts) != species.counts:
        raise ValueError(
            f'Structure composition {dict(counts)} does not match '
            f'{species.formula}: {species.counts}.'
        )
    positions = np.array([s['pos'] for s in specs])
    distances = np.linalg.norm(
        positions[:, None] - positions[None, :], axis=-1
    )
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 0.1:
        raise ValueError('Atoms cannot coincide or be less than 0.1 Å apart.')
    if np.ptp(positions, axis=0).max() > 30:
        raise ValueError(
            'The qualitative grid supports structures spanning '
            'at most 30 Å; split larger systems.'
        )
    return specs


def molecular_specs(species: Species, structure: str = '') -> list[dict]:
    if structure.strip():
        return explicit_structure(structure, species)
    if species.charge:
        raise ValueError(
            'Molecular ion bonding is not supported; use the '
            'atomic ion view for monatomic ions.'
        )
    if species.formula in MOLECULES:
        return MOLECULES[species.formula]
    if sum(species.counts.values()) == 1:
        element = next(iter(species.counts))
        if element in model.ELEMENTS:
            return [
                model._A(
                    element, 's' if element == 'H' else 'p', aim=(1, 0, 0)
                )
            ]
    raise ValueError(
        f'No unique molecular structure for {species.formula}. '
        'Supply one row per atom in Explicit structure and click Apply '
        'structure. Formula alone cannot distinguish isomers. '
        'Available neutral templates: ' + ', '.join(MOLECULES) + '.'
    )


def composition_html(left: list[Species], right: list[Species]) -> str:
    a, qa = totals(left)
    b, qb = totals(right)
    rows = ''.join(
        f'<tr><td>{symbol}</td><td>{a.get(symbol, 0)}</td>'
        f'<td>{b.get(symbol, 0) if right else "—"}</td></tr>'
        for symbol in sorted(a.keys() | b.keys(), key=ATOMIC_NUMBERS.get)
    )
    if right:
        status = (
            'Balanced in atoms and charge.'
            if a == b and qa == qb
            else 'UNBALANCED: atom counts and/or total charge differ. '
            'Visualizing species does not balance the equation.'
        )
    else:
        status = 'Single formula/composition; no reaction arrow.'
    return (
        f'<b>{status}</b> '
        f'Net charge: left {qa:+d}, right {qb:+d}.'
        '<table><tr><th>Element</th><th>Left atoms</th><th>Right atoms</th>'
        f'</tr>{rows}</table>'
        '<p>Coefficients multiply counts, not orbital populations or '
        'the number of rendered molecules. '
        'This is not a reaction animation.</p>'
    )


class EquationExplorer:
    def __init__(self):
        style = dict(description_width='120px')
        layout = widgets.Layout(width='470px')
        options = dict(style=style, layout=layout)
        self.equation = widgets.Text(
            value='2 H2 + O2 -> 2 H2O', description='Equation', **options
        )
        self.parse_button = widgets.Button(description='Parse equation')
        self.species = widgets.Dropdown(description='Species', **options)
        self.element = widgets.Dropdown(description='Atom / ion', **options)
        self.shell = widgets.Dropdown(description='Subshell', **options)
        self.orbital = widgets.Dropdown(
            description='Atomic orbital', **options
        )
        self.mode = widgets.Dropdown(
            description='View',
            options=[
                'Atomic orbital',
                'Molecule: bond contributors',
                'Molecule: bonding MO',
                'Molecule: antibonding MO',
                'Molecule: overlap',
                'Molecule: delocalized π density',
                'Molecule: delocalized π MO',
                'Molecule: DFT total density',
                'Molecule: DFT molecular orbital',
            ],
            **options,
        )
        self.bond = widgets.Dropdown(description='Molecular bond', **options)
        self.pi = widgets.Dropdown(description='Shared π MO', **options)
        self.level = widgets.FloatSlider(
            value=0.22,
            min=0.05,
            max=0.65,
            step=0.01,
            description='Lobe fraction',
            continuous_update=False,
            **options,
        )
        self.opacity = widgets.FloatSlider(
            value=0.8,
            min=0.15,
            max=1.0,
            step=0.05,
            description='Opacity',
            continuous_update=False,
            **options,
        )
        self.zeff = widgets.FloatSlider(
            value=1.0,
            min=0.5,
            max=30.0,
            step=0.5,
            description='Atomic Z effective',
            continuous_update=False,
            **options,
        )
        self.structure = widgets.Textarea(
            description='Explicit structure',
            rows=7,
            placeholder='C sp2 0 0 0 1 0 0\nH s 1.09 0 0 -1 0 0',
            **options,
        )
        self.quantum_geometry = widgets.Textarea(
            description='DFT geometry (Å)',
            rows=5,
            placeholder='O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0',
            **options,
        )
        self.multiplicity = widgets.BoundedIntText(
            value=1,
            min=1,
            max=15,
            description='Multiplicity',
            **options,
        )
        self.dft_basis = widgets.Dropdown(
            options=['def2-SVP'],
            value='def2-SVP',
            description='DFT basis',
            **options,
        )
        self.dft_functional = widgets.Dropdown(
            options=['B3LYP', 'PBE0', 'PBE'],
            value='B3LYP',
            description='DFT functional',
            **options,
        )
        self.optimization_steps = widgets.BoundedIntText(
            value=50,
            min=1,
            max=200,
            description='Opt. steps',
            **options,
        )
        self.density_spacing = widgets.FloatSlider(
            value=0.4,
            min=0.25,
            max=0.8,
            step=0.05,
            description='Grid (Å)',
            continuous_update=False,
            **options,
        )
        self.quantum_button = widgets.Button(
            description='Run PySCF DFT + optimize',
            button_style='primary',
        )
        self.quantum_mo = widgets.Dropdown(description='DFT orbital', **options)
        self.quantum_spin_channel = widgets.Dropdown(
            description='MO spin', options=[('Alpha', 0)], **options
        )
        self.density_threshold = widgets.FloatSlider(
            value=0.02,
            min=0.005,
            max=0.15,
            step=0.005,
            description='Density iso',
            continuous_update=False,
            **options,
        )
        self.mo_threshold = widgets.FloatSlider(
            value=0.12,
            min=0.03,
            max=0.4,
            step=0.01,
            description='MO iso',
            continuous_update=False,
            **options,
        )
        self.apply_button = widgets.Button(description='Apply structure')
        self.template_button = widgets.Button(
            description='Use formula template'
        )
        self.status = widgets.HTML()
        self.detail = widgets.HTML()
        self.output = widgets.Output()
        self.entries = []
        self.structures = {}
        self.quantum_geometries = {}
        self._quantum_result = None
        self.sync = False
        self.parse_button.on_click(self.parse)
        self.apply_button.on_click(self.apply_structure)
        self.template_button.on_click(self.use_template)
        self.quantum_button.on_click(self.run_quantum)
        self.species.observe(self.select_species, 'value')
        self.quantum_geometry.observe(self.save_quantum_geometry, 'value')
        self.element.observe(self.select_element, 'value')
        self.shell.observe(self.select_shell, 'value')
        for widget in (
            self.orbital,
            self.mode,
            self.bond,
            self.pi,
            self.level,
            self.opacity,
            self.zeff,
            self.quantum_mo,
            self.density_threshold,
            self.mo_threshold,
        ):
            widget.observe(self.render, 'value')
        self.quantum_spin_channel.observe(
            self.select_quantum_spin_channel, 'value'
        )
        self._molecular_result = lru_cache(maxsize=4)(self._analyze_molecule)
        self.parse()

    def parse(self, *_):
        self.sync = True
        try:
            left, right = parse_equation(self.equation.value)
            self.entries = left + right
            labels = [
                (
                    f'{"Reactant" if i < len(left) else "Product"}: '
                    f'{s.coefficient} {s.formula}'
                    f'{f" (charge {s.charge:+d})" if s.charge else ""}'
                    f'{f" ({s.phase})" if s.phase else ""}',
                    i,
                )
                for i, s in enumerate(self.entries)
            ]
            self.species.options = labels
            self.species.value = 0
            self._quantum_result = None
            self.status.value = composition_html(left, right)
        except ValueError as exc:
            self.entries = []
            self.species.options = []
            self.element.options = []
            self.shell.options = []
            self.orbital.options = []
            self.detail.value = ''
            self.status.value = f'<b>Input error:</b> {html.escape(str(exc))}'
            with self.output:
                self.output.clear_output()
        finally:
            self.sync = False
        if self.entries:
            self.select_species()

    def selected(self) -> Species:
        if self.species.value is None or not self.entries:
            raise ValueError('Parse a valid equation first.')
        return self.entries[self.species.value]

    @staticmethod
    def species_key(entry: Species) -> tuple[str, int]:
        return entry.formula, entry.charge

    def select_species(self, *_):
        if self.sync:
            return
        entry = self.selected()
        self.sync = True
        try:
            self.element.options = sorted(entry.counts, key=ATOMIC_NUMBERS.get)
            self.element.value = (
                self.element.options[0] if self.element.options else None
            )
            self.structure.value = self.structures.get(
                self.species_key(entry), ''
            )
            self.quantum_geometry.value = self.quantum_geometries.get(
                self.species_key(entry), self._template_quantum_geometry(entry)
            )
            self._quantum_result = None
        finally:
            self.sync = False
        self.select_element()

    @staticmethod
    def _template_quantum_geometry(entry: Species) -> str:
        specs = MOLECULES.get(entry.formula) if not entry.charge else None
        if specs is None:
            return ''
        atoms = model.place_atoms(specs)
        return '\n'.join(
            f'{atom["el"]} '
            + ' '.join(f'{coordinate:.6f}' for coordinate in atom['pos'])
            for atom in atoms
        )

    def save_quantum_geometry(self, change):
        if self.sync or change.get('name') != 'value' or not self.entries:
            return
        self.quantum_geometries[self.species_key(self.selected())] = (
            change['new']
        )
        self._quantum_result = None

    def select_quantum_spin_channel(self, change):
        if self.sync or change.get('name') != 'value':
            return
        self._update_quantum_orbitals()
        self.render()

    def atomic_charge(self) -> int:
        entry = self.selected()
        return entry.charge if sum(entry.counts.values()) == 1 else 0

    def select_element(self, *_):
        if self.sync:
            return
        self.sync = True
        try:
            config = (
                electron_configuration(
                    self.element.value, self.atomic_charge()
                )
                if self.element.value
                else {}
            )
            self.shell.options = [
                (f'{n}{LETTERS[ell]} ({count} e⁻)', (n, ell))
                for (n, ell), count in sorted(config.items())
            ]
            if self.shell.options:
                self.shell.value = self.shell.options[-1][1]
        except ValueError as exc:
            self.shell.options = []
            self.detail.value = f'<b>Atomic model:</b> {html.escape(str(exc))}'
        finally:
            self.sync = False
        self.select_shell()

    def select_shell(self, *_):
        if self.sync:
            return
        self.sync = True
        try:
            if self.shell.value:
                n, ell = self.shell.value
                count = electron_configuration(
                    self.element.value, self.atomic_charge()
                )[(n, ell)]
                occupations = orbital_occupancies(count, ell)
                self.orbital.options = [
                    (
                        f'{n}{model.PRETTY.get(name, name)} '
                        f'({occupations[k]} e⁻)',
                        k,
                    )
                    for k, name in enumerate(model.PURE[LETTERS[ell]])
                ]
                self.orbital.value = 0
            else:
                self.orbital.options = []
        finally:
            self.sync = False
        self.render()

    def apply_structure(self, *_):
        try:
            entry = self.selected()
            explicit_structure(self.structure.value, entry)
            self.structures[self.species_key(entry)] = self.structure.value
            if self.mode.value == 'Atomic orbital':
                self.mode.value = 'Molecule: bond contributors'
            else:
                self.render()
        except ValueError as exc:
            self.detail.value = f'<b>Structure error:</b> {
                html.escape(
                    str(exc))}'
            with self.output:
                self.output.clear_output()

    def use_template(self, *_):
        try:
            entry = self.selected()
            self.structures.pop(self.species_key(entry), None)
            self.structure.value = ''
            self._bond_key = None
            self.render()
        except ValueError as exc:
            self.detail.value = f'<b>Template error:</b> {
                html.escape(
                    str(exc))}'

    def run_quantum(self, *_):
        try:
            entry = self.selected()
            atoms = quantum_chemistry.parse_geometry(
                self.quantum_geometry.value, entry.counts
            )
            self.detail.value = (
                '<b>PySCF DFT:</b> optimizing the supplied geometry and '
                'calculating the self-consistent electron density. '
                'This may take a while.'
            )
            with self.output:
                self.output.clear_output(wait=True)
            result = quantum_chemistry.calculate_dft(
                atoms,
                charge=entry.charge,
                multiplicity=self.multiplicity.value,
                basis=self.dft_basis.value,
                functional=self.dft_functional.value,
                max_steps=self.optimization_steps.value,
                grid_spacing=self.density_spacing.value,
            )
            self.quantum_geometries[self.species_key(entry)] = (
                self.quantum_geometry.value
            )
            self._quantum_result = result
            mf = result['mean_field']
            open_shell = quantum_chemistry.is_unrestricted(mf)
            self.sync = True
            try:
                self.quantum_spin_channel.options = (
                    [('Alpha', 0), ('Beta', 1)]
                    if open_shell
                    else [('Restricted', 0)]
                )
                self.quantum_spin_channel.value = 0
                self._update_quantum_orbitals()
                self.mode.value = 'Molecule: DFT total density'
            finally:
                self.sync = False
            self.render()
        except (
            ImportError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._quantum_result = None
            self.detail.value = (
                '<b>PySCF calculation failed:</b> '
                + html.escape(str(exc))
            )
            with self.output:
                self.output.clear_output()

    def _update_quantum_orbitals(self):
        result = self._quantum_result
        if result is None:
            self.quantum_mo.options = []
            return
        mf = result['mean_field']
        channel = self.quantum_spin_channel.value or 0
        unrestricted = quantum_chemistry.is_unrestricted(mf)
        coeffs = mf.mo_coeff[channel] if unrestricted else mf.mo_coeff
        energies = mf.mo_energy[channel] if unrestricted else mf.mo_energy
        occupations = mf.mo_occ[channel] if unrestricted else mf.mo_occ
        label = next(
            (
                option[0]
                for option in self.quantum_spin_channel.options
                if option[1] == channel
            ),
            'restricted',
        ).lower()
        options = [
            (
                f'MO {index + 1} ({occupations[index]:g} e⁻, '
                f'{energies[index]:.3f} Ha)',
                index,
            )
            for index in range(len(coeffs))
        ]
        self.quantum_mo.options = options
        occupied = np.flatnonzero(np.asarray(occupations) > 0)
        self.quantum_mo.value = (
            int(occupied[-1]) if len(occupied) else 0
        )
        self.quantum_mo.description = f'{label.capitalize()} MO'

    def _analyze_molecule(self, formula: str, charge: int, structure: str):
        entry = Species(formula, 1, parse_species(formula).counts, charge)
        specs = molecular_specs(entry, structure)
        atoms = model.place_atoms(specs)
        return atoms, model.analyze(atoms)

    def render(self, *_):
        if self.sync or not self.entries:
            return
        self.sync = True
        with self.output:
            self.output.clear_output(wait=True)
            try:
                entry = self.selected()
                atomic = self.mode.value == 'Atomic orbital'
                for control in (
                    self.element,
                    self.shell,
                    self.orbital,
                    self.zeff,
                ):
                    control.disabled = not atomic
                self.bond.disabled = atomic
                self.pi.disabled = atomic
                if atomic:
                    self.render_atom(entry)
                elif self.mode.value.startswith('Molecule: DFT'):
                    self.render_quantum(entry)
                else:
                    self.render_molecule(entry)
            except ValueError as exc:
                self.detail.value = f'<b>Cannot render:</b> {
                    html.escape(
                        str(exc))}'
                self.output.clear_output()
                self.bond.options = []
                self.pi.options = []
                self.bond.disabled = True
                self.pi.disabled = True
                self._bond_key = None
                self._quantum_result = None
            finally:
                self.sync = False

    def render_quantum(self, entry: Species):
        result = self._quantum_result
        if result is None:
            self.detail.value = (
                '<b>PySCF DFT:</b> enter a geometry (or select a supplied '
                'template), set charge through the formula and spin '
                'multiplicity, then click <i>Run PySCF DFT + optimize</i>. '
                'This calculation uses the supplied geometry as its '
                'starting point; it does not infer an isomer from a formula.'
            )
            self.quantum_mo.disabled = True
            self.quantum_spin_channel.disabled = True
            return
        if (
            result['charge'] != entry.charge
            or Counter(result['symbols']) != entry.counts
        ):
            raise ValueError(
                'The displayed DFT result belongs to a different species; '
                'run the solver again for this formula.'
            )

        coordinates = result['coordinates']
        symbols = result['symbols']
        atom_names = [f'{symbol}{i + 1}' for i, symbol in enumerate(symbols)]
        distance_rows = ''.join(
            f'<tr><td>{atom_names[i]}–{atom_names[j]}</td>'
            f'<td>{distance:.3f} Å</td></tr>'
            for i, j, distance in result['contacts']
        ) or '<tr><td colspan="2">No close pairs by the contact heuristic.</td></tr>'
        angle_rows = ''.join(
            f'<tr><td>{atom_names[i]}–{atom_names[j]}–{atom_names[k]}</td>'
            f'<td>{angle:.1f}°</td></tr>'
            for i, j, k, angle in result['angles']
        ) or '<tr><td colspan="2">No angles from inferred close pairs.</td></tr>'
        energy_ev = result['energy_hartree'] * 27.211386245988
        self.detail.value = (
            f'<b>PySCF DFT ({html.escape(result["functional"])}/'
            f'{html.escape(result["basis"])}):</b> optimized total DFT '
            f'energy {result["energy_hartree"]:.8f} Ha '
            f'({energy_ev:.4f} eV); charge {result["charge"]:+d}; '
            f'multiplicity {result["multiplicity"]}. '
            'The energy includes nuclear repulsion; geometry is optimized '
            'on this energy surface, not with hand-set bond-angle penalties. '
            'It is not a bond-dissociation energy. '
            'Distances/angles below use a covalent-radius contact '
            'heuristic, not a chemical bond-order analysis.'
            '<details><summary>Optimized geometry and geometric contacts</summary>'
            '<table><tr><th>Atom</th><th>x (Å)</th><th>y (Å)</th>'
            '<th>z (Å)</th></tr>'
            + ''.join(
                f'<tr><td>{html.escape(symbols[i])}{i + 1}</td>'
                f'<td>{coordinates[i, 0]:.5f}</td>'
                f'<td>{coordinates[i, 1]:.5f}</td>'
                f'<td>{coordinates[i, 2]:.5f}</td></tr>'
                for i in range(len(symbols))
            )
            + '</table><b>Close-pair distances (heuristic)</b>'
            f'<table>{distance_rows}</table>'
            f'<b>Angles from inferred contacts</b><table>{angle_rows}</table>'
            '</details>'
            '<br>Density is a real-space DFT electron density; MO colors '
            'indicate wavefunction phase. This is a single optimized '
            'electronic state, not experimental electron-density data.'
        )
        is_orbital = self.mode.value == 'Molecule: DFT molecular orbital'
        self.quantum_mo.disabled = not is_orbital
        self.quantum_spin_channel.disabled = (
            not is_orbital or len(self.quantum_spin_channel.options) < 2
        )
        if is_orbital:
            fig, occupancy, channel = quantum_chemistry.orbital_figure(
                result,
                self.quantum_mo.value or 0,
                self.quantum_spin_channel.value or 0,
                self.mo_threshold.value,
            )
            self.detail.value += (
                f'<br>Displayed {channel} MO {self.quantum_mo.value + 1}; '
                f'occupation {occupancy:g} electrons.'
            )
        else:
            fig = quantum_chemistry.density_figure(
                result, self.density_threshold.value
            )
        self.display_figure(fig)

    @staticmethod
    def display_figure(fig: go.Figure):
        try:
            fig.show()
        except (ImportError, ValueError) as exc:
            display(
                HTML(
                    '<p>Renderer unavailable: '
                    + html.escape(str(exc))
                    + '; using embedded HTML.</p>'
                )
            )
            display(HTML(fig.to_html(include_plotlyjs=True, full_html=False)))

    def render_atom(self, entry: Species):
        symbol = self.element.value
        if not symbol:
            raise ValueError('This species is an electron, not an atom.')
        charge = self.atomic_charge()
        config = electron_configuration(symbol, charge)
        if not config:
            raise ValueError(
                'This ion has no electrons: no occupied orbitals.'
            )
        if self.shell.value is None:
            raise ValueError('No supported occupied subshell.')
        n, ell = self.shell.value
        configuration = ' '.join(
            f'{nn}{LETTERS[ll]}{count}'
            for (nn, ll), count in sorted(config.items())
        )
        note = (
            'Monatomic ion: electrons adjusted for the species charge.'
            if charge
            else 'Neutral isolated-atom reference.'
        )
        if entry.charge and sum(entry.counts.values()) > 1:
            note += (
                ' Polyatomic charge cannot be assigned uniquely to atoms; '
                'these are neutral references, not oxidation states.'
            )
        self.detail.value = (
            f'<b>{symbol}: Z={ATOMIC_NUMBERS[symbol]}, '
            f'{sum(config.values())} electrons.</b> '
            f'Configuration: {configuration}.<br>{note}<br>'
            'Aufbau estimate with common ground-state exceptions; '
            'heavy-element and ion ordering may require spectroscopic data. '
            'Orbital occupations use an illustrative Hund-rule assignment '
            'within a real s/p/d/f basis.<br>'
            f'Hydrogenic radial model with user-selected Z effective={
                self.zeff.value:g}; '
            'this controls size, not electron count. '
            'It is not a self-consistent '
            'many-electron wavefunction. Phase colours are not charge. '
            f'Radial nodes: {n - ell - 1}; angular nodes: {ell}. '
            'A high surface threshold can hide small inner lobes.'
        )
        fig = atomic_figure(
            symbol,
            charge,
            n,
            ell,
            self.orbital.value,
            self.level.value,
            self.opacity.value,
            self.zeff.value,
        )
        self.display_figure(fig)

    def render_molecule(self, entry: Species):
        atoms, res = self._molecular_result(
            entry.formula,
            entry.charge,
            self.structures.get(self.species_key(entry), ''),
        )
        old = self.bond.value
        options = [('All bonds (busy)', None)] + [
            (
                f'{model.BOND_NAMES[b["m"]]}: '
                f'{model._olabel(atoms, b["a"], b["ka"])} + '
                f'{model._olabel(atoms, b["b"], b["kb"])}',
                k,
            )
            for k, b in enumerate(res['bonds'])
        ]
        key = (
            self.species_key(entry),
            tuple((b['a'], b['ka'], b['b'], b['kb']) for b in res['bonds']),
        )
        changed = key != getattr(self, '_bond_key', None)
        self._bond_key = key
        self.bond.options = options
        self.bond.value = (
            old
            if not changed and old in [value for _, value in options]
            else next(
                (k for k, b in enumerate(res['bonds']) if b['m'] == 1),
                0 if res['bonds'] else None,
            )
        )
        modes = {
            'Molecule: bond contributors': 'Bond contributors',
            'Molecule: bonding MO': 'Bonding',
            'Molecule: antibonding MO': 'Antibonding',
            'Molecule: overlap': 'Overlap',
            'Molecule: delocalized π density': 'Delocalized π density',
            'Molecule: delocalized π MO': 'Delocalized π',
        }
        mode = modes[self.mode.value]
        nmo = min((len(p['energies']) for p in res['pi_systems']), default=0)
        old_pi = self.pi.value
        self.pi.options = [
            (f'MO {k + 1} ({res["pi_systems"][0]["occupancy"][k]} e⁻)', k)
            for k in range(nmo)
        ] or [('No shared π network', None)]
        self.pi.value = (
            old_pi
            if old_pi is not None and old_pi < nmo
            else (0 if nmo else None)
        )
        self.pi.disabled = self.mode.value != 'Molecule: delocalized π MO'
        self.bond.disabled = mode.startswith('Delocalized')
        if mode.startswith('Delocalized') and not nmo:
            raise ValueError(
                'This structure has no shared π network. '
                'Try benzene C6H6; a two-centre π bond is localized.'
            )
        self.detail.value = (
            f'<b>{
                html.escape(
                    entry.formula)}:</b> one representative molecule '
            '(stoichiometric coefficient affects counts only). '
            'Template/explicit coordinates, not a structure inferred '
            'from the equation.<br>'
            + model.summary_html(
                atoms,
                res,
                mode,
                bond=self.bond.value,
                pi_mo=self.pi.value or 0,
            )
        )
        fig = model.build_figure(
            atoms,
            res,
            mode,
            self.level.value,
            self.opacity.value,
            bond=self.bond.value,
            pi_mo=self.pi.value or 0,
        )
        self.display_figure(fig)

    def show(self):
        display(
            widgets.VBox(
                [
                    widgets.HTML(
                        '<b>Equation → composition → '
                        'atomic/molecular orbitals.</b> '
                        'Examples: 2 H2 + O2 -> 2 H2O; '
                        'Fe^2+ + 2 Cl^- -> FeCl2; '
                        'C2H4 + H2 -> C2H6. Use ^2+ charge notation. '
                        'Known molecular templates: '
                        + ', '.join(MOLECULES)
                        + '.'
                    ),
                    widgets.HBox([self.equation, self.parse_button]),
                    self.status,
                    widgets.HBox(
                        [
                            widgets.VBox(
                                [
                                    self.species,
                                    self.mode,
                                    self.element,
                                    self.shell,
                                    self.orbital,
                                    self.zeff,
                                ]
                            ),
                            widgets.VBox(
                                [
                                    self.bond,
                                    self.pi,
                                    self.level,
                                    self.opacity,
                                    self.structure,
                                    widgets.HBox(
                                        [
                                            self.apply_button,
                                            self.template_button,
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                    widgets.HTML(
                        'Explicit structure rows: '
                        '<code>element orbital-set x y z '
                        '[aim-x aim-y aim-z [twist-degrees]]</code>, '
                        'coordinates in Å. '
                        'Aim defaults to +x; supply directions '
                        'to orient bonding lobes. '
                        'Supported molecular elements: '
                        + ', '.join(model.ELEMENTS)
                        + '. This structure format is for the fast '
                        'qualitative viewer below.'
                    ),
                    widgets.HTML(
                        '<h4>Self-consistent molecular calculation '
                        '(optional PySCF DFT)</h4>'
                        '<p>This separate solver optimizes a supplied '
                        'Cartesian geometry, then computes total electron '
                        'density, molecular orbitals and electronic energy. '
                        'It does not infer structures from formulas. '
                        'Initial basis coverage is H–Kr (Z ≤ 36); '
                        'calculations are more expensive than the '
                        'qualitative viewer.</p>'
                    ),
                    widgets.HBox(
                        [
                            widgets.VBox(
                                [
                                    self.quantum_geometry,
                                    widgets.HBox(
                                        [
                                            self.multiplicity,
                                            self.dft_basis,
                                            self.dft_functional,
                                        ]
                                    ),
                                    widgets.HBox(
                                        [
                                            self.optimization_steps,
                                            self.density_spacing,
                                            self.quantum_button,
                                        ]
                                    ),
                                ]
                            ),
                            widgets.VBox(
                                [
                                    self.quantum_mo,
                                    self.quantum_spin_channel,
                                    self.density_threshold,
                                    self.mo_threshold,
                                ]
                            ),
                        ]
                    ),
                    self.detail,
                    self.output,
                ]
            )
        )


def show_equation_explorer() -> EquationExplorer:
    explorer = EquationExplorer()
    explorer.show()
    return explorer
