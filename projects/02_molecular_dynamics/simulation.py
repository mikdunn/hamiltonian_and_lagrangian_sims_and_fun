"""Interactive molecular dynamics demo with phase presets and molecule geometry views."""

from __future__ import annotations

import argparse
import io
import json
import math
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook-only dependency path
    widgets = None
    display = None


DOMAIN_KEY = "02_molecular_dynamics"
DOMAIN_TITLE = "Molecular dynamics simulations"
DEFAULT_MIXTURE = "H2O:6,N2:4,CH4:3"

ATOMIC_COLORS = {
    "H": "#f5f7fa",
    "C": "#4f5d75",
    "N": "#4f8fbf",
    "O": "#d95d39",
    "He": "#f3d250",
    "Ne": "#f78888",
}

ATOMIC_RADII = {
    "H": 35.0,
    "C": 70.0,
    "N": 72.0,
    "O": 74.0,
    "He": 48.0,
    "Ne": 60.0,
}

ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "He": 4.0026,
    "Ne": 20.1797,
}

PHASE_PRESETS = {
    "gas": {
        "damping": 0.03,
        "noise": 0.14,
        "tether": 0.0,
        "barostat": 0.05,
        "mobility": 2.1,
        "attraction": 0.28,
    },
    "liquid": {
        "damping": 0.10,
        "noise": 0.09,
        "tether": 0.0,
        "barostat": 0.08,
        "mobility": 1.35,
        "attraction": 1.0,
    },
    "solid": {
        "damping": 0.20,
        "noise": 0.035,
        "tether": 1.55,
        "barostat": 0.10,
        "mobility": 0.35,
        "attraction": 1.18,
    },
}

AVOGADRO_NUMBER = 6.02214076e23
LITER_TO_NM3 = 1.0e24
REFERENCE_TEMPERATURE_K = 300.0
REFERENCE_PRESSURE_PA = 101325.0
R_GAS_CONSTANT = 8.314462618
SIM_TIME_SCALE_S = 0.5


@dataclass(frozen=True)
class AtomSpec:
    element: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class OrbitalSpec:
    origin_atom: int
    vector: tuple[float, float, float]
    label: str


@dataclass(frozen=True)
class MoleculeTemplate:
    name: str
    geometry: str
    orbital_model: str
    atoms: tuple[AtomSpec, ...]
    bonds: tuple[tuple[int, int], ...]
    orbitals: tuple[OrbitalSpec, ...]
    sigma: float
    epsilon: float

    @property
    def mass(self) -> float:
        return float(sum(ATOMIC_MASSES[atom.element] for atom in self.atoms))


@dataclass(frozen=True)
class ReactionRule:
    equation: str
    reactants: tuple[str, str]
    stoich_reactants: dict[str, float]
    stoich_products: dict[str, float]
    product: str
    byproduct: str
    activation_temperature: float
    base_probability: float
    capture_distance_factor: float
    heat_release: float
    activation_energy: float
    pre_exponential: float
    delta_g0: float
    activation_width_kelvin: float
    allowed_phases: tuple[str, ...]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector.copy()
    return vector / norm


def _safe_sigmoid(x: float) -> float:
    # Numerically stable logistic to avoid overflow for large-magnitude inputs.
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _unit_tuple(vector: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    arr = _normalize(np.asarray(vector, dtype=float))
    return (float(arr[0]), float(arr[1]), float(arr[2]))


TETRAHEDRAL = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=float,
)
TETRAHEDRAL = TETRAHEDRAL / np.linalg.norm(TETRAHEDRAL, axis=1, keepdims=True)

H_BOND = 0.96
NH_BOND = 1.01
CH_BOND = 1.09

WATER_H1 = np.array([math.sin(math.radians(52.25)), 0.0, math.cos(math.radians(52.25))]) * H_BOND
WATER_H2 = np.array([-math.sin(math.radians(52.25)), 0.0, math.cos(math.radians(52.25))]) * H_BOND

AMMONIA_H = tuple((direction * NH_BOND) for direction in TETRAHEDRAL[:3])
METHANE_H = tuple((direction * CH_BOND) for direction in TETRAHEDRAL)


MOLECULE_LIBRARY: dict[str, MoleculeTemplate] = {
    "He": MoleculeTemplate(
        name="He",
        geometry="monatomic",
        orbital_model="1s2 closed shell",
        atoms=(AtomSpec("He", (0.0, 0.0, 0.0)),),
        bonds=(),
        orbitals=(),
        sigma=1.05,
        epsilon=0.10,
    ),
    "Ne": MoleculeTemplate(
        name="Ne",
        geometry="monatomic",
        orbital_model="2s2 2p6 closed shell",
        atoms=(AtomSpec("Ne", (0.0, 0.0, 0.0)),),
        bonds=(),
        orbitals=(),
        sigma=1.18,
        epsilon=0.14,
    ),
    "H2": MoleculeTemplate(
        name="H2",
        geometry="linear 180 deg",
        orbital_model="sigma bond along molecular axis",
        atoms=(AtomSpec("H", (-0.37, 0.0, 0.0)), AtomSpec("H", (0.37, 0.0, 0.0))),
        bonds=((0, 1),),
        orbitals=(OrbitalSpec(0, (-1.0, 0.0, 0.0), "1s"), OrbitalSpec(1, (1.0, 0.0, 0.0), "1s")),
        sigma=1.12,
        epsilon=0.18,
    ),
    "N2": MoleculeTemplate(
        name="N2",
        geometry="linear 180 deg",
        orbital_model="sp axis with pi system",
        atoms=(AtomSpec("N", (-0.55, 0.0, 0.0)), AtomSpec("N", (0.55, 0.0, 0.0))),
        bonds=((0, 1),),
        orbitals=(OrbitalSpec(0, (-1.0, 0.0, 0.0), "sp"), OrbitalSpec(1, (1.0, 0.0, 0.0), "sp")),
        sigma=1.20,
        epsilon=0.22,
    ),
    "O2": MoleculeTemplate(
        name="O2",
        geometry="linear 180 deg",
        orbital_model="sigma plus pi antibonding occupancy",
        atoms=(AtomSpec("O", (-0.60, 0.0, 0.0)), AtomSpec("O", (0.60, 0.0, 0.0))),
        bonds=((0, 1),),
        orbitals=(OrbitalSpec(0, (-1.0, 0.0, 0.0), "p"), OrbitalSpec(1, (1.0, 0.0, 0.0), "p")),
        sigma=1.22,
        epsilon=0.24,
    ),
    "CO2": MoleculeTemplate(
        name="CO2",
        geometry="linear 180 deg",
        orbital_model="central sp carbon",
        atoms=(AtomSpec("O", (-1.16, 0.0, 0.0)), AtomSpec("C", (0.0, 0.0, 0.0)), AtomSpec("O", (1.16, 0.0, 0.0))),
        bonds=((0, 1), (1, 2)),
        orbitals=(OrbitalSpec(1, (-1.0, 0.0, 0.0), "sp"), OrbitalSpec(1, (1.0, 0.0, 0.0), "sp")),
        sigma=1.48,
        epsilon=0.30,
    ),
    "H2O": MoleculeTemplate(
        name="H2O",
        geometry="bent 104.5 deg",
        orbital_model="sp3 oxygen with two lone pairs",
        atoms=(
            AtomSpec("O", (0.0, 0.0, 0.0)),
            AtomSpec("H", (float(WATER_H1[0]), float(WATER_H1[1]), float(WATER_H1[2]))),
            AtomSpec("H", (float(WATER_H2[0]), float(WATER_H2[1]), float(WATER_H2[2]))),
        ),
        bonds=((0, 1), (0, 2)),
        orbitals=(
            OrbitalSpec(0, _unit_tuple(WATER_H1), "bond"),
            OrbitalSpec(0, _unit_tuple(WATER_H2), "bond"),
            OrbitalSpec(0, _unit_tuple(TETRAHEDRAL[2]), "lone pair"),
            OrbitalSpec(0, _unit_tuple(TETRAHEDRAL[3]), "lone pair"),
        ),
        sigma=1.38,
        epsilon=0.34,
    ),
    "NH3": MoleculeTemplate(
        name="NH3",
        geometry="trigonal pyramidal 107 deg",
        orbital_model="sp3 nitrogen with one lone pair",
        atoms=(
            AtomSpec("N", (0.0, 0.0, 0.0)),
            AtomSpec("H", tuple(float(value) for value in AMMONIA_H[0])),
            AtomSpec("H", tuple(float(value) for value in AMMONIA_H[1])),
            AtomSpec("H", tuple(float(value) for value in AMMONIA_H[2])),
        ),
        bonds=((0, 1), (0, 2), (0, 3)),
        orbitals=(
            OrbitalSpec(0, _unit_tuple(AMMONIA_H[0]), "bond"),
            OrbitalSpec(0, _unit_tuple(AMMONIA_H[1]), "bond"),
            OrbitalSpec(0, _unit_tuple(AMMONIA_H[2]), "bond"),
            OrbitalSpec(0, _unit_tuple(TETRAHEDRAL[3]), "lone pair"),
        ),
        sigma=1.42,
        epsilon=0.31,
    ),
    "CH4": MoleculeTemplate(
        name="CH4",
        geometry="tetrahedral 109.5 deg",
        orbital_model="sp3 carbon",
        atoms=(
            AtomSpec("C", (0.0, 0.0, 0.0)),
            AtomSpec("H", tuple(float(value) for value in METHANE_H[0])),
            AtomSpec("H", tuple(float(value) for value in METHANE_H[1])),
            AtomSpec("H", tuple(float(value) for value in METHANE_H[2])),
            AtomSpec("H", tuple(float(value) for value in METHANE_H[3])),
        ),
        bonds=((0, 1), (0, 2), (0, 3), (0, 4)),
        orbitals=tuple(OrbitalSpec(0, _unit_tuple(direction), "bond") for direction in METHANE_H),
        sigma=1.54,
        epsilon=0.28,
    ),
}


REACTION_RULES: tuple[ReactionRule, ...] = (
    ReactionRule(
        equation="2 H2 + O2 -> 2 H2O",
        reactants=("H2", "O2"),
        stoich_reactants={"H2": 2.0, "O2": 1.0},
        stoich_products={"H2O": 2.0},
        product="H2O",
        byproduct="O2",
        activation_temperature=550.0,
        base_probability=0.06,
        capture_distance_factor=1.45,
        heat_release=0.34,
        activation_energy=65000.0,
        pre_exponential=3.0e7,
        delta_g0=-220000.0,
        activation_width_kelvin=180.0,
        allowed_phases=("gas", "liquid"),
    ),
    ReactionRule(
        equation="CH4 + 2 O2 -> CO2 + 2 H2O",
        reactants=("CH4", "O2"),
        stoich_reactants={"CH4": 1.0, "O2": 2.0},
        stoich_products={"CO2": 1.0, "H2O": 2.0},
        product="CO2",
        byproduct="H2O",
        activation_temperature=700.0,
        base_probability=0.08,
        capture_distance_factor=1.55,
        heat_release=0.46,
        activation_energy=85000.0,
        pre_exponential=5.0e7,
        delta_g0=-400000.0,
        activation_width_kelvin=220.0,
        allowed_phases=("gas", "liquid"),
    ),
    ReactionRule(
        equation="N2 + 3 H2 -> 2 NH3",
        reactants=("N2", "H2"),
        stoich_reactants={"N2": 1.0, "H2": 3.0},
        stoich_products={"NH3": 2.0},
        product="NH3",
        byproduct="H2",
        activation_temperature=650.0,
        base_probability=0.045,
        capture_distance_factor=1.40,
        heat_release=0.22,
        activation_energy=95000.0,
        pre_exponential=6.0e7,
        delta_g0=-33000.0,
        activation_width_kelvin=200.0,
        allowed_phases=("gas", "liquid"),
    ),
)


REACTION_INDEX: dict[tuple[str, str], ReactionRule] = {
    tuple(sorted(rule.reactants)): rule for rule in REACTION_RULES
}


def parse_mixture(spec: str) -> dict[str, int]:
    mixture: dict[str, int] = {}
    for chunk in spec.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Mixture entry '{entry}' must use Molecule:count format.")
        name, count_text = entry.split(":", 1)
        molecule = name.strip()
        if molecule not in MOLECULE_LIBRARY:
            supported = ", ".join(sorted(MOLECULE_LIBRARY))
            raise ValueError(f"Unknown molecule '{molecule}'. Supported molecules: {supported}")
        count = int(count_text.strip())
        if count < 0:
            raise ValueError(f"Count for '{molecule}' must be non-negative.")
        if count == 0:
            continue
        mixture[molecule] = mixture.get(molecule, 0) + count
    if not mixture:
        raise ValueError("Mixture must contain at least one molecule.")
    return mixture


def mixture_from_controls(*species_counts: tuple[str, int]) -> dict[str, int]:
    mixture: dict[str, int] = {}
    for species, count in species_counts:
        if species == "None" or count <= 0:
            continue
        mixture[species] = mixture.get(species, 0) + int(count)
    if not mixture:
        mixture["Ne"] = 6
    return mixture


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(matrix)
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    kx, ky, kz = axis
    cross = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=float,
    )
    return np.eye(3) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * (cross @ cross)


def _minimum_image(displacement: np.ndarray, box_length: float) -> np.ndarray:
    return displacement - box_length * np.rint(displacement / box_length)


def _reference_volume_liters(n_molecules: int, molarity: float) -> float:
    concentration = max(float(molarity), 1e-12)
    return max(n_molecules, 1) / (AVOGADRO_NUMBER * concentration)


def _compute_box_length(volume_liters: float, n_molecules: int, molarity: float) -> float:
    reference_liters = _reference_volume_liters(n_molecules, molarity)
    safe_volume_liters = max(0.05 * reference_liters, float(volume_liters))
    volume_nm3 = safe_volume_liters * LITER_TO_NM3
    return float(volume_nm3 ** (1.0 / 3.0))


def _initialize_centers(
    molecules: list[MoleculeTemplate],
    box_length: float,
    phase: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_molecules = len(molecules)
    side = max(1, math.ceil(n_molecules ** (1.0 / 3.0)))
    grid = np.linspace(0.16, 0.84, side)
    points = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    rng.shuffle(points)
    anchors = points[:n_molecules] * box_length
    if phase == "gas":
        avg_sigma = float(np.mean([molecule.sigma for molecule in molecules]))
        min_sep = 1.05 * avg_sigma
        centers = np.zeros((n_molecules, 3), dtype=float)
        for idx in range(n_molecules):
            placed = False
            for _ in range(240):
                candidate = rng.uniform(0.0, box_length, size=3)
                if idx == 0:
                    centers[idx] = candidate
                    placed = True
                    break
                deltas = _minimum_image(centers[:idx] - candidate, box_length)
                distances = np.linalg.norm(deltas, axis=1)
                if np.all(distances > min_sep):
                    centers[idx] = candidate
                    placed = True
                    break
            if not placed:
                centers[idx] = rng.uniform(0.0, box_length, size=3)
    elif phase == "liquid":
        centers = anchors + rng.normal(scale=0.09 * box_length / side, size=anchors.shape)
    else:
        centers = anchors + rng.normal(scale=0.03 * box_length / side, size=anchors.shape)
    centers %= box_length
    return centers, anchors


def _assemble_molecules(mixture: dict[str, int]) -> list[MoleculeTemplate]:
    molecules: list[MoleculeTemplate] = []
    for name, count in mixture.items():
        molecules.extend([MOLECULE_LIBRARY[name]] * int(count))
    return molecules


def _apply_reactions(
    *,
    centers: np.ndarray,
    velocities: np.ndarray,
    orientations: np.ndarray,
    molecules: list[MoleculeTemplate],
    box_length: float,
    temperature: float,
    pressure: float,
    phase: str,
    dt: float,
    rng: np.random.Generator,
    enabled: bool,
) -> tuple[Counter[str], float, int, dict[str, float], dict[str, dict[str, float]]]:
    products = Counter()
    total_heat = 0.0
    events = 0
    product_rates: dict[str, float] = {}
    thermodynamics: dict[str, dict[str, float]] = {}
    if not enabled:
        return products, total_heat, events, product_rates, thermodynamics

    reacted_indices: set[int] = set()
    n_molecules = len(molecules)
    temperature_k = max(temperature, 0.05)
    volume = max(box_length**3, 1e-12)
    concentration_scale = max(volume / LITER_TO_NM3, 1e-30)
    species_count = Counter(molecule.name for molecule in molecules)

    def _stochastic_round(value: float) -> int:
        value = max(0.0, float(value))
        integer = int(math.floor(value))
        return integer + int(rng.random() < (value - integer))

    for rule in REACTION_RULES:
        if phase not in rule.allowed_phases:
            continue

        reactant_a, reactant_b = rule.reactants
        n_a = int(species_count.get(reactant_a, 0))
        n_b = int(species_count.get(reactant_b, 0))
        if n_a <= 0 or n_b <= 0:
            continue

        reactant_activity = 1.0
        limiting_extent = float("inf")
        for species_name, nu in rule.stoich_reactants.items():
            count = float(species_count.get(species_name, 0.0))
            concentration_m = (count / AVOGADRO_NUMBER) / concentration_scale
            activity = max(concentration_m / 1.0, 1e-14)
            reactant_activity *= activity**nu
            limiting_extent = min(limiting_extent, count / max(nu, 1e-12))

        product_activity = 1.0
        for species_name, nu in rule.stoich_products.items():
            count = float(species_count.get(species_name, 0.0))
            concentration_m = (count / AVOGADRO_NUMBER) / concentration_scale
            activity = max(concentration_m / 1.0, 1e-14)
            product_activity *= activity**nu

        kinetic_prefactor = rule.pre_exponential * math.exp(
            -rule.activation_energy / (R_GAS_CONSTANT * max(temperature_k, 1e-9))
        )
        thermal_window = _safe_sigmoid(
            (temperature - rule.activation_temperature) / max(rule.activation_width_kelvin, 1e-9)
        )
        pressure_factor = min(1.8, max(0.55, pressure))
        delta_g0_j_mol = rule.delta_g0
        equilibrium_constant = math.exp(np.clip(-delta_g0_j_mol / (R_GAS_CONSTANT * max(temperature_k, 1e-9)), -20.0, 20.0))
        reaction_quotient = product_activity / max(reactant_activity, 1e-14)
        driving_force = float(np.clip(1.0 - reaction_quotient / max(equilibrium_constant, 1e-14), 0.0, 1.0))
        delta_g_rxn = delta_g0_j_mol + R_GAS_CONSTANT * temperature_k * math.log(max(reaction_quotient, 1e-14))
        rate_constant = kinetic_prefactor * thermal_window * pressure_factor * driving_force
        rate_constant *= max(0.05, rule.base_probability * 12.0)
        rate_constant = max(rate_constant, 0.0)

        dt_seconds = max(dt * SIM_TIME_SCALE_S, 1e-15)
        reaction_rate_molar_s = rate_constant * reactant_activity
        extent_moles = reaction_rate_molar_s * concentration_scale * dt_seconds
        extent_packets = extent_moles * AVOGADRO_NUMBER

        thermodynamics[rule.equation] = {
            "k_forward": float(rate_constant),
            "K_eq": float(equilibrium_constant),
            "Q": float(reaction_quotient),
            "delta_g_j_per_mol": float(delta_g_rxn),
            "driving_force": float(driving_force),
            "rate_molar_per_s": float(reaction_rate_molar_s),
            "extent_packets": float(extent_packets),
        }

        candidates: list[tuple[int, int, np.ndarray, float]] = []

        for i in range(n_molecules - 1):
            if i in reacted_indices:
                continue
            for j in range(i + 1, n_molecules):
                if j in reacted_indices:
                    continue
                name_i = molecules[i].name
                name_j = molecules[j].name
                if tuple(sorted((name_i, name_j))) != tuple(sorted(rule.reactants)):
                    continue

                displacement = _minimum_image(centers[j] - centers[i], box_length)
                distance = float(np.linalg.norm(displacement))
                sigma_pair = 0.5 * (molecules[i].sigma + molecules[j].sigma)
                capture = rule.capture_distance_factor * sigma_pair
                if distance <= capture:
                    candidates.append((i, j, displacement, distance))

        if not candidates:
            reactant_a_indices = [
                idx
                for idx, molecule in enumerate(molecules)
                if molecule.name == reactant_a and idx not in reacted_indices
            ]
            reactant_b_indices = [
                idx
                for idx, molecule in enumerate(molecules)
                if molecule.name == reactant_b and idx not in reacted_indices
            ]
            rng.shuffle(reactant_a_indices)
            rng.shuffle(reactant_b_indices)
            pair_count = min(len(reactant_a_indices), len(reactant_b_indices), 6)
            for pair_idx in range(pair_count):
                i = reactant_a_indices[pair_idx]
                j = reactant_b_indices[pair_idx]
                if i == j:
                    continue
                displacement = _minimum_image(centers[j] - centers[i], box_length)
                distance = float(np.linalg.norm(displacement))
                candidates.append((i, j, displacement, distance))

        if not candidates:
            continue

        collision_factor = len(candidates) / max(1.0, 0.5 * (n_a + n_b))
        expected_events = extent_packets * collision_factor
        max_events = min(len(candidates), n_a, n_b)
        if math.isfinite(limiting_extent):
            max_events = min(max_events, max(0, int(limiting_extent)))
        target_events = min(max_events, _stochastic_round(expected_events))
        if target_events <= 0:
            continue

        chosen = rng.choice(len(candidates), size=target_events, replace=False)
        for choice in np.atleast_1d(chosen):
            i, j, displacement, _ = candidates[int(choice)]
            if i in reacted_indices or j in reacted_indices:
                continue

            old_mass_i = molecules[i].mass
            old_mass_j = molecules[j].mass
            momentum = old_mass_i * velocities[i] + old_mass_j * velocities[j]

            molecules[i] = MOLECULE_LIBRARY[rule.product]
            molecules[j] = MOLECULE_LIBRARY[rule.byproduct]
            species_count[reactant_a] = max(0, species_count.get(reactant_a, 0) - 1)
            species_count[reactant_b] = max(0, species_count.get(reactant_b, 0) - 1)
            species_count[rule.product] = species_count.get(rule.product, 0) + 1
            species_count[rule.byproduct] = species_count.get(rule.byproduct, 0) + 1

            new_mass_i = molecules[i].mass
            new_mass_j = molecules[j].mass

            center_i = centers[i]
            direction = displacement
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                direction = rng.normal(size=3)
                norm = float(np.linalg.norm(direction))
            direction = direction / max(norm, 1e-9)

            midpoint = (center_i + 0.5 * displacement) % box_length
            centers[i] = (midpoint + 0.12 * molecules[i].sigma * direction) % box_length
            centers[j] = (midpoint - 0.12 * molecules[j].sigma * direction) % box_length

            base_velocity = momentum / max(new_mass_i + new_mass_j, 1e-9)
            thermal_kick = rule.heat_release * math.sqrt(max(temperature, 0.05))
            velocities[i] = base_velocity + direction * thermal_kick / max(new_mass_i, 1e-9)
            velocities[j] = base_velocity - direction * thermal_kick / max(new_mass_j, 1e-9)

            orientations[i] = _random_rotation_matrix(rng)
            orientations[j] = _random_rotation_matrix(rng)

            products[rule.product] += 1
            total_heat += rule.heat_release
            events += 1
            reacted_indices.add(i)
            reacted_indices.add(j)

    dt_safe = max(dt, 1e-12)
    for product_name, formed_count in products.items():
        product_rates[product_name] = float(formed_count) / dt_safe

    return products, total_heat, events, product_rates, thermodynamics


def _compute_kinematic_series(result: dict[str, object]) -> dict[str, np.ndarray]:
    centers_seq = result.get("trajectory_centers", [])
    box_history = np.asarray(result.get("box_length_history", []), dtype=float)
    dt = float(result.get("time_step", 1.0))
    if len(centers_seq) < 2:
        zeros = np.zeros(1, dtype=float)
        return {
            "time": zeros,
            "com_x": zeros,
            "com_y": zeros,
            "com_z": zeros,
            "mean_speed": zeros,
            "mean_acc": zeros,
        }

    centers = np.asarray(centers_seq, dtype=float)
    if box_history.size < centers.shape[0]:
        box_history = np.pad(box_history, (0, centers.shape[0] - box_history.size), mode="edge")

    vel = np.zeros((centers.shape[0] - 1, centers.shape[1], 3), dtype=float)
    com_track = np.zeros((centers.shape[0], 3), dtype=float)
    for t in range(centers.shape[0] - 1):
        disp = _minimum_image(centers[t + 1] - centers[t], float(box_history[t]))
        vel[t] = disp / max(dt, 1e-12)
        com_step = np.mean(disp, axis=0)
        com_track[t + 1] = com_track[t] + com_step

    speed = np.linalg.norm(vel, axis=2)
    n_samples = centers.shape[0]
    if vel.shape[0] >= 2:
        acc = np.linalg.norm(np.diff(vel, axis=0) / max(dt, 1e-12), axis=2)
        mean_acc_mid = np.mean(acc, axis=1)
        mean_acc = np.zeros(n_samples, dtype=float)
        mean_acc[1:-1] = mean_acc_mid
        mean_acc[0] = float(mean_acc_mid[0])
        mean_acc[-1] = float(mean_acc_mid[-1])
    else:
        mean_acc = np.zeros(n_samples, dtype=float)

    time = np.arange(centers.shape[0], dtype=float) * dt
    mean_speed = np.concatenate([[float(np.mean(speed[0]))], np.mean(speed, axis=1)])

    return {
        "time": time,
        "com_x": com_track[:, 0],
        "com_y": com_track[:, 1],
        "com_z": com_track[:, 2],
        "mean_speed": mean_speed,
        "mean_acc": mean_acc,
    }


def render_reaction_diagnostics_png(result: dict[str, object]) -> bytes:
    kin = _compute_kinematic_series(result)
    products = result.get("product_history", {})
    dt = float(result.get("time_step", 1.0))

    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), dpi=110)

    time = np.asarray(kin["time"], dtype=float)
    com_x = np.asarray(kin["com_x"], dtype=float)
    com_y = np.asarray(kin["com_y"], dtype=float)
    com_z = np.asarray(kin["com_z"], dtype=float)
    mean_speed = np.asarray(kin["mean_speed"], dtype=float)
    mean_acc = np.asarray(kin["mean_acc"], dtype=float)

    n = min(time.size, com_x.size, com_y.size, com_z.size, mean_speed.size, mean_acc.size)
    if n == 0:
        n = 1
        time = np.zeros(1, dtype=float)
        com_x = np.zeros(1, dtype=float)
        com_y = np.zeros(1, dtype=float)
        com_z = np.zeros(1, dtype=float)
        mean_speed = np.zeros(1, dtype=float)
        mean_acc = np.zeros(1, dtype=float)
    else:
        time = time[:n]
        com_x = com_x[:n]
        com_y = com_y[:n]
        com_z = com_z[:n]
        mean_speed = mean_speed[:n]
        mean_acc = mean_acc[:n]

    axes[0, 0].plot(time, com_x, label="COM x", color="#1f77b4")
    axes[0, 0].plot(time, com_y, label="COM y", color="#ff7f0e")
    axes[0, 0].plot(time, com_z, label="COM z", color="#2ca02c")
    axes[0, 0].set_title("Trajectory (center-of-mass displacement)")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("displacement")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(time, mean_speed, color="#9467bd")
    axes[0, 1].set_title("Mean Speed")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("speed")

    axes[1, 0].plot(time, mean_acc, color="#d62728")
    axes[1, 0].set_title("Mean Acceleration Magnitude")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("acceleration")

    total_products = np.zeros_like(time, dtype=float)
    for name, history in products.items():
        vals = np.asarray(history, dtype=float)
        if vals.size == 0:
            vals = np.zeros_like(time)
        elif vals.size < time.size:
            vals = np.pad(vals, (0, time.size - vals.size), mode="edge")
        vals = vals[: time.size]
        total_products += vals
        axes[1, 1].plot(time, vals, label=name)

    rate = np.gradient(total_products, max(dt, 1e-12))
    rate_axis = axes[1, 1].twinx()
    rate_axis.plot(time, rate, color="#111", linestyle="--", linewidth=1.5, label="rate")
    axes[1, 1].set_title("Products Formed and Formation Rate")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("product count")
    rate_axis.set_ylabel("d(products)/dt")
    axes[1, 1].legend(fontsize=8, loc="upper left")

    thermo_history = result.get("thermodynamics_history", {})
    thermo_lines = ["Stoichiometric kinetics + thermodynamics:"]
    for equation, snapshots in thermo_history.items():
        if not snapshots:
            continue
        latest = snapshots[-1]
        thermo_lines.append(
            f"{equation}: k={latest.get('k_forward', 0.0):.2e}, "
            f"K_eq={latest.get('K_eq', 1.0):.2e}, "
            f"Q={latest.get('Q', 0.0):.2e}, "
            f"dG={latest.get('delta_g_j_per_mol', 0.0):.2e} J/mol"
        )

    if len(thermo_lines) == 1:
        thermo_lines.append("No thermodynamic reaction snapshots recorded for this run.")

    figure.text(
        0.015,
        0.01,
        "\n".join(thermo_lines),
        ha="left",
        va="bottom",
        fontsize=8.0,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fbfbfb", alpha=0.85, edgecolor="#a9a9a9"),
    )

    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    plt.close(figure)
    return buffer.getvalue()


def _compute_forces(
    centers: np.ndarray,
    molecules: list[MoleculeTemplate],
    box_length: float,
    attraction_scale: float = 1.0,
) -> tuple[np.ndarray, float, float, list[float]]:
    n_molecules = len(molecules)
    forces = np.zeros_like(centers)
    potential = 0.0
    virial = 0.0
    normalized_distances: list[float] = []

    for i in range(n_molecules - 1):
        for j in range(i + 1, n_molecules):
            displacement = _minimum_image(centers[j] - centers[i], box_length)
            distance = float(np.linalg.norm(displacement)) + 1e-9
            sigma = 0.5 * (molecules[i].sigma + molecules[j].sigma)
            epsilon = attraction_scale * math.sqrt(molecules[i].epsilon * molecules[j].epsilon)
            cutoff = 2.7 * sigma
            if distance > cutoff:
                continue
            # Softened core avoids unbounded force spikes on rare close contacts.
            distance = max(distance, 0.82 * sigma)
            sr6 = (sigma / distance) ** 6
            sr12 = sr6 * sr6
            force = 24.0 * epsilon * (2.0 * sr12 - sr6) * displacement / (distance * distance)
            forces[i] -= force
            forces[j] += force
            potential += 4.0 * epsilon * (sr12 - sr6)
            virial += float(np.dot(displacement, force))
            normalized_distances.append(distance / sigma)

    return forces, potential, virial, normalized_distances


def simulate_molecular_dynamics(
    mixture: dict[str, int],
    phase: str,
    volume_liters: float,
    molarity: float,
    temperature_kelvin: float,
    pressure_pascal: float,
    reactions_enabled: bool = True,
    *,
    steps: int = 140,
    dt: float = 0.018,
    seed: int = 7,
) -> dict[str, object]:
    if phase not in PHASE_PRESETS:
        raise ValueError(f"Unknown phase '{phase}'.")

    molecules = _assemble_molecules(mixture)
    if len(molecules) > 24:
        raise ValueError("Keep the total molecule count at 24 or fewer for interactive use.")

    rng = np.random.default_rng(seed)
    box_length = _compute_box_length(volume_liters, len(molecules), molarity)
    temperature_reduced = max(float(temperature_kelvin) / REFERENCE_TEMPERATURE_K, 0.05)
    pressure_reduced = max(float(pressure_pascal) / REFERENCE_PRESSURE_PA, 0.05)
    centers, anchors = _initialize_centers(molecules, box_length, phase, rng)
    orientations = np.stack([_random_rotation_matrix(rng) for _ in molecules], axis=0)
    velocities = rng.normal(scale=math.sqrt(temperature_reduced), size=(len(molecules), 3))
    velocities -= np.mean(velocities, axis=0, keepdims=True)
    angular_velocity = rng.normal(scale=0.18 * math.sqrt(temperature_reduced), size=(len(molecules), 3))
    masses = np.array([molecule.mass for molecule in molecules], dtype=float)

    phase_props = PHASE_PRESETS[phase]
    trail_history: list[np.ndarray] = []
    pressure_history: list[float] = []
    kinetic_history: list[float] = []
    pair_distances: list[float] = []
    trajectory_centers: list[np.ndarray] = [centers.copy()]
    trajectory_orientations: list[np.ndarray] = [orientations.copy()]
    box_length_history: list[float] = [float(box_length)]
    product_names = sorted({rule.product for rule in REACTION_RULES})
    product_history: dict[str, list[int]] = {name: [] for name in product_names}
    product_rate_history: dict[str, list[float]] = {name: [0.0] for name in product_names}
    heat_history: list[float] = [0.0]
    reaction_events_history: list[int] = [0]
    reaction_rate_history: list[float] = [0.0]
    thermodynamics_history: dict[str, list[dict[str, float]]] = {
        rule.equation: [] for rule in REACTION_RULES
    }

    initial_counts = Counter(molecule.name for molecule in molecules)
    for name in product_names:
        product_history[name].append(int(initial_counts.get(name, 0)))

    forces, potential, virial, pair_distances = _compute_forces(
        centers,
        molecules,
        box_length,
        attraction_scale=phase_props["attraction"],
    )

    for step in range(steps):
        velocities += 0.5 * dt * forces / masses[:, None]
        centers = (centers + dt * phase_props["mobility"] * velocities) % box_length

        angular_velocity += rng.normal(scale=0.02 * math.sqrt(temperature_reduced), size=angular_velocity.shape)
        angular_velocity *= 1.0 - 0.25 * phase_props["damping"] * dt
        for index in range(len(molecules)):
            orientations[index] = _rotation_matrix_from_rotvec(angular_velocity[index] * dt) @ orientations[index]

        forces, potential, virial, pair_distances = _compute_forces(
            centers,
            molecules,
            box_length,
            attraction_scale=phase_props["attraction"],
        )

        if phase_props["tether"] > 0.0:
            tether_displacement = _minimum_image(centers - anchors, box_length)
            forces -= phase_props["tether"] * tether_displacement

        velocities += 0.5 * dt * forces / masses[:, None]
        velocities *= 1.0 - phase_props["damping"] * dt
        velocities += (
            phase_props["mobility"]
            * phase_props["noise"]
            * math.sqrt(temperature_reduced * dt)
            * rng.standard_normal(velocities.shape)
        )
        velocities -= np.mean(velocities, axis=0, keepdims=True)

        speed = np.linalg.norm(velocities, axis=1)
        max_speed = 3.8 * math.sqrt(temperature_reduced) * max(0.45, phase_props["mobility"])
        mask = speed > max_speed
        if np.any(mask):
            velocities[mask] *= (max_speed / speed[mask])[:, None]

        kinetic = 0.5 * float(np.sum(masses[:, None] * velocities * velocities))
        instantaneous_temperature = max(2.0 * kinetic / max(3 * len(molecules), 1), 1e-8)
        thermostat = math.sqrt(max(0.5, min(1.5, 1.0 + 0.18 * (temperature_reduced / instantaneous_temperature - 1.0))))
        velocities *= thermostat

        products, released_heat, events, product_rates, thermo_snapshot = _apply_reactions(
            centers=centers,
            velocities=velocities,
            orientations=orientations,
            molecules=molecules,
            box_length=box_length,
            temperature=temperature_kelvin,
            pressure=pressure_reduced,
            phase=phase,
            dt=dt,
            rng=rng,
            enabled=reactions_enabled,
        )
        if released_heat > 0.0:
            velocities += rng.normal(scale=0.12 * released_heat, size=velocities.shape)
            velocities -= np.mean(velocities, axis=0, keepdims=True)

        volume = box_length**3
        pressure_proxy = len(molecules) * instantaneous_temperature / volume + virial / (3.0 * volume)
        scale = 1.0 - phase_props["barostat"] * (pressure_reduced - pressure_proxy) * dt
        scale = float(np.clip(scale, 0.985, 1.015))
        new_box_length = box_length * scale
        centers = (centers - box_length / 2.0) * scale + new_box_length / 2.0
        anchors = (anchors - box_length / 2.0) * scale + new_box_length / 2.0
        centers %= new_box_length
        anchors %= new_box_length
        box_length = new_box_length

        pressure_history.append(float(pressure_proxy))
        kinetic_history.append(float(kinetic))
        trajectory_centers.append(centers.copy())
        trajectory_orientations.append(orientations.copy())
        box_length_history.append(float(box_length))

        species_count = Counter(molecule.name for molecule in molecules)
        for name in product_names:
            product_history[name].append(int(species_count.get(name, 0)))
            product_rate_history[name].append(float(product_rates.get(name, 0.0)))
        heat_history.append(float(released_heat))
        reaction_events_history.append(int(events))
        reaction_rate_history.append(float(events) / max(dt, 1e-12))
        for rule in REACTION_RULES:
            thermodynamics_history[rule.equation].append(
                thermo_snapshot.get(
                    rule.equation,
                    {
                        "k_forward": 0.0,
                        "K_eq": 1.0,
                        "Q": 0.0,
                        "delta_g_j_per_mol": 0.0,
                        "driving_force": 0.0,
                        "rate_molar_per_s": 0.0,
                        "extent_packets": 0.0,
                    },
                )
            )

        if step >= steps - 28:
            trail_history.append(centers.copy())

    total_energy = float(kinetic_history[-1] + potential) if kinetic_history else float(potential)
    dominant_species = max(mixture.items(), key=lambda item: item[1])[0]

    return {
        "phase": phase,
        "mixture": mixture,
        "molecules": molecules,
        "centers": centers,
        "anchors": anchors,
        "orientations": orientations,
        "box_length": float(box_length),
        "temperature": float(temperature_kelvin),
        "pressure": float(pressure_pascal),
        "temperature_reduced": float(temperature_reduced),
        "pressure_reduced": float(pressure_reduced),
        "volume_liters": float(volume_liters),
        "molarity": float(molarity),
        "potential_energy": float(potential),
        "total_energy": total_energy,
        "kinetic_history": kinetic_history,
        "pressure_history": pressure_history,
        "trail_history": trail_history,
        "trajectory_centers": trajectory_centers,
        "trajectory_orientations": trajectory_orientations,
        "pair_distances": pair_distances,
        "focus_species": dominant_species,
        "box_length_history": box_length_history,
        "product_history": product_history,
        "product_rate_history": product_rate_history,
        "reaction_heat_history": heat_history,
        "reaction_events_history": reaction_events_history,
        "reaction_rate_history": reaction_rate_history,
        "thermodynamics_history": thermodynamics_history,
        "reactions_enabled": bool(reactions_enabled),
        "time_step": float(dt),
    }


def _global_atom_positions(center: np.ndarray, orientation: np.ndarray, molecule: MoleculeTemplate) -> np.ndarray:
    local = np.array([atom.position for atom in molecule.atoms], dtype=float)
    return local @ orientation.T + center


def _blend_rotation_matrices(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    blended = (1.0 - alpha) * start + alpha * end
    q, _ = np.linalg.qr(blended)
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _upsample_trajectory(
    centers_seq: list[np.ndarray],
    orientations_seq: list[np.ndarray],
    *,
    substeps: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if substeps <= 1 or len(centers_seq) < 2 or len(orientations_seq) < 2:
        return centers_seq, orientations_seq

    smooth_centers: list[np.ndarray] = []
    smooth_orientations: list[np.ndarray] = []
    for index in range(len(centers_seq) - 1):
        c0 = np.asarray(centers_seq[index], dtype=float)
        c1 = np.asarray(centers_seq[index + 1], dtype=float)
        o0 = np.asarray(orientations_seq[index], dtype=float)
        o1 = np.asarray(orientations_seq[index + 1], dtype=float)
        for sub in range(substeps):
            alpha = sub / float(substeps)
            smooth_centers.append((1.0 - alpha) * c0 + alpha * c1)
            blended_orientation = np.stack(
                [_blend_rotation_matrices(o0[m], o1[m], alpha) for m in range(o0.shape[0])],
                axis=0,
            )
            smooth_orientations.append(blended_orientation)

    smooth_centers.append(np.asarray(centers_seq[-1], dtype=float))
    smooth_orientations.append(np.asarray(orientations_seq[-1], dtype=float))
    return smooth_centers, smooth_orientations


def _draw_box(ax: plt.Axes, box_length: float) -> None:
    edges = [
        ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
        ((1, 1, 1), (0, 1, 1)), ((1, 1, 1), (1, 0, 1)), ((1, 1, 1), (1, 1, 0)),
        ((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (1, 1, 0)),
        ((0, 1, 0), (0, 1, 1)), ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)),
    ]
    for start, end in edges:
        start_xyz = np.array(start, dtype=float) * box_length
        end_xyz = np.array(end, dtype=float) * box_length
        ax.plot(
            [start_xyz[0], end_xyz[0]],
            [start_xyz[1], end_xyz[1]],
            [start_xyz[2], end_xyz[2]],
            color="#9aa0a6",
            linewidth=0.8,
            alpha=0.6,
        )


def _render_focus_geometry(ax: plt.Axes, template: MoleculeTemplate) -> None:
    positions = np.array([atom.position for atom in template.atoms], dtype=float)
    for start_idx, end_idx in template.bonds:
        start = positions[start_idx]
        end = positions[end_idx]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="#555", linewidth=2.0)

    for atom in template.atoms:
        atom_pos = np.array(atom.position, dtype=float)
        ax.scatter(
            [atom_pos[0]],
            [atom_pos[1]],
            [atom_pos[2]],
            color=ATOMIC_COLORS.get(atom.element, "#333"),
            s=ATOMIC_RADII.get(atom.element, 55.0),
            edgecolors="black",
            linewidths=0.6,
        )
        ax.text(atom_pos[0], atom_pos[1], atom_pos[2], atom.element, fontsize=8)

    for orbital in template.orbitals:
        origin = positions[orbital.origin_atom]
        direction = np.array(orbital.vector, dtype=float)
        end = origin + 1.05 * direction
        linestyle = "--" if orbital.label == "lone pair" else ":"
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color="#00a6a6", linestyle=linestyle, linewidth=1.5)

    max_extent = float(np.max(np.linalg.norm(positions, axis=1))) if len(positions) else 1.0
    pad = max(1.6, max_extent + 0.6)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_zlim(-pad, pad)
    ax.set_title(f"{template.name}: {template.geometry}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_simulation_snapshot(
    result: dict[str, object], *, figure: plt.Figure | None = None, frame_index: int | None = None
) -> plt.Figure:
    if figure is None:
        figure = plt.figure(figsize=(14, 8))
    else:
        figure.clf()

    grid = figure.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
    ax_sim = figure.add_subplot(grid[:, 0], projection="3d")
    ax_focus = figure.add_subplot(grid[0, 1], projection="3d")
    ax_stats = figure.add_subplot(grid[1, 1])

    molecules: list[MoleculeTemplate] = result["molecules"]  # type: ignore[assignment]
    centers = np.asarray(result["centers"], dtype=float)
    orientations = np.asarray(result["orientations"], dtype=float)
    box_length = float(result["box_length"])
    trail_history = result["trail_history"]
    pair_distances = np.asarray(result["pair_distances"], dtype=float)
    trajectory_centers = result.get("trajectory_centers")
    trajectory_orientations = result.get("trajectory_orientations")

    if trajectory_centers and trajectory_orientations:
        frame_count = len(trajectory_centers)
        frame = frame_count - 1 if frame_index is None else int(np.clip(frame_index, 0, frame_count - 1))
        centers = np.asarray(trajectory_centers[frame], dtype=float)
        orientations = np.asarray(trajectory_orientations[frame], dtype=float)
        trail_start = max(0, frame - 27)
        trails = np.asarray(trajectory_centers[trail_start : frame + 1], dtype=float)
    else:
        frame = 0
        if trail_history:
            trails = np.stack(trail_history, axis=0)
        else:
            trails = np.empty((0, centers.shape[0], 3), dtype=float)

    if trails.size:
        for idx in range(trails.shape[1]):
            ax_sim.plot(trails[:, idx, 0], trails[:, idx, 1], trails[:, idx, 2], color="#7f8c8d", alpha=0.35, linewidth=1.0)

    for idx, molecule in enumerate(molecules):
        atom_positions = _global_atom_positions(centers[idx], orientations[idx], molecule)
        for bond_start, bond_end in molecule.bonds:
            start = atom_positions[bond_start]
            end = atom_positions[bond_end]
            ax_sim.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color="#444", linewidth=1.4, alpha=0.9)

        for atom_index, atom in enumerate(molecule.atoms):
            atom_pos = atom_positions[atom_index]
            ax_sim.scatter(
                [atom_pos[0]],
                [atom_pos[1]],
                [atom_pos[2]],
                color=ATOMIC_COLORS.get(atom.element, "#333"),
                s=ATOMIC_RADII.get(atom.element, 55.0),
                edgecolors="black",
                linewidths=0.4,
                alpha=0.95,
            )

    _draw_box(ax_sim, box_length)

    if trajectory_centers and frame > 0:
        previous_centers = np.asarray(trajectory_centers[frame - 1], dtype=float)
        velocity_vectors = _minimum_image(centers - previous_centers, box_length)
        speeds = np.linalg.norm(velocity_vectors, axis=1)
        scale = max(float(np.max(speeds)), 1e-9)
        velocity_vectors = velocity_vectors / scale * (0.22 * box_length / 10.0)
        ax_sim.quiver(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            velocity_vectors[:, 0],
            velocity_vectors[:, 1],
            velocity_vectors[:, 2],
            color="#2e7d32",
            linewidth=1.0,
            alpha=0.8,
        )

    ax_sim.set_xlim(0.0, box_length)
    ax_sim.set_ylim(0.0, box_length)
    ax_sim.set_zlim(0.0, box_length)
    ax_sim.set_xlabel("x")
    ax_sim.set_ylabel("y")
    ax_sim.set_zlabel("z")
    ax_sim.set_title(
        f"{result['phase'].title()} phase | frame={frame} | T={result['temperature']:.1f} K | P={result['pressure']:.0f} Pa | V={result['volume_liters']:.3e} L"
    )

    focus_name = str(result["focus_species"])
    _render_focus_geometry(ax_focus, MOLECULE_LIBRARY[focus_name])

    kinetic_history = np.asarray(result["kinetic_history"], dtype=float)
    pressure_history = np.asarray(result["pressure_history"], dtype=float)
    if pair_distances.size:
        ax_stats.hist(pair_distances, bins=np.linspace(0.7, 3.0, 18), color="#4f8fbf", alpha=0.75, label="pair distance / sigma")
        ax_stats.set_xlabel("pair distance / sigma")
        ax_stats.set_ylabel("count")
    else:
        ax_stats.text(0.05, 0.75, "No close molecular pairs\ninside the interaction cutoff.", transform=ax_stats.transAxes)
    summary_lines = [
        f"Mixture: {', '.join(f'{key}:{value}' for key, value in result['mixture'].items())}",
        f"Focus species: {focus_name} ({MOLECULE_LIBRARY[focus_name].orbital_model})",
        f"Potential energy: {result['potential_energy']:.3f}",
        f"Total energy proxy: {result['total_energy']:.3f}",
    ]
    if kinetic_history.size:
        summary_lines.append(f"Mean kinetic proxy: {float(np.mean(kinetic_history)):.3f}")
    if pressure_history.size:
        summary_lines.append(f"Mean internal pressure proxy: {float(np.mean(pressure_history)):.3f}")
    ax_stats.text(0.04, 0.95, "\n".join(summary_lines), transform=ax_stats.transAxes, va="top")
    ax_stats.set_title("Pair structure and summary")
    figure.tight_layout()
    return figure


def plot_simulation_motion_frame(
    result: dict[str, object], *, frame_index: int, figure: plt.Figure | None = None
) -> plt.Figure:
    if figure is None:
        figure = plt.figure(figsize=(8.2, 6.6))
    else:
        figure.clf()

    ax_sim = figure.add_subplot(111, projection="3d")
    molecules: list[MoleculeTemplate] = result["molecules"]  # type: ignore[assignment]
    box_length = float(result["box_length"])
    trajectory_centers = result.get("trajectory_centers")
    trajectory_orientations = result.get("trajectory_orientations")

    if not trajectory_centers or not trajectory_orientations:
        return plot_simulation_snapshot(result, figure=figure, frame_index=frame_index)

    frame_count = len(trajectory_centers)
    frame = int(np.clip(frame_index, 0, frame_count - 1))
    centers = np.asarray(trajectory_centers[frame], dtype=float)
    orientations = np.asarray(trajectory_orientations[frame], dtype=float)

    trail_start = max(0, frame - 36)
    trails = np.asarray(trajectory_centers[trail_start : frame + 1], dtype=float)
    if trails.size:
        for idx in range(trails.shape[1]):
            ax_sim.plot(
                trails[:, idx, 0],
                trails[:, idx, 1],
                trails[:, idx, 2],
                color="#6b7280",
                alpha=0.35,
                linewidth=1.1,
            )

    for idx, molecule in enumerate(molecules):
        atom_positions = _global_atom_positions(centers[idx], orientations[idx], molecule)
        for bond_start, bond_end in molecule.bonds:
            start = atom_positions[bond_start]
            end = atom_positions[bond_end]
            ax_sim.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color="#424242",
                linewidth=1.2,
                alpha=0.85,
            )

        for atom_index, atom in enumerate(molecule.atoms):
            atom_pos = atom_positions[atom_index]
            ax_sim.scatter(
                [atom_pos[0]],
                [atom_pos[1]],
                [atom_pos[2]],
                color=ATOMIC_COLORS.get(atom.element, "#333"),
                s=ATOMIC_RADII.get(atom.element, 55.0),
                edgecolors="black",
                linewidths=0.3,
                alpha=0.95,
            )

    _draw_box(ax_sim, box_length)

    if frame > 0:
        previous_centers = np.asarray(trajectory_centers[frame - 1], dtype=float)
        velocity_vectors = _minimum_image(centers - previous_centers, box_length)
        speeds = np.linalg.norm(velocity_vectors, axis=1)
        scale = max(float(np.max(speeds)), 1e-9)
        velocity_vectors = velocity_vectors / scale * (0.22 * box_length / 10.0)
        ax_sim.quiver(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            velocity_vectors[:, 0],
            velocity_vectors[:, 1],
            velocity_vectors[:, 2],
            color="#1f9d55",
            linewidth=1.0,
            alpha=0.85,
        )

    ax_sim.set_xlim(0.0, box_length)
    ax_sim.set_ylim(0.0, box_length)
    ax_sim.set_zlim(0.0, box_length)
    ax_sim.set_xlabel("x")
    ax_sim.set_ylabel("y")
    ax_sim.set_zlabel("z")
    ax_sim.set_title(
        f"{result['phase'].title()} phase motion | frame={frame} | T={result['temperature']:.1f} K | P={result['pressure']:.0f} Pa"
    )
    product_history = result.get("product_history", {})
    product_rate_history = result.get("product_rate_history", {})
    if product_history:
        overlay_lines = ["Products (count | rate):"]
        for product in sorted(product_history):
            counts = np.asarray(product_history[product], dtype=float)
            rates = np.asarray(product_rate_history.get(product, []), dtype=float)
            idx = min(frame, max(0, counts.size - 1))
            ridx = min(frame, max(0, rates.size - 1))
            count_value = counts[idx] if counts.size else 0.0
            rate_value = rates[ridx] if rates.size else 0.0
            overlay_lines.append(f"{product}: {count_value:.0f} | {rate_value:.2f}/t")

        thermo_history = result.get("thermodynamics_history", {})
        if thermo_history:
            overlay_lines.append("---")
            overlay_lines.append("Stoichiometric equations:")
            for equation, snapshots in thermo_history.items():
                if not snapshots:
                    continue
                latest = snapshots[-1]
                overlay_lines.append(
                    f"{equation} | K_eq={latest.get('K_eq', 1.0):.1e}, "
                    f"Q={latest.get('Q', 0.0):.1e}"
                )
        ax_sim.text2D(
            0.01,
            0.98,
            "\n".join(overlay_lines),
            transform=ax_sim.transAxes,
            va="top",
            ha="left",
            fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="#999"),
        )
    figure.tight_layout()
    return figure


def render_simulation_motion_frame_png(result: dict[str, object], frame_index: int) -> bytes:
    figure = plt.figure(figsize=(8.2, 6.6), dpi=108)
    plot_simulation_motion_frame(result, frame_index=frame_index, figure=figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=108)
    plt.close(figure)
    return buffer.getvalue()


def summarize_simulation(result: dict[str, object]) -> dict[str, object]:
    pressure_history = np.asarray(result["pressure_history"], dtype=float)
    kinetic_history = np.asarray(result["kinetic_history"], dtype=float)
    product_history = result.get("product_history", {})
    thermodynamics_history = result.get("thermodynamics_history", {})
    final_products: dict[str, int] = {}
    for name, hist in product_history.items():
        arr = np.asarray(hist, dtype=float)
        if arr.size:
            final_products[name] = int(arr[-1])
    final_thermodynamics: dict[str, dict[str, float]] = {}
    for equation, hist in thermodynamics_history.items():
        if hist:
            final_thermodynamics[equation] = hist[-1]

    return {
        "domain_key": DOMAIN_KEY,
        "phase": result["phase"],
        "mixture": result["mixture"],
        "focus_species": result["focus_species"],
        "box_length": round(float(result["box_length"]), 4),
        "volume_liters": float(result["volume_liters"]),
        "molarity": float(result["molarity"]),
        "temperature_kelvin": float(result["temperature"]),
        "pressure_pascal": float(result["pressure"]),
        "potential_energy": round(float(result["potential_energy"]), 6),
        "total_energy": round(float(result["total_energy"]), 6),
        "mean_pressure_proxy": round(float(np.mean(pressure_history)) if pressure_history.size else 0.0, 6),
        "mean_kinetic_proxy": round(float(np.mean(kinetic_history)) if kinetic_history.size else 0.0, 6),
        "supported_molecules": sorted(MOLECULE_LIBRARY),
        "reactions_enabled": bool(result.get("reactions_enabled", False)),
        "total_reaction_events": int(np.sum(np.asarray(result.get("reaction_events_history", [0]), dtype=int))),
        "reaction_equations": [rule.equation for rule in REACTION_RULES],
        "final_products": final_products,
        "final_thermodynamics": final_thermodynamics,
    }


def build_project_context() -> dict[str, object]:
    """Return metadata and simulator capabilities for this project."""
    return {
        "domain_key": DOMAIN_KEY,
        "domain": DOMAIN_TITLE,
        "model": "rigid-molecule Lennard-Jones coarse-grained dynamics",
        "phases": sorted(PHASE_PRESETS),
        "supported_molecules": {
            name: {
                "geometry": template.geometry,
                "orbital_model": template.orbital_model,
            }
            for name, template in sorted(MOLECULE_LIBRARY.items())
        },
        "controls": ["phase", "mixture", "volume_liters", "molarity", "temperature_kelvin", "pressure_pascal"],
        "baseline_concentration": "1.0 M",
        "reactive_products": sorted({rule.product for rule in REACTION_RULES}),
        "reaction_equations": [rule.equation for rule in REACTION_RULES],
    }


def launch_interactive_ui() -> None:
    figure = plt.figure(figsize=(14, 8))
    controls = {
        "mixture": DEFAULT_MIXTURE,
        "phase": "liquid",
        "volume_liters": _reference_volume_liters(sum(parse_mixture(DEFAULT_MIXTURE).values()), 1.0),
        "temperature_kelvin": 300.0,
        "pressure_pascal": 101325.0,
    }

    def redraw() -> None:
        try:
            mixture = parse_mixture(controls["mixture"])
            result = simulate_molecular_dynamics(
                mixture=mixture,
                phase=controls["phase"],
                volume_liters=float(controls["volume_liters"]),
                molarity=1.0,
                temperature_kelvin=float(controls["temperature_kelvin"]),
                pressure_pascal=float(controls["pressure_pascal"]),
                reactions_enabled=True,
            )
        except Exception as exc:  # pragma: no cover - interactive error surface
            figure.clf()
            axis = figure.add_subplot(111)
            axis.axis("off")
            axis.text(0.03, 0.95, f"Input error: {exc}", va="top", color="crimson", transform=axis.transAxes)
            figure.canvas.draw_idle()
            return

        plot_simulation_snapshot(result, figure=figure)
        figure.canvas.draw_idle()

    plot_simulation_snapshot(
        simulate_molecular_dynamics(
            parse_mixture(controls["mixture"]),
            controls["phase"],
            controls["volume_liters"],
            1.0,
            controls["temperature_kelvin"],
            controls["pressure_pascal"],
            True,
        ),
        figure=figure,
    )
    plt.subplots_adjust(left=0.29, bottom=0.26)

    mix_ax = plt.axes([0.03, 0.87, 0.21, 0.05])
    phase_ax = plt.axes([0.03, 0.62, 0.21, 0.19])
    volume_ax = plt.axes([0.03, 0.48, 0.21, 0.03])
    temp_ax = plt.axes([0.03, 0.40, 0.21, 0.03])
    pressure_ax = plt.axes([0.03, 0.32, 0.21, 0.03])
    reset_ax = plt.axes([0.03, 0.22, 0.10, 0.05])

    mix_box = TextBox(mix_ax, "Mixture", initial=controls["mixture"])
    phase_buttons = RadioButtons(phase_ax, ("gas", "liquid", "solid"), active=1)
    volume_slider = Slider(volume_ax, "Volume (L)", 0.2 * controls["volume_liters"], 6.0 * controls["volume_liters"], valinit=controls["volume_liters"], valstep=0.05 * controls["volume_liters"])
    temp_slider = Slider(temp_ax, "Temp (K)", 150.0, 1400.0, valinit=controls["temperature_kelvin"], valstep=10.0)
    pressure_slider = Slider(pressure_ax, "Pressure (Pa)", 2.0e4, 5.0e5, valinit=controls["pressure_pascal"], valstep=2.0e3)
    reset_button = Button(reset_ax, "Reset")

    def on_submit(text: str) -> None:
        controls["mixture"] = text
        redraw()

    def on_phase(label: str) -> None:
        controls["phase"] = label
        redraw()

    def on_slider(_: float) -> None:
        controls["volume_liters"] = volume_slider.val
        controls["temperature_kelvin"] = temp_slider.val
        controls["pressure_pascal"] = pressure_slider.val
        redraw()

    def on_reset(_: object) -> None:
        mix_box.set_val(DEFAULT_MIXTURE)
        phase_buttons.set_active(1)
        volume_slider.reset()
        temp_slider.reset()
        pressure_slider.reset()

    mix_box.on_submit(on_submit)
    phase_buttons.on_clicked(on_phase)
    volume_slider.on_changed(on_slider)
    temp_slider.on_changed(on_slider)
    pressure_slider.on_changed(on_slider)
    reset_button.on_clicked(on_reset)
    plt.show()


def build_molecular_dynamics_notebook():
    if widgets is None or display is None:
        raise RuntimeError("ipywidgets and IPython are required for notebook controls.")

    molecule_options = ["None"] + sorted(MOLECULE_LIBRARY)
    motion_image = widgets.Image(format="png")
    diagnostics_image = widgets.Image(format="png")
    status = widgets.HTML(value="")

    phase = widgets.ToggleButtons(options=[("Gas", "gas"), ("Liquid", "liquid"), ("Solid", "solid")], value="liquid", description="Phase")
    default_molecule_count = sum(parse_mixture(DEFAULT_MIXTURE).values())
    default_volume_liters = _reference_volume_liters(default_molecule_count, 1.0)
    volume = widgets.FloatLogSlider(
        description="Volume (L)",
        value=default_volume_liters,
        base=10,
        min=math.log10(default_volume_liters) - 1.0,
        max=math.log10(default_volume_liters) + 1.0,
        step=0.02,
        readout_format=".3e",
        continuous_update=False,
    )
    temperature = widgets.FloatSlider(description="Temperature (K)", value=300.0, min=120.0, max=1500.0, step=10.0, continuous_update=False)
    pressure = widgets.FloatLogSlider(
        description="Pressure (Pa)",
        value=101325.0,
        base=10,
        min=4.0,
        max=7.0,
        step=0.01,
        readout_format=".0f",
        continuous_update=False,
    )
    molarity = widgets.FloatSlider(description="Molarity (M)", value=1.0, min=0.2, max=3.0, step=0.05, continuous_update=False)
    reactions_enabled = widgets.Checkbox(value=True, description="Enable reactions")
    species1 = widgets.Dropdown(options=molecule_options, value="H2O", description="Species 1")
    count1 = widgets.IntSlider(description="Count 1", value=6, min=0, max=8, step=1, continuous_update=False)
    species2 = widgets.Dropdown(options=molecule_options, value="N2", description="Species 2")
    count2 = widgets.IntSlider(description="Count 2", value=4, min=0, max=8, step=1, continuous_update=False)
    species3 = widgets.Dropdown(options=molecule_options, value="CH4", description="Species 3")
    count3 = widgets.IntSlider(description="Count 3", value=3, min=0, max=8, step=1, continuous_update=False)
    play = widgets.Play(value=0, min=0, max=1, step=1, interval=35, description="Play", disabled=False)
    frame_slider = widgets.IntSlider(description="Frame", value=0, min=0, max=1, step=1, continuous_update=True)
    analyze_button = widgets.Button(description="Analyze Run", button_style="info")
    widgets.jslink((play, "value"), (frame_slider, "value"))

    active_result: dict[str, object] | None = None
    frame_cache: dict[int, bytes] = {}
    diagnostics_ready = False

    supported = widgets.HTML(
        value=(
            "<b>Supported molecules</b>: "
            + ", ".join(f"{name} ({template.geometry})" for name, template in sorted(MOLECULE_LIBRARY.items()))
            + "<br><i>Orbital overlays are schematic VSEPR / hybridization guides attached to rigid molecular templates.</i>"
        )
    )

    def rebuild_simulation(*_: object) -> None:
        nonlocal active_result
        nonlocal frame_cache
        nonlocal diagnostics_ready
        status.value = "<i>Building trajectory for current composition and phase...</i>"
        mixture = mixture_from_controls(
            (species1.value, count1.value),
            (species2.value, count2.value),
            (species3.value, count3.value),
        )
        active_result = simulate_molecular_dynamics(
            mixture=mixture,
            phase=phase.value,
            volume_liters=volume.value,
            molarity=molarity.value,
            temperature_kelvin=temperature.value,
            pressure_pascal=pressure.value,
            reactions_enabled=bool(reactions_enabled.value),
            steps=120,
            dt=0.022,
        )
        base_centers = active_result.get("trajectory_centers", [])
        base_orientations = active_result.get("trajectory_orientations", [])
        smooth_centers, smooth_orientations = _upsample_trajectory(
            list(base_centers),
            list(base_orientations),
            substeps=3,
        )
        active_result["trajectory_centers"] = smooth_centers
        active_result["trajectory_orientations"] = smooth_orientations

        frame_cache = {}
        frame_cache[0] = render_simulation_motion_frame_png(active_result, 0)

        total_frames = len(smooth_centers)
        play.value = 0
        frame_slider.value = 0
        play.max = max(0, total_frames - 1)
        frame_slider.max = max(0, total_frames - 1)
        motion_image.value = frame_cache[0]
        diagnostics_image.value = b""
        diagnostics_ready = False
        status.value = (
            "<b>Ready:</b> composition is initialized at 1.0 M unless changed with the molarity slider. "
            "Press Play to start motion (faster playback, longer simulation window). "
            "Use the green arrows to read direction and relative speed. "
            "Reactions use thermodynamic driving (delta G) with an integrated rate law over each timestep. "
            "Sliders are in physical units: Kelvin, Liters, and Pascals. "
            "Diagnostics plots are generated after the run reaches the final frame, or by clicking Analyze Run. "
            "To change composition, stop, edit counts/species, then press Play again."
        )

    def render_frame(*_: object) -> None:
        nonlocal frame_cache
        nonlocal diagnostics_ready
        if active_result is None:
            return
        frame = int(frame_slider.value)
        if frame not in frame_cache:
            frame_cache[frame] = render_simulation_motion_frame_png(active_result, frame)
        motion_image.value = frame_cache[frame]
        if frame >= frame_slider.max and not diagnostics_ready:
            diagnostics_image.value = render_reaction_diagnostics_png(active_result)
            diagnostics_ready = True

    def render_diagnostics(*_: object) -> None:
        nonlocal diagnostics_ready
        if active_result is None:
            return
        diagnostics_image.value = render_reaction_diagnostics_png(active_result)
        diagnostics_ready = True
        status.value = (
            "<b>Analysis updated:</b> diagnostics now reflect the latest simulation settings "
            "(phase, mixture, volume, molarity, temperature, pressure, and reaction toggle), "
            "with product rates from integrated reaction kinetics."
        )

    for control in [
        phase,
        volume,
        temperature,
        pressure,
        molarity,
        reactions_enabled,
        species1,
        species2,
        species3,
        count1,
        count2,
        count3,
    ]:
        control.observe(rebuild_simulation, names="value")
    frame_slider.observe(render_frame, names="value")
    analyze_button.on_click(render_diagnostics)

    rebuild_simulation()
    control_grid = widgets.VBox(
        [
            supported,
            status,
            phase,
            widgets.HBox([volume, molarity, temperature, pressure, reactions_enabled]),
            widgets.HBox([species1, count1]),
            widgets.HBox([species2, count2]),
            widgets.HBox([species3, count3]),
            widgets.HBox([play, frame_slider, analyze_button]),
            motion_image,
            diagnostics_image,
        ]
    )
    display(control_grid)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ui", action="store_true", help="Run a headless simulation and print a JSON summary.")
    parser.add_argument("--phase", choices=sorted(PHASE_PRESETS), default="liquid")
    parser.add_argument("--mixture", default=DEFAULT_MIXTURE, help="Comma-separated Molecule:count entries, e.g. H2O:6,N2:4")
    parser.add_argument("--volume", type=float, default=_reference_volume_liters(sum(parse_mixture(DEFAULT_MIXTURE).values()), 1.0), help="Volume in liters")
    parser.add_argument("--molarity", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in kelvin")
    parser.add_argument("--pressure", type=float, default=101325.0, help="Pressure in pascals")
    parser.add_argument("--disable-reactions", action="store_true")
    args = parser.parse_args()

    mixture = parse_mixture(args.mixture)
    result = simulate_molecular_dynamics(
        mixture=mixture,
        phase=args.phase,
        volume_liters=args.volume,
        molarity=args.molarity,
        temperature_kelvin=args.temperature,
        pressure_pascal=args.pressure,
        reactions_enabled=not args.disable_reactions,
    )

    if args.no_ui:
        print(json.dumps(summarize_simulation(result), indent=2))
        return

    launch_interactive_ui()


if __name__ == "__main__":
    main()
