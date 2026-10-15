"""A small, condition-explicit acetone aldol reaction-network demonstrator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

ACETONE_SMILES = "CC(=O)C"
ACETALDEHYDE_SMILES = "CC=O"
ALDOL_PRODUCT_SMILES = "CC(=O)CC(O)C"
ACETONE_ALPHA_PKA_AQUEOUS = 19.2


def alpha_carbon_indices(smiles: str) -> tuple[int, ...]:
    """Return graph indices of carbon atoms alpha to a carbonyl and bearing H."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "SMILES graph analysis requires RDKit in the notebook kernel. Run "
            "`%pip install \"rdkit>=2024.9\"` in a notebook cell, then restart "
            "the kernel and rerun the simulator cell."
        ) from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}.")

    carbonyl_carbons = set()
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        if first.GetAtomicNum() == 6 and second.GetAtomicNum() == 8:
            carbonyl_carbons.add(first.GetIdx())
        elif first.GetAtomicNum() == 8 and second.GetAtomicNum() == 6:
            carbonyl_carbons.add(second.GetIdx())

    alpha_carbons = set()
    for carbonyl_index in carbonyl_carbons:
        carbonyl = molecule.GetAtomWithIdx(carbonyl_index)
        for neighbor in carbonyl.GetNeighbors():
            if (
                neighbor.GetAtomicNum() == 6
                and neighbor.GetTotalNumHs(includeNeighbors=True) > 0
            ):
                alpha_carbons.add(neighbor.GetIdx())
    return tuple(sorted(alpha_carbons))


@dataclass(frozen=True)
class AldolConditions:
    """Aqueous pH and an explicitly supplied effective C-C addition rate."""

    pH: float = 14.0
    k_addition_l_mol_s: float = 0.1
    acetone_alpha_pKa: float = ACETONE_ALPHA_PKA_AQUEOUS


@dataclass(frozen=True)
class ReactionTrajectory:
    time_s: np.ndarray
    concentrations_mol_l: np.ndarray
    enolate_fraction: float

    @property
    def species(self) -> tuple[str, str, str]:
        return ("acetone", "acetaldehyde", "aldol_product")


def enolate_fraction(conditions: AldolConditions) -> float:
    """Ideal dilute-solution Henderson-Hasselbalch fraction for acetone."""
    values = (
        conditions.pH,
        conditions.k_addition_l_mol_s,
        conditions.acetone_alpha_pKa,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pH, pKa, and rate constant must be finite.")
    if not 0.0 <= conditions.pH <= 14.0:
        raise ValueError("This aqueous approximation requires pH from 0 to 14.")
    if conditions.k_addition_l_mol_s < 0.0:
        raise ValueError("The addition rate constant cannot be negative.")

    log_ratio = (conditions.pH - conditions.acetone_alpha_pKa) * math.log(10.0)
    if log_ratio >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_ratio))
    ratio = math.exp(log_ratio)
    return ratio / (1.0 + ratio)


def simulate_aldol(
    conditions: AldolConditions = AldolConditions(),
    *,
    acetone_initial_mol_l: float = 1.0,
    acetaldehyde_initial_mol_l: float = 1.0,
    duration_s: float = 3600.0,
    n_points: int = 181,
) -> ReactionTrajectory:
    """Integrate the lumped 1:1 acetone/acetaldehyde aldol concentration model."""
    alpha = enolate_fraction(conditions)
    concentrations = (acetone_initial_mol_l, acetaldehyde_initial_mol_l)
    if not all(math.isfinite(value) and value >= 0.0 for value in concentrations):
        raise ValueError("Initial concentrations must be finite and nonnegative.")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("Duration must be a finite positive number of seconds.")
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("At least two output time points are required.")

    effective_rate = conditions.k_addition_l_mol_s * alpha

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        rate = effective_rate * state[0] * state[1]
        return np.array((-rate, -rate, rate))

    times = np.linspace(0.0, duration_s, n_points)
    initial = np.array((acetone_initial_mol_l, acetaldehyde_initial_mol_l, 0.0))
    solution = solve_ivp(
        rhs,
        (0.0, duration_s),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Aldol integration failed: {solution.message}")
    return ReactionTrajectory(
        time_s=solution.t,
        concentrations_mol_l=solution.y.T,
        enolate_fraction=alpha,
    )


@dataclass(frozen=True)
class LiftedDMDModel:
    """Control-aware lifted generator fit from concentration trajectories."""

    feature_scales: np.ndarray
    generator: np.ndarray

    def derivative(
        self, state: np.ndarray, conditions: AldolConditions
    ) -> np.ndarray:
        alpha = enolate_fraction(conditions)
        effective_rate = conditions.k_addition_l_mol_s * alpha
        features = np.array(
            (state[0], state[1], state[2], effective_rate * state[0] * state[1])
        )
        return (features / self.feature_scales) @ self.generator


def fit_lifted_dmd(
    *,
    seed: int = 7,
    n_trajectories: int = 24,
    n_snapshots: int = 81,
    training_duration_s: float = 1800.0,
) -> LiftedDMDModel:
    """Fit an extended-DMD vector field on varied simulated training runs."""
    if n_trajectories < 4 or n_snapshots < 5:
        raise ValueError("DMD training needs at least 4 runs and 5 snapshots.")
    if not math.isfinite(training_duration_s) or training_duration_s <= 0.0:
        raise ValueError("Training duration must be finite and positive.")

    rng = np.random.default_rng(seed)
    feature_sets = []
    derivative_sets = []
    for _ in range(n_trajectories):
        conditions = AldolConditions(
            pH=float(rng.uniform(12.0, 14.0)),
            k_addition_l_mol_s=float(np.exp(rng.uniform(math.log(0.05), math.log(2.0)))),
        )
        trajectory = simulate_aldol(
            conditions,
            acetone_initial_mol_l=float(rng.uniform(0.3, 1.5)),
            acetaldehyde_initial_mol_l=float(rng.uniform(0.3, 1.5)),
            duration_s=training_duration_s,
            n_points=n_snapshots,
        )
        states = trajectory.concentrations_mol_l
        effective_rate = conditions.k_addition_l_mol_s * trajectory.enolate_fraction
        features = np.column_stack(
            (states, effective_rate * states[:, 0] * states[:, 1])
        )
        derivatives = np.gradient(
            states,
            trajectory.time_s,
            axis=0,
            edge_order=2,
        )
        feature_sets.append(features)
        derivative_sets.append(derivatives)

    all_features = np.vstack(feature_sets)
    all_derivatives = np.vstack(derivative_sets)
    scales = np.linalg.norm(all_features, axis=0)
    if np.any(scales <= 0.0):
        raise RuntimeError("DMD training data did not span every lifted feature.")
    generator, _, rank, _ = np.linalg.lstsq(
        all_features / scales,
        all_derivatives,
        rcond=None,
    )
    if rank < all_features.shape[1]:
        raise RuntimeError(
            "DMD training trajectories do not span the lifted feature space."
        )
    return LiftedDMDModel(feature_scales=scales, generator=generator)


def predict_with_dmd(
    model: LiftedDMDModel,
    conditions: AldolConditions,
    *,
    acetone_initial_mol_l: float = 1.0,
    acetaldehyde_initial_mol_l: float = 1.0,
    duration_s: float = 3600.0,
    n_points: int = 181,
) -> ReactionTrajectory:
    """Roll out the fitted lifted generator for one held-out condition."""
    enolate = enolate_fraction(conditions)
    initial_values = (acetone_initial_mol_l, acetaldehyde_initial_mol_l)
    if not all(math.isfinite(value) and value >= 0.0 for value in initial_values):
        raise ValueError("Initial concentrations must be finite and nonnegative.")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("Duration must be finite and positive.")
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("At least two output time points are required.")

    times = np.linspace(0.0, duration_s, n_points)
    initial = np.array((acetone_initial_mol_l, acetaldehyde_initial_mol_l, 0.0))

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return model.derivative(state, conditions)

    solution = solve_ivp(
        rhs,
        (0.0, duration_s),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"Lifted-DMD rollout failed: {solution.message}")
    return ReactionTrajectory(times, solution.y.T, enolate)


def validate_lifted_dmd(
    model: LiftedDMDModel,
    *,
    n_cases: int = 8,
    seed: int = 19,
    duration_s: float = 1200.0,
    n_points: int = 61,
) -> dict[str, float]:
    """Compare held-out surrogate trajectories with the kinetic ODE."""
    if n_cases < 1:
        raise ValueError("At least one held-out validation case is required.")
    rng = np.random.default_rng(seed)
    normalized_errors = []
    minimum_concentration = math.inf
    maximum_stoichiometric_error = 0.0
    for _ in range(n_cases):
        conditions = AldolConditions(
            pH=float(rng.uniform(12.05, 13.95)),
            k_addition_l_mol_s=float(np.exp(rng.uniform(math.log(0.06), math.log(1.8)))),
        )
        acetone_initial = float(rng.uniform(0.35, 1.4))
        aldehyde_initial = float(rng.uniform(0.35, 1.4))
        reference = simulate_aldol(
            conditions,
            acetone_initial_mol_l=acetone_initial,
            acetaldehyde_initial_mol_l=aldehyde_initial,
            duration_s=duration_s,
            n_points=n_points,
        ).concentrations_mol_l
        predicted = predict_with_dmd(
            model,
            conditions,
            acetone_initial_mol_l=acetone_initial,
            acetaldehyde_initial_mol_l=aldehyde_initial,
            duration_s=duration_s,
            n_points=n_points,
        ).concentrations_mol_l
        conversion = max(float(reference[-1, 2]), 1e-10)
        normalized_errors.append(
            float(np.sqrt(np.mean((predicted - reference) ** 2)) / conversion)
        )
        minimum_concentration = min(
            minimum_concentration,
            float(predicted.min()),
        )
        balance_error = max(
            abs((acetone_initial - predicted[-1, 0]) - predicted[-1, 2]),
            abs((aldehyde_initial - predicted[-1, 1]) - predicted[-1, 2]),
        )
        maximum_stoichiometric_error = max(
            maximum_stoichiometric_error,
            float(balance_error),
        )
    return {
        "mean_normalized_rmse": float(np.mean(normalized_errors)),
        "max_normalized_rmse": float(np.max(normalized_errors)),
        "minimum_predicted_concentration_mol_l": minimum_concentration,
        "max_stoichiometric_error_mol_l": maximum_stoichiometric_error,
    }


def build_organic_reaction_notebook():
    """Display the acetone enolate/aldol kinetics and held-out DMD comparison."""
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    acetone_alpha = alpha_carbon_indices(ACETONE_SMILES)
    aldehyde_alpha = alpha_carbon_indices(ACETALDEHYDE_SMILES)
    model = fit_lifted_dmd()
    validation = validate_lifted_dmd(model)
    pH = widgets.FloatSlider(
        value=14.0, min=12.0, max=14.0, step=0.1, description="Aqueous pH"
    )
    rate = widgets.FloatLogSlider(
        value=0.1,
        base=10,
        min=-2,
        max=1,
        step=0.1,
        description="k add (L/mol/s)",
        readout_format=".3g",
    )
    duration = widgets.IntSlider(
        value=3600, min=300, max=7200, step=300, description="Time (s)"
    )
    button = widgets.Button(description="Simulate and compare DMD", button_style="primary")
    output = widgets.Output(
        layout=widgets.Layout(border="1px solid #bbb", min_height="120px")
    )

    @output.capture(clear_output=True, wait=True)
    def _run(button_event):
        del button_event
        with output:
            clear_output(wait=True)
            conditions = AldolConditions(
                pH=float(pH.value),
                k_addition_l_mol_s=float(rate.value),
            )
            reference = simulate_aldol(conditions, duration_s=float(duration.value))
            surrogate = predict_with_dmd(
                model,
                conditions,
                duration_s=float(duration.value),
            )
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for index, name in enumerate(reference.species):
                ax.plot(
                    reference.time_s,
                    reference.concentrations_mol_l[:, index],
                    label=f"{name} (ODE)",
                )
                ax.plot(
                    surrogate.time_s,
                    surrogate.concentrations_mol_l[:, index],
                    linestyle="--",
                    alpha=0.8,
                    label=f"{name} (lifted DMD)",
                )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Concentration (mol/L)")
            ax.set_title("Acetone enolate-mediated aldol addition (illustrative kinetics)")
            ax.legend(ncol=2)
            ax.grid(alpha=0.2)
            plt.tight_layout()
            display(fig)
            plt.close(fig)
            print(
                "Acetone alpha-carbon atom indices:",
                acetone_alpha,
            )
            print(
                "Acetaldehyde alpha-carbon atom indices:",
                aldehyde_alpha,
            )
            print("Neutral aldol product SMILES:", ALDOL_PRODUCT_SMILES)
            print(f"Estimated equilibrium enolate fraction: {reference.enolate_fraction:.3e}")
            print(
                "Held-out DMD validation (normalized RMSE, positive concentrations, "
                "1:1 stoichiometric balance):",
                validation,
            )
            print(
                "The pKa estimate is aqueous at approximately 25 °C. "
                "The user-set addition rate is illustrative unless independently "
                "replaced by measured or validated data."
            )

    button.on_click(_run)
    display(widgets.VBox([
        pH, rate, duration, button,
        widgets.HTML(value="<b>Reaction trajectory and DMD validation</b>"),
        output,
    ]))
    _run(None)
