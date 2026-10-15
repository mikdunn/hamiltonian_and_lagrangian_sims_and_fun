"""Bounded organic product rules and an explicitly illustrative kinetic model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
import io
import math

import numpy as np
from scipy.integrate import solve_ivp

ELEMENTS = ("C", "H", "N", "O", "S", "F", "Cl", "Br")
CATALOG = {
    "Acetone": "CC(=O)C",
    "Propanal": "CCC=O",
    "Acetaldehyde": "CC=O",
    "Butan-2-one": "CCC(=O)C",
    "Cyclopentanone": "O=C1CCCC1",
    "Cyclohexanone": "O=C1CCCCC1",
    "Benzaldehyde": "O=Cc1ccccc1",
    "Acetic acid": "CC(=O)O",
    "Propionic acid": "CCC(=O)O",
    "Benzoic acid": "O=C(O)c1ccccc1",
    "Methanol": "CO",
    "Ethanol": "CCO",
    "Propan-1-ol": "CCCO",
    "Propan-2-ol": "CC(O)C",
    "Cyclopentanol": "OC1CCCC1",
    "Cyclohexanol": "OC1CCCCC1",
    "Dimethyl ether": "COC",
    "Ethylamine": "CCN",
    "Aniline": "Nc1ccccc1",
    "Ethane": "CC",
    "Cyclohexane": "C1CCCCC1",
    "Benzene": "c1ccccc1",
    "Fluoroethane": "CCF",
    "Chloroethane": "CCCl",
    "Bromoethane": "CCBr",
    "Ethanethiol": "CCS",
}

RULES = {
    "Aldol addition": (
        "[C;H1,H2,H3:1][C:2](=[O:3])[#6:4]."
        "[C;H1:5](=[O:6])[#6:7]>>"
        "[C:1]([C:5]([OH:6])[#6:7])[C:2](=[O:3])[#6:4]"
    ),
    "Esterification": (
        "[C:1](=[O:2])[O;H1:3].[O;H1:4][C;!$(C=O);!a:5]>>"
        "[C:1](=[O:2])[O:4][C:5].[OH2:3]"
    ),
}


def chemistry():
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            'Organic structures require RDKit in this notebook kernel. '
            'Run `%pip install "rdkit>=2024.9"`, restart, and rerun this cell.'
        ) from exc
    return Chem


def molecule(smiles: str):
    Chem = chemistry()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}.")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Each reactant must be a single connected molecule.")
    if mol.GetNumHeavyAtoms() > 24:
        raise ValueError("Interactive structures support at most 24 heavy atoms.")
    if any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        raise ValueError("These starter rules support neutral, closed-shell molecules only.")
    if any(atom.GetSymbol() not in ELEMENTS for atom in mol.GetAtoms()):
        raise ValueError(f"Supported elements: {', '.join(ELEMENTS)}.")
    if not any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
        raise ValueError("An organic reactant must contain carbon.")
    return mol


def atom_counts(smiles: str) -> dict[str, int]:
    mol = chemistry().AddHs(molecule(smiles))
    return dict(Counter(atom.GetSymbol() for atom in mol.GetAtoms()))


def formula(smiles: str) -> str:
    from rdkit.Chem import rdMolDescriptors

    return rdMolDescriptors.CalcMolFormula(molecule(smiles))


def matching_structures(counts: dict[str, int], ring: bool) -> list[tuple[str, str]]:
    if any(
        element not in ELEMENTS or not isinstance(count, int) or count < 0
        for element, count in counts.items()
    ):
        raise ValueError("Atom counts must be nonnegative integers for supported elements.")
    wanted = {element: count for element, count in counts.items() if count}
    return [
        (name, smiles) for name, smiles in CATALOG.items()
        if atom_counts(smiles) == wanted
        and bool(molecule(smiles).GetRingInfo().NumRings()) == ring
    ]


@dataclass(frozen=True)
class ProductChannel:
    family: str
    substrate: int
    partner: int
    product: str
    coproducts: tuple[str, ...]


def calculate_products(reactants: list[str], family: str) -> list[ProductChannel]:
    """Enumerate distinct regio-products of one declared reaction family."""
    from rdkit.Chem import rdChemReactions

    if not 2 <= len(reactants) <= 4:
        raise ValueError("Select two to four reactant molecules.")
    if family not in RULES:
        raise ValueError(f"Supported reaction families: {', '.join(RULES)}.")
    Chem = chemistry()
    molecules = [molecule(smiles) for smiles in reactants]
    canonical = [Chem.MolToSmiles(mol) for mol in molecules]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Combine duplicate structures into one reactant's concentration.")
    reaction = rdChemReactions.ReactionFromSmarts(RULES[family])
    if reaction is None:
        raise RuntimeError(f"Could not initialize the {family} rule.")
    found: dict[tuple[int, int, str, tuple[str, ...]], ProductChannel] = {}
    for first, second in permutations(range(len(molecules)), 2):
        for product_set in reaction.RunReactants((molecules[first], molecules[second])):
            for product_mol in product_set:
                Chem.SanitizeMol(product_mol)
            strings = tuple(Chem.MolToSmiles(mol) for mol in product_set)
            expected = Counter(atom_counts(reactants[first]))
            expected.update(atom_counts(reactants[second]))
            actual: Counter[str] = Counter()
            for product_mol in product_set:
                actual.update(atom.GetSymbol() for atom in Chem.AddHs(product_mol).GetAtoms())
            if expected != actual:
                raise RuntimeError(f"The {family} rule produced an atom-unbalanced product.")
            key = (first, second, strings[0], strings[1:])
            found[key] = ProductChannel(family, first, second, strings[0], strings[1:])
    if not found:
        raise ValueError(
            f"No supported {family} pairing was found. Aldol requires an "
            "alpha-H ketone and an aldehyde; esterification requires a "
            "carboxylic acid and an aliphatic alcohol. This is not proof that "
            "the molecules cannot react by a different mechanism."
        )
    if len(found) > 32:
        raise ValueError("More than 32 channels: simplify the input structures.")
    return list(found.values())


@dataclass(frozen=True)
class ModelConditions:
    temperature_k: float = 310.0
    duration_min: float = 60.0
    partner_multiplier: float = 1.0


@dataclass(frozen=True)
class NetworkRun:
    time_min: np.ndarray
    reactants_m: np.ndarray
    products_m: np.ndarray
    loss_m: np.ndarray
    initial_m: np.ndarray
    target_yield: float
    conditions: ModelConditions


def simulate_network(
    reactants: list[str],
    initial_m: list[float],
    channels: list[ProductChannel],
    target: int = 0,
    conditions: ModelConditions = ModelConditions(),
    n_points: int = 61,
) -> NetworkRun:
    """Competing 1:1 channels plus a tracked lumped product-loss pool."""
    initial = np.asarray(initial_m, dtype=float)
    if len(initial) != len(reactants) or not np.isfinite(initial).all() or np.any(initial <= 0):
        raise ValueError("Every reactant needs a finite positive concentration in mol/L.")
    if not channels or not 0 <= target < len(channels):
        raise ValueError("Select a valid target product channel.")
    if not 2 <= len(reactants) <= 4 or not isinstance(n_points, int) or n_points < 2:
        raise ValueError("Use two to four reactants and at least two output frames.")
    t, duration, multiplier = (
        conditions.temperature_k, conditions.duration_min, conditions.partner_multiplier
    )
    if not all(math.isfinite(value) for value in (t, duration, multiplier)):
        raise ValueError("Model conditions must be finite.")
    if not 280 <= t <= 380 or not 1 <= duration <= 120 or not 0.5 <= multiplier <= 3:
        raise ValueError("Model bounds: 280-380 K, 1-120 min, partner multiplier 0.5-3.")
    for channel in channels:
        if channel.family not in RULES or not (
            0 <= channel.substrate < len(initial)
            and 0 <= channel.partner < len(initial)
            and channel.substrate != channel.partner
        ):
            raise ValueError("Channel family or reactant indices are invalid.")
    initial = initial.copy()
    initial[channels[target].partner] *= multiplier
    r_count, p_count = len(initial), len(channels)
    formation = 0.035 * math.exp(35000 / 8.314 * (1 / 310 - 1 / t))
    decay = 0.0015 * math.exp(70000 / 8.314 * (1 / 310 - 1 / t))
    branches = Counter((c.substrate, c.partner) for c in channels)

    def rhs(_time, state):
        del _time
        derivative = np.zeros_like(state)
        for index, channel in enumerate(channels):
            forward = (
                formation * state[channel.substrate] * state[channel.partner]
                / branches[(channel.substrate, channel.partner)]
            )
            loss = decay * state[r_count + index]
            derivative[channel.substrate] -= forward
            derivative[channel.partner] -= forward
            derivative[r_count + index] += forward - loss
            derivative[r_count + p_count + index] += loss
        return derivative

    solution = solve_ivp(
        rhs, (0.0, duration), np.concatenate((initial, np.zeros(2 * p_count))),
        t_eval=np.linspace(0.0, duration, n_points), rtol=1e-8, atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Organic network integration failed: {solution.message}")
    if solution.y.min() < -1e-7:
        raise RuntimeError("Integration produced negative concentrations beyond tolerance.")
    product = solution.y[r_count:r_count + p_count].T
    lost = solution.y[r_count + p_count:].T
    remaining = solution.y[:r_count].T
    for reactant_index in range(r_count):
        consumed = sum(
            (product[:, index] + lost[:, index] for index, channel in enumerate(channels)
             if reactant_index in (channel.substrate, channel.partner)),
            np.zeros(len(solution.t)),
        )
        if not np.allclose(remaining[:, reactant_index] + consumed, initial[reactant_index], atol=1e-7):
            raise RuntimeError("Reactant/product/loss material balance failed.")
    target_yield = float(product[-1, target] / initial[channels[target].substrate])
    if not -1e-7 <= target_yield <= 1 + 1e-7:
        raise RuntimeError("Target yield lies outside its substrate-normalized bounds.")
    return NetworkRun(solution.t, remaining, product, lost, initial, target_yield, conditions)


def optimize_model_yield(
    reactants: list[str], initial_m: list[float], channels: list[ProductChannel],
    target: int, baseline: ModelConditions,
) -> tuple[NetworkRun, int]:
    """Finite search, not an experimental optimum or a continuous guarantee."""
    best = simulate_network(reactants, initial_m, channels, target, baseline)
    evaluated = 1
    for temperature in (280.0, 300.0, 320.0, 340.0, 360.0, 380.0):
        for multiplier in (0.5, 1.0, 1.5, 2.0, 3.0):
            for duration in (10.0, 30.0, 60.0, 90.0, 120.0):
                run = simulate_network(
                    reactants, initial_m, channels, target,
                    ModelConditions(temperature, duration, multiplier), n_points=2,
                )
                evaluated += 1
                if run.target_yield > best.target_yield:
                    best = run
    return (
        simulate_network(reactants, initial_m, channels, target, best.conditions),
        evaluated,
    )


def structure_png(smiles: str) -> bytes:
    from rdkit.Chem import Draw

    image = Draw.MolToImage(molecule(smiles), size=(260, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def geometry(smiles: str) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    from rdkit.Chem import AllChem

    Chem = chemistry()
    mol = Chem.AddHs(molecule(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 23
    if AllChem.EmbedMolecule(mol, parameters) != 0:
        raise RuntimeError(f"Unable to embed molecular geometry for {smiles}.")
    if not AllChem.UFFHasAllMoleculeParams(mol):
        raise ValueError(f"No complete UFF geometry parameters for {smiles}.")
    converged = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    if converged != 0:
        raise RuntimeError(f"UFF geometry minimization did not converge for {smiles}.")
    xyz = np.asarray(mol.GetConformer().GetPositions())
    xyz -= xyz.mean(axis=0)
    return (
        xyz, [atom.GetSymbol() for atom in mol.GetAtoms()],
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()],
    )


def motion_png(
    reactant_geometry: list[tuple], product_geometry: tuple,
    run: NetworkRun, target: int, frame: int,
) -> bytes:
    """Schematic cluster approach/crossfade, explicitly not atomistic dynamics."""
    import matplotlib.pyplot as plt

    if not 0 <= frame < len(run.time_min):
        raise ValueError("Frame is outside the computed trajectory.")
    fraction = frame / (len(run.time_min) - 1)
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    colors = {
        "C": "#555555", "H": "#cccccc", "O": "#d84b4b", "N": "#4477dd",
        "S": "#ddbb22", "F": "#55bb55", "Cl": "#22aa44", "Br": "#996633",
    }

    def draw(shape, offset, opacity):
        xyz, symbols, bonds = shape
        xyz = xyz + offset
        for first, second in bonds:
            ax.plot(*xyz[[first, second]].T, color="gray", alpha=opacity)
        ax.scatter(*xyz.T, c=[colors[s] for s in symbols], s=60, alpha=opacity)

    extent = max(
        10.0,
        max(float(np.abs(shape[0]).max()) for shape in [*reactant_geometry, product_geometry]) + 8.0,
    )
    for index, shape in enumerate(reactant_geometry):
        angle = 2 * math.pi * index / len(reactant_geometry)
        offset = np.array([math.cos(angle), math.sin(angle), 0]) * (6 - 3 * fraction)
        opacity = max(0.08, float(run.reactants_m[frame, index] / run.initial_m[index]))
        draw(shape, offset, opacity)
    product_max = float(run.products_m[:, target].max())
    if product_max > 0 and run.products_m[frame, target] > 0:
        draw(product_geometry, np.zeros(3), max(0.08, float(run.products_m[frame, target] / product_max)))
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
    ax.set_title(f"Schematic molecular playback | {run.time_min[frame]:.1f} min\n"
                 "Not a collision, barrier, orbital or bond-force simulation")
    ax.set_axis_off()
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    return buffer.getvalue()


def build_organic_product_notebook():
    """Molecule builders, declared reaction rules, modeled yield and playback."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from html import escape
    from IPython.display import display

    chemistry()
    output = widgets.Output(layout=widgets.Layout(border="1px solid #bbb", min_height="100px"))
    motion = widgets.Image(format="png")
    product_image = widgets.Image(format="png")
    diagnostics = widgets.Image(format="png")
    status = widgets.HTML()
    family = widgets.Dropdown(options=list(RULES), value="Esterification", description="Reaction")
    temperature = widgets.FloatSlider(value=310, min=280, max=380, step=5, description="T (K)", continuous_update=False)
    duration = widgets.IntSlider(value=60, min=1, max=120, description="Time (min)", continuous_update=False)
    ratio = widgets.FloatSlider(value=1, min=0.5, max=3, step=0.1, description="Partner x", continuous_update=False)
    calculate = widgets.Button(description="Calculate products", button_style="info")
    simulate = widgets.Button(description="Simulate selected product", button_style="primary")
    optimize = widgets.Button(description="Optimize MODEL yield", button_style="warning")
    target = widgets.Dropdown(options=[], description="Target", layout=widgets.Layout(width="95%"))
    play = widgets.Play(min=0, max=60, interval=100)
    frame = widgets.IntSlider(value=0, min=0, max=60, description="Frame")
    widgets.jslink((play, "value"), (frame, "value"))
    slots = []
    state: dict = {}

    def invalidate(*_):
        state.clear()
        target.options = []
        product_image.value = b""
        motion.value = b""
        diagnostics.value = b""
        play.value = 0
        frame.value = 0
        status.value = "<b>Inputs changed:</b> calculate products again."

    def make_slot(number, preset_name, active):
        enabled = widgets.Checkbox(value=active, description=f"Use reactant {number}")
        preset = widgets.Dropdown(options=list(CATALOG), value=preset_name, description="Example")
        counts = {
            element: widgets.BoundedIntText(value=0, min=0, max=60, description=element,
                                           layout=widgets.Layout(width="110px"))
            for element in ELEMENTS
        }
        ring = widgets.Checkbox(description="Contains ring")
        choices = widgets.Dropdown(description="Isomer", layout=widgets.Layout(width="450px"))
        custom = widgets.Text(description="Custom SMILES", placeholder="Optional explicit structure", layout=widgets.Layout(width="450px"))
        amount = widgets.FloatSlider(value=1, min=0.1, max=2, step=0.1, description="Feed (M)", continuous_update=False)
        image = widgets.Image(format="png")
        info = widgets.HTML()
        updating = False

        @output.capture(clear_output=True, wait=True)
        def refresh(*_):
            if updating:
                return
            invalidate()
            if custom.value.strip():
                choices.options = []
                image.value = b""
                smiles = custom.value.strip()
                mol = molecule(smiles)
                actual = atom_counts(smiles)
                expected = {e: control.value for e, control in counts.items() if control.value}
                if actual != expected or bool(mol.GetRingInfo().NumRings()) != ring.value:
                    choices.options = []
                    image.value = b""
                    info.value = "<b>Custom structure does not match the atom counts/ring preference.</b>"
                    return
                choices.options = [(f"Custom {formula(smiles)}", smiles)]
            else:
                previous = choices.value
                options = matching_structures({e: control.value for e, control in counts.items()}, ring.value)
                choices.options = [(f"{name}: {smiles}", smiles) for name, smiles in options]
                if previous in [smiles for _, smiles in options]:
                    choices.value = previous
            if choices.value is None:
                image.value = b""
                info.value = "<b>No catalog match.</b> Counts do not define connectivity; enter a matching custom SMILES or choose an example."
            else:
                image.value = structure_png(choices.value)
                info.value = f"<b>{escape(formula(choices.value))}</b> - choose an explicit isomer; catalog is not exhaustive."

        @output.capture(clear_output=True, wait=True)
        def use_preset(*_):
            nonlocal updating
            updating = True
            try:
                smiles = CATALOG[preset.value]
                actual = atom_counts(smiles)
                for element, control in counts.items():
                    control.value = actual.get(element, 0)
                ring.value = bool(molecule(smiles).GetRingInfo().NumRings())
                custom.value = ""
            finally:
                updating = False
            refresh()
            choices.value = smiles
            image.value = structure_png(smiles)

        for control in [*counts.values(), ring, custom]:
            control.observe(refresh, names="value")
        for control in [enabled, amount]:
            control.observe(invalidate, names="value")

        @output.capture(clear_output=True, wait=True)
        def choose(*_):
            invalidate()
            if choices.value is not None:
                image.value = structure_png(choices.value)

        choices.observe(choose, names="value")
        preset.observe(use_preset, names="value")
        use_preset()
        box = widgets.VBox([
            enabled, preset, widgets.HBox(list(counts.values())[:4]),
            widgets.HBox(list(counts.values())[4:]), ring, choices, custom, amount, info, image,
        ])
        slots.append((enabled, choices, amount, box))

    for number, (name, active) in enumerate(
        [("Acetic acid", True), ("Ethanol", True), ("Methanol", False), ("Acetone", False)], 1
    ):
        make_slot(number, name, active)

    def inputs():
        selected = [slot for slot in slots if slot[0].value]
        if not 2 <= len(selected) <= 4:
            raise ValueError("Enable two to four reactants.")
        if any(slot[1].value is None for slot in selected):
            raise ValueError("Every enabled reactant needs a matching explicit structure.")
        return [slot[1].value for slot in selected], [float(slot[2].value) for slot in selected]

    @output.capture(clear_output=True, wait=True)
    def calculate_callback(_=None):
        invalidate()
        smiles, amounts = inputs()
        channels = calculate_products(smiles, family.value)
        labels = [f"R{index + 1}" for index, slot in enumerate(slots) if slot[0].value]
        state.update(smiles=smiles, amounts=amounts, channels=channels, labels=labels)
        target.options = [
            (f"{index + 1}: {labels[c.substrate]} + {labels[c.partner]} -> {formula(c.product)} | {c.product}", index)
            for index, c in enumerate(channels)
        ]
        target.value = 0
        product_image.value = structure_png(channels[0].product)
        status.value = f"<b>{len(channels)} rule-based candidate(s).</b> Select a target; all candidates compete in the model."
        print(f"Declared family: {family.value}. No mechanism or selectivity inference.")
        for index, channel in enumerate(channels):
            print(f"{index + 1}. {channel.product}; coproducts: {', '.join(channel.coproducts) or 'none'}")
        print("Stereochemistry and catalyst/solvent effects are not predicted.")

    @output.capture()
    def render_frame(*_):
        if "run" in state:
            motion.value = motion_png(
                state["geometry"], state["product_geometry"], state["run"],
                state["target"], int(frame.value),
            )

    @output.capture(clear_output=True, wait=True)
    def run_callback(_=None, *, search=False):
        if "channels" not in state or target.value is None:
            raise ValueError("Calculate products and select a target first.")
        conditions = ModelConditions(float(temperature.value), float(duration.value), float(ratio.value))
        baseline = simulate_network(state["smiles"], state["amounts"], state["channels"], target.value, conditions)
        run = baseline
        if search:
            run, evaluations = optimize_model_yield(
                state["smiles"], state["amounts"], state["channels"], target.value, conditions,
            )
            print(f"Finite search: {evaluations} runs including baseline; "
                  f"target MODEL yield {100 * baseline.target_yield:.2f}% -> {100 * run.target_yield:.2f}%")
            print(f"Best searched settings: {run.conditions}")
            temperature.value = run.conditions.temperature_k
            duration.value = int(run.conditions.duration_min)
            ratio.value = run.conditions.partner_multiplier
        state.update(
            run=run, target=target.value,
            geometry=[geometry(smiles) for smiles in state["smiles"]],
            product_geometry=geometry(state["channels"][target.value].product),
        )
        play.value = 0
        frame.value = 0
        render_frame()
        fig, ax = plt.subplots(figsize=(9, 4))
        for index in range(len(state["smiles"])):
            ax.plot(run.time_min, run.reactants_m[:, index], label=state["labels"][index])
        for index in range(len(state["channels"])):
            ax.plot(run.time_min, run.products_m[:, index], label=f"P{index + 1}")
        if family.value == "Esterification":
            ax.plot(run.time_min, (run.products_m + run.loss_m).sum(axis=1), ":", label="Water coproduct")
        ax.plot(run.time_min, run.loss_m.sum(axis=1), "--", label="Lumped product-loss pool")
        ax.set(xlabel="Time (min)", ylabel="Concentration (M)", title="Illustrative competing reaction network")
        ax.legend()
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        plt.close(fig)
        diagnostics.value = buffer.getvalue()
        status.value = (
            f"<b>Target MODEL yield: {100 * run.target_yield:.2f}%</b> relative to initial target substrate. "
            f"T={run.conditions.temperature_k:g} K, time={run.conditions.duration_min:g} min, "
            f"partner multiplier={run.conditions.partner_multiplier:g}. "
            "Play shows schematic clusters, not molecular dynamics."
        )
        print("Fixed 1 L volume. Optimizer changes target partner feed, temperature and duration.")
        print("Actual modeled feed (M):", dict(zip(state["labels"], run.initial_m.tolist())))
        print("All rate/barrier/loss constants and equal regio-branching are illustrative. "
              "Not an experimental yield, synthesis recommendation, or equilibrium calculation.")
        print("Loss is a tracked product-equivalent pool, not a predicted degradation molecule.")

    @output.capture(clear_output=True, wait=True)
    def optimize_callback(_):
        optimize.disabled = True
        try:
            run_callback(search=True)
        finally:
            optimize.disabled = False

    def clear_run(*_):
        for key in ("run", "geometry", "product_geometry", "target"):
            state.pop(key, None)
        motion.value = b""
        diagnostics.value = b""
        play.value = 0
        frame.value = 0
        status.value = "<b>Target/conditions changed:</b> simulate or optimize again."
        if "channels" in state and target.value is not None:
            product_image.value = structure_png(state["channels"][target.value].product)

    family.observe(invalidate, names="value")
    for control in (temperature, duration, ratio, target):
        control.observe(clear_run, names="value")
    frame.observe(render_frame, names="value")
    calculate.on_click(calculate_callback)
    simulate.on_click(run_callback)
    optimize.on_click(optimize_callback)
    accordion = widgets.Accordion(children=[slot[3] for slot in slots])
    for index in range(len(slots)):
        accordion.set_title(index, f"Reactant {index + 1}")
    display(widgets.VBox([
        widgets.HTML(value="<b>Bounded organic product calculator</b><br>"
                     "Atom counts + rings filter a small structure catalog; select an isomer. "
                     "A declared reaction family supplies product rules, not universal prediction."),
        accordion, family, calculate, target, product_image,
        widgets.HBox([temperature, duration, ratio]),
        widgets.HBox([simulate, optimize]), status,
        widgets.HBox([play, frame]), widgets.HTML(value="<b>Schematic molecular playback</b>"),
        motion, widgets.HTML(value="<b>Concentration and MODEL-yield diagnostics</b>"),
        diagnostics, output,
    ]))
    calculate_callback()
    run_callback()
