from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "organic_product_test", PROJECT / "organic_product_simulator.py"
)
organic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = organic
spec.loader.exec_module(organic)


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "RDKit is optional.")
class OrganicProductTests(unittest.TestCase):
    def test_formula_is_not_a_unique_molecule(self):
        matches = organic.matching_structures({"C": 3, "H": 6, "O": 1}, False)
        self.assertEqual({name for name, _ in matches}, {"Acetone", "Propanal"})

    def test_ring_and_invalid_count_filters(self):
        self.assertEqual(
            organic.matching_structures({"C": 6, "H": 10, "O": 1}, True),
            [("Cyclohexanone", "O=C1CCCCC1")],
        )
        self.assertEqual(organic.matching_structures({"C": 99}, False), [])
        with self.assertRaises(ValueError):
            organic.matching_structures({"C": -1}, False)
        with self.assertRaises(ValueError):
            organic.matching_structures({"Xe": 1}, False)

    def test_all_catalog_structures_are_valid(self):
        for smiles in organic.CATALOG.values():
            self.assertTrue(organic.atom_counts(smiles)["C"])
            self.assertTrue(organic.formula(smiles))

    def test_esterification_product_and_water(self):
        channels = organic.calculate_products(["CC(=O)O", "CCO"], "Esterification")
        self.assertEqual(len(channels), 1)
        self.assertEqual(organic.formula(channels[0].product), "C4H8O2")
        self.assertEqual(channels[0].coproducts, ("O",))

    def test_aldol_symmetric_sites_are_deduplicated(self):
        channels = organic.calculate_products(["CC(=O)C", "CC=O"], "Aldol addition")
        self.assertEqual(len(channels), 1)
        self.assertEqual(organic.formula(channels[0].product), "C5H10O2")
        self.assertEqual(channels[0].coproducts, ())

    def test_ring_and_multiple_regio_products(self):
        channels = organic.calculate_products(
            ["O=C1CCCCC1", "O=Cc1ccccc1"], "Aldol addition"
        )
        self.assertTrue(organic.molecule(channels[0].product).GetRingInfo().NumRings())
        channels = organic.calculate_products(["CCC(=O)C", "CC=O"], "Aldol addition")
        self.assertEqual(len(channels), 2)
        self.assertEqual(len({channel.product for channel in channels}), 2)

    def test_all_supported_product_rules_balance_atoms(self):
        pairs = [
            (["CC(=O)O", "CCO"], "Esterification"),
            (["O=C(O)c1ccccc1", "OC1CCCCC1"], "Esterification"),
            (["CCC(=O)C", "CC=O"], "Aldol addition"),
        ]
        for reactants, family in pairs:
            for channel in organic.calculate_products(reactants, family):
                expected = Counter(organic.atom_counts(reactants[channel.substrate]))
                expected.update(organic.atom_counts(reactants[channel.partner]))
                actual = Counter(organic.atom_counts(channel.product))
                for smiles in channel.coproducts:
                    mol = organic.chemistry().AddHs(organic.chemistry().MolFromSmiles(smiles))
                    actual.update(atom.GetSymbol() for atom in mol.GetAtoms())
                self.assertEqual(expected, actual)

    def test_unsupported_and_duplicate_reactants_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "No supported"):
            organic.calculate_products(["CC", "CCO"], "Esterification")
        with self.assertRaisesRegex(ValueError, "No supported"):
            organic.calculate_products(["CC(=O)O", "CC=O"], "Aldol addition")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            organic.calculate_products(["CCO", "OCC"], "Esterification")
        with self.assertRaises(ValueError):
            organic.calculate_products(["CCO"], "Esterification")
        with self.assertRaisesRegex(ValueError, "Supported reaction"):
            organic.calculate_products(["CCO", "CC"], "Unknown")

    def test_invalid_structures_fail(self):
        for smiles in ("CC.[Na+]", "[CH3]", "C[NH3+]", "nonsense"):
            with self.assertRaises(ValueError):
                organic.molecule(smiles)

    def test_four_reactants_compete_and_preserve_material(self):
        smiles = ["CC(=O)O", "CCC(=O)O", "CCO", "CO"]
        channels = organic.calculate_products(smiles, "Esterification")
        self.assertEqual(len(channels), 4)
        run = organic.simulate_network(smiles, [1, 0.8, 1.1, 0.6], channels)
        self.assertGreaterEqual(run.reactants_m.min(), -1e-8)
        self.assertGreaterEqual(run.products_m.min(), -1e-8)
        for reactant in range(4):
            consumed = sum(
                (run.products_m[:, j] + run.loss_m[:, j] for j, c in enumerate(channels)
                 if reactant in (c.substrate, c.partner)),
                np.zeros(len(run.time_min)),
            )
            np.testing.assert_allclose(
                run.reactants_m[:, reactant] + consumed, run.initial_m[reactant], atol=1e-7
            )

    def test_competitor_reduces_target_yield(self):
        two = ["CC(=O)O", "CCO"]
        three = two + ["CO"]
        run_two = organic.simulate_network(two, [1, 1], organic.calculate_products(two, "Esterification"))
        run_three = organic.simulate_network(three, [1, 1, 1], organic.calculate_products(three, "Esterification"))
        self.assertLess(run_three.target_yield, run_two.target_yield)

    def test_optimizer_improves_baseline_and_is_bounded(self):
        smiles = ["CC(=O)O", "CCO", "CO"]
        channels = organic.calculate_products(smiles, "Esterification")
        baseline = organic.ModelConditions()
        reference = organic.simulate_network(smiles, [1, 1, 1], channels, 0, baseline)
        best, evaluations = organic.optimize_model_yield(smiles, [1, 1, 1], channels, 0, baseline)
        self.assertEqual(evaluations, 151)
        self.assertGreater(best.target_yield, reference.target_yield)
        self.assertLessEqual(best.target_yield, 1)
        self.assertTrue(280 <= best.conditions.temperature_k <= 380)
        self.assertTrue(0.5 <= best.conditions.partner_multiplier <= 3)
        self.assertTrue(1 <= best.conditions.duration_min <= 120)
        self.assertEqual(best.initial_m[0], 1)
        self.assertEqual(best.initial_m[1], best.conditions.partner_multiplier)

    def test_invalid_network_conditions_fail(self):
        smiles = ["CC(=O)O", "CCO"]
        channels = organic.calculate_products(smiles, "Esterification")
        for conditions in (
            organic.ModelConditions(temperature_k=500),
            organic.ModelConditions(duration_min=0),
            organic.ModelConditions(partner_multiplier=float("nan")),
        ):
            with self.assertRaises(ValueError):
                organic.simulate_network(smiles, [1, 1], channels, conditions=conditions)
        with self.assertRaises(ValueError):
            organic.simulate_network(smiles, [1, -1], channels)
        with self.assertRaises(ValueError):
            organic.simulate_network(smiles, [1, 1], channels, target=100)

    def test_molecular_frames_are_png_and_change_with_time(self):
        smiles = ["CC(=O)O", "CCO"]
        channels = organic.calculate_products(smiles, "Esterification")
        run = organic.simulate_network(smiles, [1, 1], channels)
        shapes = [organic.geometry(s) for s in smiles]
        product = organic.geometry(channels[0].product)
        first = organic.motion_png(shapes, product, run, 0, 0)
        last = organic.motion_png(shapes, product, run, 0, 60)
        self.assertTrue(first.startswith(b"\x89PNG"))
        self.assertTrue(last.startswith(b"\x89PNG"))
        self.assertNotEqual(first, last)

    def test_notebook_launch_and_controls_end_to_end(self):
        notebook = json.loads(
            (PROJECT / "02_molecular_dynamics_equations_and_derivation.ipynb").read_text()
        )
        code = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell.get("id") == "organic-aldol-simulator-controls"
        )
        with patch("IPython.display.display") as display:
            exec(compile(code, "<organic notebook>", "exec"), {})
            panel = display.call_args.args[0]
            images = [child for child in panel.children if type(child).__name__ == "Image"]
            for image in images:
                self.assertTrue(bytes(image.value).startswith(b"\x89PNG"))
            target = panel.children[4]
            accordion = panel.children[1]
            accordion.children[2].children[0].value = True
            self.assertEqual(target.options, ())
            panel.children[3].click()
            self.assertEqual(len(target.options), 2)
            target.value = 1
            simulate, optimize = panel.children[7].children
            simulate.click()
            self.assertIn("Target MODEL yield", panel.children[8].value)
            optimize.click()
            self.assertFalse(optimize.disabled)
            self.assertIn("Target MODEL yield", panel.children[8].value)
            temperature, duration, ratio = panel.children[6].children
            self.assertGreater(ratio.value, 1)
            self.assertTrue(280 <= temperature.value <= 380)
            self.assertTrue(1 <= duration.value <= 120)
            initial_frame = bytes(images[1].value)
            panel.children[9].children[1].value = 60
            self.assertNotEqual(initial_frame, bytes(images[1].value))
            self.assertTrue(bytes(images[2].value).startswith(b"\x89PNG"))

    def test_custom_smiles_must_match_formula_controls(self):
        with patch("IPython.display.display") as display:
            organic.build_organic_product_notebook()
            panel = display.call_args.args[0]
            slot = panel.children[1].children[0]
            choices, custom = slot.children[5:7]
            custom.value = "CCO"
            self.assertIsNone(choices.value)
            self.assertIn("does not match", slot.children[8].value)
            self.assertEqual(panel.children[4].options, ())
