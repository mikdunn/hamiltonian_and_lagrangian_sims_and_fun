import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

_RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

spec = importlib.util.spec_from_file_location(
    "organic_reactions",
    Path(__file__).with_name("organic_reactions.py"),
)
organic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = organic
spec.loader.exec_module(organic)


class OrganicReactionTests(unittest.TestCase):
    @unittest.skipUnless(_RDKIT_AVAILABLE, "RDKit is an optional graph dependency.")
    def test_graph_based_alpha_carbon_detection(self):
        self.assertEqual(
            organic.alpha_carbon_indices(organic.ACETONE_SMILES),
            (0, 3),
        )
        self.assertEqual(
            organic.alpha_carbon_indices(organic.ACETALDEHYDE_SMILES),
            (0,),
        )
        self.assertEqual(organic.alpha_carbon_indices("CC(=O)O"), (0,))
        from rdkit import Chem

        self.assertIsNotNone(Chem.MolFromSmiles(organic.ALDOL_PRODUCT_SMILES))

    @unittest.skipUnless(_RDKIT_AVAILABLE, "RDKit is an optional graph dependency.")
    def test_invalid_smiles_is_reported(self):
        with self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            organic.alpha_carbon_indices("this is not SMILES")

    def test_acetone_enolate_fraction_matches_henderson_hasselbalch(self):
        conditions = organic.AldolConditions(pH=14.0)
        expected = 1.0 / (
            1.0 + 10.0 ** (conditions.acetone_alpha_pKa - conditions.pH)
        )
        self.assertAlmostEqual(organic.enolate_fraction(conditions), expected)
        self.assertGreater(organic.enolate_fraction(conditions), 0.0)
        self.assertLess(organic.enolate_fraction(conditions), 1.0)

    def test_kinetic_trajectory_is_positive_and_stoichiometric(self):
        trajectory = organic.simulate_aldol(
            organic.AldolConditions(pH=14.0, k_addition_l_mol_s=1.2),
            acetone_initial_mol_l=0.8,
            acetaldehyde_initial_mol_l=1.1,
            duration_s=3600.0,
        )
        concentrations = trajectory.concentrations_mol_l
        self.assertGreaterEqual(float(concentrations.min()), -1e-10)
        np.testing.assert_allclose(
            0.8 - concentrations[:, 0],
            concentrations[:, 2],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            1.1 - concentrations[:, 1],
            concentrations[:, 2],
            atol=1e-9,
        )

    def test_rate_constant_zero_leaves_reactants_unchanged(self):
        trajectory = organic.simulate_aldol(
            organic.AldolConditions(pH=14.0, k_addition_l_mol_s=0.0)
        )
        np.testing.assert_allclose(
            trajectory.concentrations_mol_l[-1],
            (1.0, 1.0, 0.0),
            atol=1e-12,
        )

    def test_invalid_kinetic_inputs_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "pH"):
            organic.enolate_fraction(organic.AldolConditions(pH=15.0))
        with self.assertRaisesRegex(ValueError, "rate constant"):
            organic.enolate_fraction(
                organic.AldolConditions(k_addition_l_mol_s=-0.1)
            )
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            organic.simulate_aldol(acetone_initial_mol_l=-1.0)

    def test_lifted_dmd_is_evaluated_on_held_out_conditions(self):
        model = organic.fit_lifted_dmd(
            n_trajectories=16,
            n_snapshots=41,
            training_duration_s=1200.0,
        )
        metrics = organic.validate_lifted_dmd(
            model,
            n_cases=4,
            duration_s=600.0,
            n_points=31,
        )
        self.assertLess(metrics["max_normalized_rmse"], 0.08)
        self.assertGreaterEqual(
            metrics["minimum_predicted_concentration_mol_l"],
            -1e-8,
        )
        self.assertLess(metrics["max_stoichiometric_error_mol_l"], 1e-7)
