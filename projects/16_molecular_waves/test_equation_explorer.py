import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    'equation_explorer', Path(__file__).with_name('equation_explorer.py')
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EquationParsingTests(unittest.TestCase):
    def test_balanced_reaction_and_coefficients(self):
        left, right = module.parse_equation('2 H2 + O2 -> 2 H2O')
        self.assertEqual(module.totals(left), ({'H': 4, 'O': 2}, 0))
        self.assertEqual(module.totals(left), module.totals(right))
        self.assertIn('Balanced', module.composition_html(left, right))
        self.assertEqual(right[0].counts, {'H': 2, 'O': 1})

    def test_groups_hydrates_unicode_and_phases(self):
        for formula, counts in [
            ('Ca(OH)2', {'Ca': 1, 'O': 2, 'H': 2}),
            ('Al2(SO4)3', {'Al': 2, 'S': 3, 'O': 12}),
            ('K4[Fe(CN)6]', {'K': 4, 'Fe': 1, 'C': 6, 'N': 6}),
            ('CuSO4·5H2O', {'Cu': 1, 'S': 1, 'O': 9, 'H': 10}),
            ('2H₂O(l)', {'H': 2, 'O': 1}),
        ]:
            with self.subTest(formula=formula):
                self.assertEqual(module.parse_species(formula).counts, counts)
        left, right = module.parse_equation('2 H₂(g) + O₂(g) → 2 H₂O(l)')
        self.assertEqual(module.totals(left), module.totals(right))
        self.assertEqual(right[0].phase, 'l')

    def test_ionic_charge_and_electrons(self):
        for text in [
            'Fe^2+ + 2 Cl^- -> FeCl2',
            'Fe²⁺ + 2 Cl⁻ → FeCl2',
            'Fe^3+ + e^- -> Fe^2+',
            'NH4^+ + Cl^- -> NH4Cl',
            'Na^++Cl^-=NaCl',
        ]:
            with self.subTest(equation=text):
                left, right = module.parse_equation(text)
                self.assertEqual(module.totals(left), module.totals(right))
        left, right = module.parse_equation('Na -> Na^+')
        self.assertIn('UNBALANCED', module.composition_html(left, right))

    def test_single_formula_unbalanced_and_invalid_inputs(self):
        left, right = module.parse_equation('XeF4')
        self.assertEqual(left[0].counts, {'Xe': 1, 'F': 4})
        self.assertEqual(right, [])
        left, right = module.parse_equation('H2 + O2 -> H2O')
        self.assertIn('UNBALANCED', module.composition_html(left, right))
        for text in [
            '',
            'H2 ->',
            'H2 -> H2 -> H2',
            '0 H2',
            'H0',
            'Xx2',
            'Ca(OH2',
            'K[OH)',
            'H2..O',
            'C()',
            '__import__(os)',
            'Fe3+',
            'H-2',
            'H2 + + O2',
            'H2^0+',
            'H2 -> 0 H2',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.parse_equation(text)


class AtomicModelTests(unittest.TestCase):
    def test_configurations_and_ion_ordering(self):
        carbon = module.electron_configuration('C')
        self.assertEqual(carbon, {(1, 0): 2, (2, 0): 2, (2, 1): 2})
        chromium = module.electron_configuration('Cr')
        self.assertEqual(chromium[(4, 0)], 1)
        self.assertEqual(chromium[(3, 2)], 5)
        copper = module.electron_configuration('Cu')
        self.assertEqual(copper[(4, 0)], 1)
        self.assertEqual(copper[(3, 2)], 10)
        iron = module.electron_configuration('Fe', 2)
        self.assertNotIn((4, 0), iron)
        self.assertEqual(iron[(3, 2)], 6)
        self.assertEqual(module.electron_configuration('H', 1), {})
        self.assertEqual(
            sum(module.electron_configuration('Cl', -1).values()), 18
        )
        for symbol, z in module.ATOMIC_NUMBERS.items():
            config = module.electron_configuration(symbol)
            self.assertEqual(sum(config.values()), z)
            for (_, ell), count in config.items():
                self.assertTrue(0 < count <= 2 * (2 * ell + 1))
        for symbol, charge in [('C', 7), ('H', -3), ('Og', -1)]:
            with self.assertRaises(ValueError):
                module.electron_configuration(symbol, charge)

    def test_hund_occupancy(self):
        self.assertEqual(module.orbital_occupancies(2, 1), [1, 1, 0])
        self.assertEqual(module.orbital_occupancies(4, 1), [2, 1, 1])
        self.assertEqual(module.orbital_occupancies(5, 2), [1] * 5)
        self.assertEqual(module.orbital_occupancies(10, 2), [2] * 5)

    def test_hydrogenic_radial_normalization_and_shapes(self):
        # Check hydrogen 1s against its analytic field, not a mesh proxy.
        field, grid = module.atomic_field(1, 0, 's')
        points = grid['pts'].reshape((*grid['shape'], 3))
        r = np.linalg.norm(points, axis=-1)
        expected = np.exp(-r / module.model.BOHR) / np.sqrt(
            np.pi * module.model.BOHR**3
        )
        np.testing.assert_allclose(field, expected, rtol=1e-12)
        # The finite plotting grid truncates the very small exponential tails.
        self.assertAlmostEqual(
            float((field**2).sum()) * grid['h'] ** 3, 1, places=2
        )
        for n, ell, orbital in [(2, 1, 'pz'), (3, 2, 'dxy'), (4, 3, 'fxyz')]:
            field, grid = module.atomic_field(n, ell, orbital)
            self.assertTrue(np.isfinite(field).all())
            self.assertGreater(field.max(), 0)
            self.assertLess(field.min(), 0)
            fig = module.atomic_figure('C', 0, n, ell, 0, 0.22, 0.8, 1.0)
            self.assertTrue(fig.to_json())
        for n in range(1, 8):
            for ell in range(min(n, 4)):
                orbital = module.model.PURE[module.LETTERS[ell]][0]
                field, _ = module.atomic_field(n, ell, orbital)
                self.assertTrue(np.isfinite(field).all())
                self.assertGreater(float(abs(field).max()), 0)
        pz, _ = module.atomic_field(2, 1, 'pz')
        self.assertTrue(np.all(pz[:, :, 40] == 0))
        field2s, _ = module.atomic_field(2, 0, 's')
        self.assertGreater(field2s[40, 40, 40], 0)
        self.assertLess(field2s[40, 40, 48], 0)


class MolecularInputTests(unittest.TestCase):
    def test_templates_match_formulas_and_preserve_bond_counts(self):
        for formula in module.MOLECULES:
            species = module.parse_species(formula)
            specs = module.molecular_specs(species)
            counts = {}
            for spec in specs:
                counts[spec['el']] = counts.get(spec['el'], 0) + 1
            self.assertEqual(counts, species.counts)
        water = module.model.place_atoms(
            module.molecular_specs(module.parse_species('H2O'))
        )
        res = module.model.analyze(water)
        self.assertEqual(sum(b['m'] == 0 for b in res['bonds']), 2)
        self.assertEqual(res['hbonds'], [])
        hydrogen = module.parse_species('20 H2')
        self.assertEqual(len(module.molecular_specs(hydrogen)), 2)

    def test_unsupported_formula_and_ions_are_not_guessed(self):
        for formula in ('C2H6O', 'O2', 'NaCl', 'NH4^+'):
            with self.subTest(formula=formula), self.assertRaises(ValueError):
                module.molecular_specs(module.parse_species(formula))

    def test_explicit_structure_and_validation(self):
        species = module.parse_species('H2')
        text = 'H s 0 0 0 1 0 0\nH s .74 0 0 -1 0 0'
        specs = module.explicit_structure(text, species)
        atoms = module.model.place_atoms(specs)
        res = module.model.analyze(atoms)
        self.assertEqual(len(res['bonds']), 1)
        np.testing.assert_allclose(atoms[1]['pos'], [0.74, 0, 0])
        for text in [
            '',
            'H s 0 0 0',
            'H s 0 0 0\nH s 0 0 0',
            'H s nan 0 0\nH s 1 0 0',
            'H s 0 0 0\nH s 31 0 0',
            'H s 0 0 0 0 0 0\nH s 1 0 0',
            'H bad 0 0 0\nH s 1 0 0',
            'H s x 0 0\nH s 1 0 0',
            'Xe p 0 0 0',
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.explicit_structure(text, species)


class QuantumChemistryInputTests(unittest.TestCase):
    def test_solver_geometry_composition_and_coordinates(self):
        atoms = module.quantum_chemistry.parse_geometry(
            'O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0',
            {'H': 2, 'O': 1},
        )
        self.assertEqual([atom[0] for atom in atoms], ['O', 'H', 'H'])
        np.testing.assert_allclose(atoms[1][1], (0.96, 0, 0))
        for text in (
            '',
            'O p 0 0 0',
            'Og 0 0 0',
            'H nan 0 0',
            'H 0 0 0\nH 0 0 0',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.quantum_chemistry.parse_geometry(
                    text, {'H': 2, 'O': 1}
                )
        with self.assertRaisesRegex(ValueError, 'does not match'):
            module.quantum_chemistry.parse_geometry(
                'H 0 0 0\nH 0.74 0 0', {'H': 2, 'O': 1}
            )

    def test_spin_compatibility_and_element_limit(self):
        validate = module.quantum_chemistry.validate_multiplicity
        self.assertEqual(validate(['H', 'H'], 0, 1), 0)
        self.assertEqual(validate(['H'], 0, 2), 1)
        for args in (
            (['H', 'H'], 0, 2),
            (['H'], 0, 1),
            (['Kr'], -1, 1),
            (['Rb'], 0, 1),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate(*args)

    def test_contact_geometry_reports_water_angle(self):
        contacts, angles = module.quantum_chemistry.infer_geometry_contacts(
            ['O', 'H', 'H'],
            np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        )
        self.assertEqual(len(contacts), 2)
        self.assertEqual(len(angles), 1)
        self.assertAlmostEqual(angles[0][3], 104.46, delta=1)

    def test_restricted_and_unrestricted_density_shapes(self):
        qc = module.quantum_chemistry
        restricted = SimpleNamespace(
            mo_coeff=np.zeros((3, 3)),
            make_rdm1=lambda: np.eye(3),
        )
        unrestricted = SimpleNamespace(
            mo_coeff=np.zeros((2, 3, 3)),
            make_rdm1=lambda: np.stack((np.eye(3), 2 * np.eye(3))),
        )
        self.assertFalse(qc.is_unrestricted(restricted))
        self.assertTrue(qc.is_unrestricted(unrestricted))
        np.testing.assert_array_equal(
            qc._density_matrix(unrestricted), 3 * np.eye(3)
        )


class EquationWidgetTests(unittest.TestCase):
    def setUp(self):
        self.figures = []
        self.patch = patch.object(
            module.EquationExplorer,
            'display_figure',
            lambda _, fig: self.figures.append(fig),
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.ui = module.EquationExplorer()

    def test_equation_switch_atomic_and_molecular_views(self):
        ui = self.ui
        self.assertIn('Balanced', ui.status.value)
        self.assertEqual(ui.element.value, 'H')
        self.assertIn('1 electrons', ui.detail.value)
        ui.mode.value = 'Molecule: bond contributors'
        self.assertEqual(
            sum(t.type == 'mesh3d' for t in self.figures[-1].data), 2
        )
        ui.species.value = 1
        self.assertIn('No unique molecular structure for O2', ui.detail.value)
        ui.mode.value = 'Atomic orbital'
        self.assertEqual(ui.element.value, 'O')
        self.assertIn('8 electrons', ui.detail.value)
        ui.species.value = 2
        ui.mode.value = 'Molecule: bonding MO'
        self.assertIn('H2O', ui.detail.value)
        self.assertEqual(
            len(ui._molecular_result('H2O', 0, '')[1]['bonds']), 2
        )
        ui.equation.value = 'C6H6'
        ui.parse()
        ui.mode.value = 'Molecule: delocalized π density'
        self.assertIn('6 π electrons', ui.detail.value)
        ui.mode.value = 'Molecule: delocalized π MO'
        ui.pi.value = 5
        self.assertIn('Displayed π MO 6', ui.detail.value)

    def test_atomic_ions_polyatomic_references_and_error_cleanup(self):
        ui = self.ui
        ui.equation.value = 'Fe^2+ + 2 Cl^- -> FeCl2'
        ui.parse()
        self.assertIn('24 electrons', ui.detail.value)
        ui.species.value = 1
        self.assertIn('18 electrons', ui.detail.value)
        ui.equation.value = 'NH4^+'
        ui.parse()
        self.assertIn('neutral references', ui.detail.value)
        ui.equation.value = 'H^+'
        ui.parse()
        self.assertIn('no electrons', ui.detail.value)
        ui.equation.value = 'e^-'
        ui.parse()
        self.assertIn('not an atom', ui.detail.value)
        ui.equation.value = 'H2 ->'
        ui.parse()
        self.assertEqual(ui.entries, [])
        self.assertEqual(ui.species.options, ())
        self.assertIn('Input error', ui.status.value)

    def test_custom_structure_application_and_persistence(self):
        ui = self.ui
        ui.equation.value = 'H2 + O2 -> H2O'
        ui.parse()
        ui.species.value = 1
        ui.structure.value = 'O p 0 0 0\nO p 1.21 0 0 -1 0 0'
        ui.apply_structure()
        self.assertNotIn('Cannot render', ui.detail.value)
        self.assertEqual(ui.mode.value, 'Molecule: bond contributors')
        ui.species.value = 0
        ui.species.value = 1
        self.assertIn('1.21', ui.structure.value)
        ui.use_template()
        self.assertEqual(ui.structure.value, '')
        self.assertIn('No unique molecular structure for O2', ui.detail.value)
        self.assertEqual(ui.bond.options, ())
        ui.structure.value = 'O p 0 0 0\nO p 1.21 0 0 -1 0 0'
        ui.apply_structure()
        ui.structure.value = 'O p 0 0 0'
        ui.apply_structure()
        self.assertIn('Structure error', ui.detail.value)

    def test_dft_controls_are_opt_in_and_template_geometry_is_available(self):
        ui = self.ui
        ui.equation.value = 'H2O'
        ui.parse()
        self.assertIn('Molecule: DFT total density', ui.mode.options)
        self.assertEqual(ui.quantum_geometry.value.count('\n') + 1, 3)
        ui.mode.value = 'Molecule: DFT total density'
        self.assertIn('click', ui.detail.value.lower())
        self.assertIsNone(ui._quantum_result)


if __name__ == '__main__':
    unittest.main()
