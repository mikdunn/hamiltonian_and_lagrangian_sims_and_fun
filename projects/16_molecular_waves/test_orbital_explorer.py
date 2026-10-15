import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.ndimage import label


spec = importlib.util.spec_from_file_location(
    'orbital_explorer', Path(__file__).with_name('orbital_explorer.py'))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def system(prefix, **kwargs):
    name = next(name for name in model.PRESETS if name.startswith(prefix))
    atoms = model.place_atoms(model.PRESETS[name], **kwargs)
    return atoms, model.analyze(atoms)


class OrbitalExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gallery = system('Orbital gallery — d')
        cls.ethylene = system('Ethylene')
        cls.benzene = system('Benzene')

    def test_d_surfaces_have_clovers_and_torus(self):
        atoms, res = self.gallery
        for k, orb in enumerate(atoms[0]['orbs']):
            field = res['Psi'][k].reshape(res['grid']['shape'])
            level = 0.22 * abs(field).max()
            positive, negative = label(
                field > level)[1], label(
                field < -level)[1]
            self.assertEqual(
                positive + negative,
                3 if orb['label'] == 'dz²' else 4,
                orb['label'])
            meshes = model.iso_meshes(
                field,
                res['grid'],
                0.22,
                model.PHASE['ao'],
                orb['label'],
                0.8)
            self.assertEqual(len(meshes), 2)
            self.assertTrue(all(len(mesh.i) > 100 for mesh in meshes))
        fig = model.orbital_gallery(atoms, res, 0.22, 0.8, 0)
        scenes = {t.scene for t in fig.data if t.type == 'mesh3d'}
        self.assertEqual(len(scenes), 5)

    def test_s_and_p_shapes_and_nodes(self):
        for prefix, lobes in [('Orbital gallery — s', 1),
                              ('Orbital gallery — p', 2)]:
            atoms, res = system(prefix)
            for row in res['Psi']:
                field = row.reshape(res['grid']['shape'])
                level = 0.22 * abs(field).max()
                self.assertEqual(
                    label(
                        field > level)[1] +
                    label(
                        field < -
                        level)[1],
                    lobes)
        atom = system('Orbital gallery — p')[0][0]
        pz = atom['orbs'][2]
        values = model.eval_orbital(atom, pz, np.array(
            [[0, 0, 0.6], [0, 0, -0.6], [0.6, 0, 0]]))
        self.assertGreater(values[0], 0)
        self.assertAlmostEqual(values[0], -values[1])
        self.assertEqual(values[2], 0)

    def test_hybrids_are_orthonormal(self):
        for name in model.HYBRIDS:
            _, coeff = model.build_orbitals(name)
            np.testing.assert_allclose(
                coeff @ coeff.T,
                np.eye(
                    len(coeff)),
                atol=1e-12)

    def test_expected_localized_bonds_and_hydrogen_bonds(self):
        for prefix, sigma, pi, hb in [
            ('H₂', 1, 0, 0), ('N₂', 1, 2, 0), ('Ethylene', 5, 1, 0),
            ('Acetylene', 3, 2, 0), ('Water dimer', 4, 0, 1),
            ('HF dimer', 2, 0, 1), ('(HF)₄', 4, 0, 3), ('4 × H₂', 4, 0, 0),
            ('PF₅', 5, 0, 0), ('SF₆', 6, 0, 0),
        ]:
            with self.subTest(preset=prefix):
                atoms, res = system(prefix)
                self.assertEqual(sum(b['m'] == 0 for b in res['bonds']), sigma)
                self.assertEqual(sum(b['m'] == 1 for b in res['bonds']), pi)
                self.assertEqual(len(res['hbonds']), hb)
                for b in res['bonds']:
                    if b['m']:
                        self.assertFalse(
                            atoms[b['a']]['orbs'][b['ka']]['hybrid'])
                        self.assertFalse(
                            atoms[b['b']]['orbs'][b['kb']]['hybrid'])

    def test_pi_contributors_and_twist(self):
        atoms, res = self.ethylene
        index, bond = next(
            (k, b) for k, b in enumerate(
                res['bonds']) if b['m'] == 1)
        self.assertEqual([atoms[bond[x]]['orbs'][bond[y]]['label']
                          for x, y in [('a', 'ka'), ('b', 'kb')]], ['pz', 'pz'])
        fig = model.build_figure(
            atoms,
            res,
            'Bond contributors',
            0.22,
            0.8,
            bond=index)
        self.assertEqual(sum(t.type == 'mesh3d' for t in fig.data), 4)
        _, twisted = system('Ethylene', twists={1: 90})
        self.assertFalse(any(b['m'] == 1 and (
            b['a'], b['b']) == (0, 1) for b in twisted['bonds']))
        for sign in (1, -1):
            field = model.mo_field(res, bond, sign)
            self.assertAlmostEqual(
                float(
                    field @ field) *
                res['grid']['h'] ** 3,
                1,
                places=5)
        point = np.array([[0.67, 0, 0.5]])
        a = model.eval_orbital(atoms[0], atoms[0]['orbs'][bond['ka']], point)
        b = model.eval_orbital(atoms[1], atoms[1]['orbs'][bond['kb']], point)
        self.assertAlmostEqual(float((a - b)[0]), 0, places=12)
        self.assertGreater(float((a + b)[0]), 0)

    def test_benzene_shared_pi_eigenproblem_and_density(self):
        atoms, res = self.benzene
        self.assertEqual(len(res['pi_systems']), 1)
        pi = res['pi_systems'][0]
        self.assertEqual(len(pi['nodes']), 6)
        self.assertEqual(len(pi['edges']), 6)
        np.testing.assert_array_equal(pi['occupancy'], [2, 2, 2, 0, 0, 0])
        self.assertEqual(len(res['bonds']), 12)
        self.assertFalse(any(b['m'] for b in res['bonds']))
        self.assertEqual(res['lone'], [])
        indices = [res['pos_of'][node] for node in pi['nodes']]
        S = res['S'][np.ix_(indices, indices)]
        np.testing.assert_allclose(
            pi['coeff'].T @ S @ pi['coeff'],
            np.eye(6),
            atol=1e-12)
        alpha = np.array([model.orbital_alpha(
            atoms[a], atoms[a]['orbs'][k]) for a, k in pi['nodes']])
        H = 1.75 * (alpha[:, None] + alpha[None, :]) * S / 2
        np.fill_diagonal(H, alpha)
        np.testing.assert_allclose(
            H @ pi['coeff'],
            S @ pi['coeff'] *
            pi['energies'],
            atol=1e-12)
        density = np.sum(pi['occupancy'][:, None] * pi['fields'] ** 2, axis=0)
        self.assertAlmostEqual(
            float(
                density.sum()) *
            res['grid']['h'] ** 3,
            6,
            places=5)
        rotated = pi['fields'].copy()
        rotated[1:3] = np.array([[0.6, -0.8], [0.8, 0.6]]) @ rotated[1:3]
        np.testing.assert_allclose(
            density, np.sum(pi['occupancy'][:, None] * rotated ** 2, axis=0), atol=1e-12)
        fig = model.build_figure(
            atoms, res, 'Delocalized π density', 0.22, 0.8)
        self.assertEqual(sum(t.type == 'mesh3d' for t in fig.data), 1)
        self.assertIn(
            'Σ nᵢ|ψᵢ|²',
            model.summary_html(
                atoms,
                res,
                'Delocalized π density'))
        self.assertFalse(self.ethylene[1]['pi_systems'])

    def test_all_presets_and_modes_produce_serializable_figures(self):
        for name, specs in model.PRESETS.items():
            atoms, res = model.place_atoms(specs), None
            res = model.analyze(atoms)
            for mode in model.MODES:
                with self.subTest(preset=name, mode=mode):
                    fig = model.build_figure(
                        atoms, res, mode, 0.22, 0.8, orbital=0)
                    self.assertTrue(fig.to_json())
                    self.assertNotIn(
                        'Traceback', model.summary_html(
                            atoms, res, mode))

    def test_all_choices_are_preserved(self):
        model._sync_focus()
        atoms, res = self.gallery
        model._sync_choices(atoms, res)
        model._state['sync'] = True
        model.w_orbital.value = None
        model.w_bond.value = None
        model._state['sync'] = False
        model._sync_choices(atoms, res)
        self.assertIsNone(model.w_orbital.value)
        self.assertIsNone(model.w_bond.value)

    def test_widget_callbacks_select_gallery_pi_pair_and_resonance(self):
        figures = []
        with patch.object(model.go.Figure, 'show', lambda fig: figures.append(fig)):
            for prefix in ('Ethylene', 'Benzene', 'Orbital gallery — d'):
                model.w_preset.value = next(
                    name for name in model.PRESETS if name.startswith(prefix))
                self.assertNotIn('Traceback', model.out_txt.value)
                if prefix == 'Ethylene':
                    self.assertTrue(
                        model.w_mode.value.startswith('Bond contributors'))
                    atoms, res = model._state['result']
                    self.assertEqual(res['bonds'][model.w_bond.value]['m'], 1)
                    self.assertIn('pz', model.out_txt.value)
                    self.assertEqual(
                        sum(t.type == 'mesh3d' for t in figures[-1].data), 4)
                    model.w_bond.value = None
                    self.assertIsNone(model.w_bond.value)
                elif prefix == 'Benzene':
                    self.assertTrue(
                        model.w_mode.value.startswith('Delocalized π density'))
                    model.w_mode.value = 'Delocalized π — selected molecular orbital'
                    model.w_pi.value = 5
                    self.assertIn('Displayed π MO 6', model.out_txt.value)
                    self.assertNotIn('Traceback', model.out_txt.value)
                else:
                    self.assertTrue(
                        model.w_mode.value.startswith('Orbital gallery'))
                    model.w_mode.value = 'Atomic orbitals — focus atom'
                    self.assertEqual(model.w_orbital.label, 'dxy')
                    self.assertEqual(
                        sum(t.type == 'mesh3d' for t in figures[-1].data), 2)
                    model.w_set.value = 'f'
                    self.assertEqual(
                        len(model._state['result'][0][0]['orbs']), 7)
                    self.assertNotIn('Traceback', model.out_txt.value)
                    model._on_reset(None)


if __name__ == '__main__':
    unittest.main()
