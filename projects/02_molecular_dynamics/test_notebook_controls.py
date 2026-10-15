import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from traitlets import TraitError


PROJECT = Path(__file__).parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NotebookControlTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads(
            (PROJECT / "02_molecular_dynamics_equations_and_derivation.ipynb").read_text()
        )
        code = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell.get("id") == "dft-electron-density-controls"
        )
        self.namespace = {}
        with patch("IPython.display.display"):
            exec(compile(code, "<DFT controls>", "exec"), self.namespace)

    def test_dft_spin_choices_follow_electron_count(self):
        ns = self.namespace
        multiplicity = ns["dft_multiplicity"]
        self.assertEqual(multiplicity.value, 1)
        self.assertEqual(list(multiplicity.options), [1, 3, 5, 7, 9, 11])
        with self.assertRaises(TraitError):
            multiplicity.value = 59
        ns["dft_charge"].value = 1
        self.assertEqual(list(multiplicity.options), [2, 4, 6, 8, 10])
        self.assertEqual(multiplicity.value, 2)
        self.assertIn("reset to 2", ns["dft_spin_status"].value)
        ns["dft_charge"].value = 10
        self.assertTrue(ns["dft_run"].disabled)
        self.assertIn("no electrons", ns["dft_spin_status"].value)

    def test_dft_invalid_geometry_is_visible(self):
        ns = self.namespace
        ns["dft_geometry"].value = "O not coordinates"
        self.assertTrue(ns["dft_run"].disabled)
        self.assertIn("Invalid geometry/charge", ns["dft_spin_status"].value)
        ns["dft_geometry"].value = "H 0 0 0"
        self.assertFalse(ns["dft_run"].disabled)
        self.assertEqual(ns["dft_multiplicity"].value, 2)

    @unittest.skipUnless(importlib.util.find_spec("rdkit"), "RDKit is optional.")
    def test_organic_initial_plot_and_button_plot(self):
        organic = load_module("organic_ui_test", PROJECT / "organic_reactions.py")
        with patch("IPython.display.display") as display:
            organic.build_organic_reaction_notebook()
            panel = display.call_args_list[0].args[0]
            self.assertEqual(type(panel).__name__, "VBox")
            self.assertEqual(type(display.call_args_list[1].args[0]).__name__, "Figure")
            panel.children[3].click()
            self.assertEqual(display.call_count, 3)

    def test_organic_missing_dependency_fails_before_display(self):
        organic = load_module("organic_missing_test", PROJECT / "organic_reactions.py")
        with patch.dict(sys.modules, {"rdkit": None}):
            with patch("IPython.display.display") as display:
                with self.assertRaisesRegex(ImportError, "notebook kernel"):
                    organic.build_organic_reaction_notebook()
                display.assert_not_called()

    def test_phase_initial_and_updated_diagnostics(self):
        simulation = load_module("phase_ui_test", PROJECT / "simulation.py")
        with patch.object(simulation, "display") as display:
            simulation.build_molecular_dynamics_notebook()
            panel = display.call_args.args[0]
            images = [
                child for child in panel.children if type(child).__name__ == "Image"
            ]
            for image in images:
                self.assertTrue(bytes(image.value).startswith(b"\x89PNG"))
            initial_diagnostics = bytes(images[1].value)
            panel.children[2].value = "solid"
            self.assertTrue(bytes(images[1].value).startswith(b"\x89PNG"))
            self.assertNotEqual(initial_diagnostics, bytes(images[1].value))
