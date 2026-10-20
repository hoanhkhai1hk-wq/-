"""Installer failure tests never touch a venv, pip, network or model."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("local_setup", Path(__file__).resolve().parents[2] / "scripts/local/setup.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def test_existing_environment_never_reinstalls(self):
        with tempfile.TemporaryDirectory() as d, patch.object(setup, "run") as run:
            root = Path(d)
            interpreter = root / ".venv-p0/Scripts/python.exe"
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            self.assertEqual(setup.prepare_web(root, True, None), (interpreter, False))
            run.assert_not_called()

    def test_check_missing_environment_does_not_create_it(self):
        with tempfile.TemporaryDirectory() as d, patch.object(setup, "run") as run:
            with self.assertRaises(setup.SetupError):
                setup.prepare_web(Path(d), False, None)
            self.assertEqual(list(Path(d).iterdir()), [])
            run.assert_not_called()

    def test_partial_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as d, patch.object(setup, "run") as run:
            root = Path(d)
            env = root / ".venv-p0"
            env.mkdir()
            marker = env / "prior-data"
            marker.write_bytes(b"preserve")
            with self.assertRaises(setup.SetupError):
                setup.prepare_web(root, True, None)
            self.assertEqual(marker.read_bytes(), b"preserve")
            run.assert_not_called()

    def test_wrong_base_python_stops_before_venv(self):
        with tempfile.TemporaryDirectory() as d, patch.object(setup, "run", return_value="3.14") as run:
            with self.assertRaises(setup.SetupError):
                setup.prepare_web(Path(d), True, "explicit-python")
            self.assertEqual(run.call_count, 1)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_venv_failure_stops_before_pip(self):
        with tempfile.TemporaryDirectory() as d, patch.object(setup, "run", side_effect=["3.11", setup.SetupError("venv failed")]) as run:
            root = Path(d)
            (root / "requirements-p0.lock.txt").touch()
            with self.assertRaisesRegex(setup.SetupError, "venv failed"):
                setup.prepare_web(root, True, "explicit-python")
            self.assertEqual(run.call_count, 2)

    def test_pip_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "requirements-p0.lock.txt").touch()
            def simulate(command, **kwargs):
                if "-c" in command:
                    return "3.11"
                if "venv" in command:
                    web = root / ".venv-p0/Scripts/python.exe"
                    web.parent.mkdir(parents=True)
                    web.touch()
                    return ""
                raise setup.SetupError("pip failed")
            with patch.object(setup, "run", side_effect=simulate) as run:
                with self.assertRaisesRegex(setup.SetupError, "pip failed"):
                    setup.prepare_web(root, True, "explicit-python")
                self.assertEqual(run.call_count, 3)


if __name__ == "__main__":
    unittest.main()
