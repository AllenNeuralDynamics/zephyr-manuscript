"""The paper's pre-registered configs load, and ship inside the wheel."""

import shutil
import subprocess
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

from zephyr.benchmarks import CONFIGS

MANUSCRIPT = CONFIGS / "manuscript"
PACKAGE = Path(__file__).resolve().parents[1]


class ManuscriptConfigTests(unittest.TestCase):
    def test_every_experiment_names_lists_that_are_here_or_on_the_mount(self):
        paths = sorted((MANUSCRIPT / "experiments").glob("*.toml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path.name):
                data = tomllib.loads(path.read_text())
                self.assertEqual(data["seeds"], [17])
                self.assertEqual(data["output_dir"], f"/results/{path.stem}")
                for fold in data["fold"]:
                    parts = [
                        *fold["train"],
                        fold.get("val", {}),
                        *fold["test"].values(),
                    ]
                    for clips in {p["clips"] for p in parts if "clips" in p}:
                        if clips.startswith("/data/"):  # the dataset's, on Code Ocean
                            continue
                        self.assertTrue((path.parent / clips).is_file(), clips)

    def test_every_matrix_names_an_experiment_here(self):
        paths = sorted((MANUSCRIPT / "matrices").glob("*.toml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path.name):
                data = tomllib.loads(path.read_text())
                self.assertTrue((CONFIGS / data["experiment"]).is_file())
                mounts = {d["mount"] for d in data.get("data_asset", [])}
                self.assertEqual(
                    mounts - {"zephyr-df-dataset"}, set()
                )  # one cache asset

    @unittest.skipUnless(shutil.which("uv"), "needs uv to build the wheel")
    def test_the_wheel_holds_the_configs(self):
        with tempfile.TemporaryDirectory() as out:
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", out, str(PACKAGE)],
                check=True,
                capture_output=True,
            )
            (wheel,) = Path(out).glob("*.whl")
            names = set(zipfile.ZipFile(wheel).namelist())
        for path in (MANUSCRIPT / "experiments").glob("*.toml"):
            self.assertIn(
                f"zephyr/benchmarks/configs/manuscript/experiments/{path.name}", names
            )
