"""Benchmarks of zephyr on our dataset, and the methods it is compared against.

A separate distribution (``zephyr-benchmarks``) living under the ``zephyr`` import
name. It builds on zephyr; zephyr never imports it.
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent
CONFIGS = (
    _HERE / "configs" if (_HERE / "configs").is_dir() else _HERE.parents[2] / "configs"
)
"""The package's configs: inside an installed wheel, else the source tree's ``configs/``."""
