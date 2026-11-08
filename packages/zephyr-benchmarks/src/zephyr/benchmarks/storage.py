"""Where the paper's assets live (runs, cache, datasets): one prefix for all of them.

``ZEPHYR_BUCKET`` when set, else ``bucket`` of ``[tool.zephyr-manuscript]`` in the
workspace's ``pyproject.toml``. :func:`url` names an asset under it, :func:`key` the same
without ``s3://`` (as s3fs takes it), and :func:`expand` fills ``${ZEPHYR_BUCKET}`` in a
config's text (a results manifest's ``root``).
"""

import os
import tomllib
from pathlib import Path

ENV = "ZEPHYR_BUCKET"
TABLE = "zephyr-manuscript"
PLACEHOLDER = "${" + ENV + "}"


def _from_pyproject() -> str | None:
    """``bucket`` of the first ``pyproject.toml`` with the table, above this file or the
    working directory."""
    here, cwd = Path(__file__).resolve(), Path.cwd().resolve()
    for folder in (*here.parents, cwd, *cwd.parents):
        path = folder / "pyproject.toml"
        if path.is_file():
            table = tomllib.loads(path.read_text()).get("tool", {}).get(TABLE, {})
            if "bucket" in table:
                return table["bucket"]
    return None


def bucket() -> str:
    """The prefix every asset lives under (``s3://...`` or a local folder)."""
    value = os.environ.get(ENV) or _from_pyproject()
    if not value:
        raise RuntimeError(
            f"no bucket: set {ENV} or `bucket` in [tool.{TABLE}] of pyproject.toml"
        )
    return value.rstrip("/")


def url(*parts: str) -> str:
    """An asset under :func:`bucket`, e.g. ``url("zephyr-manuscript-runs")``."""
    return "/".join([bucket(), *(p.strip("/") for p in parts if p)])


def key(*parts: str) -> str:
    """:func:`url` without ``s3://``, as s3fs names objects."""
    return url(*parts).removeprefix("s3://")


def expand(text: str) -> str:
    """*text* with ``${ZEPHYR_BUCKET}`` replaced by :func:`bucket`."""
    return text.replace(PLACEHOLDER, bucket()) if PLACEHOLDER in text else text
