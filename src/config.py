"""Personal settings (Notes folder names, excluded titles), kept out of git.

Copy config.example.toml to config.toml at the repo root and edit it.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.toml"
EXAMPLE = ROOT / "config.example.toml"


def load_config():
    if not CONFIG.exists():
        raise SystemExit(
            f"Missing {CONFIG.name}. Create it from the example and set your Notes folders:\n"
            f"  cp {EXAMPLE.name} {CONFIG.name}"
        )
    with open(CONFIG, "rb") as f:
        return tomllib.load(f)
