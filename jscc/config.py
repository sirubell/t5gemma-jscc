"""YAML configuration: a file may name ``base: other.yaml`` and override any of its keys.

Relative paths in ``base`` and in ``data.ids`` / ``data.arrow`` resolve against
the file that declares them, so configs work from any working directory.
"""
from pathlib import Path

import yaml

PATH_KEYS = (("data", "ids"), ("data", "arrow"))


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path) -> dict:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    for section, key in PATH_KEYS:
        value = raw.get(section, {}).get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            raw[section][key] = str((path.parent / value).resolve())
    base = raw.pop("base", None)
    return _merge(load_config(path.parent / base), raw) if base else raw


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
