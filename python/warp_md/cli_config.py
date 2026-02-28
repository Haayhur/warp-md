from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .agent_schema import AGENT_REQUEST_SCHEMA_VERSION


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    if cfg_path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML config requires PyYAML installed") from exc
        return yaml.safe_load(cfg_path.read_text())
    return json.loads(cfg_path.read_text())


def _normalize_system_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, str):
        return {"path": spec}
    if isinstance(spec, dict):
        return spec
    raise ValueError("system spec must be a path or object")


def _normalize_traj_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, str):
        return {"path": spec}
    if isinstance(spec, dict):
        return spec
    raise ValueError("trajectory spec must be a path or object")


def _default_out(name: str, output_dir: str, used: Dict[str, int]) -> str:
    count = used.get(name, 0)
    used[name] = count + 1
    suffix = "" if count == 0 else f"_{count}"
    ext = ".json" if name == "docking" else ".npz"
    return str(Path(output_dir) / f"{name}{suffix}{ext}")


def example_config() -> None:
    example = {
        "version": AGENT_REQUEST_SCHEMA_VERSION,
        "run_id": "example-run",
        "system": {"path": "topology.pdb"},
        "trajectory": {"path": "traj.xtc"},
        "device": "auto",
        "stream": "none",
        "chunk_frames": 500,
        "output_dir": "outputs",
        "analyses": [
            {
                "name": "rg",
                "selection": "protein",
                "mass_weighted": False,
            },
            {
                "name": "rdf",
                "sel_a": "resname SOL and name OW",
                "sel_b": "resname SOL and name OW",
                "bins": 200,
                "r_max": 10.0,
            },
        ],
    }
    print(json.dumps(example, indent=2))
