# Usage:
# result = run(cfg)

from __future__ import annotations

import json
from typing import Any, Dict, Union

import numpy as np

from .config import PackConfig, PackResult


def _result_from_dict(out: Dict[str, Any]) -> PackResult:
    coords = np.asarray(out["coords"], dtype=np.float32)
    box = tuple(float(x) for x in out["box"])
    return PackResult(
        coords=coords,
        box=box,
        name=list(out["name"]),
        element=list(out["element"]),
        resname=list(out["resname"]),
        resid=[int(x) for x in out["resid"]],
        chain=[str(x) for x in out["chain"]],
        segid=[str(x) for x in out.get("segid", [])] or None,
        charge=[float(x) for x in out["charge"]],
        mol_id=[int(x) for x in out["mol_id"]],
        bonds=[(int(a), int(b)) for a, b in out.get("bonds", [])],
        record_kind=[str(x) for x in out.get("record_kind", [])] or None,
        ter_after=[int(x) for x in out.get("ter_after", [])] or None,
    )


def run(cfg: Union[PackConfig, Dict[str, Any]]) -> PackResult:
    from .. import traj_py  # type: ignore

    if not hasattr(traj_py, "pack_from_json"):
        raise RuntimeError("warp_pack bindings are unavailable in traj_py")
    if isinstance(cfg, PackConfig):
        payload = json.dumps(cfg.to_dict())
    elif isinstance(cfg, dict):
        payload = json.dumps(cfg)
    else:
        raise TypeError("run() expects PackConfig or dict")
    out: dict[str, Any] = traj_py.pack_from_json(payload)
    return _result_from_dict(out)


def parse_inp(path: str) -> Dict[str, Any]:
    from .. import traj_py  # type: ignore

    if not hasattr(traj_py, "pack_config_from_inp"):
        raise RuntimeError("warp_pack inp bindings are unavailable in traj_py")
    return traj_py.pack_config_from_inp(path)


def run_inp(path: str) -> PackResult:
    from .. import traj_py  # type: ignore

    if hasattr(traj_py, "pack_from_inp"):
        out: dict[str, Any] = traj_py.pack_from_inp(path)
        return _result_from_dict(out)
    cfg = parse_inp(path)
    return run(cfg)
