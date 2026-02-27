from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _save_output(path: str, output: Any) -> str:
    out_path = Path(path)
    suffix = out_path.suffix.lower()
    if suffix == "":
        out_path = out_path.with_suffix(".npz")
        suffix = ".npz"

    if suffix == ".npy":
        if not isinstance(output, np.ndarray):
            raise ValueError(".npy output requires a single array")
        np.save(out_path, output)
        return str(out_path)

    if suffix == ".csv":
        if not isinstance(output, np.ndarray):
            raise ValueError(".csv output requires a single array")
        np.savetxt(out_path, output, delimiter=",")
        return str(out_path)

    if suffix == ".json":
        out_path.write_text(json.dumps(_to_jsonable(output), indent=2))
        return str(out_path)

    if suffix == ".npz":
        arrays = _to_npz_dict(output)
        np.savez(out_path, **arrays)
        return str(out_path)

    raise ValueError("output extension must be .npz, .npy, .csv, or .json")


def _artifact_metadata(path: str) -> Dict[str, Any]:
    out_path = Path(path)
    if not out_path.exists():
        raise FileNotFoundError(f"output artifact not found: {path}")

    digest = hashlib.sha256()
    with out_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    fmt = out_path.suffix.lower().lstrip(".")
    if not fmt:
        fmt = "npz"
    return {
        "path": str(out_path),
        "format": fmt,
        "bytes": out_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _to_npz_dict(output: Any) -> Dict[str, np.ndarray]:
    if isinstance(output, np.ndarray):
        return {"data": output}
    if isinstance(output, dict):
        return {str(k): np.asarray(v) for k, v in output.items()}
    if isinstance(output, (list, tuple)):
        return {f"arr_{i}": np.asarray(v) for i, v in enumerate(output)}
    return {"data": np.asarray(output)}


def _to_jsonable(output: Any) -> Any:
    if isinstance(output, np.ndarray):
        return output.tolist()
    if isinstance(output, dict):
        return {str(k): _to_jsonable(v) for k, v in output.items()}
    if isinstance(output, (list, tuple)):
        return [_to_jsonable(v) for v in output]
    if isinstance(output, (np.floating, np.integer)):
        return output.item()
    return output


def _summary_from_output(output: Any, analysis: str, out_path: Path) -> Dict[str, Any]:
    if isinstance(output, np.generic):
        output = output.item()
    summary: Dict[str, Any] = {
        "analysis": analysis,
        "out": str(out_path),
    }
    if isinstance(output, np.ndarray):
        summary.update(
            {
                "kind": "array",
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "keys": ["data"],
            }
        )
        return summary
    if isinstance(output, dict):
        summary["kind"] = "dict"
        summary["keys"] = [str(k) for k in output.keys()]
        summary["shapes"] = {str(k): list(np.asarray(v).shape) for k, v in output.items()}
        return summary
    if isinstance(output, (list, tuple)):
        summary["kind"] = "tuple"
        summary["keys"] = [f"arr_{i}" for i in range(len(output))]
        summary["shapes"] = {
            f"arr_{i}": list(np.asarray(v).shape) for i, v in enumerate(output)
        }
        return summary
    summary["kind"] = "scalar"
    summary["value"] = output
    return summary


def _print_summary(summary: Dict[str, Any], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(summary, indent=2))
        return
    print(f"analysis: {summary.get('analysis')}")
    print(f"out: {summary.get('out')}")
    print(f"kind: {summary.get('kind')}")
    if "keys" in summary:
        print("keys: " + ", ".join(summary["keys"]))
    if "shape" in summary:
        print(f"shape: {summary['shape']}")
    if "dtype" in summary:
        print(f"dtype: {summary['dtype']}")
    if "value" in summary:
        print(f"value: {summary['value']}")
