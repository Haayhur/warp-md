from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ._json_types import JsonObject, JsonValue
from . import contract


def _save_output(path: str, output: Any, *, analysis_name: str = None) -> str:
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
        arrays = _to_npz_dict(output, _contract_fields(analysis_name))
        np.savez(out_path, **arrays)
        _write_npz_companions(out_path, arrays)
        return str(out_path)

    raise ValueError("output extension must be .npz, .npy, .csv, or .json")


def _artifact_metadata(
    path: str,
    *,
    analysis_name: str = None,
) -> JsonObject:
    """Generate artifact metadata with optional semantic information.

    Args:
        path: Path to the artifact file
        analysis_name: Name of the analysis that produced this artifact

    Returns:
        Dictionary with artifact metadata
    """
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

    contract_outputs = None
    if analysis_name:
        plan_contract = contract.ANALYSIS_METADATA.get(analysis_name)
        if plan_contract is None:
            plan_contract = contract.ANALYSIS_METADATA.get(analysis_name.replace("-", "_"))
        if plan_contract is not None:
            contract_outputs = plan_contract.outputs

    metadata = {
        "path": str(out_path),
        "format": fmt,
        "bytes": out_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

    # Add semantic metadata if contract info provided
    if contract_outputs and len(contract_outputs) > 0:
        output_spec = contract_outputs[0]  # Most analyses have single output
        metadata["kind"] = output_spec.kind
        if output_spec.fields:
            metadata["fields"] = output_spec.fields
        if output_spec.description:
            metadata["description"] = output_spec.description

    plot_recommendations = _artifact_plot_recommendations(out_path, contract_outputs)
    if plot_recommendations:
        metadata["plot_recommendations"] = plot_recommendations

    companions = _companion_metadata(out_path)
    if companions:
        metadata["companions"] = companions

    # Extract preview stats from NPZ files if possible
    if fmt == "npz":
        try:
            with np.load(out_path, mmap_mode="r") as data:
                preview_stats = {}
                for key in data.keys():
                    arr = data[key]
                    if arr.ndim == 1 and arr.size > 0:
                        preview_stats[f"{key}_min"] = float(arr.min())
                        preview_stats[f"{key}_max"] = float(arr.max())
                        preview_stats[f"{key}_size"] = int(arr.size)
                    elif arr.ndim == 2 and arr.size > 0:
                        preview_stats[f"{key}_shape"] = list(arr.shape)
                if preview_stats:
                    metadata["preview_stats"] = preview_stats
        except Exception:
            pass  # Preview stats are optional

    return metadata


def _contract_fields(analysis_name: Optional[str]) -> Optional[List[str]]:
    if not analysis_name:
        return None
    plan_contract = contract.ANALYSIS_METADATA.get(analysis_name)
    if plan_contract is None:
        plan_contract = contract.ANALYSIS_METADATA.get(analysis_name.replace("-", "_"))
    if plan_contract is None or not plan_contract.outputs:
        return None
    fields = plan_contract.outputs[0].fields
    return list(fields) if fields else None


def _write_npz_companions(out_path: Path, arrays: Dict[str, np.ndarray]) -> None:
    companion_dir = out_path.with_suffix("")
    companion_dir.mkdir(parents=True, exist_ok=True)

    manifest: JsonObject = {
        "source": str(out_path),
        "format": "npz_companion_v1",
        "arrays": [],
    }
    for key in sorted(arrays):
        arr = np.asarray(arrays[key])
        entry: JsonObject = {
            "key": key,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            csv_path = companion_dir / f"{key}.csv"
            _save_csv_table(csv_path, arr.reshape(-1, 1), [key])
            entry["csv"] = str(csv_path)
            entry["columns"] = [key]
        elif arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            csv_path = companion_dir / f"{key}.csv"
            columns = [f"{key}_{i}" for i in range(arr.shape[1])]
            _save_csv_table(csv_path, arr, columns)
            entry["csv"] = str(csv_path)
            entry["columns"] = columns
        elif arr.ndim in (1, 2):
            entry["csv_skipped"] = "non_numeric_dtype"
        manifest["arrays"].append(entry)

    manifest_path = companion_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _save_csv_table(path: Path, data: np.ndarray, columns: List[str]) -> None:
    header = ",".join(columns)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def _companion_metadata(out_path: Path) -> List[JsonObject]:
    if out_path.suffix.lower() != ".npz":
        return []
    companion_dir = out_path.with_suffix("")
    manifest_path = companion_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    companions: List[JsonObject] = [
        {
            "path": str(manifest_path),
            "format": "json",
            "role": "npz_companion_manifest",
        }
    ]
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return companions
    for entry in manifest.get("arrays", []):
        csv_path = entry.get("csv")
        if csv_path and Path(csv_path).exists():
            companions.append(
                {
                    "path": csv_path,
                    "format": "csv",
                    "role": "array_table",
                    "source_key": entry.get("key"),
                    "columns": entry.get("columns", []),
                }
            )
    return companions


def _artifact_plot_recommendations(
    path: Path,
    contract_outputs: Optional[List[Any]],
) -> List[JsonObject]:
    if path.suffix.lower() != ".npz" or not path.exists():
        return []
    try:
        with np.load(path, mmap_mode="r") as data:
            keys = sorted(data.keys())
            shapes = {key: list(data[key].shape) for key in keys}
    except Exception:
        return []

    if not keys or not contract_outputs:
        return []
    recommendations: List[JsonObject] = []
    for output_spec in contract_outputs:
        for recommendation in getattr(output_spec, "plot_recommendations", []):
            rec = copy.deepcopy(recommendation)
            rec["artifact"] = str(path)
            _resolve_axis_for_arrays(rec, "x", keys)
            _resolve_axis_for_arrays(rec, "y", keys)
            _resolve_axis_for_arrays(rec, "z", keys)
            for axis_name in ("y", "z"):
                axis = rec.get(axis_name)
                if isinstance(axis, dict):
                    field = axis.get("field")
                    if field in shapes:
                        rec["shape"] = shapes[field]
                        break
            recommendations.append(rec)
    return recommendations


def _resolve_axis_for_arrays(rec: JsonObject, axis_name: str, keys: List[str]) -> None:
    axis = rec.get(axis_name)
    if not isinstance(axis, dict):
        return
    field = axis.get("field")
    if field in keys:
        return
    if axis_name == "x":
        axis["field"] = "index"
        axis["units"] = axis.get("units") or "frame"
        axis["source"] = "implicit_index"


def _to_npz_dict(output: Any, fields: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    if isinstance(output, np.ndarray):
        key = _single_array_key(fields)
        return {key: output}
    if isinstance(output, dict):
        return {str(k): np.asarray(v) for k, v in output.items()}
    if isinstance(output, (list, tuple)):
        return {
            _sequence_array_key(fields, i): np.asarray(v)
            for i, v in enumerate(output)
        }
    return {_single_array_key(fields): np.asarray(output)}


def _single_array_key(fields: Optional[List[str]]) -> str:
    if not fields:
        return "data"
    if len(fields) >= 2:
        return fields[1]
    return fields[0]


def _sequence_array_key(fields: Optional[List[str]], index: int) -> str:
    if fields and index < len(fields) and fields[index] != "...":
        return fields[index]
    return f"arr_{index}"


def _to_jsonable(output: Any) -> JsonValue:
    if isinstance(output, np.ndarray):
        return output.tolist()
    if isinstance(output, dict):
        return {str(k): _to_jsonable(v) for k, v in output.items()}
    if isinstance(output, (list, tuple)):
        return [_to_jsonable(v) for v in output]
    if isinstance(output, (np.floating, np.integer)):
        return output.item()
    return output


def _summary_from_output(output: Any, analysis: str, out_path: Path) -> JsonObject:
    if isinstance(output, np.generic):
        output = output.item()
    fields = _contract_fields(analysis)
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
                "keys": [_single_array_key(fields)],
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
        summary["keys"] = [_sequence_array_key(fields, i) for i in range(len(output))]
        summary["shapes"] = {
            _sequence_array_key(fields, i): list(np.asarray(v).shape)
            for i, v in enumerate(output)
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
