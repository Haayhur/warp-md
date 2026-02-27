from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_ATLAS_API_BASE_URL = "https://www.dsimb.inserm.fr/ATLAS/api"
_DATASET_PATHS = {
    "atlas": "ATLAS",
    "chameleon": "chameleon",
    "dpf": "DPF",
}
_TRAJECTORY_KINDS = {"analysis", "protein", "total"}


class AtlasApiError(RuntimeError):
    pass


class _AtlasRetryableError(AtlasApiError):
    pass


def _normalize_dataset(dataset: str) -> str:
    token = str(dataset).strip().lower()
    if token not in _DATASET_PATHS:
        expected = ", ".join(sorted(_DATASET_PATHS))
        raise ValueError(f"unsupported dataset '{dataset}'; expected one of: {expected}")
    return token


def _normalize_kind(kind: str) -> str:
    token = str(kind).strip().lower()
    if token not in _TRAJECTORY_KINDS:
        expected = ", ".join(sorted(_TRAJECTORY_KINDS))
        raise ValueError(f"unsupported kind '{kind}'; expected one of: {expected}")
    return token


def _normalize_pdb_chain(pdb_chain: str) -> str:
    token = str(pdb_chain).strip()
    if not token:
        raise ValueError("pdb_chain is required")
    if "_" not in token:
        raise ValueError("pdb_chain must follow the format 'pdbid_chain' (example: 16pk_A)")
    return token


def _base_url_token(base_url: str) -> str:
    token = str(base_url).strip().rstrip("/")
    if not token:
        raise ValueError("base_url is required")
    return token


def build_atlas_trajectory_url(
    *,
    dataset: str,
    kind: str,
    pdb_chain: str,
    base_url: str = DEFAULT_ATLAS_API_BASE_URL,
) -> str:
    dataset_token = _normalize_dataset(dataset)
    kind_token = _normalize_kind(kind)
    pdb_chain_token = _normalize_pdb_chain(pdb_chain)
    base_url_token = _base_url_token(base_url)
    dataset_path = _DATASET_PATHS[dataset_token]
    return (
        f"{base_url_token}/{dataset_path}/{kind_token}/"
        f"{quote(pdb_chain_token, safe='')}"
    )


def default_atlas_archive_name(*, dataset: str, kind: str, pdb_chain: str) -> str:
    dataset_token = _normalize_dataset(dataset)
    kind_token = _normalize_kind(kind)
    pdb_chain_token = _normalize_pdb_chain(pdb_chain)
    if dataset_token == "atlas":
        return f"{pdb_chain_token}_{kind_token}.zip"
    return f"{pdb_chain_token}_{dataset_token}_{kind_token}.zip"


def _parse_json_message(body: bytes) -> Optional[str]:
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return None


def _raise_http_error(exc: HTTPError, url: str) -> None:
    body = b""
    try:
        if exc.fp is not None:
            body = exc.fp.read()
    except Exception:
        body = b""
    details = _parse_json_message(body)
    status = getattr(exc, "code", None)
    if details:
        raise AtlasApiError(f"ATLAS API request failed ({status}): {details} [{url}]") from exc
    raise AtlasApiError(f"ATLAS API request failed ({status}) [{url}]") from exc


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _stream_response_to_file(*, response: Any, path: Path, mode: str) -> int:
    expected_chunk_size: Optional[int] = None
    raw_size = response.headers.get("Content-Length")
    if raw_size is not None:
        try:
            expected_chunk_size = int(raw_size)
        except (TypeError, ValueError):
            expected_chunk_size = None

    written = 0
    with path.open(mode) as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)

    if expected_chunk_size is not None and written != expected_chunk_size:
        raise _AtlasRetryableError(
            "incomplete download payload: "
            f"expected {expected_chunk_size} bytes, received {written} bytes"
        )
    return written


def _download_once(
    *,
    url: str,
    out_path: Path,
    timeout: float,
    resume: bool,
) -> int:
    part_path = out_path.with_name(out_path.name + ".part")
    existing = 0
    headers = {"User-Agent": "warp-md/atlas-fetch"}

    if resume and part_path.exists():
        existing = part_path.stat().st_size
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

    request = Request(url, method="GET", headers=headers)
    with urlopen(request, timeout=timeout) as response:
        content_type = str(response.headers.get("Content-Type", "")).lower()
        if "application/json" in content_type:
            body = response.read()
            message = _parse_json_message(body)
            if message:
                raise AtlasApiError(f"ATLAS API returned JSON response: {message} [{url}]")
            raise AtlasApiError(f"ATLAS API returned JSON response [{url}]")

        status_raw = getattr(response, "status", None)
        if status_raw is None:
            getcode = getattr(response, "getcode", None)
            status_raw = getcode() if callable(getcode) else 0
        status = int(status_raw or 0)
        mode = "wb"
        base = 0
        if existing > 0:
            if status == 206:
                mode = "ab"
                base = existing
            elif status == 200:
                mode = "wb"
                base = 0
            else:
                raise _AtlasRetryableError(
                    f"unexpected HTTP status while resuming download: {status}"
                )

        written = _stream_response_to_file(response=response, path=part_path, mode=mode)
        total = base + written
        if total <= 0:
            raise _AtlasRetryableError("downloaded file is empty")

    part_path.replace(out_path)
    return total


def download_atlas_trajectory(
    *,
    dataset: str,
    kind: str,
    pdb_chain: str,
    out: Optional[str] = None,
    base_url: str = DEFAULT_ATLAS_API_BASE_URL,
    timeout: float = 120.0,
    retries: int = 3,
    retry_wait: float = 2.0,
    resume: bool = True,
) -> Dict[str, Any]:
    url = build_atlas_trajectory_url(
        dataset=dataset,
        kind=kind,
        pdb_chain=pdb_chain,
        base_url=base_url,
    )
    out_path = Path(out) if out else Path(
        default_atlas_archive_name(dataset=dataset, kind=kind, pdb_chain=pdb_chain)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if retries < 0:
        raise ValueError("retries must be >= 0")
    if retry_wait < 0:
        raise ValueError("retry_wait must be >= 0")
    if timeout <= 0:
        raise ValueError("timeout must be > 0")

    try:
        for attempt in range(1, retries + 2):
            try:
                _download_once(
                    url=url,
                    out_path=out_path,
                    timeout=timeout,
                    resume=resume,
                )
                break
            except HTTPError as exc:
                _raise_http_error(exc, url)
            except URLError as exc:
                message = f"ATLAS API request failed: {exc} [{url}]"
                if attempt > retries:
                    raise AtlasApiError(message) from exc
            except _AtlasRetryableError as exc:
                if attempt > retries:
                    raise AtlasApiError(str(exc)) from exc

            if attempt <= retries:
                time.sleep(retry_wait)
    except AtlasApiError:
        raise
    except Exception as exc:
        raise AtlasApiError(str(exc)) from exc

    n_bytes = out_path.stat().st_size
    return {
        "status": "ok",
        "dataset": _normalize_dataset(dataset),
        "kind": _normalize_kind(kind),
        "pdb_chain": _normalize_pdb_chain(pdb_chain),
        "url": url,
        "out": str(out_path),
        "bytes": n_bytes,
        "sha256": _sha256_file(out_path),
    }
