import io
import hashlib

import pytest

from warp_md import atlas_api


class _FakeResponse:
    def __init__(self, body: bytes, *, headers=None, status: int = 200):
        self._stream = io.BytesIO(body)
        self.headers = headers or {}
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_build_atlas_trajectory_url() -> None:
    url = atlas_api.build_atlas_trajectory_url(
        dataset="DPF",
        kind="total",
        pdb_chain="1bzy_A",
    )
    assert (
        url
        == "https://www.dsimb.inserm.fr/ATLAS/api/DPF/total/1bzy_A"
    )


def test_default_atlas_archive_name() -> None:
    assert (
        atlas_api.default_atlas_archive_name(
            dataset="ATLAS",
            kind="protein",
            pdb_chain="16pk_A",
        )
        == "16pk_A_protein.zip"
    )
    assert (
        atlas_api.default_atlas_archive_name(
            dataset="chameleon",
            kind="analysis",
            pdb_chain="1bcp_F",
        )
        == "1bcp_F_chameleon_analysis.zip"
    )


def test_download_atlas_trajectory_success(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "16pk_A_total.zip"
    payload = b"PK\x03\x04test-archive"

    def _fake_urlopen(request, timeout):
        assert request.full_url.endswith("/ATLAS/total/16pk_A")
        assert timeout == 12.5
        return _FakeResponse(payload, headers={"Content-Type": "application/octet-stream"})

    monkeypatch.setattr(atlas_api, "urlopen", _fake_urlopen)
    result = atlas_api.download_atlas_trajectory(
        dataset="ATLAS",
        kind="total",
        pdb_chain="16pk_A",
        out=str(out_path),
        timeout=12.5,
    )

    assert out_path.exists()
    assert out_path.read_bytes() == payload
    assert result["status"] == "ok"
    assert result["bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_download_atlas_trajectory_http_error_message(monkeypatch) -> None:
    url = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/total/missing_A"

    def _fake_urlopen(request, timeout):
        raise atlas_api.HTTPError(
            url=url,
            code=404,
            msg="not found",
            hdrs={"Content-Type": "application/json"},
            fp=io.BytesIO(b'{"message":"The zip file was not found"}'),
        )

    monkeypatch.setattr(atlas_api, "urlopen", _fake_urlopen)
    with pytest.raises(atlas_api.AtlasApiError, match="404"):
        atlas_api.download_atlas_trajectory(
            dataset="ATLAS",
            kind="total",
            pdb_chain="missing_A",
        )


def test_download_atlas_trajectory_retry_then_success(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "retry.zip"
    payload = b"PK\x03\x04retry"
    calls = {"n": 0}

    def _fake_urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise atlas_api.URLError("temporary failure")
        return _FakeResponse(payload, headers={"Content-Type": "application/octet-stream"})

    monkeypatch.setattr(atlas_api, "urlopen", _fake_urlopen)
    result = atlas_api.download_atlas_trajectory(
        dataset="ATLAS",
        kind="analysis",
        pdb_chain="16pk_A",
        out=str(out_path),
        retries=2,
        retry_wait=0.0,
    )

    assert calls["n"] == 2
    assert out_path.read_bytes() == payload
    assert result["bytes"] == len(payload)


def test_download_atlas_trajectory_resume_part_file(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "resume.zip"
    part_path = tmp_path / "resume.zip.part"
    part_path.write_bytes(b"hello")

    def _fake_urlopen(request, timeout):
        assert request.headers.get("Range") == "bytes=5-"
        return _FakeResponse(
            b"world",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": "5",
            },
            status=206,
        )

    monkeypatch.setattr(atlas_api, "urlopen", _fake_urlopen)
    result = atlas_api.download_atlas_trajectory(
        dataset="ATLAS",
        kind="total",
        pdb_chain="16pk_A",
        out=str(out_path),
        retries=0,
        retry_wait=0.0,
        resume=True,
    )

    assert out_path.read_bytes() == b"helloworld"
    assert not part_path.exists()
    assert result["bytes"] == 10


def test_download_atlas_trajectory_incomplete_content_length(tmp_path, monkeypatch) -> None:
    out_path = tmp_path / "bad.zip"

    def _fake_urlopen(request, timeout):
        return _FakeResponse(
            b"abc",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": "10",
            },
        )

    monkeypatch.setattr(atlas_api, "urlopen", _fake_urlopen)
    with pytest.raises(atlas_api.AtlasApiError, match="incomplete download payload"):
        atlas_api.download_atlas_trajectory(
            dataset="ATLAS",
            kind="analysis",
            pdb_chain="16pk_A",
            out=str(out_path),
            retries=0,
            retry_wait=0.0,
        )
