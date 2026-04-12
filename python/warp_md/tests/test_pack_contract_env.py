from __future__ import annotations

from pathlib import Path

from warp_md import pack_contract


def test_configure_native_pack_data_env(monkeypatch) -> None:
    monkeypatch.delenv("WARP_MD_PACK_DATA_DIR", raising=False)
    monkeypatch.delenv("WARP_MD_ION_REGISTRY", raising=False)
    monkeypatch.delenv("WARP_MD_SALT_REGISTRY", raising=False)

    pack_contract._configure_native_pack_data_env()

    data_dir = Path(pack_contract.os.environ["WARP_MD_PACK_DATA_DIR"])
    assert data_dir.exists()
    assert Path(pack_contract.os.environ["WARP_MD_ION_REGISTRY"]).exists()
    assert Path(pack_contract.os.environ["WARP_MD_SALT_REGISTRY"]).exists()
