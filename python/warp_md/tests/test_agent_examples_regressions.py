"""Regression tests for examples/agents utilities."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType


def _load_example_module(module_name: str, rel_path: str) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_stream_events_strips_discriminator_before_dataclass_unpack():
    warp_utils = _load_example_module("warp_utils_example_a", "examples/agents/warp_utils.py")

    class Recorder(warp_utils.WarpPackEventHandler):
        def __init__(self) -> None:
            self.started = 0
            self.completed = 0

        def on_pack_started(self, event) -> None:
            self.started += 1

        def on_pack_complete(self, event) -> None:
            self.completed += 1

    class MockStderr:
        def __init__(self):
            self._lines = [
                b'{"event":"pack_started","total_molecules":1,"box_size":[10,10,10],"box_origin":[0,0,0],"output_path":"out.pdb"}\n',
                b'{"event":"pack_complete","total_atoms":3,"total_molecules":1,"final_box_size":[10,10,10],"output_path":"out.pdb","elapsed_ms":12,"profile_ms":{"templates":1,"place_core":2,"movebad":3,"gencan":4,"relax":5}}\n',
            ]

        def readline(self):
            return self._lines.pop(0) if self._lines else b""

    class MockProc:
        def __init__(self):
            self.stderr = MockStderr()

        def wait(self):
            return 0

    handler = Recorder()
    result = warp_utils.parse_stream_events(MockProc(), handler)

    assert result is not None
    assert result["event"] == "pack_complete"
    assert handler.started == 1
    assert handler.completed == 1


def test_progress_tracker_satisfies_handler_type_checks():
    warp_utils = _load_example_module("warp_utils_example_b", "examples/agents/warp_utils.py")
    tracker = warp_utils.ProgressTracker(verbose=False)
    assert isinstance(tracker, warp_utils.WarpPackEventHandler)
    assert isinstance(tracker, warp_utils.WarpPepEventHandler)


def test_run_pack_supports_dict_and_in_memory_packconfig(monkeypatch):
    warp_pack_wrapper = _load_example_module(
        "warp_pack_wrapper_example", "examples/agents/warp_pack_wrapper.py"
    )

    staged_config_paths: list[Path] = []

    class FakeProc:
        def __init__(self):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO(
                '{"event":"pack_complete","total_atoms":1,"total_molecules":1,"final_box_size":[1,1,1],"output_path":"out.pdb","elapsed_ms":1,"profile_ms":{}}\n'
            )
            self.returncode = 0

        def wait(self):
            return 0

        def communicate(self):
            return ("", "")

    def fake_popen(cmd, **kwargs):
        cfg_idx = cmd.index("--config") + 1
        cfg_path = Path(cmd[cfg_idx])
        staged_config_paths.append(cfg_path)
        assert cfg_path.exists(), f"expected staged config at {cfg_path}"
        return FakeProc()

    monkeypatch.setattr(warp_pack_wrapper.subprocess, "Popen", fake_popen)

    dict_result = warp_pack_wrapper.run_pack(
        {"box": {"size": [10.0, 10.0, 10.0]}, "structures": [{"path": "mol.pdb", "count": 1}]},
        output="dict_out.pdb",
        stream=True,
    )
    cfg = warp_pack_wrapper.PackConfig(
        structures=[{"path": "mol.pdb", "count": 1}],
        box_size=(10.0, 10.0, 10.0),
    )
    in_memory_result = warp_pack_wrapper.run_pack(cfg, output="cfg_out.pdb", stream=True)

    assert dict_result.success
    assert in_memory_result.success
    assert len(staged_config_paths) == 2
    for staged in staged_config_paths:
        assert not staged.exists(), f"staged config should be cleaned up: {staged}"
