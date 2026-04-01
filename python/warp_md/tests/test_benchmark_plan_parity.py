from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dataset_paths(dataset_id: str) -> tuple[Path, Path]:
    root = _repo_root()
    manifest_path = root / "internal" / "benchmark" / "paper_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", [])
    row = next((item for item in datasets if item.get("id") == dataset_id), None)
    if row is None:
        raise AssertionError(f"dataset id missing in manifest: {dataset_id}")
    top = (root / str(row["topology"])).resolve()
    traj = (root / str(row["trajectory"])).resolve()
    return top, traj


def _pick_available_dataset(candidates: list[str]) -> tuple[str, Path, Path]:
    for dataset_id in candidates:
        top, traj = _dataset_paths(dataset_id)
        if top.exists() and traj.exists():
            return dataset_id, top, traj
    details = []
    for dataset_id in candidates:
        top, traj = _dataset_paths(dataset_id)
        details.append(f"{dataset_id}: top={top.exists()} traj={traj.exists()}")
    raise AssertionError("no candidate benchmark dataset present: " + "; ".join(details))


def _run_script(script_rel: str, args: list[str], out_json: Path) -> dict:
    root = _repo_root()
    script = root / script_rel
    cmd = [sys.executable, str(script), *args, "--json-out", str(out_json)]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"benchmark script failed: {script_rel}\n"
            f"exit={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return json.loads(out_json.read_text(encoding="utf-8"))


def _assert_common_metric_contract(metric_payload: dict, expected_baselines: tuple[str, ...]) -> None:
    assert str(metric_payload.get("unit", "")).strip()
    assert int(metric_payload.get("n_frames", metric_payload.get("n_lags", 0))) > 0

    speed = metric_payload.get("speed_fps", {})
    sec_mean = metric_payload.get("speed_sec_mean", {})
    errors = metric_payload.get("error_vs_warp", {})

    assert float(speed.get("warp", 0.0)) > 0.0
    assert float(sec_mean.get("warp", 0.0)) > 0.0

    for baseline in expected_baselines:
        assert baseline in speed, f"missing speed baseline: {baseline}"
        assert baseline in sec_mean, f"missing sec_mean baseline: {baseline}"
        assert baseline in errors, f"missing error baseline: {baseline}"
        assert float(speed[baseline]) > 0.0
        assert float(sec_mean[baseline]) > 0.0
        err = errors[baseline]
        assert int(err.get("n", 0)) > 0
        assert float(err.get("rmse", 0.0)) >= 0.0
        assert float(err.get("mae", 0.0)) >= 0.0


def _assert_metric_has_reference_parity(metric_payload: dict) -> None:
    speed = metric_payload.get("speed_fps", {})
    sec_mean = metric_payload.get("speed_sec_mean", {})
    assert float(speed.get("warp", 0.0)) > 0.0
    assert float(sec_mean.get("warp", 0.0)) > 0.0
    assert float(speed.get("mdanalysis", 0.0)) > 0.0
    assert float(speed.get("mdtraj", 0.0)) > 0.0
    assert float(sec_mean.get("mdanalysis", 0.0)) > 0.0
    assert float(sec_mean.get("mdtraj", 0.0)) > 0.0

    has_parity_block = any(
        key in metric_payload
        for key in (
            "error_vs_warp",
            "count_error_vs_warp",
            "autocorr_error_vs_warp",
            "scalar_abs_error_vs_warp",
        )
    )
    assert has_parity_block, "metric payload lacks parity error block"

    parity_block = None
    for key in (
        "error_vs_warp",
        "count_error_vs_warp",
        "autocorr_error_vs_warp",
        "scalar_abs_error_vs_warp",
    ):
        value = metric_payload.get(key)
        if isinstance(value, dict) and value:
            parity_block = value
            break
    assert isinstance(parity_block, dict) and parity_block, "parity block is empty or invalid"

    for baseline in ("mdanalysis", "mdtraj"):
        assert baseline in parity_block, f"missing baseline in parity block: {baseline}"
        detail = parity_block[baseline]
        assert isinstance(detail, dict), f"invalid parity detail for baseline {baseline}"
        assert int(detail.get("n", 0)) > 0, f"non-positive parity sample count for {baseline}"
        assert float(detail.get("rmse", 0.0)) >= 0.0, f"invalid rmse for {baseline}"
        assert float(detail.get("mae", 0.0)) >= 0.0, f"invalid mae for {baseline}"


def _planned_jobs_for_root(output_root: Path) -> list[dict]:
    root = _repo_root()
    plan_out = output_root / "manifest_plan.json"
    cmd = [
        sys.executable,
        str(root / "scripts" / "bench" / "run_manifest_benchmarks.py"),
        "--manifest",
        "internal/benchmark/paper_manifest.json",
        "--output-root",
        str(output_root),
        "--repeats",
        "1",
        "--max-lag",
        "32",
        "--plan-out",
        str(plan_out),
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            "manifest planning failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    plan = json.loads(plan_out.read_text(encoding="utf-8"))
    jobs = [row for row in plan.get("jobs", []) if isinstance(row, dict)]
    return jobs


def test_structural_benchmark_outputs_real_parity_contract(tmp_path: Path) -> None:
    dataset_id, top, traj = _pick_available_dataset(["alanine_xtc", "ubiquitin_dcd_q75"])
    out_json = tmp_path / "structural.json"
    payload = _run_script(
        "scripts/bench/benchmark_structural_metrics.py",
        [
            "--top",
            str(top),
            "--traj",
            str(traj),
            "--repeats",
            "1",
            "--timing-summary",
            "mean",
            "--warmup-repeats",
            "0",
        ],
        out_json,
    )

    assert payload["dataset"]["top"] == str(top)
    assert payload["dataset"]["traj"] == str(traj)
    metrics = payload.get("metrics", {})
    for metric in ("rg", "rmsd", "e2e"):
        assert metric in metrics, f"missing structural metric {metric} for dataset {dataset_id}"
        _assert_common_metric_contract(metrics[metric], expected_baselines=("mdanalysis", "mdtraj"))


def test_transport_benchmark_outputs_real_parity_contract(tmp_path: Path) -> None:
    _, top, traj = _pick_available_dataset(["water_xtc", "water_dcd"])
    out_json = tmp_path / "transport.json"
    payload = _run_script(
        "scripts/bench/benchmark_transport_metrics.py",
        [
            "--top",
            str(top),
            "--traj",
            str(traj),
            "--repeats",
            "1",
            "--max-lag",
            "32",
            "--timing-summary",
            "mean",
            "--warmup-repeats",
            "0",
        ],
        out_json,
    )

    metrics = payload.get("metrics", {})
    assert "msd" in metrics
    _assert_common_metric_contract(metrics["msd"], expected_baselines=("mdanalysis", "mdtraj"))

    conductivity = metrics.get("conductivity", {})
    if not conductivity.get("skipped", False):
        _assert_common_metric_contract(conductivity, expected_baselines=("mdanalysis", "mdtraj"))
    else:
        reason = str(conductivity.get("reason", "")).strip()
        assert reason, "conductivity skipped but missing reason"

    assert "transference" in metrics, "transport result missing transference metric contract"
    transference = metrics.get("transference", {})
    if transference.get("skipped", False):
        reason = str(transference.get("reason", "")).strip()
        assert reason, "transference skipped but missing reason"
    else:
        _assert_common_metric_contract(transference, expected_baselines=("mdanalysis", "mdtraj"))


@pytest.mark.parametrize(
    "family,required_metrics",
    [
        ("structural", ("rg", "rmsd", "e2e")),
        ("transport", ("msd",)),
    ],
)
def test_manifest_family_job_executes_and_emits_metrics_contract(
    tmp_path: Path,
    family: str,
    required_metrics: tuple[str, ...],
) -> None:
    root = _repo_root()
    manifest_path = root / "internal" / "benchmark" / "paper_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    family_benchmarks = [
        row
        for row in manifest.get("benchmarks", [])
        if isinstance(row, dict) and row.get("family") == family and bool(row.get("enabled", True))
    ]
    picked = None
    for bench in family_benchmarks:
        dataset_id = str(bench.get("dataset_id", ""))
        top, traj = _dataset_paths(dataset_id)
        if top.exists() and traj.exists():
            picked = bench
            break
    if picked is None:
        pytest.skip(f"no runnable dataset found for family={family}")

    dataset_id = str(picked["dataset_id"])
    lane = str(picked.get("lane", "cpu_baseline"))
    bench_id = str(picked["id"])

    output_root = tmp_path / "results"
    cmd = [
        sys.executable,
        str(root / "scripts" / "bench" / "run_manifest_benchmarks.py"),
        "--manifest",
        "internal/benchmark/paper_manifest.json",
        "--execute",
        "--stop-on-error",
        "--families",
        family,
        "--benchmark-ids",
        bench_id,
        "--output-root",
        str(output_root),
        "--repeats",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"manifest run failed for family={family}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    result_json = output_root / dataset_id / lane / f"{bench_id}.json"
    assert result_json.exists(), f"missing output json: {result_json}"
    payload = json.loads(result_json.read_text(encoding="utf-8"))

    metrics = payload.get("metrics", {})
    for metric in required_metrics:
        assert metric in metrics, f"missing metric {metric} from family={family} result"
        _assert_common_metric_contract(metrics[metric], expected_baselines=("mdanalysis", "mdtraj"))


def test_manifest_one_benchmark_per_family_parity_contract(tmp_path: Path) -> None:
    if os.environ.get("WARP_MD_FULL_BENCHMARK_PARITY", "0") != "1":
        pytest.skip("set WARP_MD_FULL_BENCHMARK_PARITY=1 to run full family parity benchmark test")

    root = _repo_root()
    output_root = tmp_path / "full_manifest_results"
    jobs = _planned_jobs_for_root(output_root)
    assert jobs, "no manifest jobs planned"

    first_job_by_family: dict[str, dict] = {}
    for job in jobs:
        family = str(job.get("family", "")).strip()
        if not family or family in first_job_by_family:
            continue
        top = (root / str(job.get("topology", ""))).resolve()
        traj = (root / str(job.get("trajectory", ""))).resolve()
        if top.exists() and traj.exists():
            first_job_by_family[family] = job

    assert first_job_by_family, "no family had runnable dataset assets"

    for family, job in sorted(first_job_by_family.items()):
        bench_id = str(job["bench_id"])
        cmd = [
            sys.executable,
            str(root / "scripts" / "bench" / "run_manifest_benchmarks.py"),
            "--manifest",
            "internal/benchmark/paper_manifest.json",
            "--execute",
            "--stop-on-error",
            "--families",
            family,
            "--benchmark-ids",
            bench_id,
            "--output-root",
            str(output_root),
            "--repeats",
            "1",
            "--max-lag",
            "32",
        ]
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise AssertionError(
                f"family benchmark failed: family={family} bench={bench_id}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        dataset_id = str(job["dataset_id"])
        lane = str(job.get("lane", "cpu_baseline"))
        result_json = output_root / dataset_id / lane / f"{bench_id}.json"
        assert result_json.exists(), f"missing output json for family={family}: {result_json}"
        payload = json.loads(result_json.read_text(encoding="utf-8"))

        metrics = payload.get("metrics", {})
        assert isinstance(metrics, dict) and metrics, f"metrics block missing for family={family}"
        for metric_name, metric_payload in metrics.items():
            assert isinstance(metric_payload, dict), f"invalid metric payload {metric_name} for family={family}"
            if metric_payload.get("skipped", False):
                reason = str(metric_payload.get("reason", "")).strip()
                assert reason, f"skipped metric missing reason: family={family} metric={metric_name}"
                continue
            _assert_metric_has_reference_parity(metric_payload)
