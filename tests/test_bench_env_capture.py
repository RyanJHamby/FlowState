"""Spec bench-env-capture: every benchmark run records the machine and settings it ran under."""

from __future__ import annotations

import importlib
import json
import subprocess
from importlib import metadata
from pathlib import Path

import pytest

XFAIL = pytest.mark.xfail(
    strict=True, reason="spec:bench-env-capture (remove this marker when implemented)"
)
REPO = Path(__file__).resolve().parents[1]
REQUIRED = [
    "timestamp_utc",
    "os",
    "kernel",
    "cpu_model",
    "logical_cores",
    "memory_gb",
    "python",
    "python_free_threaded",
    "packages",
    "rustc",
    "git_sha",
    "git_dirty",
    "threads",
    "cpu_governor",
    "transparent_hugepages",
    "smt",
    "numa_nodes",
    "aslr",
]


def _e():
    return importlib.import_module("benchmarks.envcapture")


@XFAIL
def test_every_required_field_is_present_and_never_empty():
    env = _e().capture_env()
    for key in REQUIRED:
        assert key in env, f"missing {key}"
        assert env[key] not in (None, ""), f"{key} is empty; use 'unknown' when it cannot be read"
    json.dumps(env)  # must be JSON-serializable


@XFAIL
def test_package_versions_match_installed_distributions():
    pk = _e().capture_env()["packages"]
    for name in ("polars", "pandas", "pyarrow", "numpy"):
        assert pk[name] == metadata.version(name)
    assert pk["flowstate"] not in ("", None)


@XFAIL
def test_unreadable_system_files_are_reported_as_unknown(monkeypatch):
    e = _e()
    monkeypatch.setattr(e, "_read_sys", lambda path: None)
    env = e.capture_env()
    for key in ("cpu_governor", "transparent_hugepages", "smt", "aslr"):
        assert env[key] == "unknown"


@XFAIL
def test_system_file_contents_are_passed_through(monkeypatch):
    e = _e()
    fake = {
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor": "performance",
        "/sys/kernel/mm/transparent_hugepage/enabled": "always [madvise] never",
        "/sys/devices/system/cpu/smt/active": "0",
        "/proc/sys/kernel/randomize_va_space": "0",
    }
    monkeypatch.setattr(e, "_read_sys", lambda path: fake.get(path))
    env = e.capture_env()
    assert env["cpu_governor"] == "performance"
    assert env["transparent_hugepages"] == "madvise"
    assert env["smt"] == "off"
    assert env["aslr"] == "off"


@XFAIL
def test_thread_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    t = _e().capture_env()["threads"]
    assert t["RAYON_NUM_THREADS"] == "4"
    assert t["POLARS_MAX_THREADS"] == "unset"


@XFAIL
def test_git_fields_describe_this_checkout():
    env = _e().capture_env()
    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert env["git_sha"] == sha
    assert isinstance(env["git_dirty"], bool)


@XFAIL
def test_write_env_round_trips(tmp_path):
    e = _e()
    path = tmp_path / "env.json"
    e.write_env(path)
    assert json.loads(path.read_text())["git_sha"] == e.capture_env()["git_sha"]
