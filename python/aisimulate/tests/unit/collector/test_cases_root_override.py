# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AIC_COLLECTOR_CASES_ROOT pinned-case-snapshot override.

All four consumers read the env var at import time (case_generator,
capabilities, model_cases, provenance), so each test spawns a subprocess with
the env fixed rather than reloading modules in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_ROOT = REPO_ROOT / "collector"


def _snapshot(tmp_path: Path, mutate: Path | None = None) -> Path:
    """Build a minimal snapshot root; optionally append a marker line to one yaml."""
    snap = tmp_path / "cases_snapshot"
    cases = snap / "collector" / "cases"
    (cases / "base_ops").mkdir(parents=True)
    (cases / "models").mkdir()
    for src in sorted((COLLECTOR_ROOT / "cases" / "base_ops").glob("*.yaml")):
        (cases / "base_ops" / src.name).write_bytes(src.read_bytes())
    for src in sorted((COLLECTOR_ROOT / "cases" / "models").glob("*.yaml")):
        (cases / "models" / src.name).write_bytes(src.read_bytes())
    if mutate is not None:
        with mutate.open("a") as fh:
            fh.write("\n# snapshot divergence marker\n")
    return snap


def _run(env_extra: dict[str, str], code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}", **env_extra},
    )


def _dirs_code():
    return (
        "import json;"
        "from collector import case_generator, capabilities, model_cases;"
        "print(json.dumps({"
        "'case_generator_base': str(case_generator.BASE_OP_CASES_DIR),"
        "'case_generator_models': str(case_generator.MODEL_CASES_DIR),"
        "'capabilities': str(capabilities._CASES_DIR),"
        "'model_cases': str(model_cases.CASE_ROOT)}))"
    )


def test_no_override_reads_repo(tmp_path):
    proc = _run({}, _dirs_code())
    assert proc.returncode == 0, proc.stderr
    dirs = json.loads(proc.stdout)
    repo_cases = str(COLLECTOR_ROOT / "cases")
    assert dirs["case_generator_base"] == repo_cases + "/base_ops"
    assert dirs["capabilities"] == repo_cases
    assert dirs["model_cases"] == repo_cases


def test_override_redirects_all_consumers(tmp_path):
    snap = _snapshot(tmp_path)
    proc = _run({"AIC_COLLECTOR_CASES_ROOT": str(snap)}, _dirs_code())
    assert proc.returncode == 0, proc.stderr
    dirs = json.loads(proc.stdout)
    assert dirs["case_generator_base"] == str(snap / "collector/cases/base_ops")
    assert dirs["case_generator_models"] == str(snap / "collector/cases/models")
    assert dirs["capabilities"] == str(snap / "collector/cases")
    assert dirs["model_cases"] == str(snap / "collector/cases")


def test_provenance_hash_tracks_snapshot_content(tmp_path):
    """Recorded hash must equal the hash of the files actually consumed."""
    snap = _snapshot(tmp_path)
    code = (
        "from collector.provenance import collector_hash, load_closures;"
        "closures = load_closures('collector/hash_closures.yaml');"
        "print(collector_hash('collector.vllm.collect_dsv4_attn_xpu', '.', closures))"
    )
    repo_proc = _run({}, code)
    assert repo_proc.returncode == 0, repo_proc.stderr
    h_repo = repo_proc.stdout.strip()

    snap_proc = _run({"AIC_COLLECTOR_CASES_ROOT": str(snap)}, code)
    assert snap_proc.returncode == 0, snap_proc.stderr
    h_snap = snap_proc.stdout.strip()
    # Snapshot copied the repo unmutated -> identical hash (key relpaths unchanged).
    assert h_repo == h_snap

    # Mutate a case yaml in the snapshot -> hash must diverge (it is what got hashed).
    mutate = snap / "collector/cases/base_ops/dsv4_attention.yaml"
    with mutate.open("a") as fh:
        fh.write("\n# snapshot divergence marker\n")
    h_mut = _run({"AIC_COLLECTOR_CASES_ROOT": str(snap)}, code).stdout.strip()
    assert h_mut != h_repo


def test_capabilities_reads_snapshot_denylist(tmp_path):
    snap = _snapshot(tmp_path)
    denylist = snap / "collector/cases/denylist.yaml"
    denylist.write_text(
        "entries:\n"
        "  - contains: __no_such_case_marker__\n"
        "    reason: selftest\n"
        "    added: 2026-10-11\n"
    )
    code = (
        "from collector.capabilities import _load_denylist;"
        "print([e[0] for e in _load_denylist()])"
    )
    snap_proc = _run({"AIC_COLLECTOR_CASES_ROOT": str(snap)}, code)
    assert snap_proc.returncode == 0, snap_proc.stderr
    assert "__no_such_case_marker__" in snap_proc.stdout
    # Repo side unaffected (repo denylist has no such marker).
    repo_proc = _run({}, code)
    assert repo_proc.returncode == 0, repo_proc.stderr
    assert "__no_such_case_marker__" not in repo_proc.stdout
