#!/usr/bin/env python3
"""Pin the local test-case snapshot used via AIC_COLLECTOR_CASES_ROOT.

Copies ``<repo>/collector/cases/`` (base_ops, models, capabilities.yaml,
denylist.yaml) into ``<out>/collector/cases/`` and writes
``<out>/SNAPSHOT_PROVENANCE.yaml`` (source commit, dirty flag, per-file
sha256). After this, the case set consumed by collect.py / case_generator /
capabilities / provenance is frozen against upstream merges: syncing upstream
code cannot change what the local runs benchmark.

The previous provenance file is archived to ``<out>.archive/<ts>/`` before an
overwrite, so snapshot history stays auditable.

Usage (local dev machine):
    python3 dsv4_xpu/scripts/snapshot_cases.py \
        --repo aisimulate-tina/python/aisimulate
Usage (container, tree synced to /workspace):
    cd /workspace && python3 scripts/snapshot_cases.py

After generating, point the runtime at it (container_env.sh does this):
    export AIC_COLLECTOR_CASES_ROOT=<snapshot parent dir>
The override expects the layout ``<root>/collector/cases/...`` — i.e. the
env var carries the snapshot root's PARENT dir (here ``<out>`` itself is the
root, since ``<out>/collector/cases`` is what gets written).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

CASES_REL = Path("collector") / "cases"
PROVENANCE_NAME = "SNAPSHOT_PROVENANCE.yaml"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(repo: Path) -> tuple[str, bool]:
    """Return (HEAD commit, cases-dir-dirty flag). Unreadable git -> ('unknown', False)."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--", str(CASES_REL)],
            capture_output=True, text=True, check=True,
        ).stdout.strip() != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False
    return commit, dirty


def archive_previous(out: Path) -> str | None:
    prov = out / PROVENANCE_NAME
    if not prov.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out.parent / (out.name + ".archive") / ts
    dest.mkdir(parents=True, exist_ok=True)
    (dest / PROVENANCE_NAME).write_text(prov.read_text())
    return str(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path.cwd(),
                        help="collector source tree (the dir containing collector/cases); default: cwd")
    parser.add_argument("--out", type=Path, default=None,
                        help="snapshot root (gets <out>/collector/cases); default: <script_dir>/../cases_snapshot")
    parser.add_argument("--tag", type=str, default="",
                        help="free-form label recorded in the provenance file (e.g. post-merge-20261011)")
    parser.add_argument("--commit", type=str, default="",
                        help="source commit override — required when the sync target is not a git "
                             "checkout (e.g. container /workspace); record the local repo HEAD here")
    parser.add_argument("--set-yaml", action="append", default=[],
                        metavar="REPO_REL_PATH=LOCAL_FILE",
                        help="inject a local campaign yaml over the snapshot copy AFTER copying "
                             "(repeatable). The provenance sha256 is computed on the injected "
                             "content and the override is recorded under `overrides` — the "
                             "snapshot stays self-describing. Use this to pin the campaign yaml "
                             "without touching the repo tree (iso mode).")
    args = parser.parse_args()

    repo = args.repo.resolve()
    src = repo / CASES_REL
    if not (src / "base_ops").is_dir():
        print(f"ERROR: {src} does not look like a collector/cases dir (missing base_ops/)", file=sys.stderr)
        return 2
    out = (args.out or Path(__file__).resolve().parent.parent / "cases_snapshot").resolve()
    out.mkdir(parents=True, exist_ok=True)

    archived = archive_previous(out)
    dst = out / CASES_REL
    if dst.exists():
        import shutil
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    copied: dict[str, str] = {}
    for path in sorted(src.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        copied[str(CASES_REL / rel)] = sha256_file(path)

    overrides: dict[str, str] = {}
    for spec in args.set_yaml:
        rel_path, _, local = spec.partition("=")
        if not rel_path or not local:
            print(f"ERROR: --set-yaml expects REPO_REL_PATH=LOCAL_FILE, got '{spec}'", file=sys.stderr)
            return 2
        target = out / rel_path  # rel_path is repo-relative; CASES_REL is already in it
        src_file = Path(local)
        if not src_file.is_file():
            print(f"ERROR: --set-yaml source file not found: {local}", file=sys.stderr)
            return 2
        if not target.is_file():
            print(f"ERROR: --set-yaml target not in snapshot: {rel_path}", file=sys.stderr)
            return 2
        target.write_bytes(src_file.read_bytes())
        provenance_rel = rel_path
        copied[provenance_rel] = sha256_file(target)
        overrides[provenance_rel] = str(src_file.resolve())

    commit, dirty = git_commit(repo)
    if args.commit:
        commit, dirty = args.commit, False
    provenance = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "source_repo": str(repo),
        "source_commit": commit,
        "cases_dirty_at_snapshot": dirty,
        "previous_provenance_archived_to": archived,
        "file_count": len(copied),
        "files_sha256": copied,
        "overrides": overrides,
        "pin_env": f"AIC_COLLECTOR_CASES_ROOT={out}",
    }
    (out / PROVENANCE_NAME).write_text(yaml.safe_dump(provenance, sort_keys=False))

    print(f"snapshot: {len(copied)} files -> {dst}")
    print(f"source:   {repo} @ {commit}{' (cases dirty)' if dirty else ''}")
    print(f"pin:      export AIC_COLLECTOR_CASES_ROOT={out}")
    if dirty:
        print("WARNING: collector/cases had uncommitted changes at snapshot time", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
