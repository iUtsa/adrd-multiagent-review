"""Freeze a configuration before the final test-set runs (task 8.3).

`create` records the SHA-256 of everything that shapes the agents' answers:
the run config, codebook, agent code (prompts live there), gold file and split.
The runner refuses to run on the test set unless the current files still match
a frozen manifest, so nothing can be tuned after the freeze without it showing.

    python -m adrd_review.freeze create configs/runs/E4_gemini.yaml
    python -m adrd_review.freeze check  configs/runs/E4_gemini.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

FROZEN_DIR = Path("configs/frozen")
REPO = Path(__file__).resolve().parents[2]
PROMPT_SOURCES = ["src/adrd_review/agents", "src/adrd_review/schemas.py", "src/adrd_review/consensus.py"]


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompts_sha() -> str:
    """One hash over every file that defines prompts, answer forms or vote-combining rules."""
    h = hashlib.sha256()
    files = []
    for src in PROMPT_SOURCES:
        p = REPO / src
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    for f in files:
        h.update(f.relative_to(REPO).as_posix().encode())  # same hash on any machine / working folder
        h.update(f.read_bytes())
    return h.hexdigest()


def fingerprint(config_path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    codebook = cfg.get("codebook", "configs/codebook.yaml")
    gold = cfg.get("gold", "data/gold/gold.csv")
    split = cfg.get("split", "configs/splits/placeholder.json")
    return {
        "config_file": str(config_path),
        "config_sha256": _sha(config_path),
        "codebook_sha256": _sha(codebook),
        "prompts_sha256": prompts_sha(),
        "gold_sha256": _sha(gold),
        "split_sha256": _sha(split),
    }


def manifest_path(config_path: str | Path) -> Path:
    return FROZEN_DIR / f"{Path(config_path).stem}.json"


def create(config_path: str | Path) -> Path:
    path = manifest_path(config_path)
    if path.exists():
        raise FileExistsError(f"{path} exists. Delete it deliberately if you really mean to re-freeze.")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest = fingerprint(config_path) | {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def check(config_path: str | Path) -> list[str]:
    """Differences between the current files and the frozen manifest ([] = still frozen)."""
    path = manifest_path(config_path)
    if not path.exists():
        return [f"no frozen manifest at {path}; run: python -m adrd_review.freeze create {config_path}"]
    frozen = json.loads(path.read_text())
    now = fingerprint(config_path)
    return [f"{k} changed since the freeze" for k in now if k.endswith("sha256") and now[k] != frozen.get(k)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m adrd_review.freeze")
    ap.add_argument("action", choices=["create", "check"])
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args(argv)
    status = 0
    for c in args.configs:
        if args.action == "create":
            print(f"frozen: {create(c)}")
        else:
            problems = check(c)
            print(f"{c}: " + ("still frozen" if not problems else "NOT frozen: " + "; ".join(problems)))
            status |= bool(problems)
    return status


if __name__ == "__main__":
    sys.exit(main())
